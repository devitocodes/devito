from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import singledispatch, cached_property
from itertools import groupby

import numpy as np
import sympy

from devito.exceptions import CompilationError
from devito.finite_differences import EvalDerivative, IndexDerivative, Weights
from devito.ir import (SEQUENTIAL, PARALLEL_IF_PVT, SEPARABLE, Forward,
                       IterationSpace, Interval, Cluster, ExprGeometry, Queue,
                       IntervalGroup, LabeledVector, Vector, normalize_properties,
                       relax_properties, unbounded, minimum, maximum, extrema,
                       vmax, vmin)
from devito.passes.clusters.cse import _cse
from devito.symbolics import (Uxmapper, estimate_cost, search, reuse_if_untouched,
                              uxreplace, sympy_dtype)
from devito.tools import (Stamp, as_mapper, as_tuple, flatten, frozendict,
                          is_integer, generator, split, timed_pass)
from devito.types import (Eq, Symbol, Temp, TempArray, TempFunction,
                          ModuloDimension, CustomDimension, IncrDimension,
                          StencilDimension, Indexed, Hyperplane)
from devito.types.grid import MultiSubDimension

__all__ = ['cire']


@timed_pass(name='cire')
def cire(clusters, mode, sregistry, options, platform):
    """
    Cross-iteration redundancies elimination.

    Parameters
    ----------
    cluster : list of Cluster
        Input Clusters, subject of the optimization pass.
    mode : str
        The transformation mode. Accepted: ['invariants', 'sops'].
        * 'invariants' is for sub-expressions that are invariant w.r.t. one or
          more Dimensions.
        * 'sops' stands for sums-of-products, that is redundancies are searched
          across all expressions in sum-of-product form.
    sregistry : SymbolRegistry
        The symbol registry, to create unique temporary names.
    options : dict
        The optimization options.
        Accepted: ['min-storage', 'cire-maxpar', 'cire-rotate'].
        * 'min-storage': if True, the pass will try to minimize the amount of
          storage introduced for the tensor temporaries. This might also reduce
          the operation count. On the other hand, this might affect fusion and
          therefore data locality. Defaults to False (legacy).
        * 'cire-maxpar': if True, privilege parallelism over working set size,
          that is the pass will try to create as many parallel loops as possible,
          even though this will require more space (Dimensions) for the temporaries.
          Defaults to False.
        * 'cire-rotate': if True, the pass will use modulo indexing for the
          outermost Dimension iterated over by the temporaries. This will sacrifice
          a parallel loop for a reduced working set size. Defaults to False (legacy).
    platform : Platform
        The underlying platform. Used to optimize the shape of the introduced
        tensor symbols.

    Examples
    --------
    1) 'invariants'. Here's an expensive expression invariant w.r.t. `t`

    t0 = (cos(a[x,y,z])*sin(b[x,y,z]))*c[t,x,y,z]

    which after CIRE becomes

    t1[x,y,z] = cos(a[x,y,z])*sin(b[x,y,z])
    t0 = t1[x,y,z]*c[t,x,y,z]

    2) 'sops'. Below we see two expressions in sum-of-product form (in this
    case, the sum degenerates to a single product).

    t0 = 2.0*a[x,y,z]*b[x,y,z]
    t1 = 3.0*a[x,y,z+1]*b[x,y,z+1]

    CIRE detects that these two expressions are actually redundant and rewrites
    them as:

    t2[x,y,z] = a[x,y,z]*b[x,y,z]
    t0 = 2.0*t2[x,y,z]
    t1 = 3.0*t2[x,y,z+1]
    """
    # NOTE: Handle prematurely expanded derivatives -- current default on
    # several backends, but soon to become legacy
    if mode == 'sops':
        if options['expand']:
            mode = 'eval-derivs'
        else:
            mode = 'index-derivs'

    for cls in modes[mode]:
        transformer = cls(sregistry, options, platform)

        clusters = transformer.process(clusters)

    return clusters


class CireTransformer:

    """
    Abstract base class for transformers implementing a CIRE variant.
    """

    def __init__(self, sregistry, options, platform):
        self.sregistry = sregistry
        self.platform = platform
        self.opt_minstorage = options['min-storage']
        self.opt_rotate = options['cire-rotate']
        self.opt_ftemps = options['cire-ftemps']
        self.opt_mingain = options['cire-mingain']
        self.opt_min_dtype = options['scalar-min-type']
        self.opt_multisubdomain = True

    def _aliases_from_clusters(self, clusters, exclude, meta):
        exprs = flatten([c.exprs for c in clusters])

        # [Clusters]_n -> [Schedule]_m
        variants = []
        for mapper in self._generate(exprs, exclude):
            # Clusters -> AliasList
            found = collect(mapper.extracted, meta.ispace, self.opt_minstorage)
            pexprs, aliases = choose(found, exprs, mapper, self.opt_mingain)

            # AliasList -> Schedule
            schedule = lower_aliases(aliases, meta, self.opt_maxpar)

            variants.append(Variant(schedule, pexprs))

        if not variants:
            return []

        # [Schedule]_m -> Schedule (s.t. best memory/flops trade-off)
        schedule, exprs = self._select(variants)

        # Schedule -> Schedule (optimization)
        if self.opt_rotate:
            schedule = optimize_schedule_rotations(schedule, self.sregistry)

        # Schedule -> [Clusters]_k
        processed, subs = lower_schedule(schedule, meta, self.sregistry,
                                         self.opt_ftemps, self.opt_min_dtype)

        # [Clusters]_k -> [Clusters]_k (optimization)
        if self.opt_multisubdomain:
            processed = optimize_clusters_msds(processed)

        # [Clusters]_k -> [Clusters]_{k+n}
        for c in clusters:
            n = len(c.exprs)
            cexprs, exprs = exprs[:n], exprs[n:]

            cexprs = [uxreplace(e, subs) for e in cexprs]

            ispace = c.ispace.augment(schedule.dmapper)
            ispace = ispace.augment(schedule.rmapper)

            processed.append(c.rebuild(exprs=cexprs, ispace=ispace))

        assert len(exprs) == 0

        return processed

    def process(self, clusters):
        raise NotImplementedError

    def _generate(self, exprs, exclude):
        """
        Generate one or more extractions from ``exprs``. An extraction is a
        set of CIRE candidates which may be turned into aliases. Two different
        extractions may contain overlapping sub-expressions and, therefore,
        should be processed and evaluated indipendently. An extraction won't
        contain any of the symbols appearing in ``exclude``.
        """
        raise NotImplementedError

    def _lookup_key(self, c):
        """
        Create a key for the given Cluster. Clusters with same key may be
        processed together in the search for CIRE candidates. Clusters should
        have a different key if they must not be processed together, e.g.,
        when this would lead to violation of data dependencies.
        """
        raise NotImplementedError

    def _select(self, variants):
        """
        Select the best variant out of a set of variants, weighing flops
        and working set.
        """
        raise NotImplementedError


class CireTransformerLegacy(CireTransformer):

    def _do_generate(self, exprs, exclude, cbk_search, cbk_compose=None):
        """
        Carry out the bulk of the work of ``_generate``.
        """
        counter = generator()
        make = lambda: Symbol(name='dummy%d' % counter(), dtype=np.float32)

        if cbk_compose is None:
            cbk_compose = lambda *args: None

        mapper = Uxmapper()
        for e in exprs:
            for i in cbk_search(e):
                if not i.is_commutative:
                    continue

                terms = cbk_compose(i)

                # Make sure we won't break any data dependencies
                if terms:
                    free_symbols = set().union(*[i.free_symbols for i in terms])
                else:
                    free_symbols = i.free_symbols
                if {a.function for a in free_symbols} & exclude:
                    continue

                mapper.add(i, make, terms)

        return mapper


class CireInvariants(CireTransformerLegacy, Queue):

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = True
        self.opt_schedule_strategy = None

    def process(self, clusters):
        return self._process_fatd(clusters, 1, xtracted=[])

    def callback(self, clusters, prefix, xtracted=None):
        if not prefix:
            return clusters
        d = prefix[-1].dim

        # Rule out extractions that would break data dependencies
        exclude = set().union(*[c.scope.writes for c in clusters])

        # Rule out extractions that depend on the Dimension currently investigated,
        # as they clearly wouldn't be invariants
        exclude.add(d)

        key = lambda c: self._lookup_key(c, d)
        processed = list(clusters)
        for ak, group in as_mapper(clusters, key=key).items():
            g = [c for c in group if c.is_dense and c not in xtracted]
            if not g:
                continue

            made = self._aliases_from_clusters(g, exclude, ak)

            if made:
                idx = processed.index(g[0])
                for n, c in enumerate(g, -len(g)):
                    processed[processed.index(c)] = made.pop(n)
                processed[idx:idx] = made

                xtracted.extend(made)

        return processed

    def _lookup_key(self, c, d):
        ispace = c.ispace.reset()
        intervals = c.ispace.intervals.drop(d).reset()
        properties = frozendict({d: relax_properties(v) for d, v in c.properties.items()})

        return AliasKey(ispace, intervals, c.dtype, c.guards, properties)

    def _select(self, variants):
        return pick_best(variants)


class CireInvariantsElementary(CireInvariants):

    def _generate(self, exprs, exclude):
        # E.g., extract `sin(x)` and `sqrt(x)` from `a*sin(x)*sqrt(x)`
        rule = lambda e: e.is_Function or (e.is_Pow and e.exp.is_Number and 0 < e.exp < 1)
        cbk_search = lambda e: search(e, rule, 'all', 'bfs_first_hit')
        basextr = self._do_generate(exprs, exclude, cbk_search)
        if not basextr:
            return
        yield basextr

        # E.g., extract `sin(x)*cos(x)` from `a*sin(x)*cos(x)`
        def cbk_search(expr):
            found, others = split(expr.args, lambda a: a in basextr)
            ret = [expr] if found else []
            for a in others:
                ret.extend(cbk_search(a))
            return ret

        cbk_compose = lambda e: split(e.args, lambda a: a in basextr)[0]
        yield self._do_generate(exprs, exclude, cbk_search, cbk_compose)


class CireInvariantsDivs(CireInvariants):

    def _generate(self, exprs, exclude):
        # E.g., extract `1/h_x`
        rule = lambda e: e.is_Pow and (not e.exp.is_Number or e.exp < 0)
        cbk_search = lambda e: search(e, rule, 'all', 'bfs_first_hit')
        yield self._do_generate(exprs, exclude, cbk_search)


class CireDerivatives(CireTransformerLegacy):

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.opt_maxpar = options['cire-maxpar']
        self.opt_schedule_strategy = options['cire-schedule']
        self.opt_multisubdomain = False

    def process(self, clusters):
        processed = []
        for c in clusters:
            if not c.is_dense:
                processed.append(c)
                continue

            # Rule out Dimension-independent dependencies, e.g.:
            # r0 = ...
            # u[x, y] = ... r0*a[x, y] ...
            exclude = {i.source.access for i in c.scope.d_flow.independent()}

            # TODO: to process third- and higher-order derivatives, we could
            # extend this by calling `_aliases_from_clusters` repeatedly until
            # `made` is empty. To be investigated
            made = self._aliases_from_clusters([c], exclude, self._lookup_key(c))

            processed.extend(flatten(made) or [c])

        return processed

    def _generate(self, exprs, exclude):
        # E.g., extract `u.dx*a*b` and `u.dx*a*c` from
        # `[(u.dx*a*b).dy`, `(u.dx*a*c).dy]`
        basextr = self._do_generate(exprs, exclude, self._cbk_search,
                                    self._cbk_compose)
        if not basextr:
            return
        yield basextr

        # E.g., extract `u.dx*a` from `[(u.dx*a*b).dy, (u.dx*a*c).dy]`
        # I.e., attempt extracting the largest common derivative-induced subexprs
        mappers = [deindexify(e) for e in basextr.extracted]
        counter = Counter(flatten(m.keys() for m in mappers))
        groups = as_mapper(counter, key=counter.get)
        grank = {k: sorted(v, key=lambda e: estimate_cost(e), reverse=True)
                 for k, v in groups.items()}

        candidates = sorted(grank, reverse=True)[:2]
        for i in candidates:
            lower_pri_elems = flatten([grank[j] for j in candidates if j != i])
            cbk_search = lambda e: self._cbk_search2(e, grank[i] + lower_pri_elems)
            yield self._do_generate(exprs, exclude, cbk_search, self._cbk_compose)

    def _lookup_key(self, c):
        return AliasKey(c.ispace, None, c.dtype, c.guards, c.properties)

    def _select(self, variants):
        if isinstance(self.opt_schedule_strategy, int):
            try:
                return variants[self.opt_schedule_strategy]
            except IndexError:
                raise CompilationError(
                    f"Illegal schedule {self.opt_schedule_strategy}; "
                    f"generated {len(variants)} schedules in total"
                )

        return pick_best(variants)


class CireEvalDerivatives(CireDerivatives):

    def _cbk_compose(self, expr):
        return split_coeff(expr)[1]

    def _cbk_search(self, expr):
        if isinstance(expr, EvalDerivative) and not expr.base.is_Function:
            return expr.args
        else:
            return flatten(e for e in [self._cbk_search(a) for a in expr.args] if e)

    def _cbk_search2(self, expr, rank):
        ret = []
        for e in self._cbk_search(expr):
            mapper = deindexify(e)
            for i in rank:
                if i in mapper:
                    ret.extend(mapper[i])
                    break
        return ret


class CireIndexDerivatives(CireDerivatives):

    def _cbk_compose(self, expr):
        if expr.is_Pow:
            return (expr,)
        terms = []
        for a in expr.args:
            try:
                if not isinstance(a.function, Weights):
                    terms.append(a)
            except AttributeError:
                terms.append(a)
        return tuple(terms)

    def _cbk_search(self, expr):
        if isinstance(expr, IndexDerivative):
            return (expr.expr,)
        else:
            return flatten(e for e in [self._cbk_search(a) for a in expr.args] if e)

    def _cbk_search2(self, expr, rank):
        ret = []
        for e in self._cbk_search(expr):
            mapper = deindexify(e)
            for i in rank:
                found = [v.expr if isinstance(v, IndexDerivative) else v
                         for v in mapper.get(i, [])]
                if found:
                    ret.extend(found)
                    break
        return ret


# Subpass mapper
modes = {
    'invariants': [CireInvariantsElementary,
                   CireInvariantsDivs],
    'eval-derivs': [CireEvalDerivatives],   # NOTE: legacy pass
    'index-derivs': [CireIndexDerivatives],
}


def collect(extracted, ispace, minstorage):
    """
    Find groups of aliasing expressions.

    We shall introduce the following (loose) terminology:

        * A ``terminal`` is the leaf of a mathematical operation. Terminals
          can be numbers (n), literals (l), or Indexeds (I).
        * ``R`` is the relaxation operator := ``R(n) = n``, ``R(l) = l``,
          ``R(I) = J``, where ``J`` has the same base as ``I`` but with all
          offsets stripped away. For example, ``R(a[i+2,j-1]) = a[i,j]``.
        * A ``relaxed expression`` is an expression in which all of the
          terminals are relaxed.

    Now we define the concept of aliasing. We say that an expression A
    aliases an expression B if:

        * ``R(A) == R(B)``
        * all pairwise Indexeds in A and B access memory locations at a
          fixed constant distance along each Dimension.

    For example, consider the following expressions:

        * a[i+1] + b[i+1]
        * a[i+1] + b[j+1]
        * a[i] + c[i]
        * a[i+2] - b[i+2]
        * a[i+2] + b[i]
        * a[i-1] + b[i-1]

    Out of the expressions above, the following alias to `a[i] + b[i]`:

        * a[i+1] + b[i+1] : same operands and operations, distance along i: 1
        * a[i-1] + b[i-1] : same operands and operations, distance along i: -1

    Whereas the following do not:

        * a[i+1] + b[j+1] : because at least one index differs
        * a[i] + c[i] : because at least one of the operands differs
        * a[i+2] - b[i+2] : because at least one operation differs
        * a[i+2] + b[i] : because the distances along ``i`` differ (+2 and +0)
    """
    # Find the potential aliases
    found = []
    for expr in extracted:
        assert not expr.is_Equality

        eg = ExprGeometry(expr)
        if eg.is_regular:
            found.append(eg)

    # Create groups of aliasing expressions
    mapper = OrderedDict()
    unseen = list(found)
    while unseen:
        c = unseen.pop(0)
        group = [c]
        for u in list(unseen):
            # Is `c` translated w.r.t. `u` ?
            if not c.translated(u):
                continue

            group.append(u)
            unseen.remove(u)
        group = Group(group, ispace=ispace)

        if minstorage:
            k = group.dimensions_translated
        else:
            k = group.dimensions
        mapper.setdefault(k, []).append(group)

    aliases = AliasList()
    queue = list(mapper.values())
    while queue:
        groups = queue.pop(0)

        while groups:
            # For each Dimension, determine the Minimum Intervals (MI) spanning
            # all of the Groups diameters
            # Example: x's largest_diameter=2  => [x[-2,0], x[-1,1], x[0,2]]
            # Note: Groups that cannot evaluate their diameter are dropped
            mapper = defaultdict(int)
            for g in list(groups):
                try:
                    mapper.update({d: max(mapper[d], v) for d, v in g.diameter.items()})
                except ValueError:
                    groups.remove(g)
            intervalss = {d: make_rotations_table(d, v) for d, v in mapper.items()}

            # For each Group, find a rotation that is compatible with a given MI
            mapper = {}
            for d, intervals in intervalss.items():
                # Not all groups may access all dimensions
                # Example: `d=t` and groups=[Group(...[t, x]...), Group(...[time, x]...)]
                impacted = [g for g in groups if d in g.dimensions]

                for interval in list(intervals):
                    found = {g: g.find_rotation_distance(d, interval) for g in impacted}
                    if all(distance is not None for distance in found.values()):
                        # `interval` is OK !
                        mapper[interval] = found
                        break

            if len(mapper) == len(intervalss):
                break

            # Try again with fewer groups
            # Heuristic: first try retaining the larger ones
            smallest = len(min(groups, key=len))
            fallback = groups
            groups, remainder = split(groups, lambda g: len(g) > smallest)
            if groups:
                queue.append(remainder)
            elif len(remainder) > 1:
                # No luck with the heuristic, e.g. there are two groups
                # and both have same `len`
                queue.append(fallback[1:])
                groups = [fallback.pop(0)]
            else:
                break

        for g in groups:
            c = g.pivot
            distances = defaultdict(int, [(i.dim, v.get(g)) for i, v in mapper.items()])

            # Create the basis alias
            offsets = [LabeledVector([(l, v[l] + distances[l]) for l in v.labels])
                       for v in c.offsets]
            subs = {i: i.function[[l + v.fromlabel(l, 0) for l in b]]
                    for i, b, v in zip(c.indexeds, c.bases, offsets)}
            pivot = uxreplace(c.expr, subs)

            # Distance of each aliased expression from the basis alias
            aliaseds = []
            distances = []
            for i in g._items:
                aliaseds.append(extracted[i.expr])

                distance = [o.distance(v) for o, v in zip(i.offsets, offsets)]
                distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]
                distances.append(LabeledVector([(d, v.pop()) for d, v in distance]))

            # Compute the alias score
            na = g.naliases
            nr = nredundants(ispace, pivot)
            score = estimate_cost(pivot, True)*((na - 1) + nr)
            aliases.add(pivot, aliaseds, list(mapper), distances, score)

    return aliases


def choose(aliases, exprs, mapper, mingain):
    """
    Analyze the detected aliases and, after applying a cost model to rule out
    the aliases with a bad memory/flops trade-off, inject them into the original
    expressions.
    """
    aliases = AliasList(aliases)

    if not aliases:
        return exprs, aliases

    # `score < m` => discarded
    # `score > M` => optimized
    # `m <= score <= M` => maybe discarded, maybe optimized; depends on heuristics
    m = mingain
    M = mingain*3

    # Filter off the aliases with low score
    key = lambda a: a.score >= m
    aliases.filter(key)

    # Project the candidate aliases into `exprs` to derive the final working set
    mapper = {k: v for k, v in mapper.items() if v.free_symbols & set(aliases.aliaseds)}
    templated = [uxreplace(e, mapper) for e in exprs]
    owset = wset(templated)

    # Filter off the aliases with a weak flop-reduction / working-set tradeoff
    key = lambda a: \
        a.score > M or \
        m <= a.score <= M and max(len(wset(a.pivot)), 1) > len(wset(a.pivot) & owset)
    aliases.filter(key)

    if not aliases:
        return exprs, aliases

    # Substitute the chosen aliasing sub-expressions
    mapper = {k: v for k, v in mapper.items() if v.free_symbols & set(aliases.aliaseds)}
    exprs = [uxreplace(e, mapper) for e in exprs]

    return exprs, aliases


def lower_aliases(aliases, meta, maxpar):
    """
    Create a Schedule from an AliasList.
    """
    stampcache = {}
    dmapper = {}
    processed = []
    for a in aliases:
        imapper = {**{i.dim: i for i in a.intervals},
                   **{i.dim.parent: i for i in a.intervals if i.dim.is_NonlinearDerived}}

        intervals = []
        writeto = []
        sub_iterators = {}
        indicess = [[] for _ in a.distances]
        for i in meta.ispace:
            try:
                interval = imapper[i.dim]
            except KeyError:
                if i.dim in a.free_symbols:
                    # Special case: the Dimension appears within the alias but
                    # not as an Indexed index. Then, it needs to be addeed to
                    # the `writeto` region too
                    interval = i
                else:
                    # E.g., `x0_blk0` or (`a[y_m+1]` => `y not in imapper`)
                    intervals.append(i)
                    continue

            if not (writeto or
                    interval != interval.zero() or
                    (maxpar and SEQUENTIAL not in meta.properties.get(i.dim))):
                # The alias doesn't require a temporary Dimension along i.dim
                intervals.append(i)
                continue

            assert not i.dim.is_NonlinearDerived

            # `i.dim` is necessarily part of the write-to region, so
            # we have to adjust the Interval's stamp. For example, consider
            # `i=x[0,0]<1>` and `interval=x[-4,4]<0>`; here we need to
            # use `<1>` as stamp, which is what appears in `ispace`
            interval = interval.lift(i.stamp)

            # We further bump the interval stamp if we were requested to trade
            # fusion for more collapse-parallelism
            if maxpar:
                stamp = stampcache.setdefault(interval.dim, Stamp())
                interval = interval.lift(stamp)

            writeto.append(interval)
            intervals.append(interval)

            if i.dim.is_Block:
                # Suitable IncrDimensions must be used to avoid OOB accesses.
                # E.g., r[xs][ys][z] => both `xs` and `ys` must be initialized such
                # that all accesses are within bounds. This requires traversing the
                # hierarchy of BlockDimensions to set `xs` (`ys`) in a way that
                # consecutive blocks access consecutive regions in `r` (e.g.,
                # `xs=x0_blk1-x0_blk0` with `blocklevels=2`; `xs=0` with
                # `blocklevels=1`, that is it degenerates in this case)
                try:
                    d = dmapper[i.dim]
                except KeyError:
                    dd = i.dim.parent
                    assert dd.is_Block
                    if dd.parent.is_Block:
                        # A BlockDimension in between BlockDimensions
                        m = i.dim.symbolic_min - i.dim.parent.symbolic_min
                    else:
                        m = 0
                    d = dmapper[i.dim] = IncrDimension("%ss" % i.dim.name, i.dim, m,
                                                       dd.symbolic_size, 1, dd.step)
                sub_iterators[i.dim] = d
            else:
                d = i.dim

            # Given the iteration `interval`, lower distances to indices
            for distance, indices in zip(a.distances, indicess):
                v = distance[interval.dim] or 0
                try:
                    indices.append(d - interval.lower + v)
                except TypeError:
                    indices.append(d)

        # The alias write-to space
        writeto = IterationSpace(IntervalGroup(writeto), sub_iterators)

        # The alias iteration space
        ispace = IterationSpace(IntervalGroup(intervals, meta.ispace.relations),
                                meta.ispace.sub_iterators,
                                meta.ispace.directions)
        ispace = ispace.augment(sub_iterators)

        processed.append(
            ScheduledAlias(a.pivot, writeto, ispace, a.aliaseds, indicess)
        )

    # The [ScheduledAliases] must be ordered so as to reuse as many of the
    # `ispace`'s IterationIntervals as possible in order to honor the
    # write-to region. Another fundamental reason for ordering is to ensure
    # deterministic code generation
    processed = sorted(processed, key=lambda i: cit(meta.ispace, i.ispace))

    return Schedule(*processed, dmapper=dmapper, is_frame=aliases.is_frame)


def optimize_schedule_rotations(schedule, sregistry):
    """
    Transform the schedule such that the tensor temporaries "rotate" along
    the outermost Dimension. This trades a parallel Dimension for a smaller
    working set size.
    """
    # The rotations Dimension is the outermost
    ridx = 0

    rmapper = defaultdict(list)
    processed = []
    for k, group in groupby(schedule, key=lambda i: i.writeto):
        g = list(group)

        try:
            candidate = k[ridx]
        except IndexError:
            # Degenerate alias (a scalar)
            processed.extend(g)
            continue
        d = candidate.dim

        try:
            ds = schedule.dmapper[d]
        except KeyError:
            # Can't do anything if `d` isn't a BlockDimension
            processed.extend(g)
            continue

        n = candidate.min_size
        assert n > 0

        iis = candidate.lower
        iib = candidate.upper

        name = sregistry.make_name(prefix='%sii' % d.root.name)
        ii = ModuloDimension(name, ds, iis, incr=iib)

        cd = CustomDimension(name='%sc' % d.root.name, symbolic_min=ii,
                             symbolic_max=iib, symbolic_size=n)
        dsi = ModuloDimension('%si' % ds.root.name, cd, cd + ds - iis, n)

        mapper = OrderedDict()
        for i in g:
            # Update `indicess` to use `xs0`, `xs1`, ...
            mds = []
            for indices in i.indicess:
                v = indices[ridx]
                try:
                    md = mapper[v]
                except KeyError:
                    name = sregistry.make_name(prefix='%sr' % d.root.name)
                    md = mapper.setdefault(v, ModuloDimension(name, ds, v, n))
                mds.append(md)
            indicess = [indices[:ridx] + [md] + indices[ridx + 1:]
                        for md, indices in zip(mds, i.indicess)]

            # Update `writeto` by switching `d` to `dsi`
            intervals = k.intervals.switch(d, dsi).zero(dsi)
            sub_iterators = dict(k.sub_iterators)
            sub_iterators[d] = dsi
            writeto = IterationSpace(intervals, sub_iterators)

            # Transform `alias` by adding `i`
            pivot = i.pivot.xreplace({d: d + cd})

            # Extend `ispace` to iterate over rotations
            d1 = writeto[ridx+1].dim  # Note: we're by construction in-bounds here
            intervals = IntervalGroup(Interval(cd))
            rispace = IterationSpace(intervals, {cd: dsi}, {cd: Forward})
            aispace = i.ispace.zero(d)
            aispace = aispace.augment({d: mds + [ii]})
            ispace = IterationSpace.union(rispace, aispace, relations={(d, cd, d1)})

            processed.append(ScheduledAlias(
                pivot, writeto, ispace, i.aliaseds, indicess,
            ))

        # Update the rotations mapper
        rmapper[d].extend(list(mapper.values()))

    return schedule.rebuild(*processed, rmapper=rmapper)


def lower_schedule(schedule, meta, sregistry, ftemps, min_dtype):
    """
    Turn a Schedule into a sequence of Clusters.
    """
    if ftemps:
        make = TempFunction
    else:
        # Typical case -- the user does *not* "see" the CIRE-created temporaries
        make = TempArray

    clusters = []
    subs = {}
    for pivot, writeto, ispace, aliaseds, indicess in schedule:
        name = sregistry.make_name()
        # Infer the dtype for the pivot
        # This prevents cases such as `floor(a*b)` with `a` and `b` floats
        # that would creat a temporary `int r = b` leading to erronous
        # numerical results

        if writeto:
            # The Dimensions defining the shape of Array
            # Note: with SubDimensions, we may have the following situation:
            #
            # for zi = z_m + zi_ltkn; zi <= z_M - zi_rtkn; ...
            #   r[zi] = ...
            #
            # Instead of `r[zi - z_m - zi_ltkn]` we have just `r[zi]`, so we'll
            # need as much room as in `zi`'s parent to avoid going OOB Aside
            # from ugly generated code, the reason we do not rather shift the
            # indices is that it prevents future passes to transform the loop
            # bounds (e.g., MPI's comp/comm overlap does that)
            dimensions = [d.parent if d.is_AbstractSub else d
                          for d in writeto.itdims]

            # The halo must be set according to the size of `writeto`
            halo = [(abs(i.lower), abs(i.upper)) for i in writeto]

            # The indices used to write into the Array
            indices = []
            for i in writeto:
                try:
                    # E.g., `xs`
                    sub_iterators = writeto.sub_iterators[i.dim]
                    assert len(sub_iterators) <= 1
                    indices.append(sub_iterators[0])
                except (KeyError, IndexError):
                    # E.g., `z` -- a non-shifted Dimension
                    indices.append(i.dim - i.lower)

            dtype = sympy_dtype(pivot, base=meta.dtype)
            obj = make(name=name, dimensions=dimensions, halo=halo, dtype=dtype)
            expression = Eq(obj[indices], uxreplace(pivot, subs))

            callback = lambda idx: obj[idx]
        else:
            # Degenerate case: scalar expression
            assert writeto.size == 0

            dtype = sympy_dtype(pivot, base=meta.dtype, smin=min_dtype)
            obj = Temp(name=name, dtype=dtype)
            expression = Eq(obj, uxreplace(pivot, subs))

            callback = lambda idx: obj

        # Create the substitution rules for the aliasing expressions
        subs.update({aliased: callback(indices)
                     for aliased, indices in zip(aliaseds, indicess)})

        properties = dict(meta.properties)

        # Drop or weaken parallelism if necessary
        for d, v in meta.properties.items():
            try:
                if any(i.is_Modulo for i in ispace.sub_iterators[d]):
                    properties[d] = normalize_properties(v, {SEQUENTIAL})
                elif d not in writeto.itdims:
                    properties[d] = normalize_properties(v, {PARALLEL_IF_PVT})
            except KeyError:
                # Non-dimension key such as (x, y) for diagonal stencil u(x+i hx, y+i hy)
                pass

        # Track star-shaped stencils for potential future optimization
        if len(writeto) > 1 and schedule.is_frame:
            properties[Hyperplane(writeto.itdims)] = {SEPARABLE}

        # Finally, build the alias Cluster
        clusters.append(Cluster(expression, ispace, meta.guards, properties))

    return clusters, subs


def optimize_clusters_msds(clusters):
    """
    Relax the clusters by letting the expressions defined over MultiSubDomains to
    rather be computed over the entire domain. This increases the likelihood of
    code lifting by later passes.
    """
    processed = []
    for c in clusters:
        msds = [d for d in c.ispace.itdims if isinstance(d, MultiSubDimension)]

        if msds:
            mapper = {d: d.root for d in msds}
            exprs = [uxreplace(e, mapper) for e in c.exprs]

            ispace = c.ispace.relaxed(msds)

            guards = {mapper.get(d, d): v for d, v in c.guards.items()}
            properties = {mapper.get(d, d): v for d, v in c.properties.items()}
            syncs = {mapper.get(d, d): v for d, v in c.syncs.items()}

            processed.append(c.rebuild(exprs=exprs, ispace=ispace, guards=guards,
                                       properties=properties, syncs=syncs))
        else:
            processed.append(c)

    return processed


def pick_best(variants):
    """
    Return the variant with the best theoretical performance.
    """
    best = None
    for i in variants:
        # Flops in the two sweeps
        flops0 = i.schedule.cost
        flops1 = estimate_cost(i.exprs, True)

        flops = flops0 + flops1

        # Estimate the data movement in the two sweeps

        # With cross-loop blocking, a Function appearing in both sweeps is
        # much more likely to be in cache during the second sweep, hence
        # we count it only once
        functions0 = set()
        functions1 = set()
        for sa in i.schedule:
            indexeds0 = search(sa.pivot, Indexed)

            if any(d.is_Block for d in sa.ispace.itdims):
                functions1.update({i.function for i in indexeds0})
            else:
                functions0.update({i.function for i in indexeds0})

        indexeds1 = search(i.exprs, Indexed)
        functions1.update({i.function for i in indexeds1})

        nfunctions0 = len(functions0)
        nfunctions1 = len(functions1)

        # All temporaries impact data movement, but some kind of temporaries
        # are more likely to be in cache than others, so they are given a
        # lighter weight
        for ii in indexeds1:
            grid = ii.function.grid
            if grid is None:
                continue

            ntemps = 0
            for sa in i.schedule:
                if len(sa.writeto) < grid.dim:
                    # Tiny temporary, extremely likely to be in cache, hardly
                    # impacting data movement in a significant way
                    ntemps += 0.1
                elif any(d.is_Block for d in sa.writeto.itdims):
                    # Cross-loop blocking temporary, likely to be in some level
                    # of cache (but unlikely to be in the fastest level)
                    ntemps += 1
                else:
                    # Grid-size temporary, likely _not_ to be in cache, and
                    # therefore requiring at least two costly accesses per
                    # grid point
                    ntemps += 2

            ntemps = int(ntemps)

            break
        else:
            ntemps = len(i.schedule)

        ws = ntemps + nfunctions0 + nfunctions1

        if best is None:
            best, best_flops, best_ws = i, flops, ws
            continue

        delta_flops = flops - best_flops
        delta_ws = ws - best_ws

        # Magic sauce
        # The coefficients were obtained empirically studying the behaviour
        # of different variants in several kernels and platforms
        # Intuitively, it's like trading 70 operations for 1 temporary
        ws_curve = lambda x: (-1 / 70)*x

        if delta_ws <= ws_curve(delta_flops):
            best, best_flops, best_ws = i, flops, ws

    return best


# Utilities


class Group(tuple):

    """
    A collection of aliasing expressions.
    """

    def __new__(cls, items, ispace=None):
        # Expand out the StencilDimensions, if any
        processed = []
        for c in items:
            sdims = [d for d in unbounded(c.expr) if d.is_Stencil]
            if not sdims:
                processed.append(c)
                continue

            f0 = lambda e: minimum(e, sdims)
            f1 = lambda e: maximum(e, sdims)

            for f in (f0, f1):
                expr = f(c.expr)
                indexeds = [f(i) for i in c.indexeds]
                offsets = [LabeledVector([(d, f(i)) for d, i in v.items()])
                           for v in c.offsets]

                processed.append(ExprGeometry(expr, indexeds, c.bases, offsets))

        obj = super().__new__(cls, processed)
        obj._items = items
        obj._ispace = ispace

        return obj

    def __repr__(self):
        return "Group(%s)" % ", ".join([str(i) for i in self])

    def find_rotation_distance(self, d, interval):
        """
        The distance from the Group pivot of a rotation along `d`
        such that it is still possible to safely iterate over `interval`.
        """
        assert d is interval.dim

        for rotation, distance in self._pivot_legal_rotations[d]:
            # Does `rotation` cover the `interval` ?
            if rotation.union(interval) != rotation:
                continue

            # Infer the `rotation`'s min_intervals from the pivot's
            min_interval = self._pivot_min_intervals[d].translate(-distance)

            # Does the `interval` actually cover the `rotation`'s `min_interval`?
            if interval.union(min_interval) == interval:
                return distance

        return None

    @cached_property
    def Toffsets(self):
        return [LabeledVector.transpose(*i) for i in zip(*[i.offsets for i in self])]

    @cached_property
    def diameter(self):
        """
        The size of the iteration space required to evaluate all aliasing
        expressions in this Group, along each Dimension.
        """
        ret = defaultdict(int)
        for i in self.Toffsets:
            for d, v in i:
                if d not in self._ispace:
                    continue
                try:
                    distance = int(max(v) - min(v))
                except TypeError:
                    # An entry in `v` has symbolic components, e.g. `x_m + 2`
                    if len(set(v)) == 1:
                        continue
                    else:
                        # Worst-case scenario, we raraly end up here
                        # Resort to the fast vector-based comparison machinery
                        # (rather than the slower sympy.simplify)
                        items = [Vector(i) for i in v]
                        distance, = vmax(*items) - vmin(*items)
                        if not is_integer(distance):
                            raise ValueError
                ret[d] = max(ret[d], distance)

        return ret

    @property
    def pivot(self):
        """
        A deterministically chosen reference for this Group.
        """
        return self[0]

    @property
    def dimensions(self):
        return frozenset(self.diameter)

    @property
    def dimensions_translated(self):
        return frozenset(d for d, v in self.diameter.items() if v > 0)

    @property
    def naliases(self):
        na = len(self._items)

        udims = set().union(*[unbounded(c.expr) for c in self._items])
        sdims = [d for d in udims if d.is_Stencil]
        implicit = int(np.prod([i._size for i in sdims]))

        return na*max(implicit, 1)

    @cached_property
    def _pivot_legal_rotations(self):
        """
        All legal rotations along each Dimension for the Group pivot.
        """
        ret = {}
        for d, (maxd, mini) in self._pivot_legal_shifts.items():
            # Rotation size = mini (min-increment) - maxd (max-decrement)
            v = mini - maxd

            # Build the table of all possible rotations
            m = make_rotations_table(d, v)

            distances = []
            for rotation in m:
                # Distance of the rotation `i` from `c`
                distance = maxd - rotation.lower
                assert distance == mini - rotation.upper
                distances.append(distance)

            ret[d] = list(zip(m, distances))

        return ret

    @cached_property
    def _pivot_min_intervals(self):
        """
        The minimum Interval along each Dimension such that by evaluating the
        pivot, all other items are evaluated too.
        """
        c = self.pivot

        ret = defaultdict(lambda: [np.inf, -np.inf])
        for i in self:
            distance = [o.distance(v) for o, v in zip(i.offsets, c.offsets)]
            distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]

            for d, v in distance:
                value = v.pop()
                try:
                    ret[d][0] = min(ret[d][0], value)
                    ret[d][1] = max(ret[d][1], value)
                except TypeError:
                    ret[d][0] = min(ret[d][0], 0)
                    ret[d][1] = max(ret[d][1], 0)

        ret = {d: Interval(d, m, M) for d, (m, M) in ret.items()}

        return ret

    @cached_property
    def _pivot_legal_shifts(self):
        """
        The max decrement and min increment along each Dimension such that the
        Group pivot does not go OOB.
        """
        c = self.pivot

        ret = defaultdict(lambda: (-np.inf, np.inf))
        for i, ofs in zip(c.indexeds, c.offsets):
            f = i.function

            for l in ofs.labels:
                if isinstance(l, StencilDimension):
                    continue

                # `f`'s cumulative halo size along `l`
                hsize = sum(f._size_halo[l])

                # Any `ofs`'s shift due to non-[0,0] iteration space
                lower, upper = self._ispace.intervals[l].offsets

                ofs0, ofs1 = extrema(ofs[l])

                try:
                    # Assume `ofs[l]` is a number (typical case)
                    maxd = min(0, max(ret[l][0], -ofs0 - lower))
                    mini = max(0, min(ret[l][1], hsize - ofs1 - upper))

                    ret[l] = (maxd, mini)
                except TypeError:
                    # E.g., `ofs[d] = x_m - x + 5`
                    ret[l] = (0, 0)

        return ret


AliasKey = namedtuple('AliasKey', 'ispace intervals dtype guards properties')
Variant = namedtuple('Variant', 'schedule exprs')


class Alias:

    def __init__(self, pivot, aliaseds, intervals, distances, score):
        self.pivot = pivot
        self.aliaseds = aliaseds
        self.intervals = intervals
        self.distances = distances
        self.score = score

    def __repr__(self):
        return "Alias<<%s>>" % self.pivot

    @property
    def free_symbols(self):
        return self.pivot.free_symbols

    @property
    def is_frame(self):
        """
        An Alias is said to be a "frame" if the `distances` of the `aliaseds`
        expressions from the `pivot` form a frame in the linear algebra sense.
        In essence, any two distances are either orthogonal or one is a multiple
        of the other by a scalar constant (i.e., a "redundant" vector w.r.t. a basis).
        """
        # NOTE: derivative => frame
        # but: frame NOT => derivative (e.g., `(1, 0, 1)`, `(0, 1, 0)`, `(2, 0, 2)` is
        # a frame, but clearly it doesn't stem from a derivative, at least not a classic
        # one. Unless perhaps one uses rotated derivatives -- but this is another story,
        # and our DSL doesn't support them yet)

        # NOTE: invariants are generally not frames. They could stem from nested
        # derivatives, thus having a non-centered access in multiple dimensions
        # E.g., `sqrt(delta[x + 5, y + 5, z + 4])` and `sqrt(delta[x + 5, y + 3, z + 4])`
        # with one possible pivot being `sqrt(delta[x + 4, y + 4, z + 4])`

        # NOTE: below is a sufficient but not necessary condition to be a frame. In
        # particular, it is a sufficient condition to be a derivative

        return all(len([e for e in i if e != 0]) <= 1 for i in self.distances)


class AliasList:

    def __init__(self, aliases=None):
        if aliases is None:
            self._list = []
        else:
            self._list = list(aliases._list)

    def __repr__(self):
        if self._list:
            return "AliasList<\n  %s\n>" % ",\n  ".join(str(i) for i in self._list)
        else:
            return "<>"

    def __len__(self):
        return self._list.__len__()

    def __iter__(self):
        for i in self._list:
            yield i

    def add(self, pivot, aliaseds, intervals, distances, score):
        assert len(aliaseds) == len(distances)
        self._list.append(Alias(pivot, aliaseds, intervals, distances, score))

    def update(self, aliases):
        self._list.extend(aliases)

    def filter(self, key):
        for i in list(self._list):
            if not key(i):
                self._list.remove(i)

    @property
    def aliaseds(self):
        return flatten(i.aliaseds for i in self._list)

    @property
    def is_frame(self):
        """
        An AliasList is said to be a "frame" if all of its Aliases are frames.
        """
        return all(i.is_frame for i in self._list)


ScheduledAlias = namedtuple('SchedAlias',
                            'pivot writeto ispace aliaseds indicess')


class Schedule(tuple):

    def __new__(cls, *items, dmapper=None, rmapper=None, is_frame=False):
        obj = super().__new__(cls, items)
        obj.dmapper = dmapper or {}
        obj.rmapper = rmapper or {}
        obj.is_frame = is_frame
        return obj

    def rebuild(self, *items, dmapper=None, rmapper=None, is_frame=False):
        return Schedule(
            *items,
            dmapper=dmapper or self.dmapper,
            rmapper=rmapper or self.rmapper,
            is_frame=is_frame or self.is_frame
        )

    @cached_property
    def cost(self):
        # Not just the sum for the individual items' cost! There might be
        # redundancies, which we factor out here...
        counter = generator()
        make = lambda _: Symbol(name='dummy%d' % counter(), dtype=np.float32)

        tot = 0
        for v in as_mapper(self, lambda i: i.ispace).values():
            exprs = [i.pivot for i in v]
            tot += estimate_cost(_cse(exprs, make), True)

        return tot


def make_rotations_table(d, v):
    """
    All possible rotations of `range(v+1)`.
    """
    m = np.array([[j-i if j > i else 0 for j in range(v+1)] for i in range(v+1)])
    m = (m - m.T)[::-1, :]

    # Shift the table so that the middle rotation is at the top
    m = np.roll(m, int(-np.floor(v/2)), axis=0)

    # Turn into a more compact representation as a list of Intervals
    m = [Interval(d, min(i), max(i)) for i in m]

    return m


def cit(ispace0, ispace1):
    """
    The Common IterationIntervals of two IterationSpaces.
    """
    found = []
    for it0, it1 in zip(ispace0.itintervals, ispace1.itintervals):
        if it0 == it1:
            found.append(it0)
        else:
            break
    return tuple(found)


def split_coeff(expr):
    """
    Split potential derivative coefficients and arguments into two groups.
    """
    # TODO: Once we'll be able to keep Derivative intact down to this point,
    # we won't probably need this function anymore, because, in essence, what
    # this function is doing is reconstructing prematurely lowered information
    grids = {getattr(i.function, 'grid', None) for i in expr.free_symbols} - {None}
    if len(grids) != 1:
        return [], None
    grid = grids.pop()

    maybe_coeff = []
    others = []
    for a in expr.args:
        indexeds = [i for i in a.free_symbols if i.is_Indexed]
        if all(not set(grid.dimensions) <= set(i.function.dimensions) for i in indexeds):
            maybe_coeff.append(a)
        else:
            others.append(a)

    return maybe_coeff, others


def nredundants(ispace, expr):
    """
    The number of redundant Dimensions in `ispace` for `expr`. A Dimension is
    redundant if it defines an iteration space for `expr` while not appearing
    among its free symbols. Note that the converse isn't generally true: there
    could be a Dimension that does not appear in the free symbols while defining
    a non-redundant iteration space (e.g., a BlockDimension).
    """
    iterated = {i.dim for i in ispace}
    used = {i for i in expr.free_symbols if i.is_Dimension}

    # "Short" dimensions won't count
    key0 = lambda d: d.is_Sub and d.local
    # StencilDimensions are like an inlined iteration space so they won't count
    key1 = lambda d: d.is_Stencil
    key = lambda d: key0(d) or key1(d)
    iterated = {d for d in iterated if not key(d)}
    used = {d for d in used if not key(d)}

    iterated = {d.root for d in iterated}
    used = {d.root for d in used}

    return len(iterated) - (len(used))


def wset(exprs):
    """
    Extract the working set out of a set of equations.
    """
    return {i.function for i in flatten([e.free_symbols for e in as_tuple(exprs)])
            if i.function.is_AbstractFunction}


def deindexify(expr):
    """
    Strip away Indexed and indices from an expression, turning them into Functions.
    This means that e.g. `deindexify(f[x+2]*3) == deindexify(f[x-4]*3) == f(x)*3`.
    This function returns a mapper that binds the de-indexified sub-expressions to
    the original counterparts.
    """
    return _deindexify(expr)[1]


@singledispatch
def _deindexify(expr):
    args = []
    mapper = defaultdict(list)
    for a in expr.args:
        arg, m = _deindexify(a)
        args.append(arg)
        for k, v in m.items():
            mapper[k].extend(v)

    rexpr = reuse_if_untouched(expr, args)
    if rexpr is not expr:
        mapper[rexpr] = [expr]

    return rexpr, mapper


@_deindexify.register(sympy.Number)
@_deindexify.register(sympy.Symbol)
@_deindexify.register(sympy.Function)
def _(expr):
    return expr, {}


@_deindexify.register(Indexed)
def _(expr):
    return expr.function, {}
