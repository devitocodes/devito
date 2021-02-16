from collections import OrderedDict, defaultdict, namedtuple
from functools import partial
from itertools import groupby

from cached_property import cached_property
import numpy as np

from devito.ir import (SEQUENTIAL, PARALLEL, PARALLEL_IF_PVT, ROUNDABLE, DataSpace,
                       Forward, IterationInstance, IterationSpace, Interval,
                       IntervalGroup, LabeledVector, Scope, detect_accesses,
                       build_intervals, normalize_properties)
from devito.passes.clusters.utils import cluster_pass, make_is_time_invariant
from devito.symbolics import (compare_ops, estimate_cost, q_constant, q_terminalop,
                              retrieve_indexed, search, uxreplace)
from devito.tools import as_tuple, flatten, split
from devito.types import (Array, TempFunction, Eq, Scalar, ModuloDimension,
                          CustomDimension, IncrDimension)

__all__ = ['cire']


@cluster_pass
def cire(cluster, mode, sregistry, options, platform):
    """
    Cross-iteration redundancies elimination.

    Parameters
    ----------
    cluster : Cluster
        Input Cluster, subject of the optimization pass.
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
        Accepted: ['min-storage', 'cire-maxpar', 'cire-rotate', 'cire-maxalias'].
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
        * 'cire-maxalias': if True, capture the largest redundancies. This will
          minimize the flop count while maximizing the number of tensor temporaries,
          thus increasing the working set size.
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
    # Relevant options
    repeats = options['cire-repeats']

    # Sanity checks
    assert mode in list(callbacks_mapper)
    assert all(i >= 0 for i in repeats.values())

    # The main CIRE loop
    processed = []
    context = cluster.exprs
    for n in reversed(range(repeats.get(mode, 1))):
        # Get the callbacks
        extract, ignore_collected, in_writeto, selector =\
            callbacks_mapper[mode](context, n, options)

        # Extract potentially aliasing expressions
        exprs, extracted = extract(cluster, sregistry)
        if not extracted:
            # Do not waste time
            continue

        # There can't be Dimension-dependent data dependences with any of
        # the `processed` Clusters, otherwise we would risk either OOB accesses
        # or reading from garbage uncomputed halo
        scope = Scope(exprs=flatten(c.exprs for c in processed) + extracted)
        if not all(i.is_indep() for i in scope.d_all_gen()):
            break

        # Search aliasing expressions
        aliases = collect(extracted, ignore_collected, options)

        # Rule out aliasing expressions with a bad flops/memory trade-off
        aliases, chosen, others = choose(exprs, aliases, selector)
        if not chosen:
            # Do not waste time
            continue

        # AliasMapper -> Schedule -> [Clusters]
        schedule = make_schedule(cluster, aliases, in_writeto, options)
        schedule = optimize_schedule(cluster, schedule, platform, sregistry, options)
        clusters, subs = lower_schedule(cluster, schedule, chosen, sregistry, options)

        # The [Clusters] must be ordered so as to reuse as many of the `cluster`'s
        # IterationIntervals as possible in order to honor the write-to region. This
        # also guarantees that fusion is maximum
        processed.extend(clusters)
        processed.sort(key=partial(cit, cluster))

        # Rebuild `cluster` so as to use the newly created aliases
        cluster = rebuild(cluster, others, schedule, subs)

        # Prepare for the next round
        context = flatten(c.exprs for c in processed) + list(cluster.exprs)

    processed.append(cluster)

    return processed


class Callbacks(object):

    """
    Interface for the callbacks needed by the CIRE loop. Each CIRE mode needs
    to provide an implementation of these callbackes in a suitable subclass.
    """

    mode = None

    def __new__(cls, context, n, options):
        min_cost = options['cire-mincost']
        max_par = options['cire-maxpar']
        max_alias = options['cire-maxalias']

        min_cost = min_cost.get(cls.mode)
        if callable(min_cost):
            min_cost = min_cost(n)

        return (partial(cls.extract, n, context, min_cost, max_alias),
                cls.ignore_collected,
                partial(cls.in_writeto, max_par),
                partial(cls.selector, min_cost))

    @classmethod
    def extract(cls, n, context, min_cost, max_alias, cluster, sregistry):
        raise NotImplementedError

    @classmethod
    def ignore_collected(cls, group):
        return False

    @classmethod
    def in_writeto(cls, max_par, dim, cluster):
        raise NotImplementedError

    @classmethod
    def selector(cls, min_cost, cost):
        return cost // min_cost


class CallbacksInvariants(Callbacks):

    mode = 'invariants'

    @classmethod
    def _extract_rule(cls, context, min_cost, cluster):
        exclude = {i.source.indexed for i in cluster.scope.d_flow.independent()}
        rule0 = lambda e: not e.free_symbols & exclude
        rule1 = make_is_time_invariant(context)
        rule2 = lambda e: estimate_cost(e, True) >= min_cost

        return lambda e: rule0(e) and rule1(e) and rule2(e)

    @classmethod
    def extract(cls, n, context, min_cost, max_alias, cluster, sregistry):
        make = lambda: Scalar(name=sregistry.make_name(), dtype=cluster.dtype).indexify()

        rule = cls._extract_rule(context, min_cost, cluster)

        extracted = []
        mapper = OrderedDict()
        for e in cluster.exprs:
            for i in search(e, rule, 'all', 'dfs_first_hit'):
                if i not in mapper:
                    symbol = make()
                    mapper[i] = symbol
                    extracted.append(e.func(symbol, i))

        processed = [uxreplace(e, mapper) for e in cluster.exprs]

        return extracted + processed, extracted

    @classmethod
    def in_writeto(cls, max_par, dim, cluster):
        return PARALLEL in cluster.properties[dim]


class CallbacksDivs(CallbacksInvariants):

    mode = 'divs'

    @classmethod
    def _extract_rule(cls, context, min_cost, cluster):
        return lambda e: (e.is_Pow and e.exp.is_Integer and e.exp < 0 and
                          all(i.function.is_const for i in e.base.free_symbols))

    @classmethod
    def selector(cls, min_cost, cost):
        return int(cost > 0)


class CallbacksSOPS(Callbacks):

    mode = 'sops'

    @classmethod
    def extract(cls, n, context, min_cost, max_alias, cluster, sregistry):
        make = lambda: Scalar(name=sregistry.make_name(), dtype=cluster.dtype).indexify()

        # The `depth` determines "how big" the extracted sum-of-products will be.
        # We observe that in typical FD codes:
        #   add(mul, mul, ...) -> stems from first order derivative
        #   add(mul(add(mul, mul, ...), ...), ...) -> stems from second order derivative
        # To search the muls in the former case, we need `depth=0`; to search the outer
        # muls in the latter case, we need `depth=2`
        depth = n

        exclude = {i.source.indexed for i in cluster.scope.d_flow.independent()}
        rule0 = lambda e: not e.free_symbols & exclude
        rule1 = lambda e: e.is_Mul and q_terminalop(e, depth)
        rule = lambda e: rule0(e) and rule1(e)

        extracted = OrderedDict()
        mapper = {}
        for e in cluster.exprs:
            for i in search(e, rule, 'all', 'bfs_first_hit'):
                if i in mapper:
                    continue

                key = lambda a: a.is_Add
                terms, others = split(list(i.args), key)

                if max_alias:
                    # Treat `e` as an FD expression and pull out the derivative
                    # coefficient from `i`
                    # Note: typically derivative coefficients are numbers, but
                    # sometimes they could be provided in symbolic form through an
                    # arbitrary Function.  In the latter case, we rely on the
                    # heuristic that such Function's basically never span the whole
                    # grid, but rather a single Grid dimension (e.g., `c[z, n]` for a
                    # stencil of diameter `n` along `z`)
                    if e.grid is not None and terms:
                        key = partial(maybe_coeff_key, e.grid)
                        others, more_terms = split(others, key)
                        terms.extend(more_terms)

                if terms:
                    k = i.func(*terms)
                    try:
                        symbol, _ = extracted[k]
                    except KeyError:
                        symbol, _ = extracted.setdefault(k, (make(), e))
                    mapper[i] = i.func(symbol, *others)

        if mapper:
            extracted = [e.func(v, k) for k, (v, e) in extracted.items()]
            processed = [uxreplace(e, mapper) for e in cluster.exprs]
            return extracted + processed, extracted
        else:
            return cluster.exprs, []

    @classmethod
    def ignore_collected(cls, group):
        return len(group) <= 1

    @classmethod
    def in_writeto(cls, max_par, dim, cluster):
        return max_par and PARALLEL in cluster.properties[dim]


callbacks_mapper = {
    CallbacksInvariants.mode: CallbacksInvariants,
    CallbacksDivs.mode: CallbacksDivs,
    CallbacksSOPS.mode: CallbacksSOPS
}


def collect(exprs, ignore_collected, options):
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
    min_storage = options['min-storage']

    # Find the potential aliases
    found = []
    for expr in exprs:
        if expr.lhs.is_Indexed or expr.is_Increment:
            continue

        indexeds = retrieve_indexed(expr.rhs)

        bases = []
        offsets = []
        for i in indexeds:
            ii = IterationInstance(i)
            if ii.is_irregular:
                break

            base = []
            offset = []
            for e, ai in zip(ii, ii.aindices):
                if q_constant(e):
                    base.append(e)
                else:
                    base.append(ai)
                    offset.append((ai, e - ai))
            bases.append(tuple(base))
            offsets.append(LabeledVector(offset))

        if not indexeds or len(bases) == len(indexeds):
            found.append(Candidate(expr, indexeds, bases, offsets))

    # Create groups of aliasing expressions
    mapper = OrderedDict()
    unseen = list(found)
    while unseen:
        c = unseen.pop(0)
        group = [c]
        for u in list(unseen):
            # Is the arithmetic structure of `c` and `u` equivalent ?
            if not compare_ops(c.expr, u.expr):
                continue

            # Is `c` translated w.r.t. `u` ?
            if not c.translated(u):
                continue

            group.append(u)
            unseen.remove(u)
        group = Group(group)

        # Apply callback to heuristically discard groups
        if ignore_collected(group):
            continue

        if min_storage:
            k = group.dimensions_translated
        else:
            k = group.dimensions
        mapper.setdefault(k, []).append(group)

    aliases = AliasMapper()
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
            alias = uxreplace(c.expr, subs)

            # All aliased expressions
            aliaseds = [i.expr for i in g]

            # Distance of each aliased expression from the basis alias
            distances = []
            for i in g:
                distance = [o.distance(v) for o, v in zip(i.offsets, offsets)]
                distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]
                distances.append(LabeledVector([(d, v.pop()) for d, v in distance]))

            aliases.add(alias, list(mapper), aliaseds, distances)

    return aliases


def choose(exprs, aliases, selector):
    """
    Use a cost model to select the aliases that are worth optimizing.
    """
    # Pass 1: a set of aliasing expressions is optimizable only if its cost
    # exceeds the mode's threshold
    candidates = OrderedDict()
    others = []
    for e in exprs:
        naliases = len(aliases.get(e.rhs))
        cost = estimate_cost(e, True)*naliases
        score = selector(cost)
        if score > 0:
            candidates[e] = score
        else:
            others.append(e)

    # Pass 2: a set of aliasing expressions is optimizable if and only if it survived
    # Pass 1 above *and* the tradeoff between operation count and working set increase
    # is favorable
    owset = wset(others)
    retained = AliasMapper(aliases)
    chosen = OrderedDict()
    others = []
    for e in exprs:
        try:
            score = candidates[e]
        except KeyError:
            score = 0
        if score > 1 or \
           score == 1 and max(len(wset(e)), 1) > len(wset(e) & owset):
            chosen[e.rhs] = e.lhs
        else:
            others.append(e)
            retained.remove(e.rhs)

    return retained, chosen, others


def make_schedule(cluster, aliases, in_writeto, options):
    """
    Create a Schedule from an AliasMapper.
    """
    max_par = options['cire-maxpar']

    dmapper = {}
    processed = []
    for alias, v in aliases.items():
        imapper = {**{i.dim: i for i in v.intervals},
                   **{i.dim.parent: i for i in v.intervals if i.dim.is_NonlinearDerived}}

        intervals = []
        writeto = []
        sub_iterators = {}
        indicess = [[] for _ in v.distances]
        for i in cluster.ispace.intervals:
            try:
                interval = imapper[i.dim]
            except KeyError:
                # E.g., `x0_blk0` or (`a[y_m+1]` => `y not in imapper`)
                intervals.append(i)
                continue

            assert i.stamp >= interval.stamp

            if not (writeto or interval != interval.zero() or in_writeto(i.dim, cluster)):
                # The alias doesn't require a temporary Dimension along i.dim
                intervals.append(i)
                continue

            assert not i.dim.is_NonlinearDerived

            # `i.dim` is necessarily part of the write-to region, so
            # we have to adjust the Interval's stamp. For example, consider
            # `i=x[0,0]<1>` and `interval=x[-4,4]<0>`; here we need to
            # use `<1>` as stamp, which is what appears in `cluster`
            interval = interval.lift(i.stamp)

            # We further bump the interval stamp if we were requested to trade
            # fusion for more collapse-parallelism
            interval = interval.lift(interval.stamp + int(max_par))

            writeto.append(interval)
            intervals.append(interval)

            if i.dim.is_Incr:
                # Suitable IncrDimensions must be used to avoid OOB accesses.
                # E.g., r[xs][ys][z] => both `xs` and `ys` must be initialized such
                # that all accesses are within bounds. This requires traversing the
                # hierarchy of IncrDimensions to set `xs` (`ys`) in a way that
                # consecutive blocks access consecutive regions in `r` (e.g.,
                # `xs=x0_blk1-x0_blk0` with `blocklevels=2`; `xs=0` with
                # `blocklevels=1`, that is it degenerates in this case)
                try:
                    d = dmapper[i.dim]
                except KeyError:
                    dd = i.dim.parent
                    assert dd.is_Incr
                    if dd.parent.is_Incr:
                        # An IncrDimension in between IncrDimensions
                        m = i.dim.symbolic_min - i.dim.parent.symbolic_min
                    else:
                        m = 0
                    d = dmapper[i.dim] = IncrDimension("%ss" % i.dim.name, i.dim, m,
                                                       dd.symbolic_size, 1, dd.step)
                sub_iterators[i.dim] = d
            else:
                d = i.dim

            # Given the iteration `interval`, lower distances to indices
            for distance, indices in zip(v.distances, indicess):
                indices.append(d - interval.lower + distance[interval.dim])

        # The alias write-to space
        writeto = IterationSpace(IntervalGroup(writeto), sub_iterators)

        # The alias iteration space
        intervals = IntervalGroup(intervals, cluster.ispace.relations)
        ispace = IterationSpace(intervals, cluster.sub_iterators, cluster.directions)
        ispace = ispace.augment(sub_iterators)

        processed.append(ScheduledAlias(alias, writeto, ispace, v.aliaseds, indicess))

    # Sort by write-to region for deterministic code generation
    processed = sorted(processed, key=lambda i: i.writeto)

    return Schedule(*processed, dmapper=dmapper)


def optimize_schedule(cluster, schedule, platform, sregistry, options):
    """
    Rewrite the schedule for performance optimization.
    """
    if options['cire-rotate']:
        schedule = _optimize_schedule_rotations(schedule, sregistry)

    schedule = _optimize_schedule_padding(cluster, schedule, platform)

    return schedule


def _optimize_schedule_rotations(schedule, sregistry):
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

        candidate = k[ridx]
        d = candidate.dim
        try:
            ds = schedule.dmapper[d]
        except KeyError:
            # Can't do anything if `d` isn't an IncrDimension over a block
            processed.extend(g)
            continue

        n = candidate.min_size
        assert n > 0

        iis = candidate.lower
        iib = candidate.upper

        ii = ModuloDimension('%sii' % d, ds, iis, incr=iib)
        cd = CustomDimension(name='%s%s' % (d, d), symbolic_min=ii, symbolic_max=iib,
                             symbolic_size=n)
        dsi = ModuloDimension('%si' % ds, cd, cd + ds - iis, n)

        mapper = OrderedDict()
        for i in g:
            # Update `indicess` to use `xs0`, `xs1`, ...
            mds = []
            for indices in i.indicess:
                v = indices[ridx]
                try:
                    md = mapper[v]
                except KeyError:
                    name = sregistry.make_name(prefix='%sr' % d.name)
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
            alias = i.alias.xreplace({d: d + cd})

            # Extend `ispace` to iterate over rotations
            d1 = writeto[ridx+1].dim  # Note: we're by construction in-bounds here
            intervals = IntervalGroup(Interval(cd, 0, 0), relations={(d, cd, d1)})
            rispace = IterationSpace(intervals, {cd: dsi}, {cd: Forward})
            aispace = i.ispace.zero(d)
            aispace = aispace.augment({d: mds + [ii]})
            ispace = IterationSpace.union(rispace, aispace)

            processed.append(ScheduledAlias(alias, writeto, ispace, i.aliaseds, indicess))

        # Update the rotations mapper
        rmapper[d].extend(list(mapper.values()))

    return Schedule(*processed, dmapper=schedule.dmapper, rmapper=rmapper)


def _optimize_schedule_padding(cluster, schedule, platform):
    """
    Round up the innermost IterationInterval of the tensor temporaries IterationSpace
    to a multiple of the SIMD vector length. This is not always possible though (it
    depends on how much halo is safely accessible in all read Functions).
    """
    processed = []
    for i in schedule:
        try:
            it = i.ispace.itintervals[-1]
            if ROUNDABLE in cluster.properties[it.dim]:
                vl = platform.simd_items_per_reg(cluster.dtype)
                ispace = i.ispace.add(Interval(it.dim, 0, it.interval.size % vl))
            else:
                ispace = i.ispace
            processed.append(ScheduledAlias(i.alias, i.writeto, ispace, i.aliaseds,
                                            i.indicess))
        except (TypeError, KeyError):
            processed.append(i)

    return Schedule(*processed, dmapper=schedule.dmapper, rmapper=schedule.rmapper)


def lower_schedule(cluster, schedule, chosen, sregistry, options):
    """
    Turn a Schedule into a sequence of Clusters.
    """
    ftemps = options['cire-ftemps']

    if ftemps:
        make = TempFunction
    else:
        # Typical case -- the user does *not* "see" the CIRE-created temporaries
        make = Array

    clusters = []
    subs = {}
    for alias, writeto, ispace, aliaseds, indicess in schedule:
        if all(i not in chosen for i in aliaseds):
            continue

        # Basic info to create the temporary that will hold the alias
        name = sregistry.make_name()
        dtype = cluster.dtype

        if writeto:
            # The Dimensions defining the shape of Array
            # Note: with SubDimensions, we may have the following situation:
            #
            # for zi = z_m + zi_ltkn; zi <= z_M - zi_rtkn; ...
            #   r[zi] = ...
            #
            # Instead of `r[zi - z_m - zi_ltkn]` we have just `r[zi]`, so we'll need
            # as much room as in `zi`'s parent to avoid going OOB
            # Aside from ugly generated code, the reason we do not rather shift the
            # indices is that it prevents future passes to transform the loop bounds
            # (e.g., MPI's comp/comm overlap does that)
            dimensions = [d.parent if d.is_Sub else d for d in writeto.itdimensions]

            # The halo must be set according to the size of writeto space
            halo = [(abs(i.lower), abs(i.upper)) for i in writeto]

            # The indices used to write into the Array
            indices = []
            for i in writeto:
                try:
                    # E.g., `xs`
                    sub_iterators = writeto.sub_iterators[i.dim]
                    assert len(sub_iterators) == 1
                    indices.append(sub_iterators[0])
                except KeyError:
                    # E.g., `z` -- a non-shifted Dimension
                    indices.append(i.dim - i.lower)

            obj = make(name=name, dimensions=dimensions, halo=halo, dtype=dtype)
            expression = Eq(obj[indices], alias)

            callback = lambda idx: obj[idx]
        else:
            # Degenerate case: scalar expression
            assert writeto.size == 0

            obj = Scalar(name=name, dtype=dtype)
            expression = Eq(obj, alias)

            callback = lambda idx: obj

        # Create the substitution rules for the aliasing expressions
        for aliased, indices in zip(aliaseds, indicess):
            subs[aliased] = callback(indices)
            if aliased in chosen:
                subs[chosen[aliased]] = callback(indices)
            else:
                # Perhaps part of a composite alias ?
                pass

        # Construct the `alias` DataSpace
        accesses = detect_accesses(expression)
        parts = {k: IntervalGroup(build_intervals(v)).add(ispace.intervals).relaxed
                 for k, v in accesses.items() if k}
        dspace = DataSpace(cluster.dspace.intervals, parts)

        # Drop or weaken parallelism if necessary
        properties = dict(cluster.properties)
        for d, v in cluster.properties.items():
            if any(i.is_Modulo for i in ispace.sub_iterators[d]):
                properties[d] = normalize_properties(v, {SEQUENTIAL})
            elif d not in writeto.dimensions:
                properties[d] = normalize_properties(v, {PARALLEL_IF_PVT})

        # Finally, build the `alias` Cluster
        clusters.append(cluster.rebuild(exprs=expression, ispace=ispace,
                                        dspace=dspace, properties=properties))

    return clusters, subs


def rebuild(cluster, others, schedule, subs):
    """
    Plug the optimized aliases into the input Cluster. This leads to creating
    a new Cluster with suitable IterationSpace and DataSpace.
    """
    exprs = [uxreplace(e, subs) for e in others]

    ispace = cluster.ispace.augment(schedule.dmapper)
    ispace = ispace.augment(schedule.rmapper)

    accesses = detect_accesses(exprs)
    parts = {k: IntervalGroup(build_intervals(v)).relaxed
             for k, v in accesses.items() if k}
    dspace = DataSpace(cluster.dspace.intervals, parts)

    return cluster.rebuild(exprs=exprs, ispace=ispace, dspace=dspace)


# Utilities


class Candidate(object):

    def __init__(self, expr, indexeds, bases, offsets):
        self.expr = expr.rhs
        self.shifts = expr.ispace.intervals
        self.indexeds = indexeds
        self.bases = bases
        self.offsets = offsets

    def __repr__(self):
        return "Candidate(expr=%s)" % self.expr

    def translated(self, other):
        """
        True if ``self`` is translated w.r.t. ``other``, False otherwise.

        Examples
        --------
        Two candidates are translated if their bases are the same and
        their offsets are pairwise translated.

        c := A[i,j] op A[i,j+1]     -> Toffsets = {i: [0,0], j: [0,1]}
        u := A[i+1,j] op A[i+1,j+1] -> Toffsets = {i: [1,1], j: [0,1]}

        Then `c` is translated w.r.t. `u` with distance `{i: 1, j: 0}`
        """
        if len(self.Toffsets) != len(other.Toffsets):
            return False
        if len(self.bases) != len(other.bases):
            return False

        # Check the bases
        if any(b0 != b1 for b0, b1 in zip(self.bases, other.bases)):
            return False

        # Check the offsets
        for (d0, o0), (d1, o1) in zip(self.Toffsets, other.Toffsets):
            if d0 is not d1:
                return False

            distance = set(o0 - o1)
            if len(distance) != 1:
                return False

        return True

    @cached_property
    def Toffsets(self):
        return LabeledVector.transpose(*self.offsets)

    @cached_property
    def dimensions(self):
        return frozenset(i for i, _ in self.Toffsets)


class Group(tuple):

    """
    A collection of aliasing expressions.
    """

    def __repr__(self):
        return "Group(%s)" % ", ".join([str(i) for i in self])

    def find_rotation_distance(self, d, interval):
        """
        The distance from the Group pivot of a rotation along Dimension ``d`` that
        can safely iterate over the ``interval``.
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
        The size of the iteration space required to evaluate all aliasing expressions
        in this Group, along each Dimension.
        """
        ret = defaultdict(int)
        for i in self.Toffsets:
            for d, v in i:
                try:
                    distance = int(max(v) - min(v))
                except TypeError:
                    # An entry in `v` has symbolic components, e.g. `x_m + 2`
                    if len(set(v)) == 1:
                        continue
                    else:
                        raise ValueError
                ret[d] = max(ret[d], distance)

        return ret

    @property
    def pivot(self):
        """
        A deterministically chosen Candidate for this Group.
        """
        return self[0]

    @property
    def dimensions(self):
        return self.pivot.dimensions

    @property
    def dimensions_translated(self):
        return frozenset(d for d, v in self.diameter.items() if v > 0)

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
        pivot, all Candidates are evaluated too.
        """
        c = self.pivot

        ret = defaultdict(lambda: [np.inf, -np.inf])
        for i in self:
            distance = [o.distance(v) for o, v in zip(i.offsets, c.offsets)]
            distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]

            for d, v in distance:
                value = v.pop()
                ret[d][0] = min(ret[d][0], value)
                ret[d][1] = max(ret[d][1], value)

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
                # `f`'s cumulative halo size along `l`
                hsize = sum(f._size_halo[l])

                # Any `ofs`'s shift due to non-[0,0] iteration space
                lower, upper = c.shifts[l].offsets

                try:
                    # Assume `ofs[d]` is a number (typical case)
                    maxd = min(0, max(ret[l][0], -ofs[l] - lower))
                    mini = max(0, min(ret[l][1], hsize - ofs[l] - upper))

                    ret[l] = (maxd, mini)
                except TypeError:
                    # E.g., `ofs[d] = x_m - x + 5`
                    ret[l] = (0, 0)

        return ret


AliasedGroup = namedtuple('AliasedGroup', 'intervals aliaseds distances')

ScheduledAlias = namedtuple('ScheduledAlias', 'alias writeto ispace aliaseds indicess')
ScheduledAlias.__new__.__defaults__ = (None,) * len(ScheduledAlias._fields)


class Schedule(tuple):

    def __new__(cls, *items, dmapper=None, rmapper=None):
        obj = super(Schedule, cls).__new__(cls, items)
        obj.dmapper = dmapper or {}
        obj.rmapper = rmapper or {}
        return obj


class AliasMapper(OrderedDict):

    """
    A mapper between aliases and collections of aliased expressions.
    """

    def add(self, alias, intervals, aliaseds, distances):
        assert len(aliaseds) == len(distances)
        self[alias] = AliasedGroup(intervals, aliaseds, distances)

    def get(self, key):
        ret = super(AliasMapper, self).get(key)
        if ret is not None:
            return ret.aliaseds
        for i in self.values():
            if key in i.aliaseds:
                return i.aliaseds
        return []

    def remove(self, key):
        for k, v in list(self.items()):
            if key in v.aliaseds:
                self.pop(k)
                return


# Utils


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


def cit(c0, c1):
    """
    The Common IterationIntervals of two given Clusters.
    """
    found = []
    for it0, it1 in zip(c0.itintervals, c1.itintervals):
        if it0 == it1:
            found.append(it0)
        else:
            break
    return tuple(found)


def maybe_coeff_key(grid, expr):
    """
    True if `expr` could be the coefficient of an FD derivative, False otherwise.
    """
    if expr.is_Number:
        return True
    indexeds = [i for i in expr.free_symbols if i.is_Indexed]
    return any(not set(grid.dimensions) <= set(i.function.dimensions) for i in indexeds)


def wset(exprs):
    """
    Extract the working set out of a set of equations.
    """
    return {i.function for i in flatten([e.free_symbols for e in as_tuple(exprs)])
            if i.function.is_AbstractFunction}
