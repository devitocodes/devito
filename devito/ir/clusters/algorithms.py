from collections import OrderedDict
from functools import partial
from itertools import groupby

import numpy as np
import sympy

from devito.exceptions import CompilationError
from devito.finite_differences.elementary import Max, Min
from devito.ir.support import (Any, Backward, Forward, IterationSpace, erange,
                               pull_dims)
from devito.ir.equations import OpMin, OpMax, identity_mapper
from devito.ir.clusters.analysis import analyze
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.ir.clusters.visitors import Queue, cluster_pass
from devito.ir.support import Scope
from devito.mpi.halo_scheme import HaloScheme, HaloTouch
from devito.mpi.reduction_scheme import DistReduce
from devito.symbolics import (limits_mapper, retrieve_indexed, uxreplace,
                              xreplace_indices)
from devito.tools import (DefaultOrderedDict, Stamp, as_mapper, flatten,
                          is_integer, split, timed_pass, toposort)
from devito.types import Array, Eq, Symbol
from devito.types.dimension import BOTTOM, ModuloDimension

__all__ = ['clusterize']


def clusterize(exprs, **kwargs):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [Cluster(e, e.ispace) for e in exprs]

    # Setup the IterationSpaces based on data dependence analysis
    clusters = impose_total_ordering(clusters)
    clusters = Schedule().process(clusters)

    # Handle SteppingDimensions
    clusters = Stepper(**kwargs).process(clusters)

    # Handle ConditionalDimensions
    clusters = guard(clusters)

    # Determine relevant computational properties (e.g., parallelism)
    clusters = analyze(clusters)

    # Input normalization
    clusters = normalize(clusters, **kwargs)

    # Derive the necessary communications for distributed-memory parallelism
    clusters = communications(clusters)

    return ClusterGroup(clusters)


def impose_total_ordering(clusters):
    """
    Create a new sequence of Clusters whose IterationSpaces are totally ordered
    according to a global set of relations.
    """
    global_relations = set().union(*[c.ispace.relations for c in clusters])
    ordering = toposort(global_relations)

    processed = []
    for c in clusters:
        key = lambda d: ordering.index(d)
        try:
            relations = {tuple(sorted(c.ispace.itdims, key=key))}
        except ValueError:
            # See issue #2204
            relations = c.ispace.relations
        ispace = c.ispace.reorder(relations=relations, mode='total')

        processed.append(c.rebuild(ispace=ispace))

    return processed


class Schedule(Queue):

    """
    This special Queue produces a new sequence of "scheduled" Clusters, which
    means that:

        * The iteration direction along each Dimension of each Cluster is such
          that the information "naturally flows from one iteration to another".
          For example, in `u[t+1, x] = u[t, x]`, the iteration Dimension `t`
          gets assigned the `Forward` direction, to honor the flow-dependence
          along `t`. Instead, in `u[t-1, x] = u[t, x]`, `t` gets assigned the
          `Backward` direction. This simple rule ensures that when we evaluate
          the LHS, the information on the RHS is up-to-date.

        * If a Cluster has both a flow- and an anti-dependence along a given
          Dimension `x`, then `x` is assigned the `Forward` direction but its
          IterationSpace is _lifted_ such that it cannot be fused with any
          other Clusters within the same iteration Dimension `x`. For example,
          consider the following coupled statements:

            - `u[t+1, x] = f(u[t, x])`
            - `v[t+1, x] = g(v[t, x], u[t, x], u[t+1, x], u[t+2, x]`

          The first statement has a flow-dependence along `t`, while the second
          one has both a flow- and an anti-dependence along `t`, hence the two
          statements will ultimately be kept in separate Clusters and then
          scheduled to different loop nests.

        * If *all* dependences across two Clusters along a given Dimension are
          backward carried dependences, then the IterationSpaces are _lifted_
          such that the two Clusters cannot be fused. This is to maximize
          the number of parallel Dimensions. Essentially, this is what low-level
          compilers call "loop fission" -- only that here it occurs at a much
          higher level of abstraction. For example:

            - `u[x+1] = w[x] + v[x]`
            - `v[x] = u[x] + w[x]

          Here, the two statements will ultimately be kept in separate Clusters
          and then scheduled to different loops; this way, `x` will be a parallel
          Dimension in both Clusters.
    """

    @timed_pass(name='schedule')
    def process(self, clusters):
        return self._process_fatd(clusters, 1)

    def callback(self, clusters, prefix, backlog=None, known_break=None):
        if not prefix:
            return clusters

        known_break = known_break or set()
        backlog = backlog or []

        # Take the innermost Dimension -- no other Clusters other than those in
        # `clusters` are supposed to share it
        candidates = prefix[-1].dim._defines

        scope = Scope(flatten(c.exprs for c in clusters))

        # Handle the nastiest case -- ambiguity due to the presence of both a
        # flow- and an anti-dependence.
        #
        # Note: in most cases, `scope.d_anti.cause == {}` -- either because
        # `scope.d_anti == {}` or because the few anti dependences are not carried
        # in any Dimension. We exploit this observation so that we only compute
        # `d_flow`, which instead may be expensive, when strictly necessary
        maybe_break = scope.d_anti.cause & candidates
        if len(clusters) > 1 and maybe_break:
            require_break = scope.d_flow.cause & maybe_break
            if require_break:
                backlog = [clusters[-1]] + backlog
                # Try with increasingly smaller ClusterGroups until the
                # ambiguity is gone
                return self.callback(clusters[:-1], prefix, backlog, require_break)

        # Schedule Clusters over different IterationSpaces if this increases
        # parallelism
        for i in range(1, len(clusters)):
            if self._break_for_parallelism(scope, candidates, i):
                return self.callback(clusters[:i], prefix, clusters[i:] + backlog,
                                     candidates | known_break)

        # Compute iteration direction
        idir = {d: Backward for d in candidates if d.root in scope.d_anti.cause}
        if maybe_break:
            idir.update({d: Forward for d in candidates if d.root in scope.d_flow.cause})
        idir.update({d: Forward for d in candidates if d not in idir})

        # Enforce iteration direction on each Cluster
        processed = []
        for c in clusters:
            ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators,
                                    {**c.ispace.directions, **idir})
            processed.append(c.rebuild(ispace=ispace))

        if not backlog:
            return processed

        # Handle the backlog -- the Clusters characterized by flow- and
        # anti-dependences along one or more Dimensions
        idir = {d: Any for d in known_break}
        stamp = Stamp()
        for i, c in enumerate(list(backlog)):
            ispace = IterationSpace(c.ispace.intervals.lift(known_break, stamp),
                                    c.ispace.sub_iterators,
                                    {**c.ispace.directions, **idir})
            backlog[i] = c.rebuild(ispace=ispace)

        return processed + self.callback(backlog, prefix)

    def _break_for_parallelism(self, scope, candidates, i):
        # `test` will be True if there's at least one data-dependence that would
        # break parallelism
        test = False
        for d in scope.d_from_access_gen(scope.a_query(i)):
            if d.is_local or d.is_storage_related(candidates):
                # Would break a dependence on storage
                return False
            if any(d.is_carried(i) for i in candidates):
                if (d.is_flow and d.is_lex_negative) or (d.is_anti and d.is_lex_positive):
                    # Would break a data dependence
                    return False
            test = test or (bool(d.cause & candidates) and not d.is_lex_equal)
        return test


@timed_pass()
def guard(clusters):
    """
    Split Clusters containing conditional expressions into separate Clusters.
    """
    processed = []
    for c in clusters:
        # Group together consecutive expressions with same ConditionalDimensions
        for cds, g in groupby(c.exprs, key=lambda e: tuple(e.conditionals)):
            exprs = list(g)

            if not cds:
                processed.append(c.rebuild(exprs=exprs))
                continue

            # Separate out the indirect ConditionalDimensions, which only serve
            # the purpose of protecting from OOB accesses
            cds = [d for d in cds if not d.indirect]

            # Chain together all `cds` conditions from all expressions in `c`
            guards = {}
            for cd in cds:
                # `BOTTOM` parent implies a guard that lives outside of
                # any iteration space, which corresponds to the placeholder None
                if cd.parent is BOTTOM:
                    d = None
                else:
                    d = cd.parent

                # If `cd` uses, as condition, an arbitrary SymPy expression, then
                # we must ensure to nest it inside the last of the Dimensions
                # appearing in `cd.condition`
                if cd._factor is not None:
                    k = d
                else:
                    dims = pull_dims(cd.condition, flag=False)
                    k = max(dims, default=d, key=lambda i: c.ispace.index(i))

                # Pull `cd` from any expr
                condition = guards.setdefault(k, [])
                for e in exprs:
                    try:
                        condition.append(e.conditionals[cd])
                        break
                    except KeyError:
                        pass

                # Remove `cd` from all `exprs` since this will be now encoded
                # globally at the Cluster level
                for i, e in enumerate(list(exprs)):
                    conditionals = dict(e.conditionals)
                    conditionals.pop(cd, None)
                    exprs[i] = e.func(*e.args, conditionals=conditionals)

            guards = {d: sympy.And(*v, evaluate=False) for d, v in guards.items()}

            # Construct a guarded Cluster
            processed.append(c.rebuild(exprs=exprs, guards=guards))

    return ClusterGroup(processed)


class Stepper(Queue):

    """
    Produce a new sequence of Clusters in which the IterationSpaces carry the
    sub-iterators induced by a SteppingDimension.
    """

    def __init__(self, sregistry=None, **kwargs):
        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        subiters = flatten([c.ispace.sub_iterators[d] for c in clusters])
        subiters = {i for i in subiters if i.is_Stepping}
        if not subiters:
            return clusters

        # Collect the index access functions along `d`, e.g., `t + 1` where `t` is
        # a SteppingDimension for `d = time`
        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(set))
        for c in clusters:
            indexeds = c.scope.indexeds

            for i in indexeds:
                try:
                    iaf = i.indices[d]
                except KeyError:
                    continue

                # Sanity checks
                sis = iaf.free_symbols & subiters
                if len(sis) == 0:
                    continue
                elif len(sis) == 1:
                    si = sis.pop()
                else:
                    raise CompilationError("Cannot use multiple SteppingDimensions "
                                           "to index into a Function")
                size = i.function.shape_allocated[d]
                assert is_integer(size)

                # Resolve StencilDimensions in case of unexpanded expressions
                # E.g. `i0 + t` -> `(t - 1, t, t + 1)`
                iafs = erange(iaf)

                mapper[size][si].update(iafs)

        # Construct the ModuloDimensions
        mds = []
        for size, v in mapper.items():
            for si, iafs in list(v.items()):
                # Offsets are sorted so that the semantic order (t0, t1, t2) follows
                # SymPy's index ordering (t, t-1, t+1) afer modulo replacement so
                # that associativity errors are consistent. This corresponds to
                # sorting offsets {-1, 0, 1} as {0, -1, 1} assigning -inf to 0
                key = lambda i: -np.inf if i - si == 0 else (i - si)
                siafs = sorted(iafs, key=key)

                for iaf in siafs:
                    name = self.sregistry.make_name(prefix='t')
                    offset = uxreplace(iaf, {si: d.root})
                    mds.append(ModuloDimension(name, si, offset, size, origin=iaf))

        # Replacement rule for ModuloDimensions
        def rule(size, e):
            try:
                return e.function.shape_allocated[d] == size
            except (AttributeError, KeyError, ValueError):
                return False

        # Reconstruct the Clusters
        processed = []
        for c in clusters:
            exprs = c.exprs

            sub_iterators = dict(c.ispace.sub_iterators)
            sub_iterators[d] = [i for i in sub_iterators[d] if i not in subiters]

            # Apply substitutions to expressions
            # Note: In an expression, there could be `u[t+1, ...]` and `v[t+1,
            # ...]`, where `u` and `v` are TimeFunction with circular time
            # buffers (save=None) *but* different modulo extent. The `t+1`
            # indices above are therefore conceptually different, so they will
            # be replaced with the proper ModuloDimension through two different
            # calls to `xreplace_indices`
            groups = as_mapper(mds, lambda d: d.modulo)
            for size, v in groups.items():
                key = partial(rule, size)
                subs = {md.origin: md for md in v}
                sub_iterators[d].extend(v)

                func = partial(xreplace_indices, mapper=subs, key=key)
                exprs = [e.apply(func) for e in exprs]

            ispace = IterationSpace(c.ispace.intervals, sub_iterators,
                                    c.ispace.directions)

            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

        return processed


@timed_pass(name='communications')
def communications(clusters):
    """
    Enrich a sequence of Clusters by adding special Clusters representing data
    communications for distributed parallelism.
    """
    clusters = HaloComms().process(clusters)
    clusters = reduction_comms(clusters)

    return clusters


class HaloComms(Queue):

    """
    Inject Clusters representing halo exchanges for distributed-memory parallelism.
    """

    _q_guards_in_key = True
    _q_properties_in_key = True

    B = Symbol(name='⊥')

    def process(self, clusters):
        return self._process_fatd(clusters, 1, seen=set())

    def callback(self, clusters, prefix, seen=None):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        # Construct a representation of the halo accesses
        processed = []
        for c in clusters:
            if c.properties.is_sequential(d) or \
               c in seen:
                continue

            hs = HaloScheme(c.exprs, c.ispace)
            if hs.is_void or \
               not d._defines & hs.distributed_aindices:
                continue

            points = set()
            for f in hs.fmapper:
                for a in c.scope.getreads(f):
                    points.add(a.access)

            # We also add all written symbols to ultimately create mock WARs
            # with `c`, which will prevent the newly created HaloTouch to ever
            # be rescheduled after `c` upon topological sorting
            points.update(a.access for a in c.scope.accesses if a.is_write)

            # Sort for determinism
            # NOTE: not sorting might impact code generation. The order of
            # the args is important because that's what search functions honor!
            points = sorted(points, key=str)

            # Construct the HaloTouch Cluster
            expr = Eq(self.B, HaloTouch(*points, halo_scheme=hs))

            key = lambda i: i in prefix[:-1] or i in hs.loc_indices
            ispace = c.ispace.project(key)
            # HaloTouches are not parallel
            properties = c.properties.sequentialize()

            halo_touch = c.rebuild(exprs=expr, ispace=ispace, properties=properties)

            processed.append(halo_touch)
            seen.update({halo_touch, c})

        processed.extend(clusters)

        return processed


def reduction_comms(clusters):
    processed = []
    fifo = []

    def _update(reductions):
        for _, reds in groupby(reductions, key=lambda r: r.ispace):
            reds = list(reds)
            exprs = flatten([dr.exprs for dr in reds])
            processed.append(reds[0].rebuild(exprs=exprs))

    for c in clusters:
        # Schedule the global distributed reductions encountered before `c`,
        # if `c`'s IterationSpace is such that the reduction can be carried out
        found, fifo = split(fifo, lambda dr: dr.ispace.is_subset(c.ispace))
        _update(found)

        # Detect the global distributed reductions in `c`
        for e in c.exprs:
            op = e.operation
            if op is None or c.is_sparse:
                continue

            var = e.lhs
            grid = c.grid
            if grid is None:
                continue

            # Is Inc/Max/Min/... actually used for a reduction?
            ispace = c.ispace.project(lambda d: d in var.free_symbols)
            if ispace.itdims == c.ispace.itdims:
                continue

            # The reduced Dimensions
            rdims = set(c.ispace.itdims) - set(ispace.itdims)

            # The reduced Dimensions inducing a global distributed reduction
            grdims = {d for d in rdims if d._defines & c.dist_dimensions}
            if not grdims:
                continue

            # The IterationSpace within which the global distributed reduction
            # must be carried out
            ispace = c.ispace.prefix(lambda d: d in var.free_symbols)
            expr = [Eq(var, DistReduce(var, op=op, grid=grid, ispace=ispace))]
            fifo.append(c.rebuild(exprs=expr, ispace=ispace))

        processed.append(c)

    # Leftover reductions are placed at the very end
    _update(fifo)

    return processed


def normalize(clusters, sregistry=None, options=None, platform=None, **kwargs):
    clusters = normalize_nested_indexeds(clusters, sregistry)
    if options['mapify-reduce']:
        clusters = normalize_reductions_dense(clusters, sregistry, platform)
    else:
        clusters = normalize_reductions_minmax(clusters)
    clusters = normalize_reductions_sparse(clusters, sregistry)

    return clusters


@cluster_pass(mode='all')
def normalize_nested_indexeds(cluster, sregistry):
    """
    Recursively extract nested Indexeds in to temporaries.
    """

    def pull_indexeds(expr, subs, mapper, parent=None):
        for i in retrieve_indexed(expr):
            if i in mapper:
                continue

            for e in i.indices:
                pull_indexeds(e, subs, mapper, parent=i)

            if parent is not None:
                # Nested Indexed, requires a temporary
                k = i.xreplace(mapper)
                v = Symbol(name=sregistry.make_name(), dtype=i.function.dtype)
                subs[k] = v

                # Update substitution status
                mapper[i] = v

    processed = []
    for e in cluster.exprs:
        subs = OrderedDict()
        pull_indexeds(e, subs, {})

        # Construct temporaries and apply substitution to `e`, in cascade
        for k, v in subs.items():
            processed.append(Eq(v, k))
            e = e.xreplace({k: v})
        processed.append(e)

    return cluster.rebuild(processed)


@cluster_pass(mode='dense')
def normalize_reductions_minmax(cluster):
    """
    Initialize the reduction variables to their neutral element and use them
    to compute the reduction.
    """
    dims = [d for d in cluster.ispace.itdims
            if cluster.properties.is_parallel_atomic(d)]
    if not dims:
        return cluster

    init = []
    processed = []
    for e in cluster.exprs:
        lhs, rhs = e.args
        f = lhs.function

        if e.operation is OpMin:
            if not f.is_Input:
                expr = Eq(lhs, limits_mapper[lhs.dtype].max)
                ispace = cluster.ispace.project(lambda i: i not in dims)
                init.append(cluster.rebuild(exprs=expr, ispace=ispace))

            processed.append(e.func(lhs, Min(lhs, rhs)))

        elif e.operation is OpMax:
            if not f.is_Input:
                expr = Eq(lhs, limits_mapper[lhs.dtype].min)
                ispace = cluster.ispace.project(lambda i: i not in dims)
                init.append(cluster.rebuild(exprs=expr, ispace=ispace))

            processed.append(e.func(lhs, Max(lhs, rhs)))

        else:
            processed.append(e)

    return init + [cluster.rebuild(processed)]


def normalize_reductions_dense(cluster, sregistry, platform):
    """
    Extract the right-hand sides of reduction Eq's in to temporaries.
    """
    return _normalize_reductions_dense(cluster, {}, sregistry, platform)


@cluster_pass(mode='dense')
def _normalize_reductions_dense(cluster, mapper, sregistry, platform):
    """
    Transform augmented expressions whose left-hand side is a scalar into
    map-reduces.

    Examples
    --------
    Given an increment expression such as

        s += f(u[x], v[x], ...)

    Turn it into

        r[x] = f(u[x], v[x], ...)
        s += r[x]
    """
    # The candidate Dimensions along which to perform the map part
    candidates = [d for d in cluster.ispace.itdims
                  if cluster.properties.is_parallel_atomic(d)]
    if not candidates:
        return cluster

    # If there are more parallel dimensions than the maximum allowed by the
    # target platform, we must restrain the number of candidates
    max_par_dims = platform.limits()['max-par-dims']
    dims = candidates[-max_par_dims:]

    # All other dimensions must be sequentialized because the output Array
    # is constrained to `dims`
    sequentialize = candidates[:-max_par_dims]

    processed = []
    properties = cluster.properties
    for e in cluster.exprs:
        if e.is_Reduction:
            lhs, rhs = e.args

            try:
                f = rhs.function
            except AttributeError:
                f = None

            if lhs.function.is_Array:
                # Probably a compiler-generated reduction, e.g. via
                # recursive compilation; it's an Array already, so nothing to do
                processed.append(e)
            elif rhs in mapper:
                # Seen this RHS already, so reuse the Array that was created for it
                processed.append(e.func(lhs, mapper[rhs].indexify()))
            elif f and f.is_Array and sum(flatten(f._size_nodomain)) == 0:
                # Special case: the RHS is an Array with no halo/padding, meaning
                # that the written data values are contiguous in memory, hence
                # we can simply reuse the Array itself as we're already in the
                # desired memory layout
                processed.append(e)
            else:
                name = sregistry.make_name()
                try:
                    grid = cluster.grid
                except ValueError:
                    grid = None
                a = mapper[rhs] = Array(name=name, dtype=e.dtype, dimensions=dims,
                                        grid=grid)

                # Populate the Array (the "map" part)
                processed.append(e.func(a.indexify(), rhs, operation=None))

                # Set all untouched entried to the identity value if necessary
                if e.conditionals:
                    nc = {d: sympy.Not(v) for d, v in e.conditionals.items()}
                    v = identity_mapper[e.lhs.dtype][e.operation]
                    processed.append(
                        e.func(a.indexify(), v, operation=None, conditionals=nc)
                    )

                processed.append(e.func(lhs, a.indexify()))

                for d in sequentialize:
                    properties = properties.sequentialize(d)
        else:
            processed.append(e)

    return cluster.rebuild(exprs=processed, properties=properties)


@cluster_pass(mode='sparse')
def normalize_reductions_sparse(cluster, sregistry):
    """
    Extract the right-hand sides of reduction Eq's in to temporaries.
    """
    processed = []
    for e in cluster.exprs:
        if e.is_Reduction and e.lhs.is_Indexed:
            # Transform `e` such that we reduce into a scalar (ultimately via
            # atomic ops, though this part is carried out by a much later pass)
            # For example, given `i = m[p_src]` (i.e., indirection array), turn:
            # `u[t, i] += f(u[t, i], src, ...)`
            # into
            # `s = f(u[t, i], src, ...)`
            # `u[t, i] += s`
            name = sregistry.make_name()
            v = Symbol(name=name, dtype=e.dtype)
            processed.extend([e.func(v, e.rhs, operation=None),
                              e.func(e.lhs, v)])
        else:
            processed.append(e)

    return cluster.rebuild(processed)
