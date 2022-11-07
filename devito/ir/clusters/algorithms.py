from collections import OrderedDict
from functools import partial
from itertools import groupby

import numpy as np
import sympy

from devito.exceptions import InvalidOperator
from devito.ir.support import (Any, Backward, Forward, IterationSpace,
                               PARALLEL_IF_ATOMIC, pull_dims)
from devito.ir.clusters.analysis import analyze
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.ir.clusters.visitors import Queue, QueueStateful, cluster_pass
from devito.symbolics import retrieve_indexed, uxreplace, xreplace_indices
from devito.tools import (DefaultOrderedDict, Stamp, as_mapper, flatten,
                          is_integer, timed_pass)
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
    clusters = Schedule().process(clusters)

    # Handle SteppingDimensions
    clusters = Stepper().process(clusters)

    # Handle ConditionalDimensions
    clusters = guard(clusters)

    # Determine relevant computational properties (e.g., parallelism)
    clusters = analyze(clusters)

    # Input normalization (e.g., SSA)
    clusters = normalize(clusters, **kwargs)

    return ClusterGroup(clusters)


class Schedule(QueueStateful):

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

        scope = self._fetch_scope(clusters)

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
                # Try with increasingly smaller ClusterGroups until the ambiguity is gone
                return self.callback(clusters[:-1], prefix, backlog, require_break)

        # Schedule Clusters over different IterationSpaces if this increases parallelism
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

        # Handle the backlog -- the Clusters characterized by flow- and anti-dependences
        # along one or more Dimensions
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
                    dims = pull_dims(cd.condition)
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

    def process(self, clusters):
        return self._process_fdta(clusters, 1)

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
                    raise InvalidOperator("Cannot use multiple SteppingDimensions "
                                          "to index into a Function")
                size = i.function.shape_allocated[d]
                assert is_integer(size)

                mapper[size][si].add(iaf)

        # Construct the ModuloDimensions
        mds = OrderedDict()
        for size, v in mapper.items():
            for si, iafs in list(v.items()):
                # Offsets are sorted so that the semantic order (t0, t1, t2) follows
                # SymPy's index ordering (t, t-1, t+1) afer modulo replacement so
                # that associativity errors are consistent. This corresponds to
                # sorting offsets {-1, 0, 1} as {0, -1, 1} assigning -inf to 0
                siafs = sorted(iafs, key=lambda i: -np.inf if i - si == 0 else (i - si))

                # Create the ModuloDimensions. Note that if `size < len(iafs)` then
                # the same ModuloDimension may be used for multiple offsets
                for iaf in siafs[:size]:
                    name = '%s%d' % (si.name, len(mds))
                    offset = uxreplace(iaf, {si: d.root})
                    md = ModuloDimension(name, si, offset, size, origin=iaf)

                    key = lambda i: i.subs(si, 0) % size
                    mds[md] = [i for i in siafs if key(i) == key(iaf)]

        # Replacement rule for ModuloDimensions
        def rule(size, e):
            try:
                return e.function.shape_allocated[d] == size
            except (AttributeError, KeyError, ValueError):
                return False

        # Reconstruct the Clusters
        processed = []
        for c in clusters:
            # Apply substitutions to expressions
            # Note: In an expression, there could be `u[t+1, ...]` and `v[t+1,
            # ...]`, where `u` and `v` are TimeFunction with circular time
            # buffers (save=None) *but* different modulo extent. The `t+1`
            # indices above are therefore conceptually different, so they will
            # be replaced with the proper ModuloDimension through two different
            # calls to `xreplace_indices`
            exprs = c.exprs
            groups = as_mapper(mds, lambda d: d.modulo)
            for size, v in groups.items():
                mapper = {}
                for md in v:
                    mapper.update({i: md for i in mds[md]})

                func = partial(xreplace_indices, mapper=mapper, key=partial(rule, size))
                exprs = [e.apply(func) for e in exprs]

            # Augment IterationSpace
            sub_iterators = dict(c.ispace.sub_iterators)
            sub_iterators[d] = tuple(i for i in sub_iterators[d] + tuple(mds)
                                     if i not in subiters)
            ispace = IterationSpace(c.ispace.intervals, sub_iterators,
                                    c.ispace.directions)

            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

        return processed


def normalize(clusters, **kwargs):
    options = kwargs['options']
    sregistry = kwargs['sregistry']

    clusters = normalize_nested_indexeds(clusters, sregistry)
    clusters = normalize_reductions(clusters, sregistry, options)

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


@cluster_pass(mode='all')
def normalize_reductions(cluster, sregistry, options):
    """
    Extract the right-hand sides of reduction Eq's in to temporaries.
    """
    opt_mapify_reduce = options['mapify-reduce']

    dims = [d for d, v in cluster.properties.items() if PARALLEL_IF_ATOMIC in v]

    if not dims:
        return cluster

    processed = []
    for e in cluster.exprs:
        if e.is_Reduction and e.lhs.is_Indexed and cluster.is_sparse:
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

        elif e.is_Reduction and e.lhs.is_Symbol and opt_mapify_reduce:
            # Transform `e` into what is in essence an explicit map-reduce
            # For example, turn:
            # `s += f(u[x], v[x], ...)`
            # into
            # `r[x] = f(u[x], v[x], ...)`
            # `s += r[x]`
            # This makes it much easier to parallelize the map part regardless
            # of the target backend
            name = sregistry.make_name()
            a = Array(name=name, dtype=e.dtype, dimensions=dims)
            processed.extend([Eq(a.indexify(), e.rhs),
                              e.func(e.lhs, a.indexify())])

        else:
            processed.append(e)

    return cluster.rebuild(processed)
