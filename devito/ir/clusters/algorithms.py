from collections import Counter
from itertools import groupby

import sympy

from devito.ir.support import Any, Backward, Forward, IterationSpace, Scope
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.symbolics import CondEq, xreplace_indices
from devito.tools import DAG, as_tuple, flatten, filter_ordered, generator
from devito.types import Scalar

__all__ = ['clusterize']


def clusterize(exprs, dse_mode=None):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [Cluster(e, e.ispace, e.dspace) for e in exprs]

    # Compute a topological ordering that honours flow- and anti-dependences.
    # This is necessary prior to enforcing the iteration direction (step below)
    clusters = Toposort().process(clusters)

    # Enforce iteration directions. This turns anti- into flow-dependences by
    # reversing the iteration direction (Backward instead of Forward). A new
    # topological sorting is then computed to expose more fusion opportunities,
    # which will be exploited within `optimize`
    clusters = Enforce().process(clusters)
    clusters = Toposort().process(clusters)

    # Apply optimizations
    clusters = optimize(clusters, dse_mode)

    # Introduce conditional Clusters
    clusters = guard(clusters)

    return ClusterGroup(clusters)


class Queue(object):

    """
    A special queue to process objects in nested IterationSpaces based on
    a divide-and-conquer algorithm.

    Notes
    -----
    Subclasses must override :meth:`callback`, which gets executed upon
    the conquer phase of the algorithm.
    """

    def callback(self, *args):
        raise NotImplementedError

    def process(self, elements):
        return self._process(elements, 1)

    def _process(self, elements, level, prefix=None):
        prefix = prefix or []

        # Divide part
        processed = []
        for pfx, g in groupby(elements, key=lambda i: i.itintervals[:level]):
            if level > len(pfx):
                # Base case
                processed.extend(list(g))
            else:
                # Recursion
                processed.extend(self._process(list(g), level + 1, pfx))

        # Conquer part (execute callback)
        processed = self.callback(processed, prefix)

        return processed


class Toposort(Queue):

    """
    Topologically sort a sequence of Clusters.

    A heuristic, which attempts to maximize Cluster fusion by bringing together
    Clusters with compatible IterationSpace, is used.
    """

    def callback(self, cgroups, prefix):
        cgroups = self._toposort(cgroups, prefix)
        cgroups = self._aggregate(cgroups, prefix)
        return cgroups

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def _toposort(self, cgroups, prefix):
        # Are there any ClusterGroups that could potentially be fused? If not,
        # don't waste time computing a new topological ordering
        counter = Counter(cg.itintervals for cg in cgroups)
        if not any(v > 1 for it, v in counter.most_common()):
            return cgroups

        # Similarly, if all ClusterGroups have the same exact prefix, no need
        # to topologically resort
        if len(counter.most_common()) == 1:
            return cgroups

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            # Heuristic 1: do not move Clusters computing Arrays (temporaries),
            # to preserve cross-loop blocking opportunities
            # Heuristic 2: prefer a node having same IterationSpace as that of
            # the last scheduled node to maximize Cluster fusion
            if not scheduled:
                return queue.pop()
            last = scheduled[-1]
            for i in list(queue):
                if any(f.is_Array for f in i.scope.writes):
                    continue
                elif i.itintervals == last.itintervals:
                    queue.remove(i)
                    return i
            return queue.popleft()

        processed = dag.topological_sort(choose_element)

        return processed

    def _aggregate(self, cgroups, prefix):
        """
        Concatenate a sequence of ClusterGroups into a new ClusterGroup.
        """
        return [ClusterGroup(cgroups, prefix)]

    def _build_dag(self, cgroups, prefix):
        """
        A DAG captures data dependences between ClusterGroups up to the iteration
        space depth dictated by ``prefix``.

        Examples
        --------
        Consider two ClusterGroups `c0` and `c1`, and ``prefix=[i]``.

        1) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j] ...
           Non-carried flow-dependence, so `cg1` must go after `cg0`.

        2) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j-1] ...
           Carried flow-dependence in `j`, so `cg1` must go after `cg0`.

        3) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j+1] ...
           Carried anti-dependence in `j`, so `cg1` must go after `cg0`.

        4) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i-1, j+1] ...
           Carried flow-dependence in `i`, so `cg1` can safely go before or after
           `cg0`. Note: the `j+1` in `cg1` has no impact -- the actual dependence
           betweeb `b[i, j]` and `b[i-1, j+1]` is along `i`.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            for cg1 in cgroups[n+1:]:
                scope = Scope(exprs=cg0.exprs + cg1.exprs)

                # Handle anti-dependences
                deps = scope.d_anti - (cg0.scope.d_anti + cg1.scope.d_anti)
                if any(i.cause & prefix for i in deps):
                    # Anti-dependences break the execution flow
                    # i) ClusterGroups between `cg0` and `cg1` must precede `cg1`
                    for cg2 in cgroups[n:cgroups.index(cg1)]:
                        dag.add_edge(cg2, cg1)
                    # ii) ClusterGroups after `cg1` cannot precede `cg1`
                    for cg2 in cgroups[cgroups.index(cg1)+1:]:
                        dag.add_edge(cg1, cg2)
                    break
                elif deps:
                    dag.add_edge(cg0, cg1)

                # Flow-dependences along one of the `prefix` Dimensions can
                # be ignored; all others require sequentialization
                deps = scope.d_flow - (cg0.scope.d_flow + cg1.scope.d_flow)
                if any(not (i.cause and i.cause & prefix) for i in deps):
                    dag.add_edge(cg0, cg1)
                    continue

                # Handle increment-after-write dependences
                deps = scope.d_output - (cg0.scope.d_output + cg1.scope.d_output)
                if any(i.is_iaw for i in deps):
                    dag.add_edge(cg0, cg1)
                    continue

        return dag


class Enforce(Queue):

    """
    Enforce the iteration direction in a sequence of Clusters based on
    data dependence analysis. The iteration direction will be such that
    the information naturally flows from one iteration to another.

    This will construct a new sequence of Clusters in which only `Forward`
    or `Backward` IterationDirections will appear (i.e., no `Any`).

    Examples
    --------
    In `u[t+1, x] = u[t, x]`, the iteration Dimension `t` gets assigned the
    `Forward` iteration direction, whereas in `u[t-1, x] = u[t, x]` it gets
    assigned `Backward`. The idea is that "to evaluate the LHS at a given
    `t`, we need up-to-date information on the RHS".
    """

    def callback(self, clusters, prefix, backlog=None, known_break=None):
        if not prefix:
            return clusters

        known_break = known_break or set()
        backlog = backlog or []

        # Take the innermost Dimension -- no other Clusters other than those in
        # `clusters` are supposed to share it
        candidates = prefix[-1].dim._defines

        scope = Scope(exprs=flatten(c.exprs for c in clusters))

        # The nastiest case:
        # eq0 := u[t+1, x] = ... u[t, x]
        # eq1 := v[t+1, x] = ... v[t, x] ... u[t, x] ... u[t+1, x] ... u[t+2, x]
        # Here, `eq0` marches forward along `t`, while `eq1` has both a flow and an
        # anti dependence with `eq0`, which ultimately will require `eq1` to go in
        # a separate t-loop
        #
        # Note: in most cases, `scope.d_anti.cause == {}` -- either because
        # `scope.d_anti == {}` or because the few anti dependences are not carried
        # in any dimension. We exploit this observation so that we only compute
        # `d_flow`, which may be expensive, when strictly necessary
        maybe_break = scope.d_anti.cause & candidates
        if len(clusters) > 1 and maybe_break:
            require_break = scope.d_flow.cause & maybe_break
            if require_break:
                backlog = [clusters[-1]] + backlog
                # Try with increasingly smaller ClusterGroups until the ambiguity is gone
                return self.callback(clusters[:-1], prefix, backlog, require_break)

        # If the flow- or anti-dependences are not coupled, one or more Clusters
        # might be scheduled separately, to increase parallelism (this is basically
        # what low-level compilers call "loop fission")
        for n, _ in enumerate(clusters):
            d_cross = scope.d_from_access(scope.a_query(n, 'R')).cross()
            if any(d.is_storage_volatile(candidates) for d in d_cross):
                break
            elif d_cross.cause & candidates:
                if n > 0:
                    return self.callback(clusters[:n], prefix, clusters[n:] + backlog,
                                         (d_cross.cause & candidates) | known_break)
                break

        # Compute iteration direction
        direction = {d: Backward for d in candidates if d.root in scope.d_anti.cause}
        direction.update({d: Forward for d in candidates if d not in direction})

        # Enforce iteration direction on each Cluster
        processed = []
        for c in clusters:
            ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators,
                                    {**c.ispace.directions, **direction})
            processed.append(Cluster(c.exprs, ispace, c.dspace))

        if not backlog:
            return processed

        # Handle the backlog -- the Clusters characterized by flow- and anti-dependences
        # along one or more Dimensions
        direction = {d: Any for d in known_break}
        for i, c in enumerate(list(backlog)):
            ispace = IterationSpace(c.ispace.intervals.lift(known_break),
                                    c.ispace.sub_iterators,
                                    {**c.ispace.directions, **direction})
            dspace = c.dspace.lift(known_break)
            backlog[i] = Cluster(c.exprs, ispace, dspace)

        return processed + self.callback(backlog, prefix)


def optimize(clusters, dse_mode):
    """
    Optimize a topologically-ordered sequence of Clusters by applying the
    following transformations:

        * [cross-cluster] Fusion
        * [intra-cluster] Several flop-reduction passes via the DSE
        * [cross-cluster] Lifting
        * [cross-cluster] Scalarization
        * [cross-cluster] Arrays Elimination
    """
    # To create temporaries
    counter = generator()
    template = lambda: "r%d" % counter()

    # Fusion
    clusters = fuse(clusters)

    from devito.dse import rewrite
    clusters = rewrite(clusters, template, mode=dse_mode)

    # Lifting
    clusters = Lift().process(clusters)

    # Lifting may create fusion opportunities
    clusters = fuse(clusters)

    # Fusion may create opportunities to eliminate Arrays (thus shrinking the
    # working set) if these store identical expressions
    clusters = eliminate_arrays(clusters, template)

    # Fusion may create scalarization opportunities
    clusters = scalarize(clusters, template)

    return ClusterGroup(clusters)


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters

        hope_invariant = {i.dim for i in prefix}

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Increments prevent lifting
            if c.has_increments:
                processed.append(c)
                continue

            # Is `c` a real candidate -- is there at least one invariant Dimension?
            if c.used_dimensions & hope_invariant:
                processed.append(c)
                continue

            impacted = set(processed) | set(clusters[n+1:])

            # None of the Functions appearing in a lifted Cluster can be written to
            if any(c.functions & set(i.scope.writes) for i in impacted):
                processed.append(c)
                continue

            # Scalars prevent lifting if they are read by another Cluster
            swrites = {f for f in c.scope.writes if f.is_Scalar}
            if any(swrites & set(i.scope.reads) for i in impacted):
                processed.append(c)
                continue

            # Perform lifting, which requires contracting the iteration space
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key).reset()
            dspace = c.dspace.project(key).reset()
            lifted.append(Cluster(c.exprs, ispace, dspace, c.guards))

        return lifted + processed


def fuse(clusters):
    """
    Fuse sub-sequences of Clusters with compatible IterationSpace.
    """
    processed = []
    for k, g in groupby(clusters, key=lambda c: set(c.itintervals)):
        maybe_fusible = list(g)

        if len(maybe_fusible) == 1 or any(c.guards for c in maybe_fusible):
            processed.extend(maybe_fusible)
        else:
            try:
                # Perform fusion
                fused = Cluster.from_clusters(*maybe_fusible)
                processed.append(fused)
            except ValueError:
                # We end up here if, for example, some Clusters have same
                # iteration Dimensions but different (partial) orderings
                processed.extend(maybe_fusible)

    return processed


def scalarize(clusters, template):
    """
    Turn local "isolated" Arrays, that is Arrays appearing only in one Cluster,
    into Scalars.
    """
    processed = []
    for c in clusters:
        # Get any Arrays appearing only in `c`
        impacted = set(clusters) - {c}
        arrays = {i for i in c.scope.writes if i.is_Array}
        arrays -= set().union(*[i.scope.reads for i in impacted])

        # Turn them into scalars
        #
        # r[x,y,z] = g(b[x,y,z])                 t0 = g(b[x,y,z])
        # ... = r[x,y,z] + r[x,y,z+1]`  ---->    t1 = g(b[x,y,z+1])
        #                                        ... = t0 + t1
        mapper = {}
        exprs = []
        for e in c.exprs:
            f = e.lhs.function
            if f in arrays:
                for i in filter_ordered(i.indexed for i in c.scope[f]):
                    mapper[i] = Scalar(name=template(), dtype=f.dtype)

                    assert len(f.indices) == len(e.lhs.indices) == len(i.indices)
                    shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                                zip(f.indices, e.lhs.indices, i.indices)}

                    handle = e.func(mapper[i], e.rhs.xreplace(mapper))
                    handle = xreplace_indices(handle, shifting)
                    exprs.append(handle)
            else:
                exprs.append(e.func(e.lhs, e.rhs.xreplace(mapper)))

        processed.append(c.rebuild(exprs))

    return processed


def eliminate_arrays(clusters, template):
    """
    Eliminate redundant expressions stored in Arrays.
    """
    mapper = {}
    processed = []
    for c in clusters:
        if not c.is_dense:
            processed.append(c)
            continue

        # Search for any redundant RHSs
        seen = {}
        for e in c.exprs:
            f = e.lhs.function
            if not f.is_Array:
                continue
            v = seen.get(e.rhs)
            if v is not None:
                # Found a redundant RHS
                mapper[f] = v
            else:
                seen[e.rhs] = f

        if not mapper:
            # Do not waste time
            processed.append(c)
            continue

        # Replace redundancies
        subs = {}
        for f, v in mapper.items():
            for i in filter_ordered(i.indexed for i in c.scope[f]):
                subs[i] = v[f.indices]
        exprs = []
        for e in c.exprs:
            if e.lhs.function in mapper:
                # Drop the write
                continue
            exprs.append(e.xreplace(subs))

        processed.append(c.rebuild(exprs))

    return processed


def guard(clusters):
    """
    Split Clusters containing conditional expressions into separate Clusters.
    """
    processed = []
    for c in clusters:
        # Group together consecutive expressions with same ConditionalDimensions
        for cds, g in groupby(c.exprs, key=lambda e: e.conditionals):
            if not cds:
                processed.append(Cluster(list(g), c.ispace, c.dspace))
                continue

            # Create a guarded Cluster
            guards = {}
            for cd in cds:
                condition = guards.setdefault(cd.parent, [])
                if cd.condition is None:
                    condition.append(CondEq(cd.parent % cd.factor, 0))
                else:
                    condition.append(cd.condition)
            guards = {k: sympy.And(*v, evaluate=False) for k, v in guards.items()}
            processed.append(Cluster(list(g), c.ispace, c.dspace, guards))

    return processed
