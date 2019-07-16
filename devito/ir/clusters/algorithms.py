from collections import Counter
from itertools import groupby

import sympy

from devito.ir.support import Any, Backward, Forward, IterationSpace, Scope
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.symbolics import CondEq
from devito.tools import DAG, as_tuple, flatten

__all__ = ['clusterize', 'optimize']


def clusterize(exprs):
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
    clusters = optimize(clusters)

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
        A DAG capturing data dependences between ClusterGroups up to a given
        iteration space depth.

        Examples
        --------
        Consider two ClusterGroups `c0` and `c1`, within the iteration space `i`.

        1) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j] ...
           Non-carried flow-dependence, so `cg1` must go after `cg0`

        2) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j+1] ...
           Anti-dependence in `j`, so `cg1` must go after `cg0`

        3) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i-1, j+1] ...
           Flow-dependence in `i`, so `cg1` can safely go before or after `cg0`
           (but clearly still within the `i` iteration space).
           Note: the `j+1` in `cg1` has no impact -- the dependence is in `i`.

        4) cg0 := b[i, j] = ...
           cg1 := ... = ... b[i, j-1] ...
           Flow-dependence in `j`, so `cg1` must go after `cg0`.
           Unlike case 3), the flow-dependence is along an inner Dimension, so
           `cg0` and `cg1 are sequentialized.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            for cg1 in cgroups[n+1:]:
                scope = Scope(exprs=cg0.exprs + cg1.exprs)

                # Handle anti-dependences
                local_deps = cg0.scope.d_anti + cg1.scope.d_anti
                if scope.d_anti - local_deps:
                    dag.add_edge(cg0, cg1)
                    break

                # Flow-dependences along one of the `prefix` Dimensions can
                # be ignored; all others require sequentialization
                local_deps = cg0.scope.d_flow + cg1.scope.d_flow
                if any(not i.cause or not (i.cause & prefix)
                       for i in scope.d_flow - local_deps):
                    dag.add_edge(cg0, cg1)
                    break

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

    def callback(self, clusters, prefix, backlog=None, known_flow_break=None):
        if not prefix:
            return clusters

        # Take the innermost Dimension -- no other Clusters other than those in
        # `clusters` are supposed to share it
        candidates = prefix[-1].dim._defines

        scope = Scope(exprs=flatten(c.exprs for c in clusters))

        # The most nasty case:
        # eq0 := u[t+1, x] = ... u[t, x]
        # eq1 := v[t+1, x] = ... v[t, x] ... u[t, x] ... u[t+1, x] ... u[t+2, x]
        # Here, `eq0` marches forward along `t`, while `eq1` has both a flow and an
        # anti dependence with `eq0`, which ultimately will require `eq1` to go in
        # a separate t-loop
        require_flow_break = (scope.d_flow.cause & scope.d_anti.cause) & candidates
        if require_flow_break and len(clusters) > 1:
            backlog = [clusters[-1]] + (backlog or [])
            # Try with increasingly smaller Cluster groups until the ambiguity is solved
            return self.callback(clusters[:-1], prefix, backlog, require_flow_break)

        # Compute iteration direction
        direction = {d: Backward for d in candidates if d.root in scope.d_anti.cause}
        direction.update({d: Forward for d in candidates if d.root in scope.d_flow.cause})
        direction.update({d: Forward for d in candidates if d not in direction})

        # Enforce iteration direction on each Cluster
        processed = []
        for c in clusters:
            ispace = IterationSpace(c.ispace.intervals, c.ispace.sub_iterators,
                                    {**c.ispace.directions, **direction})
            processed.append(Cluster(c.exprs, ispace, c.dspace))

        if backlog is None:
            return processed

        # Handle the backlog -- the Clusters characterized by flow+anti dependences along
        # one or more Dimensions
        direction = {d: Any for d in known_flow_break}
        for i, c in enumerate(as_tuple(backlog)):
            ispace = IterationSpace(c.ispace.intervals.lift(known_flow_break),
                                    c.ispace.sub_iterators,
                                    {**c.ispace.directions, **direction})
            backlog[i] = Cluster(c.exprs, ispace, c.dspace)

        return processed + self.callback(backlog, prefix)


def optimize(clusters):
    """
    Optimize a topologically-ordered sequence of Clusters by applying the
    following transformations:

        * Fusion
        * Lifting

    Notes
    -----
    This function relies on advanced data dependency analysis tools based upon classic
    Lamport theory.
    """
    # Lifting
    clusters = Lift().process(clusters)

    # Fusion
    clusters = fuse(clusters)

    return clusters


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
        candidates = [c for c in clusters if
                      any(e.is_Tensor for e in c.exprs) and  # Not just scalar exprs
                      not any(e.is_Increment for e in c.exprs) and  # No reductions
                      not c.used_dimensions & hope_invariant]  # Not an invariant ispace
        if not candidates:
            return clusters

        # Now check data dependences
        lifted = []
        processed = []
        for c in clusters:
            impacted = set(clusters) - {c}
            if c in candidates and\
                    not any(set(c.functions) & set(i.scope.writes) for i in impacted):
                # Perform lifting, which requires contracting the iteration space
                key = lambda d: d not in hope_invariant
                ispace = c.ispace.project(key)
                dspace = c.dspace.project(key)
                lifted.append(Cluster(c.exprs, ispace, dspace, guards=c.guards))
            else:
                processed.append(c)

        return lifted + processed


def fuse(clusters):
    """
    Fuse sub-sequences of Clusters with compatible IterationSpace.
    """
    processed = []
    for k, g in groupby(clusters, key=lambda cg: cg.itintervals):
        maybe_fusible = list(g)

        if len(maybe_fusible) == 1 or any(c.guards for c in maybe_fusible):
            processed.extend(maybe_fusible)
        else:
            # Perform fusion
            fused = Cluster.from_clusters(*maybe_fusible)
            processed.append(fused)

    return processed


def guard(clusters):
    """
    Split Clusters containing conditional expressions into separate Clusters.
    """
    processed = []
    for c in clusters:
        free = []
        for e in c.exprs:
            if e.conditionals:
                # Expressions that need no guarding are kept in a separate Cluster
                if free:
                    processed.append(Cluster(free, c.ispace, c.dspace))
                    free = []

                # Create a guarded Cluster
                guards = {}
                for d in e.conditionals:
                    condition = guards.setdefault(d.parent, [])
                    condition.append(d.condition or CondEq(d.parent % d.factor, 0))
                guards = {k: sympy.And(*v, evaluate=False) for k, v in guards.items()}
                processed.append(Cluster(e, c.ispace, c.dspace, guards))
            else:
                free.append(e)
        # Leftover
        if free:
            processed.append(Cluster(free, c.ispace, c.dspace))

    return processed
