from collections import Counter
from itertools import groupby

import sympy

from devito.ir.support import Any, Backward, Forward, IterationSpace, Scope
from devito.ir.clusters.analysis import analyze
from devito.ir.clusters.cluster import Cluster, ClusterGroup
from devito.ir.clusters.queue import Queue
from devito.symbolics import CondEq
from devito.tools import DAG, as_tuple, flatten, timed_pass

__all__ = ['clusterize', 'Toposort']


def clusterize(exprs):
    """
    Turn a sequence of LoweredEqs into a sequence of Clusters.
    """
    # Initialization
    clusters = [Cluster(e, e.ispace, e.dspace) for e in exprs]

    # Compute a topological ordering that honours flow- and anti-dependences
    clusters = Toposort().process(clusters)

    # Setup the IterationSpaces based on data dependence analysis
    clusters = Schedule().process(clusters)

    # Handle ConditionalDimensions
    clusters = guard(clusters)

    # Determine relevant computational properties (e.g., parallelism)
    clusters = analyze(clusters)

    return ClusterGroup(clusters)


class Toposort(Queue):

    """
    Topologically sort a sequence of Clusters.

    A heuristic, which attempts to maximize Cluster fusion by bringing together
    Clusters with compatible IterationSpace, is used.
    """

    @timed_pass(name='toposort')
    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def callback(self, cgroups, prefix):
        cgroups = self._toposort(cgroups, prefix)
        cgroups = self._aggregate(cgroups, prefix)
        return cgroups

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
                    continue

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
          backward carried depedences, then the IterationSpaces are _lifted_
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
        return self._process_fdta(clusters, 1)

    def callback(self, clusters, prefix, backlog=None, known_break=None):
        if not prefix:
            return clusters

        known_break = known_break or set()
        backlog = backlog or []

        # Take the innermost Dimension -- no other Clusters other than those in
        # `clusters` are supposed to share it
        candidates = prefix[-1].dim._defines

        scope = Scope(exprs=flatten(c.exprs for c in clusters))

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
        for i, c in enumerate(list(backlog)):
            ispace = IterationSpace(c.ispace.intervals.lift(known_break),
                                    c.ispace.sub_iterators,
                                    {**c.ispace.directions, **idir})
            dspace = c.dspace.lift(known_break)
            backlog[i] = c.rebuild(ispace=ispace, dspace=dspace)

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
        for cds, g in groupby(c.exprs, key=lambda e: e.conditionals):
            if not cds:
                processed.append(c.rebuild(exprs=list(g)))
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
            processed.append(c.rebuild(exprs=list(g), guards=guards))

    return ClusterGroup(processed)
