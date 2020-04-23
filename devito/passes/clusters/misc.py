from collections import Counter
from itertools import groupby

from devito.ir.clusters import Cluster, ClusterGroup, Queue
from devito.ir.support import TILABLE, Scope
from devito.passes.clusters.utils import cluster_pass
from devito.symbolics import pow_to_mul, uxreplace
from devito.tools import DAG, as_tuple, filter_ordered, timed_pass
from devito.types import Scalar

__all__ = ['Lift', 'fuse', 'eliminate_arrays', 'optimize_pows', 'extract_increments']


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    @timed_pass(name='lift')
    def process(self, elements):
        return super(Lift, self).process(elements)

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
            swrites = {f for f in c.scope.writes if f.is_Symbol}
            if any(swrites & set(i.scope.reads) for i in impacted):
                processed.append(c)
                continue

            # Contract iteration and data spaces for the lifted Cluster
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key).reset()
            dspace = c.dspace.project(key).reset()

            # Some properties need to be dropped
            properties = {d: v for d, v in c.properties.items() if key(d)}
            properties = {d: v - {TILABLE} for d, v in properties.items()}

            lifted.append(c.rebuild(ispace=ispace, dspace=dspace, properties=properties))

        return lifted + processed


class Fusion(Queue):

    """
    Fuse Clusters with compatible IterationSpace.
    """

    def __init__(self, toposort):
        super(Fusion, self).__init__()
        self.toposort = toposort

    def process(self, clusters):
        cgroups = [ClusterGroup(c, c.itintervals) for c in clusters]
        cgroups = self._process_fdta(cgroups, 1)
        clusters = ClusterGroup.concatenate(*cgroups)
        return clusters

    def callback(self, cgroups, prefix):
        # Toposort to maximize fusion
        if self.toposort:
            clusters = self._toposort(cgroups, prefix)
        else:
            clusters = ClusterGroup(cgroups)

        # Fusion
        processed = []
        for k, g in groupby(clusters, key=lambda c: (set(c.itintervals), c.guards)):
            maybe_fusible = list(g)

            if len(maybe_fusible) == 1:
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

        return [ClusterGroup(processed, prefix)]

    def _toposort(self, cgroups, prefix):
        # Are there any ClusterGroups that could potentially be fused? If
        # not, do not waste time computing a new topological ordering
        counter = Counter(cg.itintervals for cg in cgroups)
        if not any(v > 1 for it, v in counter.most_common()):
            return ClusterGroup(cgroups)

        # Similarly, if all ClusterGroups have the same exact prefix, no
        # need to attempt a topological sorting
        if len(counter.most_common()) == 1:
            return ClusterGroup(cgroups)

        dag = self._build_dag(cgroups, prefix)

        def choose_element(queue, scheduled):
            # Heuristic: prefer a node having same IterationSpace as that
            # of the last scheduled node to maximize Cluster fusion
            if not scheduled:
                return queue.pop()
            last = scheduled[-1]
            for i in list(queue):
                if i.itintervals == last.itintervals:
                    queue.remove(i)
                    return i
            return queue.popleft()

        return ClusterGroup(dag.topological_sort(choose_element))

    def _build_dag(self, cgroups, prefix):
        """
        A DAG representing the data dependences across the ClusterGroups within
        a given scope.
        """
        prefix = {i.dim for i in as_tuple(prefix)}

        dag = DAG(nodes=cgroups)
        for n, cg0 in enumerate(cgroups):
            for cg1 in cgroups[n+1:]:
                # A Scope to compute all cross-ClusterGroup anti-dependences
                rule = lambda i: i.is_cross
                scope = Scope(exprs=cg0.exprs + cg1.exprs, rules=rule)

                # Optimization: we exploit the following property:
                # no prefix => (edge <=> at least one (any) dependence)
                # to jump out of this potentially expensive loop as quickly as possible
                if not prefix and any(scope.d_all_gen()):
                    dag.add_edge(cg0, cg1)

                # Anti-dependences along `prefix` break the execution flow
                # (intuitively, "the loop nests are to be kept separated")
                # * All ClusterGroups between `cg0` and `cg1` must precede `cg1`
                # * All ClusterGroups after `cg1` cannot precede `cg1`
                elif any(i.cause & prefix for i in scope.d_anti_gen()):
                    for cg2 in cgroups[n:cgroups.index(cg1)]:
                        dag.add_edge(cg2, cg1)
                    for cg2 in cgroups[cgroups.index(cg1)+1:]:
                        dag.add_edge(cg1, cg2)
                    break

                # Any anti- and iaw-dependences impose that `cg1` follows `cg0`
                # while not being its immediate successor (unless it already is),
                # to avoid they are fused together (thus breaking the dependence)
                elif any(scope.d_anti_gen()) or\
                        any(i.is_iaw for i in scope.d_output_gen()):
                    dag.add_edge(cg0, cg1)
                    index = cgroups.index(cg1) - 1
                    if index > n:
                        dag.add_edge(cg0, cgroups[index])
                        dag.add_edge(cgroups[index], cg1)

                # Any flow-dependences along an inner Dimension (i.e., a Dimension
                # that doesn't appear in `prefix`) impose that `cg1` follows `cg0`
                elif any(not (i.cause and i.cause & prefix) for i in scope.d_flow_gen()):
                    dag.add_edge(cg0, cg1)

        return dag


@timed_pass()
def fuse(clusters, toposort=False):
    """
    Clusters fusion.

    If ``toposort=True``, then the Clusters are reordered to maximize the likelihood
    of fusion; the new ordering is computed such that all data dependencies are honored.
    """
    return Fusion(toposort=toposort).process(clusters)


@timed_pass()
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
                subs[i] = v[i.indices]
        exprs = []
        for e in c.exprs:
            if e.lhs.function in mapper:
                # Drop the write
                continue
            exprs.append(uxreplace(e, subs))

        processed.append(c.rebuild(exprs))

    return processed


@cluster_pass(mode='all')
def optimize_pows(cluster, *args):
    """
    Convert integer powers into Muls, such as ``a**2 => a*a``.
    """
    return cluster.rebuild(exprs=[pow_to_mul(e) for e in cluster.exprs])


@cluster_pass(mode='sparse')
def extract_increments(cluster, template, *args):
    """
    Extract the RHSs of non-local tensor expressions performing an associative
    and commutative increment, and assign them to temporaries.
    """
    processed = []
    for e in cluster.exprs:
        if e.is_Increment and e.lhs.function.is_Input:
            handle = Scalar(name=template(), dtype=e.dtype).indexify()
            if e.rhs.is_Number or e.rhs.is_Symbol:
                extracted = e.rhs
            else:
                extracted = e.rhs.func(*[i for i in e.rhs.args if i != e.lhs])
            processed.extend([e.func(handle, extracted, is_Increment=False),
                              e.func(e.lhs, handle)])
        else:
            processed.append(e)

    return cluster.rebuild(processed)
