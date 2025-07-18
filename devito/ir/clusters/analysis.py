from devito.ir.clusters.cluster import Cluster
from devito.ir.clusters.visitors import Queue
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC,
                               SEQUENTIAL, Property, Scope)
from devito.ir.support.space import IterationSpace
from devito.tools import as_tuple, flatten, timed_pass
from devito.types.dimension import Dimension

__all__ = ['analyze']


# Describes properties fetched by a `Detector`
Properties = dict[Cluster, dict[Dimension, set[Property]]]


@timed_pass()
def analyze(clusters):
    properties: Properties = {}

    # Collect properties
    clusters = Parallelism().process(clusters, properties=properties)
    clusters = Affiness().process(clusters, properties=properties)

    # Reconstruct Clusters attaching the discovered properties
    processed = [c.rebuild(properties=properties.get(c)) for c in clusters]

    return processed


class Detector(Queue):

    def process(self, clusters: list[Cluster], properties: Properties) -> list[Cluster]:
        return self._process_fatd(clusters, 1, properties=properties)

    def callback(self, clusters: list[Cluster], prefix: IterationSpace | None,
                 properties: Properties) -> list[Cluster]:
        if not prefix:
            return clusters

        # The analyzed Dimension
        d = prefix[-1].dim

        # Apply the actual callback
        retval = self._callback(clusters, d, prefix)

        # Normalize retval
        retval = set(as_tuple(retval))

        # Update `self.state`
        if retval:
            for c in clusters:
                c_properties = properties.setdefault(c, {})
                c_properties.setdefault(d, set()).update(retval)

        return clusters

    def _callback(self, clusters: list[Cluster], dim: Dimension,
                  prefix: IterationSpace | None) -> set[Property]:
        """
        Callback to be implemented by subclasses. It should return a set of
        properties for the given dimension.
        """
        raise NotImplementedError()


class Parallelism(Detector):

    """
    Detect SEQUENTIAL, PARALLEL, PARALLEL_INDEP and PARALLEL_IF_ATOMIC Dimensions.

    Consider an IterationSpace over `n` Dimensions. Let `(d_1, ..., d_n)` be the
    distance vector of a dependence. Let `i` be the `i-th` Dimension of the
    IterationSpace. Then:

        * `i` is PARALLEL_INDEP if all dependences have distance vectors:

            (d_1, ..., d_i) = 0

        * `i` is PARALLEL if all dependences have distance vectors:

            (d_1, ..., d_i) = 0, OR
            (d_1, ..., d_{i-1}) > 0

        * `i` is PARALLEL_IF_ATOMIC if all dependences have distance vectors:

            (d_1, ..., d_i) = 0, OR
            (d_1, ..., d_{i-1}) > 0, OR
            the 'write' is known to be an associative and commutative increment
    """

    def _callback(self, clusters, dim, prefix):
        # Rule out if non-unitary increment Dimension (e.g., `t0=(time+1)%2`)
        if any(c.sub_iterators[dim] for c in clusters):
            return {SEQUENTIAL}

        # All Dimensions up to and including `i-1`
        prev = flatten(i.dim._defines for i in prefix[:-1])

        is_parallel_indep = True
        is_parallel_atomic = False

        scope = Scope(flatten(c.exprs for c in clusters))
        for dep in scope.d_all_gen():
            test00 = dep.is_indep(dim) and not dep.is_storage_related(dim)
            test01 = all(dep.is_reduce_atmost(i) for i in prev)
            if test00 and test01:
                continue

            test1 = len(prev) > 0 and any(dep.is_carried(i) for i in prev)
            if test1:
                is_parallel_indep &= (dep.distance_mapper.get(dim.root) == 0)
                continue

            if dep.function in scope.initialized:
                # False alarm, the dependence is over a locally-defined symbol
                continue

            if dep.is_reduction:
                is_parallel_atomic = True
                continue

            return {SEQUENTIAL}

        if is_parallel_atomic:
            return {PARALLEL_IF_ATOMIC}
        elif is_parallel_indep:
            return {PARALLEL, PARALLEL_INDEP}
        else:
            return {PARALLEL}


class Affiness(Detector):

    """
    Detect the AFFINE Dimensions.
    """

    def _callback(self, clusters, dim, prefix):
        scope = Scope(flatten(c.exprs for c in clusters))
        accesses = [a for a in scope.accesses if not a.is_scalar]

        if all(a.is_regular and a.affine_if_present(dim._defines) for a in accesses):
            return {AFFINE}

        return set()
