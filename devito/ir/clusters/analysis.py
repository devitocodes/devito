from devito.ir.clusters.visitors import QueueStateful
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC,
                               SEQUENTIAL)
from devito.tools import as_tuple, flatten, timed_pass

__all__ = ['analyze']


@timed_pass()
def analyze(clusters):
    state = QueueStateful.State()

    # Collect properties
    clusters = Parallelism(state).process(clusters)
    clusters = Affiness(state).process(clusters)

    # Reconstruct Clusters attaching the discovered properties
    processed = [c.rebuild(properties=state.properties.get(c)) for c in clusters]

    return processed


class Detector(QueueStateful):

    def process(self, elements):
        return self._process_fatd(elements, 1)

    def callback(self, clusters, prefix):
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
                properties = self.state.properties.setdefault(c, {})
                properties.setdefault(d, set()).update(retval)

        return clusters


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

    def _callback(self, clusters, d, prefix):
        # Rule out if non-unitary increment Dimension (e.g., `t0=(time+1)%2`)
        if any(c.sub_iterators[d] for c in clusters):
            return SEQUENTIAL

        # All Dimensions up to and including `i-1`
        prev = flatten(i.dim._defines for i in prefix[:-1])

        is_parallel_indep = True
        is_parallel_atomic = False

        scope = self._fetch_scope(clusters)
        for dep in scope.d_all_gen():
            test00 = dep.is_indep(d) and not dep.is_storage_related(d)
            test01 = all(dep.is_reduce_atmost(i) for i in prev)
            if test00 and test01:
                continue

            test1 = len(prev) > 0 and any(dep.is_carried(i) for i in prev)
            if test1:
                is_parallel_indep &= (dep.distance_mapper.get(d.root) == 0)
                continue

            if dep.function in scope.initialized:
                # False alarm, the dependence is over a locally-defined symbol
                continue

            if dep.is_reduction:
                is_parallel_atomic = True
                continue

            return SEQUENTIAL

        if is_parallel_atomic:
            return PARALLEL_IF_ATOMIC
        elif is_parallel_indep:
            return {PARALLEL, PARALLEL_INDEP}
        else:
            return PARALLEL


class Affiness(Detector):

    """
    Detect the AFFINE Dimensions.
    """

    def _callback(self, clusters, d, prefix):
        scope = self._fetch_scope(clusters)
        accesses = [a for a in scope.accesses if not a.is_scalar]
        if all(a.is_regular and a.affine_if_present(d._defines) for a in accesses):
            return AFFINE
