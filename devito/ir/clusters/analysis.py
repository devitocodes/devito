from devito.ir.clusters.queue import QueueStateful
from devito.ir.support import (AFFINE, PARALLEL, PARALLEL_INDEP, PARALLEL_IF_ATOMIC,
                               ROUNDABLE, SEQUENTIAL, SKEWABLE, TILABLE, Forward)
from devito.tools import as_tuple, flatten, timed_pass

__all__ = ['analyze']


@timed_pass()
def analyze(clusters):
    state = QueueStateful.State()

    # Collect properties
    clusters = Parallelism(state).process(clusters)
    clusters = Affiness(state).process(clusters)
    clusters = Tiling(state).process(clusters)
    clusters = Skewing(state).process(clusters)
    clusters = Rounding(state).process(clusters)

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
        if any(c.sub_iterators.get(d) for c in clusters):
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

            if dep.is_increment:
                is_parallel_atomic = True
                continue

            return SEQUENTIAL

        if is_parallel_atomic:
            return PARALLEL_IF_ATOMIC
        elif is_parallel_indep:
            return {PARALLEL, PARALLEL_INDEP}
        else:
            return PARALLEL


class Rounding(Detector):

    def _callback(self, clusters, d, prefix):
        itinterval = prefix[-1]

        # The iteration direction must be Forward -- ROUNDABLE is for rounding *up*
        if itinterval.direction is not Forward:
            return

        properties = self._fetch_properties(clusters, prefix)
        if PARALLEL not in properties[d]:
            return

        scope = self._fetch_scope(clusters)

        # All accessed Functions must have enough room in the PADDING region
        # so that `i`'s trip count can safely be rounded up
        # Note: autopadding guarantees that the padding size along the
        # Fastest Varying Dimension is a multiple of the SIMD vector length
        functions = [f for f in scope.functions if f.is_AbstractFunction]
        if any(not f._honors_autopadding for f in functions):
            return

        # Mixed data types (e.g., float and double) is unsupported
        if len({f.dtype for f in functions}) > 1:
            return

        return ROUNDABLE


class Affiness(Detector):

    """
    Detect the AFFINE Dimensions.
    """

    def _callback(self, clusters, d, prefix):
        scope = self._fetch_scope(clusters)
        accesses = [a for a in scope.accesses if not a.is_scalar]
        if all(a.is_regular and a.affine_if_present(d._defines) for a in accesses):
            return AFFINE


class Tiling(Detector):

    """
    Detect the TILABLE Dimensions.
    """

    def _callback(self, clusters, d, prefix):
        # A Dimension is TILABLE only if it's PARALLEL and AFFINE
        properties = self._fetch_properties(clusters, prefix)
        if not {PARALLEL, AFFINE} <= properties[d]:
            return

        # In addition, we use the heuristic that we do not consider
        # TILABLE a Dimension that is not embedded in at least one
        # SEQUENTIAL Dimension. This is to rule out tiling when the
        # computation is not expected to be expensive
        if not any(SEQUENTIAL in properties[i.dim] for i in prefix[:-1]):
            return

        # Likewise, it won't be marked TILABLE if there's at least one
        # local SubDimension in all Clusters
        if all(any(i.dim.is_Sub and i.dim.local for i in c.itintervals)
               for c in clusters):
            return

        # If it induces dynamic bounds, then it's ruled out too
        scope = self._fetch_scope(clusters)
        if any(i.is_lex_non_stmt for i in scope.d_all_gen()):
            return

        return TILABLE


class Skewing(Detector):

    """
    Detect the SKEWABLE Dimensions.
    """

    def _callback(self, clusters, d, prefix):
        # A Dimension is SKEWABLE in case it is TILABLE
        properties = self._fetch_properties(clusters, prefix)
        if {TILABLE} <= properties[d]:
            return SKEWABLE
