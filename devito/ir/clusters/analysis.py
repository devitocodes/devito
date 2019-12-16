from functools import cmp_to_key

from devito.ir.clusters.queue import Queue
from devito.ir.support import (SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC, AFFINE,
                               WRAPPABLE, ROUNDABLE, TILABLE, Forward, Scope)
from devito.tools import as_tuple, flatten

__all__ = ['analyze']


def analyze(clusters):
    state = State()

    clusters = Parallelism(state).process(clusters)
    clusters = Affiness(state).process(clusters)
    clusters = Blocking(state).process(clusters)
    clusters = Wrapping(state).process(clusters)
    clusters = Rounding(state).process(clusters)

    # Group properties by Cluster
    properties = {}
    for k, v in state.properties.items():
        for c in k:
            properties.setdefault(c, {}).update(v)

    # Rebuild Clusters to attach the discovered properties
    processed = [c.rebuild(properties=properties.get(c)) for c in clusters]

    return processed


class State(object):

    def __init__(self):
        self.properties = {}
        self.scopes = {}


class Detector(Queue):

    def __init__(self, state):
        super(Detector, self).__init__()
        self.state = state

    def _fetch_scope(self, clusters):
        key = as_tuple(clusters)
        if key not in self.state.scopes:
            self.state.scopes[key] = Scope(flatten(c.exprs for c in key))
        return self.state.scopes[key]

    def process(self, elements):
        return self._process_fatd(elements, 1)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        # Apply the actual callback
        retval = self._callback(clusters, prefix)

        # Update `self.state`
        if retval is not None:
            key = as_tuple(clusters)
            properties = self.state.properties.setdefault(key, {})
            properties.setdefault(prefix[-1].dim, []).append(retval)

        return clusters


class Parallelism(Detector):

    """
    Detect SEQUENTIAL, PARALLEL, and PARALLEL_IF_ATOMIC Dimensions.
    """

    def _callback(self, clusters, prefix):
        # The analyzed Dimension
        i = prefix[-1].dim

        # All Dimensions up to and including `i-1`
        prev = flatten(d.dim._defines for d in prefix[:-1])

        # The i-th Dimension is PARALLEL if for all dependences (d_1, ..., d_n):
        # test0 := (d_1, ..., d_i) = 0, OR
        # test1 := (d_1, ..., d_{i-1}) > 0

        # The i-th Dimension is PARALLEL_IF_ATOMIC if for all dependeces:
        # test0 OR test1 OR the write is an associative and commutative increment

        is_parallel_atomic = False

        scope = self._fetch_scope(clusters)
        for dep in scope.d_all_gen():
            test00 = dep.is_indep(i) and not dep.is_storage_related(i)
            test01 = all(dep.is_reduce_atmost(d) for d in prev)
            if test00 and test01:
                continue

            test1 = len(prev) > 0 and any(dep.is_carried(d) for d in prev)
            if test1:
                continue

            if not dep.is_increment:
                return SEQUENTIAL

            # At this point, if it's not SEQUENTIAL, it can only be PARALLEL_IF_ATOMIC
            is_parallel_atomic = True

        if is_parallel_atomic:
            return PARALLEL_IF_ATOMIC
        else:
            return PARALLEL


class Wrapping(Detector):

    """
    Detect the WRAPPABLE Dimensions.
    """

    def _callback(self, clusters, prefix):
        # The analyzed Dimension
        i = prefix[-1].dim

        if not i.is_Time:
            return

        scope = self._fetch_scope(clusters)
        accesses = [a for a in scope.accesses if a.function.is_TimeFunction]

        # If not using modulo-buffered iteration, then `i` is surely not WRAPPABLE
        if not accesses or any(not a.function._time_buffering_default for a in accesses):
            return

        stepping = {a.function.time_dim for a in accesses}
        if len(stepping) > 1:
            # E.g., with ConditionalDimensions we may have `stepping={t, tsub}`
            return
        stepping = stepping.pop()

        # All accesses must be affine in `stepping`
        if any(not a.affine_if_present(stepping._defines) for a in accesses):
            return

        # Pick the `back` and `front` slots accessed
        try:
            compareto = cmp_to_key(lambda a0, a1: a0.distance(a1, stepping))
            accesses = sorted(accesses, key=compareto)
            back, front = accesses[0][stepping], accesses[-1][stepping]
        except TypeError:
            return

        # Check we're not accessing (read, write) always the same slot
        if back == front:
            return

        accesses_back = [a for a in accesses if a[stepping] == back]

        # There must be NO writes to the `back` timeslot
        if any(a.is_write for a in accesses_back):
            return

        # There must be NO further accesses to the `back` timeslot after
        # any earlier timeslot is written
        # Note: potentially, this can be relaxed by replacing "any earlier timeslot"
        # with the `front timeslot`
        if not all(all(d.sink is not a or d.source.lex_ge(a) for d in scope.d_flow)
                   for a in accesses_back):
            return

        return WRAPPABLE


class Rounding(Detector):

    def _callback(self, clusters, prefix):
        itinterval = prefix[-1]

        # The iteration direction must be Forward -- ROUNDABLE is for rounding *up*
        if itinterval.direction is not Forward:
            return

        # The analyzed Dimension
        i = itinterval.dim

        properties = self.state.properties.get(as_tuple(clusters), {})
        if PARALLEL not in properties.get(i, []):
            return

        scope = self._fetch_scope(clusters)

        # All non-scalar writes must be over Arrays, that is temporaries, otherwise
        # we would end up overwriting user data
        writes = [w for w in scope.writes if w.is_Tensor]
        if any(not w.is_Array for w in writes):
            return

        # All accessed Functions must have enough room in the PADDING region
        # so that `i`'s trip count can safely be rounded up
        # Note: autopadding guarantees that the padding size along the
        # Fastest Varying Dimension is a multiple of the SIMD vector length
        functions = [f for f in scope.functions if f.is_Tensor]
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

    def _callback(self, clusters, prefix):
        # The analyzed Dimension
        i = prefix[-1].dim

        scope = self._fetch_scope(clusters)

        accesses = [a for a in scope.accesses if not a.is_scalar]
        if all(a.is_regular and a.affine_if_present(i._defines) for a in accesses):
            return AFFINE


class Blocking(Detector):

    def process(self, elements):
        return self._process_fdta(elements, 1)

    def _callback(self, clusters, prefix):
        # The analyzed Dimension
        i = prefix[-1].dim

        # A Dimension TILABLE only if it's PARALLEL and AFFINE
        properties = self.state.properties.get(as_tuple(clusters), {})
        if not {PARALLEL, AFFINE} <= set(properties.get(i, [])):
            return

        # In addition, we use the heuristic that we do not consider
        # TILEABLE a Dimension that is not embedded in at least one
        # SEQUENTIAL Dimension. This is to rule out tiling when the
        # computation is not expected to be expensive
        if not any(SEQUENTIAL in properties.get(j.dim, []) for j in prefix[:-1]):
            return

        # Likewise, it won't be marked TILABLE if there's at least one
        # local SubDimension in all Clusters
        if all(any(j.dim.is_Sub and j.dim.local for j in c.itintervals)
               for c in clusters):
            return

        # If it induces dynamic bounds in any of the inner Iterations,
        # then it's ruled out too
        scope = self._fetch_scope(clusters)
        if any(d.is_lex_non_stmt for d in scope.d_all_gen()):
            return

        return TILABLE
