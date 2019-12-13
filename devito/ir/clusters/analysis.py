from devito.ir.clusters.queue import Queue
from devito.ir.support import SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC, Scope
from devito.tools import as_tuple, flatten

__all__ = ['analyze']


def analyze(clusters):
    state = State()

    clusters = Parallelism(state).process(clusters)

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

    def _make_scope(self, clusters):
        key = as_tuple(clusters)
        if key not in self.state.scopes:
            self.state.scopes[key] = Scope(flatten(c.exprs for c in key))
        return self.state.scopes[key]

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

    def process(self, elements):
        return self._process_fatd(elements, 1)

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

        scope = self._make_scope(clusters)
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
