from collections import OrderedDict

from devito.ir.clusters.queue import Queue
from devito.ir.support import (SEQUENTIAL, PARALLEL, PARALLEL_IF_ATOMIC, VECTOR,
                               TILABLE, WRAPPABLE)
from devito.tools import timed_pass

__all__ = ['analyze']


class State(object):

    def __init__(self):
        self.properties = OrderedDict()
        self.scopes = OrderedDict()


class Detector(Queue):

    def __init__(self, state):
        super(Detector, self).__init__()
        self.state = state

    def callback(self, clusters, prefix):
        self._callback(clusters, prefix)
        return clusters


class Parallelism(Detector):

    def _callback(self, clusters, prefix):
        properties = OrderedDict()


def analyze(clusters):
    state = State()

    clusters = Parallelism(state).process(clusters)

    return clusters
