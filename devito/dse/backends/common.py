import abc
from collections import OrderedDict
from time import time

from devito.dse.inspection import estimate_cost
from devito.dse.manipulation import freeze_expression

from devito.logger import dse
from devito.tools import flatten

__all__ = ['AbstractRewriter', 'State', 'dse_pass']


def dse_pass(func):

    def wrapper(self, state, **kwargs):
        tic = time()
        state.update(flatten([func(self, c, **kwargs) for c in state.clusters]))
        toc = time()

        key = '%s%d' % (func.__name__, len(self.timings))
        self.timings[key] = toc - tic
        if self.profile:
            candidates = [c.exprs for c in state.clusters if c.is_dense]
            self.ops[key] = estimate_cost(flatten(candidates))

    return wrapper


class State(object):

    def __init__(self, cluster):
        self.clusters = [cluster]

    def update(self, clusters):
        self.clusters = clusters or self.clusters


class AbstractRewriter(object):
    """
    Transform a cluster of SymPy expressions into one or more clusters, overall
    performing fewer arithmetic operations.
    """

    __metaclass__ = abc.ABCMeta

    """
    Name conventions for new temporaries.
    """
    conventions = {
        'redundancy': 'r',
        'sum-of-product': 'sop',
        'time-invariant': 'ti',
        'time-dependent': 'td',
        'temporary': 'tcse'
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'min-cost-space-hoist': 10,
        'min-cost-time-hoist': 200,
        'min-cost-factorize': 100,
        'max-operands': 40,
    }

    def __init__(self, profile=True):
        self.profile = profile

        self.ops = OrderedDict()
        self.timings = OrderedDict()

    def run(self, cluster):
        state = State(cluster)

        self._pipeline(state)

        self._finalize(state)

        self._summary()

        return state.clusters

    @abc.abstractmethod
    def _pipeline(self, state):
        return

    @dse_pass
    def _finalize(self, cluster, **kwargs):
        """
        Finalize the DSE output: ::

            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``cluster.exprs`` will not alter the effect of
              the DSE passes.
        """
        return cluster.rebuild([freeze_expression(e) for e in cluster.exprs])

    def _summary(self):
        """
        Print a summary of the DSE transformations
        """

        if self.profile:
            row = "%s [flops: %s, elapsed: %.2f]"
            summary = " >>\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(),
                                                              k[1:])),
                                               str(self.ops.get(k, "?")), v)
                                        for k, v in self.timings.items())
            elapsed = sum(self.timings.values())
            dse("%s\n     [Total elapsed: %.2f s]" % (summary, elapsed))
