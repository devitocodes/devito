import abc
from collections import OrderedDict
from time import time

from devito.symbolics import estimate_cost, freeze_expression, pow_to_mul

from devito.logger import dse
from devito.tools import flatten

__all__ = ['AbstractRewriter', 'State', 'dse_pass']


def dse_pass(func):

    def wrapper(self, state, **kwargs):
        # A template to construct temporaries
        tempname = self.conventions.get(func.__name__)
        if tempname:
            start = kwargs.get('start')
            tempname += '%d' if start is None else (('_%d_' % start) + '%d')
            template = lambda i: tempname % i
        else:
            template = None

        # Invoke the DSE pass
        tic = time()
        state.update(flatten([func(self, c, template, **kwargs)
                              for c in state.clusters]))
        toc = time()

        # Profiling
        key = '%s%d' % (func.__name__, len(self.timings))
        self.timings[key] = toc - tic
        if self.profile:
            candidates = [c.exprs for c in state.clusters if c.is_dense]
            self.ops[key] = estimate_cost(flatten(candidates))

    return wrapper


class State(object):

    def __init__(self, cluster):
        self.clusters = [cluster]
        self._has_changed = False

    def update(self, clusters):
        clusters = clusters or self.clusters
        self._has_changed = len(clusters) != len(self.clusters) or\
            any(c1.exprs != c2.exprs for c1, c2 in zip(clusters, self.clusters))
        self.clusters = clusters

    @property
    def has_changed(self):
        return self._has_changed


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
        '_extract_sum_of_products': 'sop',
        '_extract_time_invariants': 'ti',
        '_extract_time_varying': 'td',
        '_eliminate_intra_stencil_redundancies': 'tcse',
        '_eliminate_inter_stencil_redundancies': 'r'
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'min-cost-alias': 10,
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
    def _finalize(self, cluster, *args, **kwargs):
        """
        Finalize the DSE output: ::

            * Pow-->Mul. Convert integer powers in an expression to Muls,
              like a**2 => a*a.
            * Freezing. Make sure that subsequent SymPy operations applied to
              the expressions in ``cluster.exprs`` will not alter the effect of
              the DSE passes.
        """
        exprs = [pow_to_mul(e) for e in cluster.exprs]
        return cluster.rebuild([freeze_expression(e) for e in exprs])

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
