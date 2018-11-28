import abc
from collections import OrderedDict
from time import time

from devito.logger import dle


__all__ = ['AbstractRewriter', 'State', 'dle_pass']


def dle_pass(func):

    def wrapper(self, state, **kwargs):
        tic = time()
        # Processing
        processed, extra = func(self, state.nodes, state)
        for i, nodes in enumerate(list(state.efuncs)):
            state.efuncs[i], _ = func(self, nodes, state)
        # State update
        state.update(processed, **extra)
        toc = time()

        self.timings[func.__name__] = toc - tic

    return wrapper


class State(object):

    """Represent the output of the DLE."""

    def __init__(self, nodes):
        self.nodes = nodes

        self.efuncs = []
        self.dimensions = []
        self.input = []
        self.includes = []

    def update(self, nodes, **kwargs):
        self.nodes = nodes

        self.efuncs.extend(list(kwargs.get('efuncs', [])))
        self.dimensions.extend(list(kwargs.get('dimensions', [])))
        self.input.extend(list(kwargs.get('input', [])))
        self.includes.extend(list(kwargs.get('includes', [])))


class AbstractRewriter(object):
    """
    Transform Iteration/Expression trees to generate high performance C.

    This is just an abstract class. Actual transformers should implement the
    abstract method ``_pipeline``, which performs a sequence of AST transformations.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, nodes, params):
        self.nodes = nodes
        self.params = params

        self.timings = OrderedDict()

    def run(self):
        """The optimization pipeline, as a sequence of AST transformation passes."""
        state = State(self.nodes)

        self._pipeline(state)

        self._summary()

        return state

    @abc.abstractmethod
    def _pipeline(self, state):
        return

    def _summary(self):
        """
        Print a summary of the DLE transformations
        """

        row = "%s [elapsed: %.2f]"
        out = " >>\n     ".join(row % ("".join(filter(lambda c: not c.isdigit(), k[1:])),
                                       v)
                                for k, v in self.timings.items())
        elapsed = sum(self.timings.values())
        dle("%s\n     [Total elapsed: %.2f s]" % (out, elapsed))
