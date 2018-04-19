import abc
from collections import OrderedDict, defaultdict
from time import time

from devito.logger import dle
from devito.tools import as_tuple


__all__ = ['AbstractRewriter', 'Arg', 'BlockingArg', 'State', 'dle_pass']


def dle_pass(func):

    def wrapper(self, state, **kwargs):
        tic = time()
        # Processing
        processed, extra = func(self, state.nodes, state)
        for i, nodes in enumerate(list(state.elemental_functions)):
            state.elemental_functions[i], _ = func(self, nodes, state)
        # State update
        state.update(processed, **extra)
        toc = time()

        self.timings[func.__name__] = toc - tic

    return wrapper


class State(object):

    """Represent the output of the DLE."""

    def __init__(self, nodes):
        self.nodes = nodes

        self.elemental_functions = []
        self.arguments = []
        self.includes = []
        self.flags = defaultdict(bool)

    def update(self, nodes, **kwargs):
        self.nodes = nodes

        self.elemental_functions.extend(list(kwargs.get('elemental_functions', [])))
        self.arguments.extend(list(kwargs.get('arguments', [])))
        self.includes.extend(list(kwargs.get('includes', [])))
        self.flags.update({i: True for i in as_tuple(kwargs.get('flags', ()))})


class Arg(object):

    """A DLE-produced argument."""

    def __init__(self, argument, value):
        self.argument = argument
        self.value = value

    def __repr__(self):
        return "DLE-GenericArg"


class BlockingArg(Arg):

    def __init__(self, blocked_dim, iteration, value):
        """
        Represent an argument introduced in the kernel by Rewriter._loop_blocking.

        :param blocked_dim: The blocked :class:`Dimension`.
        :param iteration: The :class:`Iteration` object from which the ``blocked_dim``
                          was derived.
        :param value: A suggested value determined by the DLE.
        """
        super(BlockingArg, self).__init__(blocked_dim, value)
        self.iteration = iteration

    def __repr__(self):
        return "DLE-BlockingArg[%s,%s,suggested=%s]" %\
            (self.argument, self.original_dim, self.value)

    @property
    def original_dim(self):
        return self.iteration.dim


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
