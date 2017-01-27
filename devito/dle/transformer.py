from collections import OrderedDict, Sequence

from time import time

from devito.logger import dle, dle_warning
from devito.visitors import FindSections, FunctionBuilder


def transform(node, mode='basic'):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. Currently, only the default
                 'basic' option is available.
    """

    if isinstance(node, Sequence):
        assert all(n.is_Node for n in node)
        node = list(node)
    elif node.is_Node:
        node = [node]
    else:
        raise ValueError("Got illegal node of type %s." % type(node))

    if not mode:
        return State(node)
    elif isinstance(mode, str):
        mode = set([mode])
    else:
        try:
            mode = set(mode)
        except TypeError:
            dle_warning("Arg mode must be str or tuple (got %s)" % type(mode))
            return State(node)
    if mode.isdisjoint({'noop', 'basic'}):
        dle_warning("Unknown transformer mode(s) %s" % str(mode))
        return State(node)
    else:
        return Transformer(node).run(mode)

    return Transformer(node).run(mode)


def dle_transformation(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.__name__])):
            tic = time()
            state.update(**func(self, state))
            toc = time()

            key = '%s%d' % (func.__name__, len(self.timings))
            self.timings[key] = toc - tic

    return wrapper


class State(object):

    def __init__(self, nodes):
        self.nodes = nodes

    def update(self, nodes=None):
        self.nodes = nodes or self.nodes


class Transformer(object):

    triggers = {
        '_extract_loops': ('basic',)
    }

    def __init__(self, nodes):
        self.nodes = nodes

        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.nodes)

        self._extract_loops(state, mode=mode)

        self._summary(mode)

        return state

    @dle_transformation
    def _extract_loops(self, state, **kwargs):
        """
        Move inner loops to separate functions.
        """

        return {'nodes': state.nodes}

    def _summary(self, mode):
        """
        Print a summary of the DLE transformations
        """

        if mode.intersection({'basic'}):
            steps = " --> ".join("(%s)" % i for i in self.timings.keys())
            elapsed = sum(self.timings.values())
            dle("%s [%.2f s]" % (steps, elapsed))
