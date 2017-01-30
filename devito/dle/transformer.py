from __future__ import absolute_import

from collections import OrderedDict, Sequence
from time import time

import numpy as np

import cgen as c

from devito.dimension import Dimension
from devito.dle.inspection import retrieve_iteration_tree
from devito.interfaces import ScalarData, SymbolicData
from devito.logger import dle, dle_warning
from devito.nodes import Element, Function
from devito.tools import as_tuple
from devito.visitors import FindSymbols, Transformer


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
        return Rewriter(node).run(mode)

    return Rewriter(node).run(mode)


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
        self.nodes = as_tuple(nodes)
        self.elemental_functions = []

    def update(self, nodes=None, elemental_functions=None):
        self.nodes = as_tuple(nodes) or self.nodes
        self.elemental_functions = as_tuple(elemental_functions) or\
            self.elemental_functions


class Rewriter(object):

    triggers = {
        '_create_elemental_functions': ('basic',)
    }

    def __init__(self, nodes):
        self.nodes = nodes

        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.nodes)

        self._create_elemental_functions(state, mode=mode)

        self._summary(mode)

        return state

    @dle_transformation
    def _create_elemental_functions(self, state, **kwargs):
        """
        Move :class:`Iteration` sub-trees to separate functions.

        By default, inner iteration trees are moved. To move different types of
        :class:`Iteration`, one can provide a lambda function in ``kwargs['rule']``,
        taking as input an iterable of :class:`Iteration` and returning an iterable
        of :class:`Iteration` (eg, a subset, the whole iteration tree).
        """

        rule = kwargs.get('rule', lambda tree: tree[-1:])

        functions = []
        processed = []
        for i, node in enumerate(state.nodes):
            mapper = {}
            for j, tree in enumerate(retrieve_iteration_tree(node)):
                if len(tree) <= 1:
                    continue

                name = "f_%d_%d" % (i, j)

                candidate = rule(tree)
                leftover = tuple(k for k in tree if k not in candidate)

                args = FindSymbols().visit(candidate)
                args += [k.dim for k in leftover if k not in args and k.is_Closed]

                known = [k.name for k in args]
                known += [k.index for k in candidate]
                maybe_unknown = FindSymbols(mode='free-symbols').visit(candidate)
                args += [k for k in maybe_unknown if k.name not in known]

                call = []
                parameters = []
                for k in args:
                    if isinstance(k, Dimension):
                        call.append(k.ccode)
                        parameters.append(k)
                    elif isinstance(k, SymbolicData):
                        call.append("%s_vec" % k.name)
                        parameters.append(k)
                    else:
                        call.append(k.name)
                        parameters.append(ScalarData(name=k.name, dtype=np.int32))

                root = candidate[0]

                # Track info to transform the main tree
                call = '%s(%s)' % (name, ','.join(call))
                mapper[root] = Element(c.Statement(call))

                # Produce the new function
                functions.append(Function(name, root, 'void', parameters, ('static',)))

            # Transform the main tree
            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed, 'elemental_functions': functions}

    def _summary(self, mode):
        """
        Print a summary of the DLE transformations
        """

        if mode.intersection({'basic'}):
            steps = " --> ".join("(%s)" % i for i in self.timings.keys())
            elapsed = sum(self.timings.values())
            dle("%s [%.2f s]" % (steps, elapsed))
