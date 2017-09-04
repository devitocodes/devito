# Types used by all DLE backends

import abc
from collections import OrderedDict, defaultdict
from time import time

from devito.dse import as_symbol, terminals
from devito.logger import dle
from devito.nodes import Iteration, SEQUENTIAL, PARALLEL, VECTOR
from devito.tools import as_tuple
from devito.visitors import FindSections, NestedTransformer


__all__ = ['AbstractRewriter', 'Arg', 'BlockingArg', 'State', 'dle_pass']


def dle_pass(func):

    def wrapper(self, state, **kwargs):
        tic = time()
        state.update(**func(self, state))
        toc = time()

        self.timings[func.__name__] = toc - tic

    return wrapper


class State(object):

    """Represent the output of the DLE."""

    def __init__(self, nodes):
        self.nodes = as_tuple(nodes)

        self.elemental_functions = ()
        self.arguments = ()
        self.includes = ()
        self.flags = defaultdict(bool)

    def update(self, nodes=None, elemental_functions=None, arguments=None,
               includes=None, flags=None):
        self.nodes = as_tuple(nodes) or self.nodes
        self.elemental_functions = as_tuple(elemental_functions) or\
            self.elemental_functions
        self.arguments += as_tuple(arguments)
        self.includes += as_tuple(includes)
        self.flags.update({i: True for i in as_tuple(flags)})


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

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'collapse': 32,  # Available physical cores
        'max_fission': 800,  # Statements
        'min_fission': 20  # Statements
    }

    def __init__(self, nodes, params):
        self.nodes = nodes
        self.params = params

        self.timings = OrderedDict()

    def run(self):
        """The optimization pipeline, as a sequence of AST transformation passes."""
        state = State(self.nodes)

        self._analyze(state)

        self._pipeline(state)

        self._summary()

        return state

    @dle_pass
    def _analyze(self, state):
        """
        Analyze the Iteration/Expression trees in ``state.nodes`` to detect
        information useful to the subsequent DLE passes.

        In particular, fully-parallel or "outermost-sequential inner-parallel"
        (OSIP) :class:`Iteration` trees are searched tracked. In an OSIP
        :class:`Iteration` tree, the outermost :class:`Iteration` represents
        a sequential dimension, whereas all inner :class:`Iteration` objects
        represent parallel dimensions.
        """
        nodes = state.nodes
        sections = FindSections().visit(nodes)

        # The analysis below may return "false positives" (ie, absence of fully-
        # parallel or OSIP trees when this is actually false), but this should
        # never be the case in practice, given the targeted stencil codes.
        mapper = OrderedDict()
        for tree, nexprs in sections.items():
            exprs = [e.expr for e in nexprs]

            # "Prefetch" objects to speed up the analsys
            terms = {e: tuple(terminals(e.rhs)) for e in exprs}

            # Determine whether the Iteration tree ...
            is_FP = True  # ... is fully parallel (FP)
            is_OP = True  # ... has an outermost parallel dimension (OP)
            is_OSIP = True  # ... is of type OSIP
            is_US = True  # ... has a unit-strided innermost dimension (US)
            for lhs in [e.lhs for e in exprs if not e.lhs.is_Symbol]:
                for e in exprs:
                    for i in [j for j in terms[e] if as_symbol(j) == as_symbol(lhs)]:
                        is_FP &= lhs.indices == i.indices

                        is_OP &= lhs.indices[0] == i.indices[0] and\
                            all(lhs.indices[0].free_symbols.isdisjoint(j.free_symbols)
                                for j in i.indices[1:])  # not A[x,y] = A[x,x+1]

                        is_US &= lhs.indices[-1] == i.indices[-1]

                        lhs_function, i_function = lhs.base.function, i.base.function
                        is_OSIP &= lhs_function.indices[0] == i_function.indices[0] and\
                            (lhs.indices[0] != i.indices[0] or len(lhs.indices) == 1 or
                             lhs.indices[1] == i.indices[1])

            # Build a node->property mapper
            if is_FP:
                for i in tree:
                    mapper.setdefault(i, []).append(PARALLEL)
            elif is_OP:
                mapper.setdefault(tree[0], []).append(PARALLEL)
            elif is_OSIP:
                mapper.setdefault(tree[0], []).append(SEQUENTIAL)
                for i in tree[1:]:
                    mapper.setdefault(i, []).append(PARALLEL)
            if is_FP or is_OSIP or is_US:
                # Vectorizable
                if len(tree) > 1 and SEQUENTIAL not in mapper.get(tree[-2], []):
                    # Heuristic: there's at least an outer parallel Iteration
                    mapper.setdefault(tree[-1], []).append(VECTOR)

        # Store the discovered properties in the Iteration/Expression tree
        for k, v in list(mapper.items()):
            args = k.args
            # SEQUENTIAL kills PARALLEL
            properties = SEQUENTIAL if (SEQUENTIAL in v or not k.is_Linear) else v
            properties = as_tuple(args.pop('properties')) + as_tuple(properties)
            mapper[k] = Iteration(properties=properties, **args)
        nodes = NestedTransformer(mapper).visit(nodes)

        return {'nodes': nodes}

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
