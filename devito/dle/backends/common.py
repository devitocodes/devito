# Types used by all DLE backends

import abc
from collections import OrderedDict, defaultdict
from time import time

from devito.dse import as_symbol, terminals
from devito.logger import dle
from devito.nodes import Iteration
from devito.tools import as_tuple, flatten
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
        try:
            self.flags.update({i: True for i in flags})
        except TypeError:
            self.flags[flags] = True

    @property
    def trees(self):
        return self.nodes, self.elemental_functions

    @property
    def has_applied_blocking(self):
        """True if loop blocking was applied, False otherwise."""
        return 'blocking' in self.flags

    @property
    def func_table(self):
        """Return a mapper from elemental function names to :class:`Function`."""
        return OrderedDict([(i.name, i) for i in self.elemental_functions])


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
        'elemental': 30,  # Operations
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
        candidate = max(list(sections), key=lambda i: len(i))
        candidates = [i for i in sections if len(i) == len(candidate)]

        # The analysis below may return "false positives" (ie, absence of fully-
        # parallel or OSIP trees when this is actually false), but this should
        # never be the case in practice, given the targeted stencil codes.
        mapper = OrderedDict()
        for tree in candidates:
            exprs = [e.expr for e in sections[tree]]

            # "Prefetch" objects to speed up the analsys
            terms = {e: tuple(terminals(e.rhs)) for e in exprs}
            writes = {e.lhs for e in exprs if not e.is_Symbol}

            # Does the Iteration index only appear in the outermost dimension ?
            has_parallel_dimension = True
            for k, v in terms.items():
                for i in writes:
                    maybe_dependencies = [j for j in v if as_symbol(i) == as_symbol(j)
                                          and not j.is_Symbol]
                    for j in maybe_dependencies:
                        handle = flatten(k.atoms() for k in j.indices[1:])
                        has_parallel_dimension &= not (i.indices[0] in handle)
            if not has_parallel_dimension:
                continue

            # Determine if fully-parallel (FP), OSIP, unit-stride (in innermost dim)
            is_FP = True
            is_OSIP = False
            is_US = True
            for e1 in exprs:
                lhs = e1.lhs
                if lhs.is_Symbol:
                    continue
                for e2 in exprs:
                    handle = [i for i in terms[e2] if as_symbol(i) == as_symbol(lhs)]
                    is_FP &= len(handle) == 0
                    is_OSIP |= any(lhs.indices[0] != i.indices[0] for i in handle)
                    is_US &= all(lhs.indices[-1] == i.indices[-1] for i in handle)

            # Is the innermost Iteration vectorizable?
            is_Vectorizable = is_FP or is_OSIP or is_US

            # Track the discovered properties
            if is_OSIP:
                mapper.setdefault(tree[0], []).append('sequential')
            for i in tree[is_OSIP:]:
                mapper.setdefault(i, []).append('parallel')
            if is_Vectorizable:
                mapper.setdefault(tree[-1], []).append('vector-dim')

        # Introduce the discovered properties in the Iteration/Expression tree
        for k, v in list(mapper.items()):
            args = k.args
            # 'sequential' kills 'parallel'
            properties = ('sequential',) if 'sequential' in v else tuple(v)
            properties = as_tuple(args.pop('properties')) + properties
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
