from __future__ import absolute_import

from collections import OrderedDict, Sequence, namedtuple
from itertools import combinations
from time import time

import numpy as np

import cgen as c

from devito.dimension import Dimension
from devito.dle.inspection import retrieve_iteration_tree
from devito.dle.manipulation import compose_nodes
from devito.dse import terminals
from devito.interfaces import ScalarData, SymbolicData
from devito.logger import dle, dle_warning
from devito.nodes import Element, Function, Iteration, List
from devito.tools import as_tuple, flatten
from devito.visitors import FindSections, FindSymbols, IsPerfectIteration, Transformer


def transform(node, mode='basic'):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. Available modes are: ::

                    * 'noop': Do nothing.
                    * 'blocking': Apply loop blocking
                    * 'advanced': Identify elemental functions and apply loop blocking.
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
    if mode.isdisjoint({'noop', 'blocking', 'advanced'}):
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

    """Represent the output of the DLE."""

    def __init__(self, nodes):
        self.nodes = as_tuple(nodes)

        self.elemental_functions = ()
        self.arguments = ()

    def update(self, nodes=None, elemental_functions=None, arguments=None):
        self.nodes = as_tuple(nodes) or self.nodes
        self.elemental_functions = as_tuple(elemental_functions) or\
            self.elemental_functions
        self.arguments += as_tuple(arguments)


class Arg(object):

    """A DLE-produced argument."""

    from_Blocking = False

    def __init__(self, argument, value):
        self.argument = argument
        self.value = value

    def __repr__(self):
        return "DLE-GenericArg"


class BlockingArg(Arg):

    from_Blocking = True

    def __init__(self, blocked_dim, original_dim, value):
        """
        Represent an argument introduced in the kernel by Rewriter._loop_blocking.

        :param blocked_dim: The blocked :class:`Dimension`.
        :param original_dim: The original :class:`Dimension` corresponding
                             to ``blocked_dim``.
        :param value: A suggested value determined by the DLE.
        """
        super(BlockingArg, self).__init__(blocked_dim, value)
        self.original_dim = original_dim

    def __repr__(self):
        bsize = self.value if self.value else '<unused>'
        return "DLE-BlockingArg[%s,%s,%s]" % (self.argument, self.original_dim, bsize)


class Rewriter(object):

    triggers = {
        '_create_elemental_functions': ('advanced',),
        '_loop_blocking': ('blocking', 'advanced')
    }

    def __init__(self, nodes, params):
        self.nodes = nodes
        self.params = params

        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.nodes)

        self._analyze_and_decorate(state)

        self._create_elemental_functions(state, mode=mode)
        self._loop_blocking(state, mode=mode)

        self._summary(mode)

        return state

    def _analyze_and_decorate(self, state):
        """
        Visit the Iteration/Expression trees in ``state.nodes`` and collect
        useful information:

            * Identification of deepest loop nest(s), the candidates for most
                loop optimizations applied by the DLE;
            * Presence of "outermost-sequential inner-parallel" (OSIP) loop trees:
                that is, Iteration/Expression subtrees in which the outermost
                :class:`Iteration` represents an inherently sequential dimension,
                whereas all inner :class:`Iteration` represent parallelizable
                dimensions.

        The presence of OSIP subtrees is marked in the input Iteration/Expression
        tree by introducing suitable mixin nodes.
        """

        nodes = state.nodes

        sections = FindSections().visit(nodes)
        trees = sections.keys()
        deepest = max(trees, key=lambda i: len(i))
        deepest = [i for i in trees if len(i) == len(deepest)]

        # The analysis below may return "false positives" (ie, existance of a
        # LCD when this is actually not true), but we expect this to never be the
        # case given the stencil codes that the DLE will attempt to optimize.
        for k in deepest:
            exprs = [e.stencil for e in sections[k]]

            # Retain only expressions that may induce true dependencies (ie, tensors)
            exprs = [e for e in exprs if not e.lhs.is_Symbol]

            # Is the loop nest of type "outermost sequential (LCD), inners parallel" ?
            match = True
            for e1 in exprs:
                lhs = e1.lhs
                for e2 in exprs:
                    terms = [i for i in terminals(e2.rhs)
                             if i.base.label == lhs.base.label]

                    # Check is of type -- a[j][][] = a[i][][] + b[i][][] + c[i][][]
                    match = (all(len(i.indices) == len(lhs.indices) for i in terms) and
                             all(i.indices[0] == terms[0].indices[0] for i in terms) and
                             lhs.indices[0] != terms[0].indices[0])
                    if not match:
                        break
                if not match:
                    break

            if match:
                mapper = {i: Property(i, ('parallel',)) for i in k[1:]}
                nodes = Transformer(mapper).visit(nodes)

        state.update(nodes=nodes)

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

    @dle_transformation
    def _loop_blocking(self, state, **kwargs):
        """
        Apply loop blocking to :class:`Iteartion` trees.

        By default, the blocked :class:`Iteration` objects and the block size are
        determined heuristically. The heuristic consists of searching the deepest
        Iteration/Expression tree and blocking all dimensions except:

            * The innermost (eg, to retain SIMD vectorization);
            * Those dimensions inducing loop-carried dependencies.

        The caller may take over the heuristic through ``kwargs['blocking']``,
        a dictionary indicating the block size of each blocked dimension. For
        example, for the :class:`Iteration` tree below: ::

            for i
              for j
                for k
                  ...

        one may pass in ``kwargs['blocking'] = {i: 4, j: 7}``, in which case the
        two outer loops would be blocked, and the resulting 2-dimensional block
        would be of size 4x7.
        """

        Region = namedtuple('Region', 'main leftover')

        is_InnerBlockable = self.params.get('blockinner',
                                            len(state.elemental_functions) > 0)

        dims = OrderedDict()
        processed = []
        for node in state.nodes:
            mapper = {}
            for tree in retrieve_iteration_tree(node):
                # Is the Iteration tree blockable ?
                iterations = [i for i in tree if 'parallel' in i.properties]
                iterations = iterations if is_InnerBlockable else iterations[:-1]
                if not iterations:
                    continue
                root = iterations[0]
                if not IsPerfectIteration().visit(root):
                    continue

                # Construct the blocked loop nest, as well as all necessary
                # remainder loops
                regions = {}
                blocked_iterations = []
                for i in iterations:
                    # Build Iteration over blocks
                    dim = dims.setdefault((i.dim, 'inter-block'),
                                          Dimension("%sb" % i.dim.name))

                    block_size = dim.symbolic_size
                    iter_size = i.dim.symbolic_size

                    start = i.limits[0] - i.offsets[0]
                    finish = iter_size - ((iter_size - i.offsets[1]) % block_size)
                    inter_block = Iteration([], dim, [start, finish, block_size])

                    # Build Iteration within a block
                    dim = dims.setdefault((i.dim, 'intra-block'),
                                          Dimension(i.dim.name))

                    start = inter_block.dim
                    finish = start + block_size
                    intra_block = Iteration([], dim, [start, finish, 1], i.index)

                    blocked_iterations.append((inter_block, intra_block))

                    # Build unitary-increment Iteration over the 'main' region
                    # (the one blocked); necessary to generate code iterating over
                    # non-blocked ("remainder") iterations.
                    start = inter_block.limits[0]
                    finish = inter_block.limits[1]
                    main = Iteration([], i.dim, [start, finish, 1], i.index)

                    # Build unitary-increment Iteration over the 'leftover' region:
                    # again as above, this may be necessary when the dimension size
                    # is not a multiple of the block size.
                    start = inter_block.limits[1]
                    finish = iter_size - i.offsets[1]
                    leftover = Iteration([], i.dim, [start, finish, 1], i.index)

                    regions[i] = Region(main, leftover)

                blocked_tree = list(flatten(zip(*blocked_iterations)))
                blocked = compose_nodes(blocked_tree + [iterations[-1].nodes])

                # Build remainder loops
                remainder_tree = []
                for n in range(len(iterations)):
                    for i in combinations(iterations, n + 1):
                        nodes = [v.leftover if k in i else v.main
                                 for k, v in regions.items()]
                        nodes += [iterations[-1].nodes]
                        remainder_tree.append(compose_nodes(nodes))

                # Will replace with blocked loop tree
                mapper[root] = List(body=[blocked] + remainder_tree)

            rebuilt = Transformer(mapper).visit(node)

            processed.append(rebuilt)

        # All blocked dimensions
        blocked = OrderedDict([(k, v) for k, v in dims.items() if k[1] == 'inter-block'])
        if not blocked:
            return {'nodes': processed}

        # Determine the block shape
        blockshape = self.params.get('blockshape')
        if not blockshape:
            # Use trivial heuristic for a suitable blockshape
            def heuristic(dim_size):
                ths = 8  # FIXME: This really needs to be improved
                return ths if dim_size > ths else 1
            blockshape = {k: heuristic for k in blocked.keys()}
        else:
            try:
                nitems, nrequired = len(blockshape), len(blocked)
                blockshape = {k: v for k, v in zip(blocked, blockshape)}
                if nitems > nrequired:
                    dle_warning("Provided 'blockshape' has more entries than "
                                "blocked loops; dropping entries ...")
                if nitems < nrequired:
                    dle_warning("Provided 'blockshape' has fewer entries than "
                                "blocked loops; dropping dimensions ...")
            except TypeError:
                blockshape = {blocked.keys()[0]: blockshape}
            blockshape.update({k: None for k in blocked.keys()
                               if k not in blockshape})

        # Track any additional arguments required to execute /state.nodes/
        arguments = [BlockingArg(v, k[0], blockshape[k]) for k, v in blocked.items()]

        return {'nodes': processed, 'arguments': arguments}

    def _summary(self, mode):
        """
        Print a summary of the DLE transformations
        """

        if mode.intersection({'blocking', 'advanced'}):
            steps = " --> ".join("(%s)" % i for i in self.timings.keys())
            elapsed = sum(self.timings.values())
            dle("%s [%.2f s]" % (steps, elapsed))
