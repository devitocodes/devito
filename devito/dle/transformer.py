from __future__ import absolute_import

from collections import OrderedDict, Sequence, namedtuple
from itertools import combinations
from time import time

import numpy as np
import psutil

import cgen as c

from devito.dimension import Dimension
from devito.dle.inspection import retrieve_iteration_tree
from devito.dle.manipulation import compose_nodes
from devito.dse import terminals
from devito.interfaces import ScalarData, SymbolicData
from devito.logger import dle, dle_warning
from devito.nodes import Denormals, Element, Function, Iteration, List
from devito.tools import as_tuple, flatten
from devito.visitors import (FindNodeType, FindSections, FindSymbols,
                             IsPerfectIteration, Transformer)


def transform(node, mode='basic', compiler=None):
    """
    Transform Iteration/Expression trees to generate highly optimized C code.

    :param node: The Iteration/Expression tree to be transformed, or an iterable
                 of Iteration/Expression trees.
    :param mode: Drive the tree transformation. ``mode`` can be a string indicating
                 a pre-established optimization sequence or a tuple of individual
                 transformations; in the latter case, the specified transformations
                 are composed. We use the following convention: ::

                    * [S]: a pre-defined sequence of transformations.
                    * [T]: a single transformation.
                    * [sT]: a single "speculative" transformation; that is, a
                            transformation that might increase, or even decrease,
                            performance.

                 The keywords usable in/as ``mode`` are: ::

                    * 'noop': Do nothing -- [S]
                    * 'basic': Apply all of the available legal transformations
                               that are most likely to increase performance (ie, all
                               [T] listed below), except for loop blocking -- [S]
                    * 'advanced': Like 'basic', but also switches on loop blocking -- [S]
                    * 'speculative': Apply all of the 'advanced' transformations,
                                     plus other transformations that might increase
                                     (or possibly decrease) performance -- [S]
                    * 'blocking': Apply loop blocking -- [T]
                    * 'split': Identify and split elemental functions -- [T]
                    * 'simd': Add pragmas to trigger compiler auto-vectorization -- [T]
                    * 'ntstores': Add pragmas to issue nontemporal stores -- [sT]

                 If ``mode`` is a tuple, the last entry may be used to provide optional
                 arguments to the DLE transformations. Accepted key-value pairs are: ::

                    * 'blockshape': A tuple representing the shape of a block created
                                    by loop blocking.
                    * 'blockinner': By default, loop blocking is not applied to the
                                    innermost dimension of an Iteration/Expression tree
                                    to maximize the chances of SIMD vectorization. To
                                    force the blocking of this loop, the ``blockinner``
                                    flag should be set to True.
                    * 'openmp': True to emit OpenMP code, False otherwise.
    :param compiler: Compiler class used to perform JIT compilation. Useful to
                     introduce compiler-specific vectorization pragmas.
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
        mode = list([mode])
    elif not isinstance(mode, Sequence):
        return State(node)
    else:
        try:
            mode = list(mode)
        except TypeError:
            dle_warning("Arg mode must be str or tuple (got %s)" % type(mode))
            return State(node)

    params = {}
    if isinstance(mode[-1], dict):
        params = mode.pop(-1)
    for k in params.keys():
        if k not in ('blockshape', 'blockinner', 'openmp'):
            dle_warning("Illegal DLE parameter '%s'" % str(k))
            params.pop(k)

    mode = set(mode)
    if params.pop('openmp', False):
        mode |= {'openmp'}
    if mode.isdisjoint({'noop', 'basic', 'advanced', 'speculative',
                        'blocking', 'split', 'simd', 'openmp', 'ntstores'}):
        dle_warning("Unknown transformer mode(s) %s" % str(mode))
        return State(node)
    else:
        return Rewriter(node, params, compiler).run(mode)


def dle_transformation(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.__name__])):
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
        self.flags = ()

    def update(self, nodes=None, elemental_functions=None, arguments=None,
               includes=None, flags=None):
        self.nodes = as_tuple(nodes) or self.nodes
        self.elemental_functions = as_tuple(elemental_functions) or\
            self.elemental_functions
        self.arguments += as_tuple(arguments)
        self.includes += as_tuple(includes)
        self.flags += as_tuple(flags)

    @property
    def _has_ntstores(self):
        """True if nontemporal stores will be generated, False otherwise."""
        return 'ntstores' in self.flags


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

    """
    Track what options trigger a given transformation.
    """
    triggers = {
        '_avoid_denormals': ('basic', 'advanced', 'speculative'),
        '_create_elemental_functions': ('split', 'basic', 'advanced', 'speculative'),
        '_loop_blocking': ('blocking', 'advanced', 'speculative'),
        '_simdize': ('simd', 'basic', 'advanced', 'speculative'),
        '_nontemporal_stores': ('ntstores', 'speculative'),
        '_ompize': ('openmp',)
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'collapse': 32
    }

    def __init__(self, nodes, params, compiler):
        self.nodes = nodes
        self.params = params
        self.compiler = compiler

        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.nodes)

        self._analyze_and_decorate(state)

        self._avoid_denormals(state, mode=mode)
        self._create_elemental_functions(state, mode=mode)
        self._loop_blocking(state, mode=mode)
        self._simdize(state, mode=mode)
        self._nontemporal_stores(state, mode=mode)
        self._ompize(state, mode=mode)

        self._summary(mode)

        return state

    def _analyze_and_decorate(self, state):
        """
        Analyze the Iteration/Expression trees in ``state.nodes`` and track
        useful information for the subsequent DLE's transformation steps.

        In particular, the presence of fully-parallel or "outermost-sequential
        inner-parallel" (OSIP) :class:`Iteration` trees is tracked. In an OSIP
        :class:`Iteration` tree, outermost :class:`Iteration` objects represent
        an inherently sequential dimension, whereas all inner :class:`Iteration`
        objects represent parallelizable dimensions.
        """

        nodes = state.nodes

        sections = FindSections().visit(nodes)
        trees = sections.keys()
        candidate = max(trees, key=lambda i: len(i))
        candidates = [i for i in trees if len(i) == len(candidate)]

        # The analysis below may return "false positives" (ie, absence of fully-
        # parallel or OSIP trees when this is actually false), but this should
        # never be the case in practice, given the targeted stencil codes.
        for tree in candidates:
            exprs = [e.stencil for e in sections[tree]]

            # "Prefetch" terminals to speed up the checks below
            terms = {e: tuple(terminals(e.rhs)) for e in exprs}

            # Does the Iteration index only appear in the outermost dimension ?
            has_parallel_dimension = True
            sample = None
            for k, v in terms.items():
                if v:
                    handle = v[0]
                    sample = sample or handle.indices[0]
                    if any(sample in i.indices[1:] or sample != i.indices[0] for i in v):
                        has_parallel_dimension = False
                        break
            if not has_parallel_dimension:
                continue

            # Is the Iteration tree fully-parallel or OSIP?
            is_OSIP = False
            for e in exprs:
                lhs = e.lhs
                if lhs.is_Symbol:
                    continue
                handle = [i for i in terms[e] if i.base.label == lhs.base.label]
                if any(lhs.indices[0] != i.indices[0] for i in handle):
                    is_OSIP = True
                    break

            # Track the discovered properties in the Iteration/Expression tree
            if is_OSIP:
                args = tree[0].args
                properties = as_tuple(args.pop('properties')) + ('sequential',)
                mapper = {tree[0]: Iteration(properties=properties, **args)}
                nodes = Transformer(mapper).visit(nodes)
            mapper = {i: ('parallel',) for i in tree[is_OSIP:-1]}
            mapper[tree[-1]] = ('vector-dim',)
            for i in tree[is_OSIP:]:
                args = i.args
                properties = as_tuple(args.pop('properties')) + mapper[i]
                propertized = Iteration(properties=properties, **args)
                nodes = Transformer({i: propertized}).visit(nodes)

        state.update(nodes=nodes)

    @dle_transformation
    def _avoid_denormals(self, state, **kwargs):
        """
        Introduce nodes in the Iteration/Expression tree that will generate macros
        to avoid computing with denormal numbers. These are normally flushed away
        when using SSE-like instruction sets in a complete C program, but when
        compiling shared objects specific instructions must instead be inserted.
        """
        return {'nodes': (Denormals(),) + state.nodes,
                'includes': ('xmmintrin.h', 'pmmintrin.h')}

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
                                          Dimension("%s_block" % i.dim.name))

                    block_size = dim.symbolic_size
                    iter_size = i.dim.symbolic_size

                    start = i.limits[0] - i.offsets[0]
                    finish = iter_size - ((iter_size - i.offsets[1]) % block_size)
                    inter_block = Iteration([], dim, [start, finish, block_size],
                                            properties=('parallel', 'blocked'))

                    # Build Iteration within a block
                    dim = dims.setdefault((i.dim, 'intra-block'),
                                          Dimension("%s_intrab" % i.dim.name))

                    start = inter_block.dim
                    finish = start + block_size
                    intra_block = Iteration([], dim, [start, finish, 1], i.index,
                                            properties=('parallel',))

                    blocked_iterations.append((inter_block, intra_block))

                    # Build unitary-increment Iteration over the 'main' region
                    # (the one blocked); necessary to generate code iterating over
                    # non-blocked ("remainder") iterations.
                    start = inter_block.limits[0]
                    finish = inter_block.limits[1]
                    main = Iteration([], i.dim, [start, finish, 1], i.index,
                                     properties=('parallel',))

                    # Build unitary-increment Iteration over the 'leftover' region:
                    # again as above, this may be necessary when the dimension size
                    # is not a multiple of the block size.
                    start = inter_block.limits[1]
                    finish = iter_size - i.offsets[1]
                    leftover = Iteration([], i.dim, [start, finish, 1], i.index,
                                         properties=('parallel',))

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
                blockshape = {list(blocked)[0]: blockshape}
            blockshape.update({k: None for k in blocked.keys()
                               if k not in blockshape})

        # Track any additional arguments required to execute /state.nodes/
        arguments = [BlockingArg(v, k[0], blockshape[k]) for k, v in blocked.items()]

        return {'nodes': processed, 'arguments': arguments}

    @dle_transformation
    def _ompize(self, state, **kwargs):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit parallel code
        """

        processed = []
        for node in state.nodes:

            # Handle denormals
            denormals = FindNodeType(Denormals).visit(state.nodes)
            mapper = {i: Denormals(header=omplang['par-region']) for i in denormals}

            # Handle parallelizable loops
            for tree in retrieve_iteration_tree(node):
                # Note: a 'blocked' Iteration is guaranteed to be 'parallel' too
                blocked = [i for i in tree if 'blocked' in i.properties]
                parallelizable = [i for i in tree if 'parallel' in i.properties]
                candidates = blocked or parallelizable
                if not candidates:
                    continue

                # Heuristic: if at least two parallel loops are available and the
                # physical core count is greater than self.thresholds['collapse'],
                # then omp-collapse the loops
                if psutil.cpu_count(logical=False) < self.thresholds['collapse'] or\
                        len(candidates) < 2:
                    n = candidates[0]
                    mapper[n] = List(header=omplang['par-for'], body=n)
                else:
                    nodes = candidates[:2]
                    mapper.update({n: List(header=omplang['par-for-collapse2'], body=n)
                                   for n in nodes})

            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed}

    @dle_transformation
    def _simdize(self, state, **kwargs):
        """
        Add compiler-specific or, if not available, OpenMP pragmas to the
        Iteration/Expression tree to emit SIMD-friendly code.
        """
        if self.compiler:
            key = self.compiler.__class__.__name__
            complang = complang_ALL.get(key, {})
        else:
            complang = {}

        pragmas = [complang.get('ignore-deps', omplang['simd-for'])]

        def decorate(nodes):
            processed = []
            for node in nodes:
                mapper = {}
                for tree in retrieve_iteration_tree(node):
                    mapper.update({i: List(pragmas, i) for i in tree
                                   if 'vector-dim' in i.properties})
                processed.append(Transformer(mapper).visit(node))
            return processed

        return {'nodes': decorate(state.nodes),
                'elemental_functions': decorate(state.elemental_functions)}

    @dle_transformation
    def _nontemporal_stores(self, state, **kwargs):
        """
        Add compiler-specific pragmas and instructions to generate nontemporal
        stores (ie, non-cached stores).
        """
        if self.compiler:
            key = self.compiler.__class__.__name__
            complang = complang_ALL.get(key, {})
        else:
            complang = {}

        pragma = complang.get('ntstores')
        fence = complang.get('storefence')
        if not pragma or not fence:
            return {}

        def decorate(nodes):
            processed = []
            for node in nodes:
                mapper = {}
                for tree in retrieve_iteration_tree(node):
                    fenced = False
                    for i in tree:
                        if not fenced and 'parallel' in i.properties:
                            mapper[i] = List(body=i, footer=fence)
                            fenced = True
                        if 'vector-dim' in i.properties:
                            mapper[i] = List(header=pragma, body=i)
                processed.append(Transformer(mapper).visit(node))
            return processed

        return {'nodes': decorate(state.nodes),
                'elemental_functions': decorate(state.elemental_functions),
                'flags': 'ntstores'}

    def _summary(self, mode):
        """
        Print a summary of the DLE transformations
        """

        if mode.intersection({'blocking', 'advanced'}):
            steps = " --> ".join("(%s)" % i for i in self.timings.keys())
            elapsed = sum(self.timings.values())
            dle("%s [%.2f s]" % (steps, elapsed))


# Utilities

"""
A dictionary to quickly access standard OpenMP pragmas
"""
omplang = {
    'par-region': c.Pragma('omp parallel'),
    'par-for': c.Pragma('omp parallel for schedule(static)'),
    'par-for-collapse2': c.Pragma('omp parallel for collapse(2) schedule(static)'),
    'simd-for': c.Pragma('omp simd')
}

"""
Compiler-specific language
"""
complang_ALL = {
    'IntelCompiler': {'ignore-deps': c.Pragma('ivdep'),
                      'ntstores': c.Pragma('vector nontemporal'),
                      'storefence': c.Statement('_mm_sfence()')}
}
