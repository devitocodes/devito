from __future__ import absolute_import

from collections import OrderedDict, Sequence, namedtuple
from itertools import combinations
from operator import attrgetter
from time import time

import cgen as c
import cpuinfo
import numpy as np
import psutil

from devito.dimension import Dimension
from devito.dle import (compose_nodes, copy_arrays,
                        filter_iterations, retrieve_iteration_tree)
from devito.dse import (as_symbol, estimate_cost, promote_scalar_expressions, terminals)
from devito.interfaces import ScalarFunction, TensorFunction
from devito.logger import dle, dle_warning
from devito.nodes import (Block, Denormals, Element, Expression, FunCall,
                          Function, Iteration, List)
from devito.tools import as_tuple, filter_sorted, flatten, grouper, roundm
from devito.visitors import (FindNodes, FindSections, FindSymbols,
                             IsPerfectIteration, SubstituteExpression,
                             NestedTransformer, Transformer)


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
                    * '3D-advanced': Like 'advanced', but apply 3D loop blocking
                                     if there are at least the perfectly nested
                                     parallel iteration spaces -- [S]
                    * 'speculative': Apply all of the 'advanced' transformations,
                                     plus other transformations that might increase
                                     (or possibly decrease) performance -- [S]
                    * 'fission': Apply loop fission -- [T]
                    * 'blocking': Apply loop blocking -- [T]
                    * 'split': Identify and split elemental functions -- [T]
                    * 'padding': Introduce "shadow" arrays, padded to the nearest
                                 multiple of the vector length, to maximize data
                                 alignment -- [T]
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
    if '3D-advanced' in mode:
        params['blockinner'] = True
        mode.remove('3D-advanced')
        mode.add('advanced')
    if 'noop' not in mode and params.pop('openmp', False):
        mode |= {'openmp'}
    if mode.isdisjoint(set(Rewriter.modes)):
        dle_warning("Unknown transformer mode(s) %s" % str(mode))
        return State(node)
    else:
        return Rewriter(node, params, compiler).run(mode)


def dle_pass(func):

    def wrapper(self, state, **kwargs):
        if kwargs['mode'].intersection(set(self.triggers[func.__name__])):
            tic = time()
            state.update(**func(self, state))
            toc = time()

            self.timings[func.__name__] = toc - tic

    return wrapper


class State(object):

    """Represent the output of the DLE."""

    def __init__(self, nodes, mode='noop'):
        self.nodes = as_tuple(nodes)
        self.mode = mode

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
    def has_applied_nontemporal_stores(self):
        """True if nontemporal stores will be generated, False otherwise."""
        return 'ntstores' in self.flags

    @property
    def has_applied_blocking(self):
        """True if loop blocking was applied, False otherwise."""
        return 'blocking' in self.flags

    @property
    def func_table(self):
        """Return a mapper from elemental function names to :class:`Function`."""
        return OrderedDict([(i.name, i) for i in self.elemental_functions])

    @property
    def needs_aggressive_autotuning(self):
        return self.mode in ['aggressive']


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


class Rewriter(object):

    """
    Transform Iteration/Expression trees to generate high performance C.
    """

    """
    All DLE transformation modes.
    """
    modes = ('noop', 'basic', 'advanced', '3D-advanced', 'speculative',
             'fission', 'padding', 'blocking', 'split', 'simd', 'openmp', 'ntstores')

    """
    Track what options trigger a given transformation.
    """
    triggers = {
        '_analyze': modes,
        '_avoid_denormals': ('basic', 'advanced', 'speculative'),
        '_loop_fission': ('fission', 'split', 'advanced', 'speculative'),
        '_create_elemental_functions': ('split', 'basic', 'advanced', 'speculative'),
        '_loop_blocking': ('blocking', 'advanced', 'speculative'),
        '_padding': ('padding', 'speculative'),
        '_simdize': ('simd', 'basic', 'advanced', 'speculative'),
        '_nontemporal_stores': ('ntstores', 'speculative'),
        '_ompize': ('openmp',)
    }

    """
    Bag of thresholds, to be used to trigger or prevent certain transformations.
    """
    thresholds = {
        'collapse': 32,  # Available physical cores
        'elemental': 30,  # Operations
        'max_fission': 8000000,  # Statements
        'min_fission': 1  # Statements
    }

    def __init__(self, nodes, params, compiler):
        self.nodes = nodes
        self.params = params
        self.compiler = compiler

        self.timings = OrderedDict()

    def run(self, mode):
        state = State(self.nodes, mode)

        self._analyze(state, mode=mode)

        self._avoid_denormals(state, mode=mode)
        self._loop_fission(state, mode=mode)
        self._padding(state, mode=mode)
        self._create_elemental_functions(state, mode=mode)
        # FIXME: still unsupported
        #self._loop_blocking(state, mode=mode)
        self._simdize(state, mode=mode)
        self._nontemporal_stores(state, mode=mode)
        self._ompize(state, mode=mode)

        self._summary(mode)

        return state

    @dle_pass
    def _analyze(self, state, **kwargs):
        """
        Analyze the Iteration/Expression trees in ``state.nodes`` to detect
        information useful in subsequent DLE passes.

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

            # Is the Iteration tree fully-parallel or OSIP?
            is_OSIP = False
            for e1 in exprs:
                lhs = e1.lhs
                if lhs.is_Symbol:
                    continue
                for e2 in exprs:
                    handle = [i for i in terms[e2] if as_symbol(i) == as_symbol(lhs)]
                    if any(lhs.indices[0] != i.indices[0] for i in handle):
                        is_OSIP = True
                        break

            # Track the discovered properties
            if is_OSIP:
                mapper.setdefault(tree[0], []).append('sequential')
            for i in tree[is_OSIP:-1]:
                mapper.setdefault(i, []).append('parallel')
            mapper.setdefault(tree[-1], []).extend(['parallel', 'vector-dim'])

        # Introduce the discovered properties in the Iteration/Expression tree
        for k, v in list(mapper.items()):
            args = k.args
            # 'sequential' has obviously precedence over 'parallel'
            properties = ('sequential',) if 'sequential' in v else tuple(v)
            properties = as_tuple(args.pop('properties')) + properties
            mapper[k] = Iteration(properties=properties, **args)
        nodes = NestedTransformer(mapper).visit(nodes)

        return {'nodes': nodes}

    @dle_pass
    def _avoid_denormals(self, state, **kwargs):
        """
        Introduce nodes in the Iteration/Expression tree that will generate macros
        to avoid computing with denormal numbers. These are normally flushed away
        when using SSE-like instruction sets in a complete C program, but when
        compiling shared objects specific instructions must instead be inserted.
        """
        return {'nodes': (Denormals(),) + state.nodes,
                'includes': ('xmmintrin.h', 'pmmintrin.h')}

    @dle_pass
    def _padding(self, state, **kwargs):
        """
        Introduce temporary buffers padded to the nearest multiple of the vector
        length, to maximize data alignment. At the bottom of the kernel, the
        values in the padded temporaries will be copied back into the input arrays.
        """

        mapper = OrderedDict()
        for node in state.nodes:
            # Assess feasibility of the transformation
            handle = FindSymbols('symbolics-writes').visit(node)
            if not handle:
                continue

            shape = max([i.shape for i in handle], key=len)
            if not shape:
                continue

            candidates = [i for i in handle if i.shape[-1] == shape[-1]]
            if not candidates:
                continue

            # Retrieve the maximum number of items in a SIMD register when processing
            # the expressions in /node/
            exprs = FindNodes(Expression).visit(node)
            exprs = [e for e in exprs if e.output_function in candidates]
            assert len(exprs) > 0
            dtype = exprs[0].dtype
            assert all(e.dtype == dtype for e in exprs)
            try:
                simd_items = get_simd_items(dtype)
            except KeyError:
                # Fallback to 16 (maximum expectable padding, for AVX512 registers)
                simd_items = simdinfo['avx512f'] / np.dtype(dtype).itemsize

            shapes = {k: k.shape[:-1] + (roundm(k.shape[-1], simd_items),)
                      for k in candidates}
            mapper.update(OrderedDict([(k.indexed,
                                        TensorFunction(name='p%s' % k.name,
                                                       shape=shapes[k],
                                                       dimensions=k.indices,
                                                       onstack=k._mem_stack).indexed)
                          for k in candidates]))

        # Substitute original arrays with padded buffers
        processed = [SubstituteExpression(mapper).visit(n) for n in state.nodes]

        # Build Iteration trees for initialization and copy-back of padded arrays
        mapper = OrderedDict([(k, v) for k, v in mapper.items()
                              if k.function.is_SymbolicData])
        init = copy_arrays(mapper, reverse=True)
        copyback = copy_arrays(mapper)

        processed = init + as_tuple(processed) + copyback

        return {'nodes': processed}

    @dle_pass
    def _create_elemental_functions(self, state, **kwargs):
        """
        Move :class:`Iteration` sub-trees to separate functions.

        By default, inner iteration trees are moved. To move different types of
        :class:`Iteration`, one can provide a lambda function in ``kwargs['rule']``,
        taking as input an iterable of :class:`Iteration` and returning an iterable
        of :class:`Iteration` (eg, a subset, the whole iteration tree).
        """
        noinline = self._compiler_decoration('noinline', c.Comment('noinline?'))
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
                root = candidate[0]
                expressions = FindNodes(Expression).visit(candidate)

                # Heuristic: create elemental functions only if more than
                # self.thresholds['elemental_functions'] operations are present
                ops = estimate_cost([e.expr for e in expressions])
                if ops < self.thresholds['elemental'] and not root.is_Elementizable:
                    continue

                # Determine the elemental function's arguments ...
                already_in_scope = [k.dim for k in candidate]
                required = [k for k in FindSymbols(mode='free-symbols').visit(candidate)
                            if k not in already_in_scope]
                required += [as_symbol(k) for k in
                             set(flatten(k.free_symbols for k in root.bounds_symbolic))]
                required = set(required)

                args = []
                seen = {e.output for e in expressions if e.is_scalar}
                for d in FindSymbols('symbolics').visit(candidate):
                    # Add a necessary Symbolic object
                    handle = "(float*) %s" if d.is_SymbolicFunction else "%s_vec"
                    args.append((handle % d.name, d))
                    seen |= {as_symbol(d)}
                    # Add necessary information related to Dimensions
                    for k in d.indices:
                        if k.size is not None:
                            continue
                        # Dimension size
                        size = k.symbolic_size
                        if size not in seen:
                            args.append((k.ccode, k))
                            seen |= {size}
                        # Dimension index may be required too
                        if k in required - seen:
                            index_arg = (k.name, ScalarFunction(name=k.name,
                                                                dtype=np.int32))
                            args.append(index_arg)
                            seen |= {k}

                # Add non-temporary scalars to the elemental function's arguments
                handle = filter_sorted(required - seen, key=attrgetter('name'))
                args.extend([(k.name, ScalarFunction(name=k.name, dtype=np.int32))
                             for k in handle])

                # Track info to transform the main tree
                call, parameters = zip(*args)
                mapper[root] = List(header=noinline, body=FunCall(name, call))

                # Produce the new function
                functions.append(Function(name, root, 'void', parameters, ('static',)))

            # Transform the main tree
            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed, 'elemental_functions': functions}

    @dle_pass
    def _loop_fission(self, state, **kwargs):
        """
        Apply loop fission to innermost :class:`Iteartion` objects. This pass
        is not applied if the number of statements in an Iteration's body is
        lower than ``self.thresholds['fission'].``
        """

        processed = []
        for node in state.nodes:
            mapper = {}
            for tree in retrieve_iteration_tree(node):
                if len(tree) <= 1:
                    # Heuristically avoided
                    continue

                candidate = tree[-1]
                expressions = [e for e in candidate.nodes if e.is_Expression]

                if len(expressions) < self.thresholds['max_fission']:
                    # Heuristically avoided
                    continue
                if len(expressions) != len(candidate.nodes):
                    # Dangerous for correctness
                    continue

                functions = list(set.union(*[set(e.functions) for e in expressions]))
                wrapped = [e.expr for e in expressions]

                if not functions or not wrapped:
                    # Heuristically avoided
                    continue

                # Promote temporaries from scalar to tensors
                handle = functions[0]
                dim = handle.indices[-1]
                size = handle.shape[-1]
                if any(dim != i.indices[-1] for i in functions):
                    # Dangerous for correctness
                    continue

                wrapped = promote_scalar_expressions(wrapped, (size,), (dim,), True)

                assert len(wrapped) == len(expressions)
                rebuilt = [Expression(s, e.dtype) for s, e in zip(wrapped, expressions)]

                # Group statements
                # TODO: Need a heuristic here to maximize reuse
                args_frozen = candidate.args_frozen
                properties = as_tuple(args_frozen['properties']) + ('elemental',)
                args_frozen['properties'] = properties
                n = self.thresholds['min_fission']
                fissioned = [Iteration(g, **args_frozen) for g in grouper(rebuilt, n)]

                mapper[candidate] = List(body=fissioned)

            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed}

    @dle_pass
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

        blocked = OrderedDict()
        processed = []
        for node in state.nodes:
            mapper = {}
            for tree in retrieve_iteration_tree(node):
                # Is the Iteration tree blockable ?
                iterations = [i for i in tree if i.is_Parallel]
                if 'blockinner' not in self.params:
                    iterations = [i for i in iterations if not i.is_Vectorizable]
                if not iterations:
                    continue
                root = iterations[0]
                if not IsPerfectIteration().visit(root):
                    continue

                # Construct the blocked loop nest, as well as all necessary
                # remainder loops
                regions = OrderedDict()
                blocked_iterations = []
                for i in iterations:
                    # Build Iteration over blocks
                    dim = blocked.setdefault(i, Dimension("%s_block" % i.dim.name))
                    block_size = dim.symbolic_size
                    iter_size = i.dim.size or i.dim.symbolic_size
                    start = i.limits[0] - i.offsets[0]
                    finish = iter_size - i.offsets[1]
                    finish = finish - ((finish - i.offsets[1]) % block_size)
                    inter_block = Iteration([], dim, [start, finish, block_size],
                                            properties=as_tuple('parallel'))

                    # Build Iteration within a block
                    start = inter_block.dim
                    finish = start + block_size
                    properties = 'vector-dim' if i.is_Vectorizable else None
                    intra_block = Iteration([], i.dim, [start, finish, 1], i.index,
                                            properties=as_tuple(properties))

                    blocked_iterations.append((inter_block, intra_block))

                    # Build unitary-increment Iteration over the 'main' region
                    # (the one blocked); necessary to generate code iterating over
                    # non-blocked ("remainder") iterations.
                    start = inter_block.limits[0]
                    finish = inter_block.limits[1]
                    main = Iteration([], i.dim, [start, finish, 1], i.index,
                                     properties=i.properties)

                    # Build unitary-increment Iteration over the 'leftover' region:
                    # again as above, this may be necessary when the dimension size
                    # is not a multiple of the block size.
                    start = inter_block.limits[1]
                    finish = iter_size - i.offsets[1]
                    leftover = Iteration([], i.dim, [start, finish, 1], i.index,
                                         properties=i.properties)

                    regions[i] = Region(main, leftover)

                blocked_tree = list(flatten(zip(*blocked_iterations)))
                blocked_tree = compose_nodes(blocked_tree + [iterations[-1].nodes])

                # Build remainder loops
                remainder_tree = []
                for n in range(len(iterations)):
                    for i in combinations(iterations, n + 1):
                        nodes = [v.leftover if k in i else v.main
                                 for k, v in regions.items()]
                        nodes += [iterations[-1].nodes]
                        remainder_tree.append(compose_nodes(nodes))

                # Will replace with blocked loop tree
                mapper[root] = List(body=[blocked_tree] + remainder_tree)

            rebuilt = Transformer(mapper).visit(node)

            processed.append(rebuilt)

        # All blocked dimensions
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
        arguments = [BlockingArg(v, k, blockshape[k]) for k, v in blocked.items()]

        return {'nodes': processed, 'arguments': arguments, 'flags': 'blocking'}

    @dle_pass
    def _ompize(self, state, **kwargs):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit parallel code
        """

        processed = []
        for node in state.nodes:

            # Reset denormals flag each time a parallel region is entered
            denormals = FindNodes(Denormals).visit(state.nodes)
            mapper = {i: List(c.Comment('DLE: moved denormals flag')) for i in denormals}

            # Handle parallelizable loops
            for tree in retrieve_iteration_tree(node):
                # Determine the number of consecutive parallelizable Iterations
                key = lambda i: i.is_Parallel and not i.is_Vectorizable
                candidates = filter_iterations(tree, key=key, stop='consecutive')
                if not candidates:
                    continue

                # Heuristic: if at least two parallel loops are available and the
                # physical core count is greater than self.thresholds['collapse'],
                # then omp-collapse the loops
                nparallel = len(candidates)
                if psutil.cpu_count(logical=False) < self.thresholds['collapse'] or\
                        nparallel < 2:
                    parallelism = omplang['for']
                else:
                    parallelism = omplang['collapse'](nparallel)

                root = candidates[0]
                mapper[root] = Block(header=omplang['par-region'],
                                     body=denormals + [Element(parallelism), root])

            processed.append(Transformer(mapper).visit(node))

        return {'nodes': processed}

    @dle_pass
    def _simdize(self, state, **kwargs):
        """
        Add compiler-specific or, if not available, OpenMP pragmas to the
        Iteration/Expression tree to emit SIMD-friendly code.
        """
        ignore_deps = as_tuple(self._compiler_decoration('ignore-deps'))

        def decorate(nodes):
            processed = []
            for node in nodes:
                mapper = {}
                for tree in retrieve_iteration_tree(node):
                    vector_iterations = [i for i in tree if i.is_Vectorizable]
                    for i in vector_iterations:
                        handle = FindSymbols('symbolics').visit(i)
                        try:
                            aligned = [j for j in handle
                                       if j.shape[-1] % get_simd_items(j.dtype) == 0]
                        except KeyError:
                            aligned = []
                        if aligned:
                            simd = omplang['simd-for-aligned']
                            simd = simd(','.join([j.name for j in aligned]),
                                        simdinfo[get_simd_flag()])
                        else:
                            simd = omplang['simd-for']
                        mapper[i] = List(ignore_deps + as_tuple(simd), i)
                processed.append(Transformer(mapper).visit(node))
            return processed

        return {'nodes': decorate(state.nodes),
                'elemental_functions': decorate(state.elemental_functions)}

    @dle_pass
    def _nontemporal_stores(self, state, **kwargs):
        """
        Add compiler-specific pragmas and instructions to generate nontemporal
        stores (ie, non-cached stores).
        """
        pragma = self._compiler_decoration('ntstores')
        fence = self._compiler_decoration('storefence')
        if not pragma or not fence:
            return {}

        def decorate(nodes):
            processed = []
            for node in nodes:
                mapper = {}
                for tree in retrieve_iteration_tree(node):
                    for i in tree:
                        if i.is_Parallel:
                            mapper[i] = List(body=i, footer=fence)
                            break
                transformed = Transformer(mapper).visit(node)
                mapper = {}
                for tree in retrieve_iteration_tree(transformed):
                    for i in tree:
                        if i.is_Vectorizable:
                            mapper[i] = List(header=pragma, body=i)
                transformed = Transformer(mapper).visit(transformed)
                processed.append(transformed)
            return processed

        return {'nodes': decorate(state.nodes),
                'elemental_functions': decorate(state.elemental_functions),
                'flags': 'ntstores'}

    def _compiler_decoration(self, name, default=None):
        if self.compiler:
            key = self.compiler.__class__.__name__
            complang = complang_ALL.get(key, {})
        else:
            complang = {}
        return complang.get(name, default)

    def _summary(self, mode):
        """
        Print a summary of the DLE transformations
        """

        if mode.intersection({'blocking', 'basic', 'advanced', 'speculative'}):
            row = "%s [elapsed: %.2f]"
            out = " >>\n     ".join(row % (filter(lambda c: not c.isdigit(), k[1:]), v)
                                    for k, v in self.timings.items())
            elapsed = sum(self.timings.values())
            dle("%s\n     [Total elapsed: %.2f s]" % (out, elapsed))


# Utilities

"""
A dictionary to quickly access standard OpenMP pragmas
"""
omplang = {
    'for': c.Pragma('omp for schedule(static)'),
    'collapse': lambda i: c.Pragma('omp for collapse(%d) schedule(static)' % i),
    'par-region': c.Pragma('omp parallel'),
    'par-for': c.Pragma('omp parallel for schedule(static)'),
    'simd-for': c.Pragma('omp simd'),
    'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j))
}

"""
Compiler-specific language
"""
complang_ALL = {
    'IntelCompiler': {'ignore-deps': c.Pragma('ivdep'),
                      'ntstores': c.Pragma('vector nontemporal'),
                      'storefence': c.Statement('_mm_sfence()'),
                      'noinline': c.Pragma('noinline')}
}
complang_ALL['IntelKNLCompiler'] = complang_ALL['IntelCompiler']

"""
SIMD generic info
"""
simdinfo = {
    # Sizes in bytes of a vector register
    'sse': 16, 'see4_2': 16,
    'avx': 32, 'avx2': 32,
    'avx512f': 64
}


def get_simd_flag():
    """Retrieve the best SIMD flag on the current architecture."""
    if get_simd_flag.flag is None:
        ordered_known = ('sse', 'sse4_2', 'avx', 'avx2', 'avx512f')
        flags = cpuinfo.get_cpu_info().get('flags')
        if not flags:
            return None
        for i in reversed(ordered_known):
            if i in flags:
                get_simd_flag.flag = i
                return i
    else:
        # "Cached" because calls to cpuingo are expensive
        return get_simd_flag.flag
get_simd_flag.flag = None  # noqa


def get_simd_items(dtype):
    """Determine the number of items of type ``dtype`` that can fit in a SIMD
    register on the current architecture."""

    simd_size = simdinfo[get_simd_flag()]
    return simd_size / np.dtype(dtype).itemsize
