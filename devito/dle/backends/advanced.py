# The internal loop engine

from __future__ import absolute_import

from collections import OrderedDict, namedtuple
from itertools import combinations

import numpy as np
import psutil

import cgen as c

from devito.dimension import Dimension
from devito.dle import (compose_nodes, copy_arrays, filter_iterations,
                        fold_blockable_tree, unfold_blocked_tree,
                        retrieve_iteration_tree)
from devito.dle.backends import (BasicRewriter, BlockingArg, dle_pass, omplang,
                                 simdinfo, get_simd_flag, get_simd_items)
from devito.dse import promote_scalar_expressions
from devito.exceptions import DLEException
from devito.interfaces import TensorFunction
from devito.logger import dle_warning
from devito.nodes import Block, Denormals, Element, Expression, Iteration, List
from devito.tools import as_tuple, flatten, grouper, roundm
from devito.visitors import (FindNodes, FindSymbols, IsPerfectIteration,
                             SubstituteExpression, Transformer)


class DevitoRewriter(BasicRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_fission(state)
        #self._create_elemental_functions(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._ompize(state)

    @dle_pass
    def _loop_fission(self, state, **kwargs):
        """
        Apply loop fission to innermost :class:`Iteration` objects. This pass
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
        Apply loop blocking to :class:`Iteration` trees.

        Blocking is applied to parallel iteration trees. Heuristically, innermost
        dimensions are not blocked to maximize the trip count of the SIMD loops.

        Different heuristics may be specified via ``kwargs['blockshape']`` and
        ``kwargs['blockinner']``. The former, a dictionary, is used to indicate
        a specific block size for each blocked dimension. For example, for the
        :class:`Iteration` tree: ::

            for i
              for j
                for k
                  ...

        one may provide ``kwargs['blockshape'] = {i: 4, j: 7}``, in which case the
        two outer loops will blocked, and the resulting 2-dimensional block will
        have size 4x7. The latter may be set to True to also block innermost parallel
        :class:`Iteration` objects.
        """
        Region = namedtuple('Region', 'main leftover')
        exclude_innermost = 'blockinner' not in self.params

        blocked = OrderedDict()
        processed = []
        for node in state.nodes:
            # Make sure loop blocking will span as many Iterations as possible
            fold = fold_blockable_tree(node, exclude_innermost)

            mapper = {}
            for tree in retrieve_iteration_tree(fold):
                # Is the Iteration tree blockable ?
                iterations = [i for i in tree if i.is_Parallel]
                if exclude_innermost:
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
                    intra_block = i._rebuild([], limits=[start, finish, 1], offsets=None,
                                             properties=as_tuple(properties))

                    blocked_iterations.append((inter_block, intra_block))

                    # Build unitary-increment Iteration over the 'main' region
                    # (the one blocked); necessary to generate code iterating over
                    # non-blocked ("remainder") iterations.
                    start = inter_block.limits[0]
                    finish = inter_block.limits[1]
                    main = i._rebuild([], limits=[start, finish, 1], offsets=None,
                                      properties=i.properties)

                    # Build unitary-increment Iteration over the 'leftover' region:
                    # again as above, this may be necessary when the dimension size
                    # is not a multiple of the block size.
                    start = inter_block.limits[1]
                    finish = iter_size - i.offsets[1]
                    leftover = i._rebuild([], limits=[start, finish, 1], offsets=None,
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

            rebuilt = Transformer(mapper).visit(fold)

            # Finish unrolling any previously folded Iterations
            processed.append(unfold_blocked_tree(rebuilt))

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


class DevitoSpeculativeRewriter(DevitoRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_fission(state)
        self._padding(state)
        self._create_elemental_functions(state)
        self._loop_blocking(state)
        self._simdize(state)
        self._nontemporal_stores(state)
        if self.params['openmp'] is True:
            self._ompize(state)

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


class DevitoCustomRewriter(DevitoSpeculativeRewriter):

    passes_mapper = {
        'fission': DevitoSpeculativeRewriter._loop_fission,
        'padding': DevitoSpeculativeRewriter._padding,
        'split': DevitoSpeculativeRewriter._create_elemental_functions
    }

    def __init__(self, nodes, passes, params):
        try:
            passes = passes.split(',')
        except AttributeError:
            raise DLEException
        if not all(i in DevitoCustomRewriter.passes_mapper for i in passes):
            raise DLEException
        self.passes = passes
        super(DevitoCustomRewriter, self).__init__(nodes, params)

    def _pipeline(self, state):
        for i in self.passes:
            DevitoCustomRewriter.passes_mapper[i](self, state)
