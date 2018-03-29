# The internal loop engine

from __future__ import absolute_import

from collections import OrderedDict
from itertools import combinations

import cgen
import numpy as np
import psutil

from devito.cgen_utils import ccode
from devito.dimension import Dimension
from devito.dle import fold_blockable_tree, unfold_blocked_tree
from devito.dle.backends import (BasicRewriter, BlockingArg, dle_pass, omplang,
                                 simdinfo, get_simd_flag, get_simd_items)
from devito.dse import promote_scalar_expressions
from devito.exceptions import DLEException
from devito.ir.iet import (Block, Expression, Iteration, List,
                           PARALLEL, ELEMENTAL, REMAINDER, tagger,
                           FindNodes, FindSymbols, IsPerfectIteration, Transformer,
                           compose_nodes, retrieve_iteration_tree, filter_iterations)
from devito.logger import dle_warning
from devito.tools import as_tuple, grouper


class DevitoRewriter(BasicRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_fission(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._ompize(state)
        self._create_elemental_functions(state)
        self._minimize_remainders(state)

    @dle_pass
    def _loop_fission(self, nodes, state):
        """
        Apply loop fission to innermost :class:`Iteration` objects. This pass
        is not applied if the number of statements in an Iteration's body is
        lower than ``self.thresholds['fission'].``
        """

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
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
            properties = as_tuple(args_frozen['properties']) + (ELEMENTAL,)
            args_frozen['properties'] = properties
            n = self.thresholds['min_fission']
            fissioned = [Iteration(g, **args_frozen) for g in grouper(rebuilt, n)]

            mapper[candidate] = List(body=fissioned)

        processed = Transformer(mapper).visit(nodes)

        return processed, {}

    @dle_pass
    def _loop_blocking(self, nodes, state):
        """
        Apply loop blocking to :class:`Iteration` trees.

        Blocking is applied to parallel iteration trees. Heuristically, innermost
        dimensions are not blocked to maximize the trip count of the SIMD loops.

        Different heuristics may be specified by passing the keywords ``blockshape``
        and ``blockinner`` to the DLE. The former, a dictionary, is used to indicate
        a specific block size for each blocked dimension. For example, for the
        :class:`Iteration` tree: ::

            for i
              for j
                for k
                  ...

        one may provide ``blockshape = {i: 4, j: 7}``, in which case the
        two outer loops will blocked, and the resulting 2-dimensional block will
        have size 4x7. The latter may be set to True to also block innermost parallel
        :class:`Iteration` objects.
        """
        exclude_innermost = not self.params.get('blockinner', False)
        ignore_heuristic = self.params.get('blockalways', False)

        # Make sure loop blocking will span as many Iterations as possible
        fold = fold_blockable_tree(nodes, exclude_innermost)

        mapper = {}
        blocked = OrderedDict()
        for tree in retrieve_iteration_tree(fold):
            # Is the Iteration tree blockable ?
            iterations = [i for i in tree if i.is_Parallel]
            if exclude_innermost:
                iterations = [i for i in iterations if not i.is_Vectorizable]
            if len(iterations) <= 1:
                continue
            root = iterations[0]
            if not IsPerfectIteration().visit(root):
                # Illegal/unsupported
                continue
            if not tree[0].is_Sequential and not ignore_heuristic:
                # Heuristic: avoid polluting the generated code with blocked
                # nests (thus increasing JIT compilation time and affecting
                # readability) if the blockable tree isn't embedded in a
                # sequential loop (e.g., a timestepping loop)
                continue

            # Decorate intra-block iterations with an IterationProperty
            TAG = tagger(len(mapper))

            # Build all necessary Iteration objects, individually. These will
            # subsequently be composed to implement loop blocking.
            inter_blocks = []
            intra_blocks = []
            remainders = []
            for i in iterations:
                name = "%s%d_block" % (i.dim.name, len(mapper))

                # Build Iteration over blocks
                dim = blocked.setdefault(i, Dimension(name=name))
                bsize = dim.symbolic_size
                bstart = i.limits[0]
                binnersize = i.dim.symbolic_extent + (i.offsets[1] - i.offsets[0])
                bfinish = i.dim.symbolic_end - (binnersize % bsize) - 1
                inter_block = Iteration([], dim, [bstart, bfinish, bsize],
                                        offsets=i.offsets, properties=PARALLEL)
                inter_blocks.append(inter_block)

                # Build Iteration within a block
                limits = (dim, dim + bsize - 1, 1)
                intra_block = i._rebuild([], limits=limits, offsets=(0, 0),
                                         properties=i.properties + (TAG, ELEMENTAL))
                intra_blocks.append(intra_block)

                # Build unitary-increment Iteration over the 'leftover' region.
                # This will be used for remainder loops, executed when any
                # dimension size is not a multiple of the block size.
                remainder = i._rebuild([], limits=[bfinish + 1, i.dim.symbolic_end, 1],
                                       offsets=(i.offsets[1], i.offsets[1]))
                remainders.append(remainder)

            # Build blocked Iteration nest
            blocked_tree = compose_nodes(inter_blocks + intra_blocks +
                                         [iterations[-1].nodes])

            # Build remainder Iterations
            remainder_trees = []
            for n in range(len(iterations)):
                for c in combinations([i.dim for i in iterations], n + 1):
                    # First all inter-block Interations
                    nodes = [b._rebuild(properties=b.properties + (REMAINDER,))
                             for b, r in zip(inter_blocks, remainders)
                             if r.dim not in c]
                    # Then intra-block or remainder, for each dim (in order)
                    properties = (REMAINDER, TAG, ELEMENTAL)
                    for b, r in zip(intra_blocks, remainders):
                        handle = r if b.dim in c else b
                        nodes.append(handle._rebuild(properties=properties))
                    nodes.extend([iterations[-1].nodes])
                    remainder_trees.append(compose_nodes(nodes))

            # Will replace with blocked loop tree
            mapper[root] = List(body=[blocked_tree] + remainder_trees)

        rebuilt = Transformer(mapper).visit(fold)

        # Finish unrolling any previously folded Iterations
        processed = unfold_blocked_tree(rebuilt)

        # All blocked dimensions
        if not blocked:
            return processed, {}

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

        return processed, {'arguments': arguments, 'flags': 'blocking'}

    @dle_pass
    def _simdize(self, nodes, state):
        """
        Add compiler-specific or, if not available, OpenMP pragmas to the
        Iteration/Expression tree to emit SIMD-friendly code.
        """
        ignore_deps = as_tuple(self._compiler_decoration('ignore-deps'))

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            for i in vector_iterations:
                handle = FindSymbols('symbolics').visit(i)
                try:
                    aligned = [j for j in handle if j.is_Tensor and
                               j.shape[-1] % get_simd_items(j.dtype) == 0]
                except KeyError:
                    aligned = []
                if aligned:
                    simd = omplang['simd-for-aligned']
                    simd = as_tuple(simd(','.join([j.name for j in aligned]),
                                    simdinfo[get_simd_flag()]))
                else:
                    simd = as_tuple(omplang['simd-for'])
                mapper[i] = i._rebuild(pragmas=i.pragmas + ignore_deps + simd)

        processed = Transformer(mapper).visit(nodes)

        return processed, {}

    @dle_pass
    def _ompize(self, nodes, state):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit parallel code
        """
        # Group by outer loop so that we can embed within the same parallel region
        was_tagged = False
        groups = OrderedDict()
        for tree in retrieve_iteration_tree(nodes):
            # Determine the number of consecutive parallelizable Iterations
            key = lambda i: i.is_ParallelRelaxed and\
                not (i.is_Elementizable or i.is_Vectorizable)
            candidates = filter_iterations(tree, key=key, stop='asap')
            if not candidates:
                was_tagged = False
                continue
            # Consecutive tagged Iteration go in the same group
            is_tagged = any(i.tag is not None for i in tree)
            key = len(groups) - (is_tagged & was_tagged)
            handle = groups.setdefault(key, OrderedDict())
            handle[candidates[0]] = candidates
            was_tagged = is_tagged

        # Handle parallelizable loops
        mapper = OrderedDict()
        for group in groups.values():
            private = []
            for root, tree in group.items():
                # Heuristic: if at least two parallel loops are available and the
                # physical core count is greater than self.thresholds['collapse'],
                # then omp-collapse the loops
                nparallel = len(tree)
                if psutil.cpu_count(logical=False) < self.thresholds['collapse'] or\
                        nparallel < 2:
                    parallel = omplang['for']
                else:
                    parallel = omplang['collapse'](nparallel)

                # Introduce the `omp parallel` pragma
                if root.is_ParallelAtomic:
                    # Introduce the `omp atomic` pragmas
                    exprs = FindNodes(Expression).visit(root)
                    subs = {i: List(header=omplang['atomic'], body=i)
                            for i in exprs if i.is_increment}
                    handle = Transformer(subs).visit(root)
                    mapper[root] = handle._rebuild(pragmas=root.pragmas + (parallel,))
                else:
                    mapper[root] = root._rebuild(pragmas=root.pragmas + (parallel,))

                # Track the thread-private and thread-shared variables
                private.extend([i for i in FindSymbols('symbolics').visit(root)
                                if i.is_Array and i._mem_stack])

            # Build the parallel region
            private = sorted(set([i.name for i in private]))
            private = ('private(%s)' % ','.join(private)) if private else ''
            rebuilt = [v for k, v in mapper.items() if k in group]
            par_region = Block(header=omplang['par-region'](private), body=rebuilt)
            for k, v in list(mapper.items()):
                if isinstance(v, Iteration):
                    mapper[k] = None if v.is_Remainder else par_region

        processed = Transformer(mapper).visit(nodes)

        return processed, {}

    @dle_pass
    def _minimize_remainders(self, nodes, state):
        """
        Reshape temporary tensors and adjust loop trip counts to prevent as many
        compiler-generated remainder loops as possible.
        """
        # The innermost dimension is the one that might get padded
        p_dim = -1

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            if not vector_iterations or len(vector_iterations) > 1:
                continue
            root = vector_iterations[0]
            if root.tag is None:
                continue

            # Padding
            writes = [i for i in FindSymbols('symbolics-writes').visit(root)
                      if i.is_Array]
            padding = []
            for i in writes:
                try:
                    simd_items = get_simd_items(i.dtype)
                except KeyError:
                    # Fallback to 16 (maximum expectable padding, for AVX512 registers)
                    simd_items = simdinfo['avx512f'] / np.dtype(i.dtype).itemsize
                padding.append(simd_items - i.shape[-1] % simd_items)
            if len(set(padding)) == 1:
                padding = padding[0]
                for i in writes:
                    padded = (i._padding[p_dim][0], i._padding[p_dim][1] + padding)
                    i.update(padding=i._padding[:p_dim] + (padded,))
            else:
                # Padding must be uniform -- not the case, so giving up
                continue

            # Dynamic trip count adjustment
            endpoint = root.end_symbolic
            if not endpoint.is_Symbol:
                continue
            condition = []
            externals = set(i.symbolic_shape[-1] for i in FindSymbols().visit(root)
                            if i.is_Tensor)
            for i in root.uindices:
                for j in externals:
                    condition.append(root.end_symbolic + padding < j)
            condition = ' && '.join(ccode(i) for i in condition)
            endpoint_padded = endpoint.func('_%s' % endpoint.name)
            init = cgen.Initializer(
                cgen.Value("const int", endpoint_padded),
                cgen.Line('(%s) ? %s : %s' % (condition,
                                              ccode(endpoint + padding),
                                              endpoint))
            )

            # Update the Iteration bound
            limits = list(root.limits)
            limits[1] = endpoint_padded.func(endpoint_padded.name)
            rebuilt = list(tree)
            rebuilt[rebuilt.index(root)] = root._rebuild(limits=limits)

            mapper[tree[0]] = List(header=init, body=compose_nodes(rebuilt))

        processed = Transformer(mapper).visit(nodes)

        return processed, {}


class DevitoRewriterSafeMath(DevitoRewriter):

    """
    This Rewriter is slightly less aggressive than :class:`DevitoRewriter`, as it
    doesn't drop denormal numbers, which may sometimes harm the numerical precision.
    Loop fission is also avoided (to avoid reassociation of operations).
    """

    def _pipeline(self, state):
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._ompize(state)
        self._create_elemental_functions(state)
        self._minimize_remainders(state)


class DevitoSpeculativeRewriter(DevitoRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._loop_fission(state)
        self._loop_blocking(state)
        self._simdize(state)
        self._nontemporal_stores(state)
        if self.params['openmp'] is True:
            self._ompize(state)
        self._create_elemental_functions(state)
        self._minimize_remainders(state)

    @dle_pass
    def _nontemporal_stores(self, nodes, state):
        """
        Add compiler-specific pragmas and instructions to generate nontemporal
        stores (ie, non-cached stores).
        """
        pragma = self._compiler_decoration('ntstores')
        fence = self._compiler_decoration('storefence')
        if not pragma or not fence:
            return {}

        mapper = {}
        for tree in retrieve_iteration_tree(nodes):
            for i in tree:
                if i.is_Parallel:
                    mapper[i] = List(body=i, footer=fence)
                    break
        processed = Transformer(mapper).visit(nodes)

        mapper = {}
        for tree in retrieve_iteration_tree(processed):
            for i in tree:
                if i.is_Vectorizable:
                    mapper[i] = List(header=pragma, body=i)
        processed = Transformer(mapper).visit(processed)

        return processed, {'flags': 'ntstores'}


class DevitoCustomRewriter(DevitoSpeculativeRewriter):

    passes_mapper = {
        'denormals': DevitoSpeculativeRewriter._avoid_denormals,
        'blocking': DevitoSpeculativeRewriter._loop_blocking,
        'openmp': DevitoSpeculativeRewriter._ompize,
        'simd': DevitoSpeculativeRewriter._simdize,
        'fission': DevitoSpeculativeRewriter._loop_fission,
        'split': DevitoSpeculativeRewriter._create_elemental_functions
    }

    def __init__(self, nodes, passes, params):
        try:
            passes = passes.split(',')
        except AttributeError:
            # Already in tuple format
            if not all(i in DevitoCustomRewriter.passes_mapper for i in passes):
                raise DLEException
        self.passes = passes
        super(DevitoCustomRewriter, self).__init__(nodes, params)

    def _pipeline(self, state):
        for i in self.passes:
            DevitoCustomRewriter.passes_mapper[i](self, state)
