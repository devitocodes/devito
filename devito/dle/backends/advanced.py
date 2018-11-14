from collections import OrderedDict
from itertools import combinations

import cgen
import numpy as np

from devito.cgen_utils import ccode
from devito.dle import BlockDimension, fold_blockable_tree, unfold_blocked_tree
from devito.dle.backends import (BasicRewriter, Ompizer, dle_pass, simdinfo,
                                 get_simd_flag, get_simd_items)
from devito.exceptions import DLEException
from devito.ir.iet import (Expression, Iteration, List, HaloSpot, PARALLEL, ELEMENTAL,
                           REMAINDER, tagger, FindSymbols, FindNodes, Transformer,
                           IsPerfectIteration, compose_nodes, retrieve_iteration_tree)
from devito.logger import perf_adv
from devito.tools import as_tuple


class AdvancedRewriter(BasicRewriter):

    _parallelizer = Ompizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halo_updates(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._create_efuncs(state)
        self._minimize_remainders(state)

    @dle_pass
    def _loop_wrapping(self, iet, state):
        """
        Emit a performance warning if WRAPPABLE :class:`Iteration`s are found,
        as these are a symptom that unnecessary memory is being allocated.
        """
        for i in FindNodes(Iteration).visit(iet):
            if not i.is_Wrappable:
                continue
            perf_adv("Functions using modulo iteration along Dimension `%s` "
                     "may safely allocate a one slot smaller buffer" % i.dim)
        return iet, {}

    @dle_pass
    def _optimize_halo_updates(self, iet, state):
        """
        Drop unnecessary halo exchanges, or shuffle them around to improve
        computation-communication overlap.
        """
        hss = FindNodes(HaloSpot).visit(iet)
        mapper = {i: None for i in hss if i.is_Redundant}
        processed = Transformer(mapper, nested=True).visit(iet)

        return processed, {}

    @dle_pass
    def _loop_blocking(self, nodes, state):
        """
        Apply loop blocking to PARALLEL :class:`Iteration` trees.
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
            if not tree.root.is_Sequential and not ignore_heuristic:
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
                # Build Iteration over blocks
                name = "%s%d_block" % (i.dim.name, len(mapper))
                dim = blocked.setdefault(i, BlockDimension(i.dim, name=name))
                binnersize = i.symbolic_extent + (i.offsets[1] - i.offsets[0])
                bfinish = i.dim.symbolic_end - (binnersize % dim.step)
                inter_block = Iteration([], dim, bfinish, offsets=i.offsets,
                                        properties=PARALLEL)
                inter_blocks.append(inter_block)

                # Build Iteration within a block
                limits = (dim, dim + dim.step - 1, 1)
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

        return processed, {'dimensions': list(blocked.values())}

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
                    simd = Ompizer.lang['simd-for-aligned']
                    simd = as_tuple(simd(','.join([j.name for j in aligned]),
                                    simdinfo[get_simd_flag()]))
                else:
                    simd = as_tuple(Ompizer.lang['simd-for'])
                mapper[i] = i._rebuild(pragmas=i.pragmas + ignore_deps + simd)

        processed = Transformer(mapper).visit(nodes)

        return processed, {}

    @dle_pass
    def _parallelize(self, iet, state):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit parallel code
        """
        def key(i):
            return i.is_ParallelRelaxed and not (i.is_Elementizable or i.is_Vectorizable)
        return self._parallelizer(key).make_parallel(iet)

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
            writes = [i.write for i in FindNodes(Expression).visit(root)
                      if i.write.is_Array]
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
            endpoint = root.symbolic_end
            if not endpoint.is_Symbol:
                continue
            condition = []
            externals = set(i.symbolic_shape[-1] for i in FindSymbols().visit(root)
                            if i.is_Tensor)
            for i in root.uindices:
                for j in externals:
                    condition.append(root.symbolic_end + padding < j)
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


class AdvancedRewriterSafeMath(AdvancedRewriter):

    """
    This Rewriter is slightly less aggressive than :class:`AdvancedRewriter`, as it
    doesn't drop denormal numbers, which may sometimes harm the numerical precision.
    """

    def _pipeline(self, state):
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._create_efuncs(state)
        self._minimize_remainders(state)


class SpeculativeRewriter(AdvancedRewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halo_updates(state)
        self._loop_wrapping(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp'] is True:
            self._parallelize(state)
        self._create_efuncs(state)
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

        return processed, {}


class CustomRewriter(SpeculativeRewriter):

    passes_mapper = {
        'denormals': SpeculativeRewriter._avoid_denormals,
        'wrapping': SpeculativeRewriter._loop_wrapping,
        'blocking': SpeculativeRewriter._loop_blocking,
        'openmp': SpeculativeRewriter._parallelize,
        'simd': SpeculativeRewriter._simdize,
        'split': SpeculativeRewriter._create_efuncs
    }

    def __init__(self, nodes, passes, params):
        try:
            passes = passes.split(',')
            if 'openmp' not in passes and params['openmp']:
                passes.append('openmp')
        except AttributeError:
            # Already in tuple format
            if not all(i in CustomRewriter.passes_mapper for i in passes):
                raise DLEException
        self.passes = passes
        super(CustomRewriter, self).__init__(nodes, params)

    def _pipeline(self, state):
        for i in self.passes:
            CustomRewriter.passes_mapper[i](self, state)
