import abc
from collections import OrderedDict
from itertools import product
from functools import partial
from time import time

import cgen

from devito.cgen_utils import ccode
from devito.dle.blocking_utils import (BlockDimension, fold_blockable_tree,
                                       unfold_blocked_tree)
from devito.dle.parallelizer import Ompizer
from devito.exceptions import DLEException
from devito.ir.iet import (Call, Expression, Iteration, List, HaloSpot, Prodder, PARALLEL,
                           AFFINE, FindSymbols, FindNodes, FindAdjacent, MapNodes,
                           Transformer, compose_nodes, filter_iterations, make_efunc,
                           retrieve_iteration_tree)
from devito.logger import perf_adv
from devito.mpi import HaloExchangeBuilder
from devito.parameters import configuration
from devito.tools import DAG, as_tuple, filter_ordered, flatten

__all__ = ['PlatformRewriter', 'CPU64Rewriter', 'Intel64Rewriter', 'PowerRewriter',
           'ArmRewriter', 'SpeculativeRewriter', 'DeviceOffloadingRewriter',
           'CustomRewriter']


class State(object):

    def __init__(self, iet):
        self._efuncs = OrderedDict([('root', iet)])
        self._dimensions = []
        self._includes = []

        # Track performance of each pass
        self._timings = OrderedDict()

    @property
    def root(self):
        return self._efuncs['root']

    @property
    def efuncs(self):
        return tuple(v for k, v in self._efuncs.items() if k != 'root')

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def includes(self):
        return self._includes

    @property
    def timings(self):
        return self._timings


def process(func, state):
    """
    Apply ``func`` to the IETs in ``state._efuncs``, and update ``state`` accordingly.
    """
    # Create a Call graph. `func` will be applied to each node in the Call graph.
    # `func` might change an `efunc` signature; the Call graph will be used to
    # propagate such change through the `efunc` callers
    dag = DAG(nodes=['root'])
    queue = ['root']
    while queue:
        caller = queue.pop(0)
        callees = FindNodes(Call).visit(state._efuncs[caller])
        for callee in filter_ordered([i.name for i in callees]):
            if callee in state._efuncs:  # Exclude foreign Calls, e.g., MPI calls
                try:
                    dag.add_node(callee)
                    queue.append(callee)
                except KeyError:
                    # `callee` already in `dag`
                    pass
                dag.add_edge(callee, caller)
    assert dag.size == len(state._efuncs)

    # Apply `func`
    for i in dag.topological_sort():
        state._efuncs[i], metadata = func(state._efuncs[i])

        # Track any new Dimensions introduced by `func`
        state._dimensions.extend(list(metadata.get('dimensions', [])))

        # Track any new #include required by `func`
        state._includes.extend(list(metadata.get('includes', [])))
        state._includes = filter_ordered(state._includes)

        # Track any new ElementalFunctions
        state._efuncs.update(OrderedDict([(i.name, i)
                                          for i in metadata.get('efuncs', [])]))

        # If there's a change to the `args` and the `iet` is an efunc, then
        # we must update the call sites as well, as the arguments dropped down
        # to the efunc have just increased
        args = as_tuple(metadata.get('args'))
        if args:
            # `extif` avoids redundant updates to the parameters list, due
            # to multiple children wanting to add the same input argument
            extif = lambda v: list(v) + [e for e in args if e not in v]
            stack = [i] + dag.all_downstreams(i)
            for n in stack:
                efunc = state._efuncs[n]
                calls = [c for c in FindNodes(Call).visit(efunc) if c.name in stack]
                mapper = {c: c._rebuild(arguments=extif(c.arguments)) for c in calls}
                efunc = Transformer(mapper).visit(efunc)
                if efunc.is_Callable:
                    efunc = efunc._rebuild(parameters=extif(efunc.parameters))
                state._efuncs[n] = efunc


def dle_pass(func):
    def wrapper(self, state, **kwargs):
        tic = time()
        process(partial(func, self), state)
        toc = time()
        state.timings[func.__name__] = toc - tic
    return wrapper


class AbstractRewriter(object):
    """
    Transform Iteration/Expression trees (IETs) to generate high performance C.

    This is just an abstract class. Actual transformers should implement the
    abstract method ``_pipeline``, which performs a sequence of IET transformations.
    """

    __metaclass__ = abc.ABCMeta

    _node_parallelizer_type = None
    """The local-node IET parallelizer. To be specified by subclasses."""

    def __init__(self, params, platform):
        self.params = params
        self.platform = platform

        self._node_parallelizer = self._node_parallelizer_type()

    def run(self, iet):
        """The optimization pipeline, as a sequence of AST transformation passes."""
        state = State(iet)

        self._pipeline(state)

        return state

    @abc.abstractmethod
    def _pipeline(self, state):
        return


class PlatformRewriter(AbstractRewriter):

    """No-op rewriter."""

    lang = {}
    """
    Collection of backend-compiler-specific pragmas.
    """

    def _pipeline(self, state):
        return

    def _backend_compiler_pragma(self, name, default=None):
        key = configuration['compiler'].__class__.__name__
        return self.lang.get(key, {}).get(name, default)

    @dle_pass
    def _avoid_denormals(self, iet):
        """
        Introduce nodes in the Iteration/Expression tree that will expand to C
        macros telling the CPU to flush denormal numbers in hardware. Denormals
        are normally flushed when using SSE-based instruction sets, except when
        compiling shared objects.
        """
        return iet, {}

    @dle_pass
    def _loop_wrapping(self, iet):
        """
        Emit a performance warning if WRAPPABLE Iterations are found,
        as these are a symptom that unnecessary memory is being allocated.
        """
        for i in FindNodes(Iteration).visit(iet):
            if not i.is_Wrappable:
                continue
            perf_adv("Functions using modulo iteration along Dimension `%s` "
                     "may safely allocate a one slot smaller buffer" % i.dim)
        return iet, {}

    @dle_pass
    def _optimize_halospots(self, iet):
        """
        Optimize the HaloSpots in ``iet``.

        * Remove all USELESS HaloSpots;
        * Merge all hoistable HaloSpots with their root HaloSpot, thus
          removing redundant communications and anticipating communications
          that will be required by later Iterations.
        """
        # Drop USELESS HaloSpots
        mapper = {hs: hs.body for hs in FindNodes(HaloSpot).visit(iet) if hs.is_Useless}
        iet = Transformer(mapper, nested=True).visit(iet)

        # Handle `hoistable` HaloSpots
        mapper = {}
        for halo_spots in MapNodes(Iteration, HaloSpot).visit(iet).values():
            root = halo_spots[0]
            halo_schemes = [hs.halo_scheme.project(hs.hoistable) for hs in halo_spots[1:]]
            mapper[root] = root._rebuild(halo_scheme=root.halo_scheme.union(halo_schemes))
            mapper.update({hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(hs.hoistable))
                           for hs in halo_spots[1:]})
        iet = Transformer(mapper, nested=True).visit(iet)

        # At this point, some HaloSpots may have become empty (i.e., requiring
        # no communications), hence they can be removed
        #
        # <HaloSpot(u,v)>           HaloSpot(u,v)
        #   <A>                       <A>
        # <HaloSpot()>      ---->   <B>
        #   <B>
        mapper = {i: i.body for i in FindNodes(HaloSpot).visit(iet) if i.is_empty}
        iet = Transformer(mapper, nested=True).visit(iet)

        # Finally, we try to move HaloSpot-free Iteration nests within HaloSpot
        # subtrees, to overlap as much computation as possible. The HaloSpot-free
        # Iteration nests must be fully affine, otherwise we wouldn't be able to
        # honour the data dependences along the halo
        #
        # <HaloSpot(u,v)>            HaloSpot(u,v)
        #   <A>             ---->      <A>
        # <B>              affine?     <B>
        #
        # Here, <B> doesn't require any halo exchange, but it might still need the
        # output of <A>; thus, if we do computation/communication overlap over <A>
        # *and* want to embed <B> within the HaloSpot, then <B>'s iteration space
        # will have to be split as well. For this, <B> must be affine.
        mapper = {}
        for v in FindAdjacent((HaloSpot, Iteration)).visit(iet).values():
            for g in v:
                root = None
                for i in g:
                    if i.is_HaloSpot:
                        root = i
                        mapper[root] = [root.body]
                    elif root and all(j.is_Affine for j in FindNodes(Iteration).visit(i)):
                        mapper[root].append(i)
                        mapper[i] = None
                    else:
                        root = None
        mapper = {k: k._rebuild(body=List(body=v)) if v else v for k, v in mapper.items()}
        iet = Transformer(mapper).visit(iet)

        return iet, {}

    @dle_pass
    def _loop_blocking(self, iet):
        """
        Apply loop blocking to PARALLEL Iteration trees.
        """
        blockinner = bool(self.params.get('blockinner'))
        blockalways = bool(self.params.get('blockalways'))

        # Make sure loop blocking will span as many Iterations as possible
        iet = fold_blockable_tree(iet, blockinner)

        mapper = {}
        efuncs = []
        block_dims = []
        for tree in retrieve_iteration_tree(iet):
            # Is the Iteration tree blockable ?
            iterations = filter_iterations(tree, lambda i: i.is_Parallel)
            if not blockinner:
                iterations = iterations[:-1]
            if len(iterations) <= 1:
                continue
            root = iterations[0]
            if not (tree.root.is_Sequential or iet.is_Callable) and not blockalways:
                # Heuristic: avoid polluting the generated code with blocked
                # nests (thus increasing JIT compilation time and affecting
                # readability) if the blockable tree isn't embedded in a
                # sequential loop (e.g., a timestepping loop)
                continue

            # Apply loop blocking to `tree`
            interb = []
            intrab = []
            for i in iterations:
                d = BlockDimension(i.dim, name="%s%d_blk" % (i.dim.name, len(mapper)))
                block_dims.append(d)
                # Build Iteration over blocks
                properties = (PARALLEL,) + ((AFFINE,) if i.is_Affine else ())
                interb.append(Iteration([], d, d.symbolic_max, properties=properties))
                # Build Iteration within a block
                intrab.append(i._rebuild([], limits=(d, d+d.step-1, 1), offsets=(0, 0)))

            # Construct the blocked tree
            blocked = compose_nodes(interb + intrab + [iterations[-1].nodes])
            blocked = unfold_blocked_tree(blocked)

            # Promote to a separate Callable
            dynamic_parameters = flatten((bi.dim, bi.dim.symbolic_size) for bi in interb)
            efunc = make_efunc("bf%d" % len(mapper), blocked, dynamic_parameters)
            efuncs.append(efunc)

            # Compute the iteration ranges
            ranges = []
            for i, bi in zip(iterations, interb):
                maxb = i.symbolic_max - (i.symbolic_size % bi.dim.step)
                ranges.append(((i.symbolic_min, maxb, bi.dim.step),
                               (maxb + 1, i.symbolic_max, i.symbolic_max - maxb)))

            # Build Calls to the `efunc`
            body = []
            for p in product(*ranges):
                dynamic_args_mapper = {}
                for bi, (m, M, b) in zip(interb, p):
                    dynamic_args_mapper[bi.dim] = (m, M)
                    dynamic_args_mapper[bi.dim.step] = (b,)
                call = efunc.make_call(dynamic_args_mapper)
                body.append(List(body=call))

            mapper[root] = List(body=body)

        iet = Transformer(mapper).visit(iet)

        return iet, {'dimensions': block_dims, 'efuncs': efuncs,
                     'args': [i.step for i in block_dims]}

    @dle_pass
    def _dist_parallelize(self, iet):
        """
        Add MPI routines performing halo exchanges to emit distributed-memory
        parallel code.
        """
        sync_heb = HaloExchangeBuilder('basic')
        user_heb = HaloExchangeBuilder(self.params['mpi'])
        mapper = {}
        for i, hs in enumerate(FindNodes(HaloSpot).visit(iet)):
            heb = user_heb if hs.is_Overlappable else sync_heb
            mapper[hs] = heb.make(hs, i)
        efuncs = sync_heb.efuncs + user_heb.efuncs
        objs = sync_heb.objs + user_heb.objs
        iet = Transformer(mapper, nested=True).visit(iet)

        return iet, {'includes': ['mpi.h'], 'efuncs': efuncs, 'args': objs}

    @dle_pass
    def _simdize(self, iet):
        """
        Add pragmas to the Iteration/Expression tree to enforce SIMD auto-vectorization
        by the backend compiler.
        """
        ignore_deps = as_tuple(self._backend_compiler_pragma('ignore-deps'))

        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            for i in vector_iterations:
                aligned = [j for j in FindSymbols('symbolics').visit(i)
                           if j.is_DiscreteFunction]
                if aligned:
                    simd = Ompizer.lang['simd-for-aligned']
                    simd = as_tuple(simd(','.join([j.name for j in aligned]),
                                    self.platform.simd_reg_size))
                else:
                    simd = as_tuple(Ompizer.lang['simd-for'])
                mapper[i] = i._rebuild(pragmas=i.pragmas + ignore_deps + simd)

        processed = Transformer(mapper).visit(iet)

        return processed, {}

    @dle_pass
    def _node_parallelize(self, iet):
        """
        Add OpenMP pragmas to the Iteration/Expression tree to emit shared-memory
        parallel code.
        """
        return self._node_parallelizer.make_parallel(iet)

    @dle_pass
    def _hoist_prodders(self, iet):
        """
        Move Prodders within the outer levels of an Iteration tree.
        """
        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            for prodder in FindNodes(Prodder).visit(tree.root):
                if prodder._periodic:
                    try:
                        key = lambda i: isinstance(i.dim, BlockDimension)
                        candidate = filter_iterations(tree, key)[-1]
                    except IndexError:
                        # Fallback: use the outermost Iteration
                        candidate = tree.root
                    mapper[candidate] = candidate._rebuild(nodes=((prodder._rebuild(),) +
                                                                  candidate.nodes))
                    mapper[prodder] = None

        iet = Transformer(mapper, nested=True).visit(iet)

        return iet, {}


class CPU64Rewriter(PlatformRewriter):

    _node_parallelizer_type = Ompizer

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halospots(state)
        if self.params['mpi']:
            self._dist_parallelize(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp']:
            self._node_parallelize(state)
        self._hoist_prodders(state)


class Intel64Rewriter(CPU64Rewriter):

    lang_intel_common = {
        'ignore-deps': cgen.Pragma('ivdep'),
        'ntstores': cgen.Pragma('vector nontemporal'),
        'storefence': cgen.Statement('_mm_sfence()'),
        'noinline': cgen.Pragma('noinline')
    }
    lang = {
        'IntelCompiler': lang_intel_common,
        'IntelKNLCompiler': lang_intel_common
    }
    """
    Collection of backend-compiler-specific pragmas.
    """

    @dle_pass
    def _avoid_denormals(self, iet):
        header = [cgen.Comment('Flush denormal numbers to zero in hardware'),
                  cgen.Statement('_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)'),
                  cgen.Statement('_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)')]
        iet = List(header=header, body=iet)
        return iet, {'includes': ('xmmintrin.h', 'pmmintrin.h')}


PowerRewriter = CPU64Rewriter
ArmRewriter = CPU64Rewriter


class DeviceOffloadingRewriter(PlatformRewriter):

    _node_parallelizer_type = Ompizer

    def _pipeline(self, state):
        self._optimize_halospots(state)
        if self.params['mpi']:
            self._dist_parallelize(state)
        self._simdize(state)
        self._node_parallelize(state)
        self._hoist_prodders(state)

    @dle_pass
    def _node_parallelize(self, iet):
        """
        Add OpenMP pragmas to offload PARALLEL Iteration nests onto a device.
        """
        # TODO: this is still to be implemented -- something other than
        # `.make_parallel` will have to be used, e.g., `.make_offloadable`
        return self._node_parallelizer.make_parallel(iet)


class SpeculativeRewriter(CPU64Rewriter):

    def _pipeline(self, state):
        self._avoid_denormals(state)
        self._optimize_halospots(state)
        self._loop_wrapping(state)
        if self.params['mpi']:
            self._dist_parallelize(state)
        self._loop_blocking(state)
        self._simdize(state)
        if self.params['openmp']:
            self._node_parallelize(state)
        self._minimize_remainders(state)
        self._hoist_prodders(state)

    @dle_pass
    def _nontemporal_stores(self, nodes):
        """
        Add compiler-specific pragmas and instructions to generate nontemporal
        stores (ie, non-cached stores).
        """
        pragma = self._backend_compiler_pragma('ntstores')
        fence = self._backend_compiler_pragma('storefence')
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

    @dle_pass
    def _minimize_remainders(self, iet):
        """
        Reshape temporary tensors and adjust loop trip counts to prevent as many
        compiler-generated remainder loops as possible.
        """
        # The innermost dimension is the one that might get padded
        p_dim = -1

        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            vector_iterations = [i for i in tree if i.is_Vectorizable]
            if not vector_iterations or len(vector_iterations) > 1:
                continue
            root = vector_iterations[0]

            # Padding
            writes = [i.write for i in FindNodes(Expression).visit(root)
                      if i.write.is_Array]
            padding = []
            for i in writes:
                try:
                    simd_items = self.platform.simd_items_per_reg(i.dtype)
                except KeyError:
                    return iet, {}
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
            endpoint = root.symbolic_max
            if not endpoint.is_Symbol:
                continue
            condition = []
            externals = set(i.symbolic_shape[-1] for i in FindSymbols().visit(root)
                            if i.is_Tensor)
            for i in root.uindices:
                for j in externals:
                    condition.append(root.symbolic_max + padding < j)
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

        processed = Transformer(mapper).visit(iet)

        return processed, {}


class CustomRewriter(SpeculativeRewriter):

    passes_mapper = {
        'denormals': SpeculativeRewriter._avoid_denormals,
        'optcomms': SpeculativeRewriter._optimize_halospots,
        'wrapping': SpeculativeRewriter._loop_wrapping,
        'blocking': SpeculativeRewriter._loop_blocking,
        'openmp': SpeculativeRewriter._node_parallelize,
        'mpi': SpeculativeRewriter._dist_parallelize,
        'simd': SpeculativeRewriter._simdize,
        'minrem': SpeculativeRewriter._minimize_remainders,
        'prodders': SpeculativeRewriter._hoist_prodders
    }

    def __init__(self, passes, params, platform):
        try:
            passes = passes.split(',')
            if 'openmp' not in passes and params['openmp']:
                passes.append('openmp')
        except AttributeError:
            # Already in tuple format
            if not all(i in CustomRewriter.passes_mapper for i in passes):
                raise DLEException("Unknown passes `%s`" % str(passes))
        self.passes = passes
        super(CustomRewriter, self).__init__(params, platform)

    def _pipeline(self, state):
        for i in self.passes:
            CustomRewriter.passes_mapper[i](self, state)
