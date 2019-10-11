import abc
from collections import OrderedDict
from functools import partial
from time import time

import cgen

from devito.dle.blocking_utils import Blocker, BlockDimension
from devito.dle.parallelizer import Ompizer
from devito.exceptions import DLEException
from devito.ir.iet import (Call, Iteration, List, HaloSpot, Prodder, PARALLEL,
                           FindSymbols, FindNodes, FindAdjacent, MapNodes, Transformer,
                           filter_iterations, retrieve_iteration_tree)
from devito.logger import perf_adv, dle_warning as warning
from devito.mpi import HaloExchangeBuilder, HaloScheme
from devito.parameters import configuration
from devito.tools import DAG, as_tuple, filter_ordered, generator

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

    def __init__(self, params, platform):
        self.params = params
        self.platform = platform

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

    _node_parallelizer_type = None
    """The local-node IET parallelizer. To be specified by subclasses."""

    _default_blocking_levels = 1
    """
    Depth of the loop blocking hierarchy. 1 => "blocks", 2 => "blocks" and "sub-blocks",
    3 => "blocks", "sub-blocks", and "sub-sub-blocks", ...
    """

    def __init__(self, params, platform):
        super(PlatformRewriter, self).__init__(params, platform)

        # Iteration blocker (i.e., for "loop blocking")
        self._node_blocker = Blocker(
            params.get('blockinner'),
            params.get('blockalways'),
            params.get('blocklevels') or self._default_blocking_levels
        )

        # Shared-memory parallelizer
        self._node_parallelizer = self._node_parallelizer_type()

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
        Emit a performance message if WRAPPABLE Iterations are found,
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

        * Remove all ``useless`` HaloSpots;
        * Merge all ``hoistable`` HaloSpots with their root HaloSpot, thus
          removing redundant communications and anticipating communications
          that will be required by later Iterations.
        """
        # Drop `useless` HaloSpots
        mapper = {hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(hs.useless))
                  for hs in FindNodes(HaloSpot).visit(iet)}
        iet = Transformer(mapper, nested=True).visit(iet)

        # Handle `hoistable` HaloSpots
        # First, we merge `hoistable` HaloSpots together, to anticipate communications
        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            halo_spots = FindNodes(HaloSpot).visit(tree.root)
            if not halo_spots:
                continue
            root = halo_spots[0]
            if root in mapper:
                continue
            hss = [root.halo_scheme]
            hss.extend([hs.halo_scheme.project(hs.hoistable) for hs in halo_spots[1:]])
            try:
                mapper[root] = root._rebuild(halo_scheme=HaloScheme.union(hss))
            except ValueError:
                # HaloSpots have non-matching `loc_indices` and therefore can't be merged
                warning("Found hoistable HaloSpots with disjoint loc_indices, "
                        "skipping optimization")
                continue
            for hs in halo_spots[1:]:
                halo_scheme = hs.halo_scheme.drop(hs.hoistable)
                if halo_scheme.is_void:
                    mapper[hs] = hs.body
                else:
                    mapper[hs] = hs._rebuild(halo_scheme=halo_scheme)
        iet = Transformer(mapper, nested=True).visit(iet)

        # Then, we make sure the halo exchanges get performed *before*
        # the first distributed Dimension. Again, we do this to anticipate
        # communications, which hopefully has a pay off in performance
        #
        # <Iteration x>                    <HaloSpot(u)>, in y
        #   <HaloSpot(u)>, in y    ---->   <Iteration x>
        #   <Iteration y>                    <Iteration y>
        mapper = {}
        for i, halo_spots in MapNodes(Iteration, HaloSpot).visit(iet).items():
            hoistable = [hs for hs in halo_spots if hs.hoistable]
            if not hoistable:
                continue
            elif len(hoistable) > 1:
                # We should never end up here, but for now we can't prove it formally
                warning("Found multiple hoistable HaloSpots, skipping optimization")
                continue
            hs = hoistable.pop()
            if hs in mapper:
                continue
            if i.dim.root in hs.dimensions:
                halo_scheme = hs.halo_scheme.drop(hs.hoistable)
                if halo_scheme.is_void:
                    mapper[hs] = hs.body
                else:
                    mapper[hs] = hs._rebuild(halo_scheme=halo_scheme)

                halo_scheme = hs.halo_scheme.project(hs.hoistable)
                mapper[i] = hs._rebuild(halo_scheme=halo_scheme, body=i._rebuild())
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
        Apply hierarchical loop blocking to PARALLEL Iteration trees.
        """
        return self._node_blocker.make_blocking(iet)

    @dle_pass
    def _dist_parallelize(self, iet):
        """
        Add MPI routines performing halo exchanges to emit distributed-memory
        parallel code.
        """
        # To produce unique object names
        generators = {'msg': generator(), 'comm': generator(), 'comp': generator()}
        sync_heb = HaloExchangeBuilder('basic', **generators)
        user_heb = HaloExchangeBuilder(self.params['mpi'], **generators)
        mapper = {}
        for hs in FindNodes(HaloSpot).visit(iet):
            heb = user_heb if hs.is_Overlappable else sync_heb
            mapper[hs] = heb.make(hs)
        efuncs = sync_heb.efuncs + user_heb.efuncs
        objs = sync_heb.objs + user_heb.objs
        iet = Transformer(mapper, nested=True).visit(iet)

        # Must drop the PARALLEL tag from the Iterations within which halo
        # exchanges are performed
        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            for i in reversed(tree):
                if i in mapper:
                    # Already seen this subtree, skip
                    break
                if FindNodes(Call).visit(i):
                    mapper.update({n: n._rebuild(properties=set(n.properties)-{PARALLEL})
                                   for n in tree[:tree.index(i)+1]})
                    break
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
    def _minimize_remainders(self, iet):
        """
        Adjust ROUNDABLE Iteration bounds so as to avoid the insertion of remainder
        loops by the backend compiler.
        """
        roundable = [i for i in FindNodes(Iteration).visit(iet) if i.is_Roundable]

        mapper = {}
        for i in roundable:
            functions = FindSymbols().visit(i)

            # Get the SIMD vector length
            dtypes = {f.dtype for f in functions if f.is_Tensor}
            assert len(dtypes) == 1
            vl = configuration['platform'].simd_items_per_reg(dtypes.pop())

            # Round up `i`'s max point so that at runtime only vector iterations
            # will be performed (i.e., remainder loops won't be necessary)
            m, M, step = i.limits
            limits = (m, M + (i.symbolic_size % vl), step)

            mapper[i] = i._rebuild(limits=limits)

        iet = Transformer(mapper).visit(iet)

        return iet, {}

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
                    mapper[candidate] = candidate._rebuild(nodes=(candidate.nodes +
                                                                  (prodder._rebuild(),)))
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
        self._minimize_remainders(state)
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
