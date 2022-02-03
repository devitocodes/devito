import numpy as np
import cgen as c
from sympy import And, Max

from devito.data import FULL
from devito.ir import (Conditional, DummyEq, Dereference, Expression, ExpressionBundle,
                       FindSymbols, FindNodes, ParallelTree, Prodder, List, Transformer,
                       IsPerfectIteration, filter_iterations, retrieve_iteration_tree,
                       VECTORIZED)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB, LangTransformer, DeviceAwareMixin
from devito.passes.iet.misc import is_on_device
from devito.symbolics import INT, ccode
from devito.tools import as_tuple, prod
from devito.types import Symbol, NThreadsBase, Wildcard

__all__ = ['PragmaSimdTransformer', 'PragmaShmTransformer',
           'PragmaDeviceAwareTransformer', 'PragmaLangBB']


class PragmaTransformer(LangTransformer):

    """
    Abstract base class for LangTransformers that parallelize Iterations
    as well as manage data allocation with pragmas.
    """

    pass


class PragmaSimdTransformer(PragmaTransformer):

    """
    Abstract base class for PragmaTransformers capable of emitting SIMD-parallel IETs.
    """

    @property
    def simd_reg_size(self):
        return self.platform.simd_reg_size

    @iet_pass
    def make_simd(self, iet):
        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            candidates = [i for i in tree if i.is_ParallelRelaxed]

            # As long as there's an outer level of parallelism, the innermost
            # PARALLEL Iteration gets vectorized
            if len(candidates) < 2:
                continue
            candidate = candidates[-1]

            # Only fully-parallel Iterations will be SIMD-ized (ParallelRelaxed
            # might not be enough then)
            if not candidate.is_Parallel:
                continue

            # Add SIMD pragma
            indexeds = FindSymbols('indexeds').visit(candidate)
            aligned = {i.name for i in indexeds if i.function.is_DiscreteFunction}
            if aligned:
                simd = self.lang['simd-for-aligned']
                simd = as_tuple(simd(','.join(sorted(aligned)), self.simd_reg_size))
            else:
                simd = as_tuple(self.lang['simd-for'])
            pragmas = candidate.pragmas + simd

            # Add VECTORIZED property
            properties = list(candidate.properties) + [VECTORIZED]

            mapper[candidate] = candidate._rebuild(pragmas=pragmas, properties=properties)

        iet = Transformer(mapper).visit(iet)

        return iet, {}


class PragmaShmTransformer(PragmaSimdTransformer):

    """
    Abstract base class for PragmaTransformers capable of emitting SIMD-parallel
    and shared-memory-parallel IETs.
    """

    def __init__(self, sregistry, options, platform):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to access the symbols appearing in an IET.
        options : dict
             The optimization options. Accepted: ['par-collapse-ncores',
             'par-collapse-work', 'par-chunk-nonaffine', 'par-dynamic-work', 'par-nested']
             * 'par-collapse-ncores': use a collapse clause if the number of
               available physical cores is greater than this threshold.
             * 'par-collapse-work': use a collapse clause if the trip count of the
               collapsable Iterations is statically known to exceed this threshold.
             * 'par-chunk-nonaffine': coefficient to adjust the chunk size in
               non-affine parallel Iterations.
             * 'par-dynamic-work': use dynamic scheduling if the operation count per
               iteration exceeds this threshold. Otherwise, use static scheduling.
             * 'par-nested': nested parallelism if the number of hyperthreads per core
               is greater than this threshold.
        platform : Platform
            The underlying platform.
        """
        key = lambda i: i.is_ParallelRelaxed and not i.is_Vectorized
        super().__init__(key, sregistry, platform)

        self.collapse_ncores = options['par-collapse-ncores']
        self.collapse_work = options['par-collapse-work']
        self.chunk_nonaffine = options['par-chunk-nonaffine']
        self.dynamic_work = options['par-dynamic-work']
        self.nested = options['par-nested']

    @property
    def ncores(self):
        return self.platform.cores_physical

    @property
    def nhyperthreads(self):
        return self.platform.threads_per_core

    @property
    def nthreads(self):
        return self.sregistry.nthreads

    @property
    def nthreads_nested(self):
        return self.sregistry.nthreads_nested

    @property
    def nthreads_nonaffine(self):
        return self.sregistry.nthreads_nonaffine

    @property
    def threadid(self):
        return self.sregistry.threadid

    def _select_candidates(self, candidates):
        assert candidates

        if self.ncores < self.collapse_ncores:
            return candidates[0], []

        mapper = {}
        for n0, root in enumerate(candidates):

            collapsable = []
            for n, i in enumerate(candidates[n0+1:], n0+1):
                # The Iteration nest [root, ..., i] must be perfect
                if not IsPerfectIteration(depth=i).visit(root):
                    break

                # Loops are collapsable only if none of the iteration variables appear
                # in initializer expressions. For example, the following two loops
                # cannot be collapsed
                #
                # for (i = ... )
                #   for (j = i ...)
                #     ...
                #
                # Here, we make sure this won't happen
                if any(j.dim in i.symbolic_min.free_symbols for j in candidates[n0:n]):
                    break

                # Also, we do not want to collapse SIMD-vectorized Iterations
                if i.is_Vectorized:
                    break

                # Would there be enough work per parallel iteration?
                nested = candidates[n+1:]
                if nested:
                    try:
                        work = prod([int(j.dim.symbolic_size) for j in nested])
                        if work < self.collapse_work:
                            break
                    except TypeError:
                        pass

                collapsable.append(i)

            # Give a score to this candidate, based on the number of fully-parallel
            # Iterations and their position (i.e. outermost to innermost) in the nest
            score = (
                int(root.is_ParallelNoAtomic),
                int(len([i for i in collapsable if i.is_ParallelNoAtomic]) >= 1),
                int(len([i for i in collapsable if i.is_ParallelRelaxed]) >= 1),
                -(n0 + 1)  # The outermost, the better
            )

            mapper[(root, tuple(collapsable))] = score

        # Retrieve the candidates with highest score
        root, collapsable = max(mapper, key=mapper.get)

        return root, list(collapsable)

    def _make_reductions(self, partree):
        if not any(i.is_ParallelAtomic for i in partree.collapsed):
            return partree

        exprs = [i for i in FindNodes(Expression).visit(partree) if i.is_Increment]
        reduction = [i.output for i in exprs]
        if all(i.is_Affine for i in partree.collapsed) or \
           all(not i.is_Indexed for i in reduction):
            # Implement reduction
            mapper = {partree.root: partree.root._rebuild(reduction=reduction)}
        else:
            # Make sure the increment is atomic
            mapper = {i: i._rebuild(pragmas=self.lang['atomic']) for i in exprs}

        partree = Transformer(mapper).visit(partree)

        return partree

    def _make_threaded_prodders(self, partree):
        mapper = {i: self.Prodder(i) for i in FindNodes(Prodder).visit(partree)}
        partree = Transformer(mapper).visit(partree)
        return partree

    def _make_partree(self, candidates, nthreads=None):
        assert candidates

        # Get the collapsable Iterations
        root, collapsable = self._select_candidates(candidates)
        ncollapse = 1 + len(collapsable)

        # Prepare to build a ParallelTree
        if all(i.is_Affine for i in candidates):
            bundles = FindNodes(ExpressionBundle).visit(root)
            sops = sum(i.ops for i in bundles)
            if sops >= self.dynamic_work:
                schedule = 'dynamic'
            else:
                schedule = 'static'
            if nthreads is None:
                # pragma ... for ... schedule(..., 1)
                nthreads = self.nthreads
                body = self.HostIteration(schedule=schedule, ncollapse=ncollapse,
                                          **root.args)
            else:
                # pragma ... parallel for ... schedule(..., 1)
                body = self.HostIteration(schedule=schedule, parallel=True,
                                          ncollapse=ncollapse, nthreads=nthreads,
                                          **root.args)
            prefix = []
        else:
            # pragma ... for ... schedule(..., expr)
            assert nthreads is None
            nthreads = self.nthreads_nonaffine
            chunk_size = Symbol(name='chunk_size')
            body = self.HostIteration(ncollapse=ncollapse, chunk_size=chunk_size,
                                      **root.args)

            niters = prod([root.symbolic_size] + [j.symbolic_size for j in collapsable])
            value = INT(Max(niters / (nthreads*self.chunk_nonaffine), 1))
            prefix = [Expression(DummyEq(chunk_size, value, dtype=np.int32))]

        # Create a ParallelTree
        partree = ParallelTree(prefix, body, nthreads=nthreads)

        return root, partree

    def _make_parregion(self, partree, parrays):
        if not any(i.is_ParallelPrivate for i in partree.collapsed):
            return self.Region(partree)

        # Vector-expand all written Arrays within `partree`, since at least
        # one of the parallelized Iterations requires thread-private Arrays
        # E.g. a(x, y) -> b(tid, x, y), where `tid` is the ThreadID Dimension
        vexpandeds = []
        for n in FindNodes(Expression).visit(partree):
            i = n.write
            if not (i.is_Array or i.is_TempFunction):
                continue
            elif i in parrays:
                pi = parrays[i]
            else:
                pi = parrays.setdefault(i, i._make_pointer(dim=self.threadid))
            vexpandeds.append(VExpanded(i, pi))

        if vexpandeds:
            init = c.Initializer(c.Value(self.threadid._C_typedata, self.threadid.name),
                                 self.lang['thread-num'])
            prefix = List(header=init,
                          body=vexpandeds + list(partree.prefix),
                          footer=c.Line())
            partree = partree._rebuild(prefix=prefix)

        return self.Region(partree)

    def _make_guard(self, parregion):
        return parregion

    def _make_nested_partree(self, partree):
        # Apply heuristic
        if self.nhyperthreads <= self.nested:
            return partree

        # Note: there might be multiple sub-trees amenable to nested parallelism,
        # hence we loop over all of them
        #
        # for (i = ... )  // outer parallelism
        #   for (j0 = ...)  // first source of nested parallelism
        #     ...
        #   for (j1 = ...)  // second source of nested parallelism
        #     ...
        mapper = {}
        for tree in retrieve_iteration_tree(partree):
            outer = tree[:partree.ncollapsed]
            inner = tree[partree.ncollapsed:]

            # Heuristic: nested parallelism is applied only if the top nested
            # parallel Iteration iterates *within* the top outer parallel Iteration
            # (i.e., the outer is a loop over blocks, while the nested is a loop
            # within a block)
            candidates = []
            for i in inner:
                if self.key(i) and any((j.dim.root is i.dim.root) for j in outer):
                    candidates.append(i)
                elif candidates:
                    # If there's at least one candidate but `i` doesn't honor the
                    # heuristic above, then we break, as the candidates must be
                    # perfectly nested
                    break
            if not candidates:
                continue

            # Introduce nested parallelism
            subroot, subpartree = self._make_partree(candidates, self.nthreads_nested)

            mapper[subroot] = subpartree

        partree = Transformer(mapper).visit(partree)

        return partree

    def _make_parallel(self, iet):
        mapper = {}
        parrays = {}
        for tree in retrieve_iteration_tree(iet, mode='superset'):
            # Get the parallelizable Iterations in `tree`
            candidates = filter_iterations(tree, key=self.key)
            if not candidates:
                continue

            # Outer parallelism
            root, partree = self._make_partree(candidates)
            if partree is None or root in mapper:
                continue

            # Nested parallelism
            partree = self._make_nested_partree(partree)

            # Handle reductions
            partree = self._make_reductions(partree)

            # Atomicize and optimize single-thread prodders
            partree = self._make_threaded_prodders(partree)

            # Wrap within a parallel region
            parregion = self._make_parregion(partree, parrays)

            # Protect the parallel region if necessary
            parregion = self._make_guard(parregion)

            mapper[root] = parregion

        iet = Transformer(mapper).visit(iet)

        # The new arguments introduced by this pass
        args = [i for i in FindSymbols().visit(iet) if isinstance(i, (NThreadsBase))]
        for n in FindNodes(VExpanded).visit(iet):
            args.extend([(n.pointee, True), n.pointer])

        return iet, {'args': args, 'includes': [self.lang['header']]}

    @iet_pass
    def make_parallel(self, iet):
        return self._make_parallel(iet)


class PragmaDeviceAwareTransformer(DeviceAwareMixin, PragmaShmTransformer):

    """
    Abstract base class for PragmaTransformers capable of emitting SIMD-parallel,
    shared-memory-parallel, and device-parallel IETs.
    """

    def __init__(self, sregistry, options, platform):
        super().__init__(sregistry, options, platform)

        self.gpu_fit = options['gpu-fit']
        self.par_tile = options['par-tile']
        self.par_disabled = options['par-disabled']

    def _is_offloadable(self, iet):
        expressions = FindNodes(Expression).visit(iet)
        if any(not is_on_device(e.write, self.gpu_fit) for e in expressions):
            return False

        functions = FindSymbols().visit(iet)
        buffers = [f for f in functions if f.is_Array and f._mem_mapped]
        hostfuncs = [f for f in functions if not is_on_device(f, self.gpu_fit)]
        return not (buffers and hostfuncs)

    def _make_threaded_prodders(self, partree):
        if isinstance(partree.root, self.DeviceIteration):
            # no-op for now
            return partree
        else:
            return super()._make_threaded_prodders(partree)

    def _make_partree(self, candidates, nthreads=None):
        """
        Parallelize the `candidates` Iterations. In particular:

            * A PARALLEL Iteration writing (reading) a mapped Array while
              reading (writing) a host Function (that is, all Functions `f`
              such that `is_on_device(f)` gives False) is parallelized
              on the host. These are essentially the Iterations that initialize
              or dump the Devito-created buffers.
            * All other PARALLEL Iterations (typically, the majority) are
              offloaded to the device.
        """
        assert candidates

        root, collapsable = self._select_candidates(candidates)

        if self._is_offloadable(root):
            body = self.DeviceIteration(gpu_fit=self.gpu_fit,
                                        ncollapse=len(collapsable) + 1,
                                        **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)

            return root, partree
        elif not self.par_disabled:
            # Resort to host parallelism
            return super()._make_partree(candidates, nthreads)
        else:
            return root, None

    def _make_parregion(self, partree, *args):
        if isinstance(partree.root, self.DeviceIteration):
            # no-op for now
            return partree
        else:
            return super()._make_parregion(partree, *args)

    def _make_guard(self, parregion, *args):
        partrees = FindNodes(ParallelTree).visit(parregion)
        if not any(isinstance(i.root, self.DeviceIteration) for i in partrees):
            return super()._make_guard(parregion, *args)

        cond = []
        # There must be at least one iteration or potential crash
        if not parregion.is_Affine:
            trees = retrieve_iteration_tree(parregion.root)
            tree = trees[0][:parregion.ncollapsed]
            cond.extend([i.symbolic_size > 0 for i in tree])

        # SparseFunctions may occasionally degenerate to zero-size arrays. In such
        # a case, a copy-in produces a `nil` pointer on the device. To fire up a
        # parallel loop we must ensure none of the SparseFunction pointers are `nil`
        symbols = FindSymbols().visit(parregion)
        sfs = [i for i in symbols if i.is_SparseFunction]
        if sfs:
            size = [prod(f._C_get_field(FULL, d).size for d in f.dimensions) for f in sfs]
            cond.extend([i > 0 for i in size])

        # Combine all cond elements
        if cond:
            parregion = List(body=[Conditional(And(*cond), parregion)])

        return parregion

    def _make_nested_partree(self, partree):
        if isinstance(partree.root, self.DeviceIteration):
            # no-op for now
            return partree
        else:
            return super()._make_nested_partree(partree)


class PragmaLangBB(LangBB):

    @classmethod
    def _make_symbolic_sections_from_imask(cls, f, imask):
        datasize = cls._map_data(f)
        if imask is None:
            imask = [FULL]*len(datasize)

        sections = []
        for i, j in zip(imask, datasize):
            if i is FULL:
                start, size = 0, j
            else:
                try:
                    start, size = i
                except TypeError:
                    start, size = i, 1
            sections.append((start, size))

        # Unroll (or "flatten") the remaining Dimensions not captured by `imask`
        if len(imask) < len(datasize):
            try:
                start, size = sections.pop(-1)
            except IndexError:
                start, size = (0, 1)
            remainder_size = prod(datasize[len(imask):])
            # The reason we may see a Wildcard is detailed in the `linearize_transfer`
            # pass, take a look there for more info. Basically, a Wildcard here means
            # that the symbol `start` is actually a temporary whose value already
            # represents the unrolled size
            if not isinstance(start, Wildcard):
                start *= remainder_size
            size *= remainder_size
            sections.append((start, size))

        return sections

    @classmethod
    def _make_sections_from_imask(cls, f, imask):
        sections = cls._make_symbolic_sections_from_imask(f, imask)

        sections = ['[%s:%s]' % (ccode(start), ccode(size)) for start, size in sections]
        sections = ''.join(sections)

        return sections

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _map_to(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-enter-to'](f.name, sections)

    _map_to_wait = _map_to

    @classmethod
    def _map_alloc(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-enter-alloc'](f.name, sections)

    @classmethod
    def _map_present(cls, f, imask=None):
        return

    @classmethod
    def _map_wait(cls, queueid=None):
        try:
            return cls.mapper['map-wait'](queueid)
        except KeyError:
            # Not all languages may provide an explicit wait construct
            return None

    @classmethod
    def _map_update(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update'](f.name, sections)

    @classmethod
    def _map_update_host(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-host'](f.name, sections)

    _map_update_host_async = _map_update_host

    @classmethod
    def _map_update_device(cls, f, imask=None, queueid=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-update-device'](f.name, sections)

    _map_update_device_async = _map_update_device

    @classmethod
    def _map_release(cls, f, imask=None, devicerm=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.mapper['map-release'](f.name, sections,
                                         (' if(%s)' % devicerm.name) if devicerm else '')

    @classmethod
    def _map_delete(cls, f, imask=None, devicerm=None):
        sections = cls._make_sections_from_imask(f, imask)
        # This ugly condition is to avoid a copy-back when, due to
        # domain decomposition, the local size of a Function is 0, which
        # would cause a crash
        items = []
        if devicerm is not None:
            items.append(devicerm.name)
        items.extend(['(%s != 0)' % i for i in cls._map_data(f)])
        cond = ' if(%s)' % ' && '.join(items)
        return cls.mapper['map-exit-delete'](f.name, sections, cond)


# Utils

class VExpanded(Dereference):
    pass
