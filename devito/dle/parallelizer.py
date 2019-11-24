from collections import OrderedDict
import os

import numpy as np
import cgen as c
from sympy import Function, Or, Max, Not

from devito.data import FULL
from devito.ir import (DummyEq, Conditional, Block, Expression, List, Prodder,
                       While, FindSymbols, FindNodes, MapNodes, Return, COLLAPSED,
                       Transformer, IsPerfectIteration, retrieve_iteration_tree,
                       filter_iterations)
from devito.symbolics import CondEq, INT
from devito.parameters import configuration
from devito.tools import as_tuple, is_integer, flatten, prod
from devito.types import Constant, Symbol


def ncores():
    return configuration['platform'].cores_physical


def nhyperthreads():
    return configuration['platform'].threads_per_core


class NThreadsMixin(object):

    is_PerfKnob = True

    def __new__(cls, **kwargs):
        name = kwargs.get('name', cls.name)
        value = cls.default_value()
        obj = Constant.__new__(cls, name=name, dtype=np.int32, value=value)
        obj.aliases = as_tuple(kwargs.get('aliases')) + (name,)
        return obj

    @property
    def _arg_names(self):
        return self.aliases

    def _arg_values(self, **kwargs):
        for i in self.aliases:
            if i in kwargs:
                return {self.name: kwargs.pop(i)}
        # Fallback: as usual, pick the default value
        return self._arg_defaults()


class NThreads(NThreadsMixin, Constant):

    name = 'nthreads'

    @classmethod
    def default_value(cls):
        return int(os.environ.get('OMP_NUM_THREADS', ncores()))


class NThreadsNested(NThreadsMixin, Constant):

    name = 'nthreads_nested'

    @classmethod
    def default_value(cls):
        return nhyperthreads()


class NThreadsNonaffine(NThreads):

    name = 'nthreads_nonaffine'


class ParallelRegion(Block):

    def __init__(self, body, nthreads, private=None):
        header = ParallelRegion._make_header(nthreads, private)
        super(ParallelRegion, self).__init__(header=header, body=body)
        self.nthreads = nthreads

    @classmethod
    def _make_header(cls, nthreads, private):
        private = ('private(%s)' % ','.join(private)) if private else ''
        return c.Pragma('omp parallel num_threads(%s) %s' % (nthreads.name, private))


class ParallelTree(List):

    """
    This class is to group together a parallel for-loop with some setup
    statements, for example:

        .. code-block:: C

          int chunk_size = ...
          #pragma omp ... schedule(..., chunk_size)
          for (int i = ...)
          {
            ...
          }
    """

    _traversable = ['prefix', 'body']

    def __init__(self, prefix, body, nthreads=None):
        super(ParallelTree, self).__init__(body=body)
        self.prefix = as_tuple(prefix)
        self.nthreads = nthreads

    def __getattr__(self, name):
        if 'body' in self.__dict__:
            # During unpickling, `__setattr__` calls `__getattr__(..., 'body')`,
            # which would cause infinite recursion if we didn't check whether
            # 'body' is present or not
            return getattr(self.body[0], name)
        raise AttributeError

    @property
    def functions(self):
        return as_tuple(self.nthreads)


class ThreadedProdder(Conditional, Prodder):

    _traversable = []

    def __init__(self, prodder):
        # Atomic-ize any single-thread Prodders in the parallel tree
        condition = CondEq(Function('omp_get_thread_num')(), 0)

        # Prod within a while loop until all communications have completed
        # In other words, the thread delegated to prodding is entrapped for as long
        # as it's required
        prod_until = Not(Function(prodder.name)(*[i.name for i in prodder.arguments]))
        then_body = List(header=c.Comment('Entrap thread until comms have completed'),
                         body=While(prod_until))

        Conditional.__init__(self, condition, then_body)
        Prodder.__init__(self, prodder.name, prodder.arguments, periodic=prodder.periodic)


class Ompizer(object):

    NESTED = 2
    """
    Use nested parallelism if the number of hyperthreads per core is greater
    than this threshold.
    """

    COLLAPSE_NCORES = 4
    """
    Use a collapse clause if the number of available physical cores is greater
    than this threshold.
    """

    COLLAPSE_WORK = 100
    """
    Use a collapse clause if the trip count of the collapsable Iterations
    exceeds this threshold. Note however the trip count is rarely known at
    compilation time (e.g., this may happen when DefaultDimensions are used).
    """

    CHUNKSIZE_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in parallelized non-affine Iterations.
    """

    lang = {
        'for': lambda i, cs:
            c.Pragma('omp for collapse(%d) schedule(dynamic,%s)' % (i, cs)),
        'par-for': lambda i, cs, nt:
            c.Pragma('omp parallel for collapse(%d) schedule(dynamic,%s) num_threads(%s)'
                     % (i, cs, nt)),
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update')
    }
    """
    Shortcuts for the OpenMP language.
    """

    def __init__(self, key=None):
        """
        Parameters
        ----------
        key : callable, optional
            Return True if an Iteration can be parallelized, False otherwise.
        """
        if key is not None:
            self.key = key
        else:
            self.key = lambda i: i.is_ParallelRelaxed and not i.is_Vectorizable
        self.nthreads = NThreads(aliases='nthreads0')
        self.nthreads_nested = NThreadsNested(aliases='nthreads1')
        self.nthreads_nonaffine = NThreadsNonaffine(aliases='nthreads2')

    def _find_collapsable(self, root, candidates):
        collapsable = []
        if ncores() >= self.COLLAPSE_NCORES and IsPerfectIteration().visit(root):
            for n, i in enumerate(candidates[1:], 1):
                # The OpenMP specification forbids collapsed loops to use iteration
                # variables in initializer expressions. E.g., the following is forbidden:
                #
                # #pragma omp ... collapse(2)
                # for (i = ... )
                #   for (j = i ...)
                #     ...
                #
                # Here, we make sure this won't happen
                if any(j.dim in i.symbolic_min.free_symbols for j in candidates[:n]):
                    break

                # Also, we do not want to collapse vectorizable Iterations
                if i.is_Vectorizable:
                    break

                # Would there be enough work per parallel iteration?
                try:
                    work = prod([int(j.dim.symbolic_size) for j in candidates[n+1:]])
                    if work < Ompizer.COLLAPSE_WORK:
                        break
                except TypeError:
                    pass

                collapsable.append(i)
        return collapsable

    def _make_atomic_incs(self, partree):
        if not partree.is_ParallelAtomic:
            return partree
        # Introduce one `omp atomic` pragma for each increment
        exprs = FindNodes(Expression).visit(partree)
        exprs = [i for i in exprs if i.is_Increment and not i.is_ForeignExpression]
        mapper = {i: List(header=self.lang['atomic'], body=i) for i in exprs}
        partree = Transformer(mapper).visit(partree)
        return partree

    def _make_threaded_prodders(self, partree):
        mapper = {i: ThreadedProdder(i) for i in FindNodes(Prodder).visit(partree)}
        partree = Transformer(mapper).visit(partree)
        return partree

    def _make_partree(self, candidates, nthreads=None):
        """Parallelize the `candidates` Iterations attaching suitable OpenMP pragmas."""
        assert candidates
        root = candidates[0]

        # Get the collapsable Iterations
        collapsable = self._find_collapsable(root, candidates)
        ncollapse = 1 + len(collapsable)

        # Prepare to build a ParallelTree
        prefix = []
        if all(i.is_Affine for i in candidates):
            if nthreads is None:
                # pragma omp for ... schedule(..., 1)
                nthreads = self.nthreads
                omp_pragma = self.lang['for'](ncollapse, 1)
            else:
                # pragma omp parallel for ... schedule(..., 1)
                omp_pragma = self.lang['par-for'](ncollapse, 1, nthreads)
        else:
            # pragma omp for ... schedule(..., expr)
            assert nthreads is None
            nthreads = self.nthreads_nonaffine

            chunk_size = Symbol(name='chunk_size')
            omp_pragma = self.lang['for'](ncollapse, chunk_size)

            niters = prod([root.symbolic_size] + [j.symbolic_size for j in collapsable])
            value = INT(Max(niters / (nthreads*self.CHUNKSIZE_NONAFFINE), 1))
            prefix.append(Expression(DummyEq(chunk_size, value, dtype=np.int32)))

        # Create a ParallelTree
        body = root._rebuild(pragmas=root.pragmas + (omp_pragma,),
                             properties=root.properties + (COLLAPSED(ncollapse),))
        partree = ParallelTree(prefix, body, nthreads=nthreads)

        collapsed = [partree] + collapsable

        return root, partree, collapsed

    def _make_parregion(self, partree):
        # Build the `omp-parallel` region
        private = [i for i in FindSymbols().visit(partree)
                   if i.is_Array and i._mem_stack]
        private = sorted(set([i.name for i in private]))
        return ParallelRegion(partree, partree.nthreads, private)

    def _make_guard(self, partree, collapsed):
        # Do not enter the parallel region if the step increment is 0; this
        # would raise a `Floating point exception (core dumped)` in some OpenMP
        # implementations. Note that using an OpenMP `if` clause won't work
        cond = [CondEq(i.step, 0) for i in collapsed if isinstance(i.step, Symbol)]
        cond = Or(*cond)
        if cond != False:  # noqa: `cond` may be a sympy.False which would be == False
            partree = List(body=[Conditional(cond, Return()), partree])
        return partree

    def _make_nested_partree(self, partree):
        # Apply heuristic
        if nhyperthreads() <= Ompizer.NESTED:
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
                if any(is_integer(j.step - i.symbolic_size) for j in outer):
                    candidates.append(i)
                elif candidates:
                    # If there's at least one candidate but `i` doesn't honor the
                    # heuristic above, then we break, as the candidates must be
                    # perfectly nested
                    break
            if not candidates:
                continue

            # Introduce nested parallelism
            subroot, subpartree, _ = self._make_partree(candidates, self.nthreads_nested)

            mapper[subroot] = subpartree

        partree = Transformer(mapper).visit(partree)

        return partree

    def _make_data_transfers(self, iet):
        # No data transfers needed
        return iet

    def make_parallel(self, iet):
        """Transform ``iet`` by introducing shared-memory parallelism."""
        mapper = OrderedDict()
        for tree in retrieve_iteration_tree(iet):
            # Get the omp-parallelizable Iterations in `tree`
            candidates = filter_iterations(tree, key=self.key)
            if not candidates:
                continue

            # Outer parallelism
            root, partree, collapsed = self._make_partree(candidates)

            # Nested parallelism
            partree = self._make_nested_partree(partree)

            # Ensure increments are atomic
            partree = self._make_atomic_incs(partree)

            # Atomicize and optimize single-thread prodders
            partree = self._make_threaded_prodders(partree)

            # Wrap within a parallel region, declaring private and shared variables
            parregion = self._make_parregion(partree)

            # Protect the parallel region in case of 0-valued step increments
            parregion = self._make_guard(parregion, collapsed)

            mapper[root] = parregion

        iet = Transformer(mapper).visit(iet)

        # Add the data transfers
        iet = self._make_data_transfers(iet)

        # The used `nthreads` arguments
        args = [i for i in FindSymbols().visit(iet) if isinstance(i, (NThreadsMixin))]

        return iet, {'args': args, 'includes': ['omp.h']}


class OmpizerGPU(Ompizer):

    COLLAPSE_NCORES = 1
    """
    Always collapse when possible.
    """

    def map_to(f):
        omp_pragma = 'omp target enter data map(to: %s%s)'
        var_name = f.name
        var_data = ''.join('[0:%s]' % f._C_get_field(FULL, d).size for d in f.dimensions)
        return c.Pragma(omp_pragma % (var_name, var_data))

    def map_from(f):
        omp_pragma = 'omp target exit data map(from: %s%s)'
        var_name = f.name
        var_data = ''.join('[0:%s]' % f._C_get_field(FULL, d).size for d in f.dimensions)
        return c.Pragma(omp_pragma % (var_name, var_data))

    lang = dict(Ompizer.lang)
    lang.update({
        'par-for-teams': lambda i:
            c.Pragma('omp target teams distribute parallel for collapse(%d)' % i),
        'map-to': map_to,
        'map-from': map_from
    })

    def __init__(self, key=None):
        if key is None:
            key = lambda i: i.is_ParallelRelaxed
        super(OmpizerGPU, self).__init__(key=key)

    def _make_threaded_prodders(self, partree):
        # no-op for now
        return partree

    def _make_partree(self, candidates, nthreads=None):
        """
        Parallelize the `candidates` Iterations attaching suitable OpenMP pragmas
        for GPU offloading.
        """
        assert candidates
        root = candidates[0]

        # Get the collapsable Iterations
        collapsable = self._find_collapsable(root, candidates)
        ncollapse = 1 + len(collapsable)

        # Prepare to build a ParallelTree
        omp_pragma = self.lang['par-for-teams'](ncollapse)

        # Create a ParallelTree
        body = root._rebuild(pragmas=root.pragmas + (omp_pragma,),
                             properties=root.properties + (COLLAPSED(ncollapse),))
        partree = ParallelTree([], body, nthreads=nthreads)

        collapsed = [partree] + collapsable

        return root, partree, collapsed

    def _make_parregion(self, partree):
        # no-op for now
        return partree

    def _make_guard(self, partree, *args):
        # no-op for now
        return partree

    def _make_nested_partree(self, partree):
        # no-op for now
        return partree

    def _make_data_transfers(self, iet):
        data_transfers = {}
        for iteration, partree in MapNodes(child_types=ParallelTree).visit(iet).items():
            # The data that need to be transfered between host and device
            exprs = FindNodes(Expression).visit(partree)
            reads = set().union(*[i.reads for i in exprs])
            writes = set([i.write for i in exprs])
            data = {i for i in reads | writes if i.is_Tensor}

            # At what depth in the IET should the data transfer be performed?
            candidate = iteration
            for tree in retrieve_iteration_tree(iet):
                if iteration not in tree:
                    continue
                for i in reversed(tree[:tree.index(iteration)+1]):
                    found = False
                    for k, v in MapNodes('any', Expression, 'groupby').visit(i).items():
                        test0 = any(isinstance(n, ParallelTree) for n in k)
                        test1 = set().union(*[e.functions for e in v]) & data
                        if not test0 and test1:
                            found = True
                            break
                    if found:
                        break
                    candidate = i
                break

            # Create the omp pragmas for the data transfer
            map_tos = [self.lang['map-to'](i) for i in data]
            map_froms = [self.lang['map-from'](i) for i in writes if i.is_Tensor]
            data_transfers.setdefault(candidate, []).append((map_tos, map_froms))

        # Now create a new IET with the data transfer
        mapper = {}
        for i, v in data_transfers.items():
            map_tos, map_froms = zip(*v)
            mapper[i] = List(header=flatten(map_tos), body=i, footer=flatten(map_froms))
        iet = Transformer(mapper).visit(iet)

        return iet
