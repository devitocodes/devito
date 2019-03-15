from collections import OrderedDict
import os

import numpy as np
import cgen as c
import psutil
from sympy import Function, Or

from devito.ir.iet import (Call, Conditional, Block, Expression, List, Prodder,
                           FindSymbols, FindNodes, Return, COLLAPSED, Transformer,
                           IsPerfectIteration, retrieve_iteration_tree, filter_iterations)
from devito.symbolics import CondEq
from devito.parameters import configuration
from devito.tools import memoized_func
from devito.types import Constant, Symbol


@memoized_func
def ncores():
    try:
        return configuration['cross-compile'].cpu_count(logical=False)
    except AttributeError:
        return psutil.cpu_count(logical=False)


@memoized_func
def nhyperthreads():
    try:
        logical = configuration['cross-compile'].cpu_count(logical=True)
    except AttributeError:
        logical = psutil.cpu_count(logical=True)
    physical = ncores()
    if logical % physical > 0:
        dle_warning("Couldn't detect number of hyperthreads per core, assuming 1")
        return 1
    else:
        return logical // physical


class NThreads(Constant):

    @classmethod
    def default_value(cls):
        return int(os.environ.get('OMP_NUM_THREADS', ncores()))

    def __new__(cls, **kwargs):
        return super(NThreads, cls).__new__(cls, name=kwargs['name'], dtype=np.int32,
                                            value=NThreads.default_value())


class ParallelRegion(Block):

    def __init__(self, body, nthreads, private=None):
        header = ParallelRegion._make_header(nthreads, private)
        super(ParallelRegion, self).__init__(header=header, body=body)
        self.nthreads = nthreads

    @classmethod
    def _make_header(cls, nthreads, private):
        private = ('private(%s)' % ','.join(private)) if private else ''
        return c.Pragma('omp parallel num_threads(%s) %s' % (nthreads.name, private))

    @property
    def functions(self):
        return (self.nthreads,)


class SingleThreadProdder(Conditional, Prodder):

    _traversable = []

    def __init__(self, prodder):
        condition = CondEq(Function('omp_get_thread_num')(), 0)
        then_body = Call(prodder.name, prodder.arguments)
        Conditional.__init__(self, condition, then_body)
        Prodder.__init__(self, prodder.name, prodder.arguments, periodic=prodder.periodic)


class Ompizer(object):

    COLLAPSE = 32
    """
    Use a collapse clause if the number of available physical cores is greater
    than this threshold.
    """

    lang = {
        'for': lambda i: c.Pragma('omp for collapse(%d) schedule(static)' % i),
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
        self.nthreads = NThreads(name='nthreads')

    def _find_collapsable(self, root, candidates):
        # Apply heuristic
        if ncores() < Ompizer.COLLAPSE:
            return [root]

        # To be collapsable, two Iterations must be perfectly nested
        if not IsPerfectIteration().visit(root):
            return [root]

        # The OpenMP specification forbids collapsed loops to use iteration variables
        # in initializer expressions. For example, the following is forbidden:
        #
        # #pragma omp ... collapse(2)
        # for (int i = ... )
        #   for (int j = i ...)
        #     ...
        #
        # Below, we make sure this won't happen
        collapsable = []
        for n, i in enumerate(candidates):
            if any(j.dim in i.symbolic_min.free_symbols for j in candidates[:n]):
                break
            collapsable.append(i)
        return collapsable

    def _make_parallel_tree(self, root, collapsable):
        """Parallelize the IET rooted in `root`."""
        ncollapse = len(collapsable)
        parallel = self.lang['for'](ncollapse)

        pragmas = root.pragmas + (parallel,)
        properties = root.properties + (COLLAPSED(ncollapse),)

        # Introduce the `omp for` pragma
        mapper = OrderedDict()
        if root.is_ParallelAtomic:
            # Introduce the `omp atomic` pragmas
            exprs = FindNodes(Expression).visit(root)
            exprs = [i for i in exprs if i.is_Increment and not i.is_ForeignExpression]
            subs = {i: List(header=self.lang['atomic'], body=i) for i in exprs}
            handle = Transformer(subs).visit(root)
            mapper[root] = handle._rebuild(pragmas=pragmas, properties=properties)
        else:
            mapper[root] = root._rebuild(pragmas=pragmas, properties=properties)
        root = Transformer(mapper).visit(root)

        # Atomic-ize any single-thread Prodders in the parallel tree
        mapper = {i: SingleThreadProdder(i) for i in FindNodes(Prodder).visit(root)}
        root = Transformer(mapper).visit(root)

        return root

    def make_parallel(self, iet):
        """Transform ``iet`` by introducing shared-memory parallelism."""
        mapper = OrderedDict()
        for tree in retrieve_iteration_tree(iet):
            # Get the first omp-parallelizable Iteration in `tree`
            candidates = filter_iterations(tree, key=self.key)
            if not candidates:
                continue
            root = candidates[0]

            # Determine the number of collapsable Iterations
            collapsable = self._find_collapsable(root, candidates)

            # Build the `omp-for` tree
            partree = self._make_parallel_tree(root, collapsable)

            # Find out the thread-private and thread-shared variables
            private = [i for i in FindSymbols().visit(partree)
                       if i.is_Array and i._mem_stack]

            # Build the `omp-parallel` region
            private = sorted(set([i.name for i in private]))
            partree = ParallelRegion(partree, self.nthreads, private)

            # Do not enter the parallel region if the step increment is 0; this
            # would raise a `Floating point exception (core dumped)` in some OpenMP
            # implementations. Note that using an OpenMP `if` clause won't work
            cond = [CondEq(i.step, 0) for i in collapsable if isinstance(i.step, Symbol)]
            cond = Or(*cond)
            if cond != False:  # noqa: `cond` may be a sympy.False which would be == False
                partree = List(body=[Conditional(cond, Return()), partree])

            mapper[root] = partree
        iet = Transformer(mapper).visit(iet)

        return iet, {'args': [self.nthreads] if mapper else [],
                     'includes': ['omp.h']}
