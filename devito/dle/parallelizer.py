from collections import OrderedDict

import numpy as np
import cgen as c
import psutil

from devito.ir.iet import (Conditional, Block, Element, Expression, List, FindSymbols,
                           FindNodes, Transformer, IsPerfectIteration, COLLAPSED,
                           retrieve_iteration_tree, filter_iterations)
from devito.symbolics import CondEq
from devito.parameters import configuration
from devito.types import Constant, Symbol


def ncores():
    try:
        return configuration['cross-compile'].cpu_count()
    except AttributeError:
        return psutil.cpu_count(logical=False)


class NThreads(Constant):

    def __new__(cls, **kwargs):
        return super(NThreads, cls).__new__(cls, name=kwargs['name'], dtype=np.int32,
                                            value=ncores())


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


class Ompizer(object):

    COLLAPSE = 32
    """Use a collapse clause if the number of available physical cores is
    greater than this threshold."""

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

    def _ncollapse(self, root, candidates):
        # The OpenMP specification forbids collapsed loops to use iteration variables
        # in initializer expressions. For example, the following is forbidden:
        #
        # #pragma omp ... collapse(2)
        # for (int i = ... )
        #   for (int j = i ...)
        #     ...
        #
        # Below, we make sure this won't happen
        for n, i in enumerate(candidates):
            if any(j.dim in i.symbolic_min.free_symbols for j in candidates[:n]):
                break
        candidates = candidates[:n]
        # Heuristic: if at least two parallel loops are available and the
        # physical core count is greater than COLLAPSE, then omp-collapse them
        nparallel = len(candidates)
        isperfect = IsPerfectIteration().visit(root)
        if ncores() < Ompizer.COLLAPSE or nparallel < 2 or not isperfect:
            return 1
        else:
            return nparallel

    def _make_parallel_tree(self, root, candidates):
        """Parallelize the IET rooted in `root`."""
        ncollapse = self._ncollapse(root, candidates)
        parallel = self.lang['for'](ncollapse)

        pragmas = root.pragmas + (parallel,)
        properties = root.properties + (COLLAPSED(ncollapse),)

        # Introduce the `omp for` pragma
        mapper = OrderedDict()
        if root.is_ParallelAtomic:
            # Introduce the `omp atomic` pragmas
            exprs = FindNodes(Expression).visit(root)
            subs = {i: List(header=self.lang['atomic'], body=i)
                    for i in exprs if i.is_Increment}
            handle = Transformer(subs).visit(root)
            mapper[root] = handle._rebuild(pragmas=pragmas, properties=properties)
        else:
            mapper[root] = root._rebuild(pragmas=pragmas, properties=properties)

        root = Transformer(mapper).visit(root)

        return root

    def make_parallel(self, iet):
        """Transform ``iet`` by introducing shared-memory parallelism."""
        mapper = OrderedDict()
        for tree in retrieve_iteration_tree(iet):
            # Get the first omp-parallelizable Iteration in `tree`
            candidates = filter_iterations(tree, key=self.key, stop='asap')
            if not candidates:
                continue
            root = candidates[0]

            # Build the `omp-for` tree
            partree = self._make_parallel_tree(root, candidates)

            # Find out the thread-private and thread-shared variables
            private = [i for i in FindSymbols().visit(partree)
                       if i.is_Array and i._mem_stack]

            # Build the `omp-parallel` region
            private = sorted(set([i.name for i in private]))
            partree = ParallelRegion(partree, self.nthreads, private)

            # Do not enter the parallel region if the step increment might be 0; this
            # would raise a `Floating point exception (core dumped)` in some OpenMP
            # implementation. Note that using an OpenMP `if` clause won't work
            if isinstance(root.step, Symbol):
                cond = Conditional(CondEq(root.step, 0), Element(c.Statement('return')))
                partree = List(body=[cond, partree])

            mapper[root] = partree
        iet = Transformer(mapper).visit(iet)

        return iet, {'input': [self.nthreads] if mapper else []}
