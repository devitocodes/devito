from collections import OrderedDict

import numpy as np
import cgen as c
import psutil

from devito.function import Constant
from devito.ir.equations import DummyEq
from devito.ir.iet import (FindSymbols, FindNodes, Transformer, Block, Expression,
                           LocalExpression, List, Iteration, retrieve_iteration_tree,
                           filter_iterations, IsPerfectIteration, COLLAPSED)
from devito.parameters import configuration
from devito.types import Symbol


def ncores():
    try:
        return configuration['cross-compile'].cpu_count()
    except AttributeError:
        return psutil.cpu_count(logical=False)


class NThreads(Constant):

    def __new__(cls, **kwargs):
        return super(NThreads, cls).__new__(cls, name='nthreads', dtype=np.int32,
                                            value=ncores())


class Ompizer(object):

    COLLAPSE = 32
    """Use a collapse clause if the number of available physical cores is
    greater than this threshold."""

    lang = {
        'for': lambda i: c.Pragma('omp for collapse(%d) schedule(static)' % i),
        'par-region': lambda i: c.Pragma('omp parallel num_threads(nt) %s' % i),
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update')
    }
    """
    Shortcuts for the OpenMP language.
    """

    def __init__(self, key):
        """
        Parameters
        ----------
        key : callable
            Return True if an Iteration can be parallelized, False otherwise.
        """
        self.key = key

    def _ncollapse(self, root, candidates):
        # Heuristic: if at least two parallel loops are available and the
        # physical core count is greater than COLLAPSE, then omp-collapse them
        nparallel = len(candidates)
        isperfect = IsPerfectIteration().visit(root)
        if ncores() < Ompizer.COLLAPSE or nparallel < 2 or not isperfect:
            return 1
        else:
            return nparallel

    def _make_parallel_tree(self, root, candidates):
        """
        Return a mapper to parallelize the :class:`Iteration`s within ``root``.
        """
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

        return mapper

    def make_parallel(self, iet):
        """
        Transform ``iet`` by decorating its parallel :class:`Iteration`s with
        suitable ``#pragma omp ...`` for thread-level parallelism.
        """
        # Group sequences of loops that should go within the same parallel region
        was_tagged = False
        groups = OrderedDict()
        for tree in retrieve_iteration_tree(iet):
            # Determine the number of consecutive parallelizable Iterations
            candidates = filter_iterations(tree, key=self.key, stop='asap')
            if not candidates:
                was_tagged = False
                continue
            # Consecutive tagged Iteration go in the same group
            is_tagged = any(i.tag is not None for i in tree)
            key = len(groups) - (is_tagged & was_tagged)
            handle = groups.setdefault(key, OrderedDict())
            handle[candidates[0]] = candidates
            was_tagged = is_tagged

        mapper = OrderedDict()
        for group in groups.values():
            private = []
            for root, candidates in group.items():
                mapper.update(self._make_parallel_tree(root, candidates))

                # Track the thread-private and thread-shared variables
                private.extend([i for i in FindSymbols('symbolics').visit(root)
                                if i.is_Array and i._mem_stack])

            # Build the parallel region
            private = sorted(set([i.name for i in private]))
            private = ('private(%s)' % ','.join(private)) if private else ''
            rebuilt = [v for k, v in mapper.items() if k in group]
            par_region = Block(header=self.lang['par-region'](private), body=rebuilt)
            for k, v in list(mapper.items()):
                if isinstance(v, Iteration):
                    mapper[k] = None if v.is_Remainder else par_region
        processed = Transformer(mapper).visit(iet)

        # Hack/workaround to the fact that the OpenMP pragmas are not true
        # IET nodes, so the `nthreads` variables won't be detected as a
        # Callable parameter unless inserted in a mock Expression
        if mapper:
            nt = NThreads()
            eq = LocalExpression(DummyEq(Symbol(name='nt', dtype=np.int32), nt))
            return List(body=[eq, processed]), {'input': [nt]}
        else:
            return List(body=processed), {}
