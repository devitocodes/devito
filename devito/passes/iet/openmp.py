import os

import numpy as np
import cgen as c
from sympy import Or, Max, Not

from devito.data import FULL
from devito.ir import (DummyEq, Conditional, Dereference, Expression, ExpressionBundle,
                       List, Prodder, ParallelIteration, ParallelBlock, While,
                       FindSymbols, FindNodes, Return, COLLAPSED, VECTORIZED, Transformer,
                       IsPerfectIteration, retrieve_iteration_tree, filter_iterations)
from devito.symbolics import CondEq, DefFunction, INT
from devito.parameters import configuration
from devito.passes.iet.engine import iet_pass
from devito.tools import as_tuple, is_integer, prod
from devito.types import Constant, PointerArray, Symbol

__all__ = ['NThreads', 'NThreadsNested', 'NThreadsNonaffine', 'Ompizer',
           'OpenMPIteration', 'ParallelTree']


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


class OpenMPRegion(ParallelBlock):

    def __init__(self, body, nthreads, private=None):
        header = OpenMPRegion._make_header(nthreads, private)
        super(OpenMPRegion, self).__init__(header=header, body=body)
        self.nthreads = nthreads

    @classmethod
    def _make_header(cls, nthreads, private=None):
        private = ('private(%s)' % ','.join(private)) if private else ''
        return c.Pragma('omp parallel num_threads(%s) %s' % (nthreads.name, private))


class OpenMPIteration(ParallelIteration):

    def __init__(self, *args, **kwargs):
        kwargs.pop('pragmas', None)
        pragma = self._make_header(**kwargs)

        properties = as_tuple(kwargs.pop('properties', None))
        properties += (COLLAPSED(kwargs.get('ncollapse', 1)),)

        self.schedule = kwargs.pop('schedule', None)
        self.parallel = kwargs.pop('parallel', False)
        self.ncollapse = kwargs.pop('ncollapse', None)
        self.chunk_size = kwargs.pop('chunk_size', None)
        self.nthreads = kwargs.pop('nthreads', None)
        self.reduction = kwargs.pop('reduction', None)

        super(OpenMPIteration, self).__init__(*args, pragmas=[pragma],
                                              properties=properties, **kwargs)

    @classmethod
    def _make_header(cls, **kwargs):
        construct = cls._make_construct(**kwargs)
        clauses = cls._make_clauses(**kwargs)

        header = ' '.join([construct] + clauses)

        return c.Pragma(header)

    @classmethod
    def _make_construct(cls, parallel=False, **kwargs):
        if parallel:
            return 'omp parallel for'
        else:
            return 'omp for'

    @classmethod
    def _make_clauses(cls, ncollapse=None, chunk_size=None, nthreads=None,
                      reduction=None, schedule=None, **kwargs):
        clauses = []

        clauses.append('collapse(%d)' % (ncollapse or 1))

        if chunk_size is not False:
            clauses.append('schedule(%s,%s)' % (schedule or 'dynamic',
                                                chunk_size or 1))

        if nthreads:
            clauses.append('num_threads(%s)' % nthreads)

        if reduction:
            args = []
            for i in reduction:
                if i.is_Indexed:
                    # OpenMP expects a size not an index as input of reduction,
                    # such as reduction(+:f[d_m:d_M+1])
                    f = i.function
                    bounds = []
                    for k, d in zip(i.indices, f.dimensions):
                        if k.is_Number:
                            bounds.append('[%s:%s]' % (str(k), str(k + 1)))
                        else:
                            bounds.append('[0:%s]' % f._C_get_field(FULL, d).size)
                    args.append('%s%s' % (i.name, ''.join(bounds)))
                else:
                    args.append(str(i))
            clauses.append('reduction(+:%s)' % ','.join(args))

        return clauses


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
        # Normalize and sanity-check input
        body = as_tuple(body)
        assert len(body) == 1 and body[0].is_Iteration

        self.prefix = as_tuple(prefix)
        self.nthreads = nthreads

        super(ParallelTree, self).__init__(body=body)

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

    @property
    def root(self):
        return self.body[0]


class ThreadedProdder(Conditional, Prodder):

    _traversable = []

    def __init__(self, prodder):
        # Atomic-ize any single-thread Prodders in the parallel tree
        condition = CondEq(Ompizer.lang['thread-num'], 0)

        # Prod within a while loop until all communications have completed
        # In other words, the thread delegated to prodding is entrapped for as long
        # as it's required
        prod_until = Not(DefFunction(prodder.name, [i.name for i in prodder.arguments]))
        then_body = List(header=c.Comment('Entrap thread until comms have completed'),
                         body=While(prod_until))

        Conditional.__init__(self, condition, then_body)
        Prodder.__init__(self, prodder.name, prodder.arguments, periodic=prodder.periodic)


class Ompizer(object):

    lang = {
        'simd-for': c.Pragma('omp simd'),
        'simd-for-aligned': lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j)),
        'atomic': c.Pragma('omp atomic update'),
        'thread-num': DefFunction('omp_get_thread_num')
    }
    """
    Shortcuts for the OpenMP language.
    """

    _Region = OpenMPRegion
    _Iteration = OpenMPIteration

    def __init__(self, sregistry, options, key=None):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET (e.g., `sregistry.threadid`, `sregistry.nthreads`).
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
        key : callable, optional
            Return True if an Iteration can be parallelized, False otherwise.
        """
        self.sregistry = sregistry

        self.collapse_ncores = options['par-collapse-ncores']
        self.collapse_work = options['par-collapse-work']
        self.chunk_nonaffine = options['par-chunk-nonaffine']
        self.dynamic_work = options['par-dynamic-work']
        self.nested = options['par-nested']

        if key is not None:
            self.key = key
        else:
            self.key = lambda i: i.is_ParallelRelaxed and not i.is_Vectorized

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

    def _find_collapsable(self, root, candidates):
        collapsable = []
        if ncores() >= self.collapse_ncores:
            for n, i in enumerate(candidates[1:], 1):
                # The Iteration nest [root, ..., i] must be perfect
                if not IsPerfectIteration(depth=i).visit(root):
                    break

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
        return collapsable

    @classmethod
    def _make_tid(cls, tid):
        return c.Initializer(c.Value(tid._C_typedata, tid.name), cls.lang['thread-num'])

    def _make_reductions(self, partree, collapsed):
        if not any(i.is_ParallelAtomic for i in collapsed):
            return partree

        # Collect expressions inducing reductions
        exprs = FindNodes(Expression).visit(partree)
        exprs = [i for i in exprs if i.is_Increment and not i.is_ForeignExpression]

        reduction = [i.output for i in exprs]
        if (all(i.is_Affine for i in collapsed)
                or all(not i.is_Indexed for i in reduction)):
            # Introduce reduction clause
            mapper = {partree.root: partree.root._rebuild(reduction=reduction)}
        else:
            # Introduce one `omp atomic` pragma for each increment
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
        if all(i.is_Affine for i in candidates):
            bundles = FindNodes(ExpressionBundle).visit(root)
            sops = sum(i.ops for i in bundles)
            if sops >= self.dynamic_work:
                schedule = 'dynamic'
            else:
                schedule = 'static'
            if nthreads is None:
                # pragma omp for ... schedule(..., 1)
                nthreads = self.nthreads
                body = OpenMPIteration(schedule=schedule, ncollapse=ncollapse,
                                       **root.args)
            else:
                # pragma omp parallel for ... schedule(..., 1)
                body = OpenMPIteration(schedule=schedule, parallel=True,
                                       ncollapse=ncollapse, nthreads=nthreads,
                                       **root.args)
            prefix = []
        else:
            # pragma omp for ... schedule(..., expr)
            assert nthreads is None
            nthreads = self.nthreads_nonaffine
            chunk_size = Symbol(name='chunk_size')
            body = OpenMPIteration(ncollapse=ncollapse, chunk_size=chunk_size,
                                   **root.args)

            niters = prod([root.symbolic_size] + [j.symbolic_size for j in collapsable])
            value = INT(Max(niters / (nthreads*self.chunk_nonaffine), 1))
            prefix = [Expression(DummyEq(chunk_size, value, dtype=np.int32))]

        # Create a ParallelTree
        partree = ParallelTree(prefix, body, nthreads=nthreads)

        collapsed = [partree] + collapsable

        return root, partree, collapsed

    def _make_parregion(self, partree, parrays):
        arrays = [i for i in FindSymbols().visit(partree) if i.is_Array]

        # Detect thread-private arrays on the heap and "map" them to shared
        # vector-expanded (one entry per thread) Arrays
        heap_private = [i for i in arrays if i._mem_heap and i._mem_local]
        heap_globals = []
        for i in heap_private:
            if i in parrays:
                pi = parrays[i]
            else:
                pi = parrays.setdefault(i, PointerArray(name=self.sregistry.make_name(),
                                                        dimensions=(self.threadid,),
                                                        array=i))
            heap_globals.append(Dereference(i, pi))
        if heap_globals:
            body = List(header=self._make_tid(self.threadid),
                        body=heap_globals+[partree], footer=c.Line())
        else:
            body = partree

        return OpenMPRegion(body, partree.nthreads)

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
        if nhyperthreads() <= self.nested:
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
                if self.key(i) and any(is_integer(j.step-i.symbolic_size) for j in outer):
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

    def _make_parallel(self, iet):
        mapper = {}
        parrays = {}
        for tree in retrieve_iteration_tree(iet):
            # Get the omp-parallelizable Iterations in `tree`
            candidates = filter_iterations(tree, key=self.key)
            if not candidates:
                continue

            # Outer parallelism
            root, partree, collapsed = self._make_partree(candidates)
            if root in mapper:
                continue

            # Nested parallelism
            partree = self._make_nested_partree(partree)

            # Handle reductions
            partree = self._make_reductions(partree, collapsed)

            # Atomicize and optimize single-thread prodders
            partree = self._make_threaded_prodders(partree)

            # Wrap within a parallel region, declaring private and shared variables
            parregion = self._make_parregion(partree, parrays)

            # Protect the parallel region in case of 0-valued step increments
            parregion = self._make_guard(parregion, collapsed)

            mapper[root] = parregion

        iet = Transformer(mapper).visit(iet)

        # The new arguments introduced by this pass
        args = [i for i in FindSymbols().visit(iet) if isinstance(i, (NThreadsMixin))]
        for n in FindNodes(Dereference).visit(iet):
            args.extend([(n.array, True), n.parray])

        return iet, {'args': args, 'includes': ['omp.h']}

    @iet_pass
    def make_parallel(self, iet):
        """
        Create a new IET with shared-memory parallelism via OpenMP pragmas.
        """
        return self._make_parallel(iet)

    @iet_pass
    def make_simd(self, iet, **kwargs):
        """
        Create a new IET with SIMD parallelism via OpenMP pragmas.
        """
        simd_reg_size = kwargs.pop('simd_reg_size')

        mapper = {}
        for tree in retrieve_iteration_tree(iet):
            candidates = [i for i in tree if i.is_Parallel]

            # As long as there's an outer level of parallelism, the innermost
            # PARALLEL Iteration gets vectorized
            if len(candidates) < 2:
                continue
            candidate = candidates[-1]

            # Construct OpenMP SIMD pragma
            aligned = [j for j in FindSymbols('symbolics').visit(candidate)
                       if j.is_DiscreteFunction]
            if aligned:
                simd = self.lang['simd-for-aligned']
                simd = as_tuple(simd(','.join([j.name for j in aligned]),
                                simd_reg_size))
            else:
                simd = as_tuple(self.lang['simd-for'])
            pragmas = candidate.pragmas + simd

            # Add VECTORIZED property
            properties = list(candidate.properties) + [VECTORIZED]

            mapper[candidate] = candidate._rebuild(pragmas=pragmas, properties=properties)

        iet = Transformer(mapper).visit(iet)

        return iet, {}
