import cgen as c
from sympy import Not

from devito.data import FULL
from devito.ir import (Conditional, List, Prodder, ParallelIteration, ParallelBlock,
                       ParallelTree, While, COLLAPSED, FindNodes, Transformer)
from devito.symbolics import CondEq, DefFunction
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.language import (Constructs, HostPragmaParallelizer,
                                        DevicePragmaParallelizer)
from devito.tools import as_tuple

__all__ = ['Ompizer', 'OpenMPIteration', 'OpenMPRegion']


class OpenMPRegion(ParallelBlock):

    def __init__(self, body, private=None):
        # Normalize and sanity-check input. A bit ugly, but it makes everything
        # much simpler to manage and reconstruct
        body = as_tuple(body)
        assert len(body) == 1
        body = body[0]
        assert body.is_List
        if isinstance(body, ParallelTree):
            partree = body
        elif body.is_List:
            assert len(body.body) == 1 and isinstance(body.body[0], ParallelTree)
            assert len(body.footer) == 0
            partree = body.body[0]
            partree = partree._rebuild(prefix=(List(header=body.header,
                                                    body=partree.prefix)))

        header = OpenMPRegion._make_header(partree.nthreads, private)
        super(OpenMPRegion, self).__init__(header=header, body=partree)

    @property
    def partree(self):
        return self.body[0]

    @property
    def root(self):
        return self.partree.root

    @property
    def nthreads(self):
        return self.partree.nthreads

    @classmethod
    def _make_header(cls, nthreads, private=None):
        private = ('private(%s)' % ','.join(private)) if private else ''
        return c.Pragma('omp parallel num_threads(%s) %s' % (nthreads.name, private))


class OpenMPIteration(ParallelIteration):

    def __init__(self, *args, **kwargs):
        pragmas, kwargs = self._make_header(**kwargs)

        properties = as_tuple(kwargs.pop('properties', None))
        properties += (COLLAPSED(kwargs.get('ncollapse', 1)),)

        self.schedule = kwargs.pop('schedule', None)
        self.parallel = kwargs.pop('parallel', False)
        self.ncollapse = kwargs.pop('ncollapse', None)
        self.chunk_size = kwargs.pop('chunk_size', None)
        self.nthreads = kwargs.pop('nthreads', None)
        self.reduction = kwargs.pop('reduction', None)

        super(OpenMPIteration, self).__init__(*args, pragmas=pragmas,
                                              properties=properties, **kwargs)

    @classmethod
    def _make_header(cls, **kwargs):
        kwargs.pop('pragmas', None)

        construct = cls._make_construct(**kwargs)
        clauses = cls._make_clauses(**kwargs)
        header = c.Pragma(' '.join([construct] + clauses))

        return (header,), kwargs

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
                    f = i.function
                    bounds = []
                    for k, d in zip(i.indices, f.dimensions):
                        if k.is_Number:
                            bounds.append('[%s]' % k)
                        else:
                            # OpenMP expects a range as input of reduction,
                            # such as reduction(+:f[0:f_vec->size[1]])
                            bounds.append('[0:%s]' % f._C_get_field(FULL, d).size)
                    args.append('%s%s' % (i.name, ''.join(bounds)))
                else:
                    args.append(str(i))
            clauses.append('reduction(+:%s)' % ','.join(args))

        return clauses


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


class Ompizer(HostPragmaParallelizer):

    lang = Constructs([
        ('simd-for', c.Pragma('omp simd')),
        ('simd-for-aligned', lambda i, j: c.Pragma('omp simd aligned(%s:%d)' % (i, j))),
        ('atomic', c.Pragma('omp atomic update')),
        ('thread-num', DefFunction('omp_get_thread_num')),
        ('header', 'omp.h')
    ])
    #TODO: TRY CONSTRUCTS via make_simd...

    _Region = OpenMPRegion
    _Iteration = OpenMPIteration

    def _make_threaded_prodders(self, partree):
        mapper = {i: ThreadedProdder(i) for i in FindNodes(Prodder).visit(partree)}
        partree = Transformer(mapper).visit(partree)
        return partree
