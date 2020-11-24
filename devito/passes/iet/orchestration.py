from collections import namedtuple

import cgen as c
from sympy import Or
import numpy as np

from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Callable, Conditional, List, Iteration, PointerCast,
                           SyncSpot, While, FindNodes, LocalExpression, Transformer,
                           BlankLine, DummyExpr, derive_parameters, make_tfunc)
from devito.ir.support import Forward
from devito.passes.iet.engine import iet_pass
from devito.symbolics import CondEq, CondNe, FieldFromComposite, ListInitializer
from devito.tools import (as_mapper, as_list, filter_ordered, filter_sorted,
                          is_integer)
from devito.types import (NThreadsSTD, STDThreadArray, WaitLock, WithLock,
                          FetchWait, FetchWaitPrefetch, Delete, SharedData, Symbol)

__init__ = ['Orchestrator', 'BusyWait']


class Orchestrator(object):

    """
    Coordinate host and device in a heterogeneous system (e.g., CPU and GPU).
    This boils down to introducing data transfers, synchronizations, asynchronous
    computation and so on.
    """

    _Parallelizer = None
    """
    To be specified by the subclasses. This is used to generate IETs for the
    data transfers between host and device.
    """

    def __init__(self, sregistry):
        self.sregistry = sregistry

        if self._Parallelizer is None:
            raise NotImplementedError

    @property
    def _P(self):
        """Shortcut for self._Parallelizer."""
        return self._Parallelizer

    def __make_threads(self, value=None):
        name = self.sregistry.make_name(prefix='threads')

        if value is None:
            threads = STDThreadArray(name=name, nthreads_std=1)
        else:
            nthreads_std = NThreadsSTD(name='np%s' % name, value=value)
            threads = STDThreadArray(name=name, nthreads_std=nthreads_std)

        return threads

    def __make_init_threads(self, threads, tfunc, pieces):
        d = threads.index
        sdata = tfunc.sdata
        if threads.size == 1:
            callback = lambda body: body
        else:
            callback = lambda body: Iteration(body, d, threads.size - 1)

        idinit = DummyExpr(FieldFromComposite(sdata._field_id, sdata[d]),
                           1 + sum(i.size for i in pieces.threads) + d)
        arguments = list(tfunc.parameters)
        arguments[-1] = sdata.symbolic_base + d
        call = Call('std::thread', Call(tfunc.name, arguments, is_indirect=True),
                    retobj=threads[d])
        threadsinit = List(
            header=c.Comment("Fire up and initialize `%s`" % threads.name),
            body=callback([idinit, call])
        )

        return threadsinit

    def __make_finalize_threads(self, threads, sdata):
        d = threads.index
        if threads.size == 1:
            callback = lambda body: body
        else:
            callback = lambda body: Iteration(body, d, threads.size - 1)

        threadswait = List(
            header=c.Comment("Wait for completion of `%s`" % threads.name),
            body=callback([
                While(CondEq(FieldFromComposite(sdata._field_flag, sdata[d]), 2)),
                DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 0),
                Call(FieldFromComposite('join', threads[d]))
            ])
        )

        return threadswait

    def __make_activate_thread(self, threads, sdata, sync_ops):
        if threads.size == 1:
            d = threads.index
        else:
            d = Symbol(name=self.sregistry.make_name(prefix=threads.index.name))

        sync_locks = [s for s in sync_ops if s.is_SyncLock]
        condition = Or(*([CondNe(s.handle, 2) for s in sync_locks] +
                         [CondNe(FieldFromComposite(sdata._field_flag, sdata[d]), 1)]))

        if threads.size == 1:
            activation = [BusyWait(condition)]
        else:
            activation = [DummyExpr(d, 0),
                          BusyWait(condition, DummyExpr(d, (d + 1) % threads.size))]

        activation.extend([DummyExpr(FieldFromComposite(i.name, sdata[d]), i)
                           for i in sdata.fields])
        activation.extend([DummyExpr(s.handle, 0) for s in sync_locks])
        activation.append(DummyExpr(FieldFromComposite(sdata._field_flag, sdata[d]), 2))
        activation = List(
            header=[c.Line(), c.Comment("Activate `%s`" % threads.name)],
            body=activation,
            footer=c.Line()
        )

        return activation

    def _make_waitlock(self, iet, sync_ops, *args):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.target.name for s in sync_ops)),
            body=BusyWait(Or(*[CondEq(s.handle, 0) for s in sync_ops])),
            footer=c.Line()
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet

    def _make_withlock(self, iet, sync_ops, pieces, root):
        locks = sorted({s.lock for s in sync_ops}, key=lambda i: i.name)

        threads = self.__make_threads(value=min(i.size for i in locks))

        preactions = []
        postactions = []
        for s in sync_ops:
            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]

            preactions.append(List(body=[
                BlankLine,
                List(header=self._P._map_update_wait_host(s.target, imask,
                                                          SharedData._field_id)),
                DummyExpr(s.handle, 1)
            ]))
            postactions.append(DummyExpr(s.handle, 2))
        preactions.append(BlankLine)
        postactions.insert(0, BlankLine)

        # Turn `iet` into an ElementalFunction so that it can be
        # executed asynchronously by `threadhost`
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = List(body=tuple(preactions) + iet.body + tuple(postactions))
        tfunc = make_tfunc(name, body, root, threads, self.sregistry)
        pieces.funcs.append(tfunc)

        sdata = tfunc.sdata

        # Schedule computation to the first available thread
        iet = self.__make_activate_thread(threads, sdata, sync_ops)

        # Initialize the locks
        for i in locks:
            values = np.full(i.shape, 2, dtype=np.int32).tolist()
            pieces.init.append(LocalExpression(DummyEq(i, ListInitializer(values))))

        # Fire up the threads
        pieces.init.append(self.__make_init_threads(threads, tfunc, pieces))
        pieces.threads.append(threads)

        # Final wait before jumping back to Python land
        pieces.finalize.append(self.__make_finalize_threads(threads, sdata))

        return iet

    def _make_fetchwait(self, iet, sync_ops, *args):
        # Construct fetches
        fetches = []
        for s in sync_ops:
            fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetches.append(self._P._map_to(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=fetches, body=iet)

        return iet

    def _make_fetchwaitprefetch(self, iet, sync_ops, pieces, root):
        threads = self.__make_threads()

        fetches = []
        prefetches = []
        presents = []
        for s in sync_ops:
            if s.direction is Forward:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                fsize = s.function._C_get_field(FULL, s.dim).size
                fc_cond = fc + (s.size - 1) < fsize
                pfc = s.fetch + 1
                pfc_cond = pfc + (s.size - 1) < fsize
            else:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_max)
                fc_cond = fc >= 0
                pfc = s.fetch - 1
                pfc_cond = pfc >= 0

            # Construct fetch IET
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetch = List(header=self._P._map_to(s.function, imask))
            fetches.append(Conditional(fc_cond, fetch))

            # Construct present clauses
            imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            presents.extend(as_list(self._P._map_present(s.function, imask)))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            prefetch = List(header=self._P._map_to_wait(s.function, imask,
                                                        SharedData._field_id))
            prefetches.append(Conditional(pfc_cond, prefetch))

        functions = filter_ordered(s.function for s in sync_ops)
        casts = [PointerCast(f) for f in functions]

        # Turn init IET into a Callable
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=casts + fetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        func = Callable(name, body, 'void', parameters, 'static')
        pieces.funcs.append(func)

        # Perform initial fetch by the main thread
        pieces.init.append(List(
            header=c.Comment("Initialize data stream for `%s`" % threads.name),
            body=[Call(name, func.parameters), BlankLine]
        ))

        # Turn prefetch IET into a threaded Callable
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(header=c.Line(), body=casts + prefetches)
        tfunc = make_tfunc(name, body, root, threads, self.sregistry)
        pieces.funcs.append(tfunc)

        sdata = tfunc.sdata

        # Glue together all the IET pieces, including the activation bits
        iet = List(body=[
            BlankLine,
            BusyWait(CondNe(FieldFromComposite(sdata._field_flag,
                                               sdata[threads.index]), 1)),
            List(header=presents),
            iet,
            self.__make_activate_thread(threads, sdata, sync_ops)
        ])

        # Fire up the threads
        pieces.init.append(self.__make_init_threads(threads, tfunc, pieces))
        pieces.threads.append(threads)

        # Final wait before jumping back to Python land
        pieces.finalize.append(self.__make_finalize_threads(threads, sdata))

        return iet

    def _make_delete(self, iet, sync_ops, *args):
        # Construct deletion clauses
        deletions = []
        for s in sync_ops:
            if s.dim.is_Custom:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                imask = [(fc, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            else:
                imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            deletions.append(self._P._map_delete(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=c.Line(), body=iet, footer=[c.Line()] + deletions)

        return iet

    @iet_pass
    def process(self, iet):

        def key(s):
            # The SyncOps are to be processed in the following order
            return [WaitLock, WithLock, Delete, FetchWait, FetchWaitPrefetch].index(s)

        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            FetchWait: self._make_fetchwait,
            FetchWaitPrefetch: self._make_fetchwaitprefetch,
            Delete: self._make_delete
        }

        sync_spots = FindNodes(SyncSpot).visit(iet)

        if not sync_spots:
            return iet, {}

        pieces = namedtuple('Pieces', 'init finalize funcs threads')([], [], [], [])

        subs = {}
        for n in sync_spots:
            mapper = as_mapper(n.sync_ops, lambda i: type(i))
            for _type in sorted(mapper, key=key):
                subs[n] = callbacks[_type](subs.get(n, n), mapper[_type], pieces, iet)

        iet = Transformer(subs).visit(iet)

        # Add initialization and finalization code
        init = List(body=pieces.init, footer=c.Line())
        finalize = List(header=c.Line(), body=pieces.finalize)
        iet = iet._rebuild(body=(init,) + iet.body + (finalize,))

        return iet, {'efuncs': pieces.funcs,
                     'includes': ['thread'],
                     'args': [i.size for i in pieces.threads if not is_integer(i.size)]}


# Utils

class BusyWait(While):
    pass
