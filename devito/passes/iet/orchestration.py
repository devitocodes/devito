from collections import namedtuple

import cgen as c
from sympy import Or
import numpy as np

from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Callable, Conditional, List, SyncSpot, While,
                           FindNodes, LocalExpression, Transformer, BlankLine,
                           PragmaList, DummyExpr, derive_parameters, make_thread_ctx)
from devito.ir.support import Forward
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe, FieldFromComposite, ListInitializer
from devito.tools import (as_mapper, as_list, filter_ordered, filter_sorted,
                          is_integer)
from devito.types import (WaitLock, WithLock, FetchWait, FetchWaitPrefetch,
                          Delete, SharedData)

__init__ = ['Orchestrator', 'BusyWait']


class Orchestrator(object):

    """
    Lower the SyncSpot in IET for efficient host-device asynchronous computation.
    """

    lang = LangBB
    """
    The language used to implement host-device data movements.
    """

    def __init__(self, sregistry):
        self.sregistry = sregistry

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
        # Sorting for deterministic code gen
        locks = sorted({s.lock for s in sync_ops}, key=lambda i: i.name)

        # The `min` is used to pick the maximum possible degree of parallelism.
        # For example, assume there are two locks in the given `sync_ops`, `lock0(i)`
        # and `lock1(j)`. If, say, `lock0` protects 3 entries of a certain Function
        # `u`, while `lock1` protects 2 entries of the Function `v`, then there
        # will never be more than 2 threads in flight concurrently
        npthreads = min(i.size for i in locks)

        preactions = []
        postactions = []
        for s in sync_ops:
            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]

            update = List(header=self.lang._map_update_wait_host(s.target, imask,
                                                                 SharedData._field_id))
            preactions.append(List(body=[BlankLine, update, DummyExpr(s.handle, 1)]))
            postactions.append(DummyExpr(s.handle, 2))
        preactions.append(BlankLine)
        postactions.insert(0, BlankLine)

        # Turn `iet` into a ThreadFunction so that it can be executed
        # asynchronously by a pthread in the `npthreads` pool
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = List(body=tuple(preactions) + iet.body + tuple(postactions))
        tctx = make_thread_ctx(name, body, root, npthreads, sync_ops, self.sregistry)
        pieces.funcs.extend(tctx.funcs)

        # Schedule computation to the first available thread
        iet = tctx.activate

        # Initialize the locks
        for i in locks:
            values = np.full(i.shape, 2, dtype=np.int32).tolist()
            pieces.init.append(LocalExpression(DummyEq(i, ListInitializer(values))))

        # Fire up the threads
        pieces.init.append(tctx.init)
        pieces.threads.append(tctx.threads)

        # Final wait before jumping back to Python land
        pieces.finalize.append(tctx.finalize)

        return iet

    def _make_fetchwait(self, iet, sync_ops, *args):
        # Construct fetches
        fetches = []
        for s in sync_ops:
            fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetches.append(self.lang._map_to(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=fetches, body=iet)

        return iet

    def _make_fetchwaitprefetch(self, iet, sync_ops, pieces, root):
        fetches = []
        prefetches = []
        presents = []
        for s in sync_ops:
            if s.direction is Forward:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                pfc = s.fetch + 1
                fc_cond = s.next_cbk(s.dim.symbolic_min)
                pfc_cond = s.next_cbk(s.dim + 1)
            else:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_max)
                pfc = s.fetch - 1
                fc_cond = s.next_cbk(s.dim.symbolic_max)
                pfc_cond = s.next_cbk(s.dim - 1)

            # Construct init IET
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetch = PragmaList(self.lang._map_to(s.function, imask),
                               {s.function} | fc.free_symbols)
            fetches.append(Conditional(fc_cond, fetch))

            # Construct present clauses
            imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            presents.extend(as_list(self.lang._map_present(s.function, imask)))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            prefetch = PragmaList(self.lang._map_to_wait(s.function, imask,
                                                         SharedData._field_id),
                                  {s.function} | pfc.free_symbols)
            prefetches.append(Conditional(pfc_cond, prefetch))

        # Turn init IET into a Callable
        functions = filter_ordered(s.function for s in sync_ops)
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=fetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        func = Callable(name, body, 'void', parameters, 'static')
        pieces.funcs.append(func)

        # Perform initial fetch by the main thread
        pieces.init.append(List(
            header=c.Comment("Initialize data stream"),
            body=[Call(name, parameters), BlankLine]
        ))

        # Turn prefetch IET into a ThreadFunction
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(header=c.Line(), body=prefetches)
        tctx = make_thread_ctx(name, body, root, None, sync_ops, self.sregistry)
        pieces.funcs.extend(tctx.funcs)

        # Glue together all the IET pieces, including the activation logic
        sdata = tctx.sdata
        threads = tctx.threads
        iet = List(body=[
            BlankLine,
            BusyWait(CondNe(FieldFromComposite(sdata._field_flag,
                                               sdata[threads.index]), 1)),
            List(header=presents),
            iet,
            tctx.activate
        ])

        # Fire up the threads
        pieces.init.append(tctx.init)
        pieces.threads.append(threads)

        # Final wait before jumping back to Python land
        pieces.finalize.append(tctx.finalize)

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
            deletions.append(self.lang._map_delete(s.function, imask))

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
                     'includes': ['pthread.h'],
                     'args': [i.size for i in pieces.threads if not is_integer(i.size)]}


# Utils

class BusyWait(While):
    pass
