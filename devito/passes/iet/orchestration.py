from collections import namedtuple

import cgen as c
from sympy import Or

from devito.data import FULL
from devito.ir.iet import (Call, Callable, Conditional, List, SyncSpot, FindNodes,
                           Transformer, BlankLine, BusyWait, Pragma, PragmaTransfer,
                           DummyExpr, derive_parameters, make_thread_ctx)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe, FieldFromComposite
from devito.tools import as_mapper, filter_ordered, filter_sorted, flatten, is_integer
from devito.types import (WaitLock, WithLock, FetchUpdate, FetchPrefetch,
                          PrefetchUpdate, WaitPrefetch, Delete, SharedData)

__init__ = ['Orchestrator']


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

        preactions = [BlankLine]
        for s in sync_ops:
            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]
            update = PragmaTransfer(self.lang._map_update_host_async, s.target,
                                    imask=imask, queueid=SharedData._field_id)
            preactions.append(update)
        wait = self.lang._map_wait(SharedData._field_id)
        if wait is not None:
            preactions.append(Pragma(wait))
        preactions.extend([DummyExpr(s.handle, 1) for s in sync_ops])
        preactions.append(BlankLine)

        postactions = [BlankLine]
        postactions.extend([DummyExpr(s.handle, 2) for s in sync_ops])

        # Turn `iet` into a ThreadFunction so that it can be executed
        # asynchronously by a pthread in the `npthreads` pool
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = List(body=tuple(preactions) + iet.body + tuple(postactions))
        tctx = make_thread_ctx(name, body, root, npthreads, sync_ops, self.sregistry)
        pieces.funcs.extend(tctx.funcs)

        # Schedule computation to the first available thread
        iet = tctx.activate

        # Fire up the threads
        pieces.init.append(tctx.init)

        # Final wait before jumping back to Python land
        pieces.finalize.append(tctx.finalize)

        # Keep track of created objects
        pieces.objs.add(sync_ops, tctx.sdata, tctx.threads)

        return iet

    def _make_fetchupdate(self, iet, sync_ops, pieces, *args):
        # Construct fetches
        postactions = []
        for s in sync_ops:
            # The condition is already encoded in `iet` with a Conditional,
            # which stems from the originating Cluster's guards
            assert s.fcond is None

            imask = [(s.tstore, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            postactions.append(
                PragmaTransfer(self.lang._map_update_device, s.target, imask=imask)
            )

        # Turn init IET into a Callable
        functions = filter_ordered(flatten([(s.target, s.function) for s in sync_ops]))
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=iet.body + tuple(postactions))
        parameters = filter_sorted(functions + derive_parameters(body))
        func = Callable(name, body, 'void', parameters, 'static')
        pieces.funcs.append(func)

        # Perform initial fetch by the main thread
        iet = List(
            header=c.Comment("Initialize data stream"),
            body=Call(name, parameters)
        )

        return iet

    def _make_prefetchupdate(self, iet, sync_ops, pieces, root):
        fid = SharedData._field_id

        postactions = [BlankLine]
        for s in sync_ops:
            # The condition is already encoded in `iet` with a Conditional,
            # which stems from the originating Cluster's guards
            assert s.pcond is None

            imask = [(s.tstore, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            postactions.append(PragmaTransfer(self.lang._map_update_device_async,
                                              s.target, imask=imask, queueid=fid))
        wait = self.lang._map_wait(fid)
        if wait is not None:
            postactions.append(Pragma(wait))

        # Turn prefetch IET into a ThreadFunction
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(body=iet.body + tuple(postactions))
        tctx = make_thread_ctx(name, body, root, None, sync_ops, self.sregistry)
        pieces.funcs.extend(tctx.funcs)

        # The IET degenerates to the threads activation logic
        iet = tctx.activate

        # Fire up the threads
        pieces.init.append(tctx.init)

        # Final wait before jumping back to Python land
        pieces.finalize.append(tctx.finalize)

        # Keep track of created objects
        pieces.objs.add(sync_ops, tctx.sdata, tctx.threads)

        return iet

    def _make_waitprefetch(self, iet, sync_ops, pieces, *args):
        ff = SharedData._field_flag

        waits = []
        objs = filter_ordered(pieces.objs.get(s) for s in sync_ops)
        for sdata, threads in objs:
            wait = BusyWait(CondNe(FieldFromComposite(ff, sdata[threads.index]), 1))
            waits.append(wait)

        iet = List(
            header=c.Comment("Wait for the arrival of prefetched data"),
            body=waits + [BlankLine, iet]
        )

        return iet

    def _make_fetchprefetch(self, iet, sync_ops, pieces, root):
        fid = SharedData._field_id

        fetches = []
        prefetches = []
        presents = []
        for s in sync_ops:
            f = s.function
            dimensions = s.dimensions
            fc = s.fetch
            ifc = s.ifetch
            pfc = s.pfetch
            fcond = s.fcond
            pcond = s.pcond

            # Construct init IET
            imask = [(ifc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            fetch = PragmaTransfer(self.lang._map_to, f, imask=imask)
            fetches.append(Conditional(fcond, fetch))

            # Construct present clauses
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            presents.append(PragmaTransfer(self.lang._map_present, f, imask=imask))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            prefetch = PragmaTransfer(self.lang._map_to_wait, f, imask=imask, queueid=fid)
            prefetches.append(Conditional(pcond, prefetch))

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
                                               sdata[threads.index]), 1))
        ] + presents + [
            iet,
            tctx.activate
        ])

        # Fire up the threads
        pieces.init.append(tctx.init)

        # Final wait before jumping back to Python land
        pieces.finalize.append(tctx.finalize)

        # Keep track of created objects
        pieces.objs.add(sync_ops, sdata, threads)

        return iet

    def _make_delete(self, iet, sync_ops, *args):
        # Construct deletion clauses
        deletions = []
        for s in sync_ops:
            dimensions = s.dimensions
            fc = s.fetch

            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            deletions.append(
                PragmaTransfer(self.lang._map_delete, s.function, imask=imask)
            )

        # Glue together the new IET pieces
        iet = List(header=c.Line(), body=[iet, BlankLine] + deletions)

        return iet

    @iet_pass
    def process(self, iet):
        sync_spots = FindNodes(SyncSpot).visit(iet)
        if not sync_spots:
            return iet, {}

        def key(s):
            # The SyncOps are to be processed in the following order
            return [WaitLock,
                    WithLock,
                    Delete,
                    FetchUpdate,
                    FetchPrefetch,
                    PrefetchUpdate,
                    WaitPrefetch].index(s)

        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            Delete: self._make_delete,
            FetchUpdate: self._make_fetchupdate,
            FetchPrefetch: self._make_fetchprefetch,
            PrefetchUpdate: self._make_prefetchupdate
        }
        postponed_callbacks = {
            WaitPrefetch: self._make_waitprefetch
        }
        all_callbacks = [callbacks, postponed_callbacks]

        pieces = namedtuple('Pieces', 'init finalize funcs objs')([], [], [], Objs())

        # The processing is a two-step procedure; first, we apply the `callbacks`;
        # then, the `postponed_callbacks`, as these depend on objects produced by the
        # `callbacks`
        subs = {}
        for cbks in all_callbacks:
            for n in sync_spots:
                mapper = as_mapper(n.sync_ops, lambda i: type(i))
                for _type in sorted(mapper, key=key):
                    try:
                        subs[n] = cbks[_type](subs.get(n, n), mapper[_type], pieces, iet)
                    except KeyError:
                        pass

        iet = Transformer(subs).visit(iet)

        # Add initialization and finalization code
        init = List(body=pieces.init, footer=c.Line())
        finalize = List(header=c.Line(), body=pieces.finalize)
        body = iet.body._rebuild(body=(init,) + iet.body.body + (finalize,))
        iet = iet._rebuild(body=body)

        return iet, {
            'efuncs': pieces.funcs,
            'includes': ['pthread.h'],
            'args': [i.size for i in pieces.objs.threads if not is_integer(i.size)]
        }


# Utils

class Objs(object):

    def __init__(self):
        self.data = {}

    def __repr__(self):
        return self.data.__repr__()

    @classmethod
    def _askey(cls, sync_op):
        if sync_op.is_SyncLock:
            return sync_op.target
        else:
            return (sync_op.target, sync_op.function)

    def add(self, sync_ops, sdata, threads):
        for s in sync_ops:
            key = self._askey(s)
            self.data[key] = (sdata, threads)

    def get(self, sync_op):
        key = self._askey(sync_op)
        return self.data[key]

    @property
    def threads(self):
        return [v for _, v in self.data.values()]
