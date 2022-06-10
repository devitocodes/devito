from collections import namedtuple

import cgen as c
from sympy import Or

from devito.data import FULL
from devito.ir.iet import (Call, Callable, Conditional, List, SyncSpot, FindNodes,
                           Transformer, BlankLine, BusyWait, DummyExpr, AsyncCall,
                           AsyncCallable, derive_parameters)
from devito.ir.support import (WaitLock, WithLock, ReleaseLock, FetchUpdate,
                               FetchPrefetch, PrefetchUpdate, WaitPrefetch, Delete)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe, FieldFromComposite
from devito.tools import as_mapper, filter_ordered, filter_sorted, flatten
from devito.types import SharedData, QueueID

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

    def _make_releaselock(self, iet, sync_ops, *args):
        preactions = []
        preactions.append(BusyWait(Or(*[CondNe(s.handle, 2) for s in sync_ops])))
        preactions.extend(DummyExpr(s.handle, 0) for s in sync_ops)

        iet = List(
            header=c.Comment("Release lock(s) as soon as possible"),
            body=preactions + [iet]
        )

        return iet

    def _make_withlock(self, iet, sync_ops, pieces, root):
        qid = QueueID()

        preactions = []
        for s in sync_ops:
            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]
            preactions.append(self.lang._map_update_host_async(s.target, imask, qid))
        if self.lang._map_wait is not None:
            preactions.append(self.lang._map_wait(qid))
        preactions.extend([DummyExpr(s.handle, 1) for s in sync_ops])
        preactions.append(BlankLine)

        postactions = [BlankLine]
        postactions.extend([DummyExpr(s.handle, 2) for s in sync_ops])

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = List(body=tuple(preactions) + iet.body + tuple(postactions))
        func = AsyncCallable(name, body)
        pieces.funcs.append(func)

        # The corresponding AsyncCall
        iet = AsyncCall(name, func.parameters)

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
            postactions.append(self.lang._map_update_device(s.target, imask))

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
        qid = QueueID()

        postactions = [BlankLine]
        for s in sync_ops:
            # The condition is already encoded in `iet` with a Conditional,
            # which stems from the originating Cluster's guards
            assert s.pcond is None

            imask = [(s.tstore, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            postactions.append(self.lang._map_update_device_async(s.target, imask, qid))
        if self.lang._map_wait is not None:
            postactions.append(self.lang._map_wait(qid))

        # Turn prefetch IET into a AsyncCallable
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(body=iet.body + tuple(postactions))
        tctx = make_thread_ctx(name, body, root, None, self.sregistry)
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
        qid = QueueID()

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
            fetch = self.lang._map_to(f, imask)
            fetches.append(Conditional(fcond, fetch))

            # Construct present clauses
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            presents.append(self.lang._map_present(f, imask))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            prefetch = self.lang._map_to_wait(f, imask, qid)
            prefetches.append(Conditional(pcond, prefetch))

        # Turn init IET into a Callable
        functions = filter_ordered(s.function for s in sync_ops)
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=fetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        pieces.funcs.append(Callable(name, body, 'void', parameters, 'static'))

        # Perform initial fetch by the main thread
        pieces.init.append(List(
            header=c.Comment("Initialize data stream"),
            body=[Call(name, parameters), BlankLine]
        ))

        # Turn prefetch IET into a AsyncCallable
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        func = AsyncCallable(name, prefetches)
        pieces.funcs.append(func)

        # Glue together all the IET pieces, including the activation logic
        iet = List(body=presents + [iet, AsyncCall(name, func.parameters)])

        return iet

    def _make_delete(self, iet, sync_ops, *args):
        # Construct deletion clauses
        deletions = []
        for s in sync_ops:
            dimensions = s.dimensions
            fc = s.fetch

            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in dimensions]
            deletions.append(self.lang._map_delete(s.function, imask))

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
                    ReleaseLock,
                    Delete,
                    FetchUpdate,
                    FetchPrefetch,
                    PrefetchUpdate,
                    WaitPrefetch].index(s)

        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            ReleaseLock: self._make_releaselock,
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

        # Inject initialization code
        body = iet.body._rebuild(body=tuple(pieces.init) + iet.body.body)
        iet = iet._rebuild(body=body)

        return iet, {
            'efuncs': pieces.funcs,
            'includes': ['pthread.h'],
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
