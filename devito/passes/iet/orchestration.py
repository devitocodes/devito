from collections import OrderedDict

import cgen as c
from sympy import Or

from devito.ir.iet import (Call, Callable, List, SyncSpot, FindNodes,
                           Transformer, BlankLine, BusyWait, DummyExpr, AsyncCall,
                           AsyncCallable, derive_parameters)
from devito.ir.support import (WaitLock, WithLock, ReleaseLock, FetchUpdate,
                               PrefetchUpdate)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe
from devito.tools import as_mapper
from devito.types import QueueID

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

    def _make_waitlock(self, iet, sync_ops):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.function.name for s in sync_ops)),
            body=BusyWait(Or(*[CondEq(s.handle, 0) for s in sync_ops])),
            footer=c.Line()
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet, []

    def _make_releaselock(self, iet, sync_ops):
        preactions = []
        preactions.append(BusyWait(Or(*[CondNe(s.handle, 2) for s in sync_ops])))
        preactions.extend(DummyExpr(s.handle, 0) for s in sync_ops)

        iet = List(
            header=c.Comment("Release lock(s) as soon as possible"),
            body=preactions + [iet]
        )

        return iet, []

    def _make_withlock(self, iet, sync_ops):
        qid = QueueID()

        preactions = [self.lang._map_update_host_async(s.function, s.imask, qid)
                      for s in sync_ops]
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
        parameters = derive_parameters(body)
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    def _make_fetchupdate(self, iet, sync_ops):
        postactions = [self.lang._map_update_device(s.target, s.imask)
                       for s in sync_ops]

        # Turn init IET into a Callable
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=iet.body + tuple(postactions))
        parameters = derive_parameters(body)
        efunc = Callable(name, body, 'void', parameters, 'static')

        # Perform initial fetch by the main thread
        iet = List(
            header=c.Comment("Initialize data stream"),
            body=Call(name, parameters)
        )

        return iet, [efunc]

    def _make_prefetchupdate(self, iet, sync_ops):
        qid = QueueID()

        postactions = [self.lang._map_update_device_async(s.target, s.imask, qid)
                       for s in sync_ops]
        if self.lang._map_wait is not None:
            postactions.append(self.lang._map_wait(qid))
        postactions.append(BlankLine)
        postactions.extend([DummyExpr(s.handle, 2) for s in sync_ops])

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(body=iet.body + (BlankLine,) + tuple(postactions))
        parameters = derive_parameters(body)
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    @iet_pass
    def process(self, iet):
        sync_spots = FindNodes(SyncSpot).visit(iet)
        if not sync_spots:
            return iet, {}

        callbacks = OrderedDict([
            (WaitLock, self._make_waitlock),
            (WithLock, self._make_withlock),
            (FetchUpdate, self._make_fetchupdate),
            (PrefetchUpdate, self._make_prefetchupdate),
            (ReleaseLock, self._make_releaselock),
        ])

        # The SyncOps are to be processed in a given order
        key = lambda s: list(callbacks).index(s)

        efuncs = []
        subs = {}
        for n in sync_spots:
            mapper = as_mapper(n.sync_ops, lambda i: type(i))
            for t in sorted(mapper, key=key):
                subs[n], v = callbacks[t](subs.get(n, n), mapper[t])
                efuncs.extend(v)

        iet = Transformer(subs).visit(iet)

        return iet, {'efuncs': efuncs}
