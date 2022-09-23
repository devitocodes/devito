from collections import OrderedDict
from functools import singledispatch

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

    def _lower_tasks(self, sync_ops, func):
        """
        Utility method to lower a list of SyncOps into an IET implementing
        the core of the task.
        """
        actions = []
        prefixes = set()
        for s in sync_ops:
            v, prefix = func(s.function, s, self.lang, self.sregistry)
            actions.extend(v)
            prefixes.add(prefix)

        # Only homogeneous tasks supported for now
        assert len(prefixes) == 1
        prefix = prefixes.pop()

        return actions, prefix

    def _make_waitlock(self, iet, sync_ops):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.target.name for s in sync_ops)),
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
        preactions, prefix = self._lower_tasks(sync_ops, task_withlock)

        preactions.extend([DummyExpr(s.handle, 1) for s in sync_ops])
        preactions.append(BlankLine)

        postactions = [BlankLine]
        postactions.extend([DummyExpr(s.handle, 2) for s in sync_ops])

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=tuple(preactions) + iet.body + tuple(postactions))
        parameters = derive_parameters(body)
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    def _make_fetchupdate(self, iet, sync_ops):
        postactions, prefix = self._lower_tasks(sync_ops, task_fetchupdate)

        # Turn init IET into a Callable
        name = self.sregistry.make_name(prefix=prefix)
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
        postactions, prefix = self._lower_tasks(sync_ops, task_prefetchupdate)

        postactions.append(BlankLine)
        postactions.extend([DummyExpr(s.handle, 2) for s in sync_ops])

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
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


# Task handlers


@singledispatch
def task_withlock(function, s, lang, sregistry):
    """
    The only known handler of a WithLock task is the copy from device
    to host. However, we make it single-dispatchable for foreign modules.
    """
    qid = lang.AsyncQueue(name='qid')

    actions = [lang._map_update_host_async(s.target, s.imask, qid)]
    if lang._map_wait is not None:
        actions.append(lang._map_wait(qid))

    return actions, 'copy_device_to_host'


@singledispatch
def task_fetchupdate(function, s, lang, sregistry):
    """
    The only known handler of a FetchUpdate task is the synchronous copy from
    host to device. However, we make it single-dispatchable for foreign modules.
    """
    actions = [lang._map_update_device(s.target, s.imask)]

    return actions, 'init_device'


@singledispatch
def task_prefetchupdate(function, s, lang, sregistry):
    """
    The only known handler of a PrefetchUpdate task is the asynchronous copy
    from device to host. However, we make it single-dispatchable for foreign modules.
    """
    qid = lang.AsyncQueue(name='qid')

    actions = [lang._map_update_device_async(s.target, s.imask, qid)]
    if lang._map_wait is not None:
        actions.append(lang._map_wait(qid))

    return actions, 'prefetch_host_to_device'
