from collections import OrderedDict
from functools import singledispatch

import cgen as c
from sympy import Or

from devito.exceptions import CompilationError
from devito.ir.iet import (Call, Callable, List, SyncSpot, FindNodes,
                           Transformer, BlankLine, BusyWait, DummyExpr, AsyncCall,
                           AsyncCallable, derive_parameters)
from devito.ir.support import (WaitLock, WithLock, ReleaseLock, FetchUpdate,
                               PrefetchUpdate)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe
from devito.tools import as_mapper
from devito.types import HostLayer

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

        return iet, []

    def _make_releaselock(self, iet, sync_ops, *args):
        pre = []
        pre.append(BusyWait(Or(*[CondNe(s.handle, 2) for s in sync_ops])))
        pre.extend(DummyExpr(s.handle, 0) for s in sync_ops)

        iet = List(
            header=c.Comment("Release lock(s) as soon as possible"),
            body=pre + [iet]
        )

        return iet, []

    def _make_withlock(self, iet, sync_ops, layer):
        body, prefix = withlock(layer, iet, sync_ops, self.lang, self.sregistry)

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
        parameters = derive_parameters(body)
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    def _make_fetchupdate(self, iet, sync_ops, layer):
        body, prefix = fetchupdate(layer, iet, sync_ops, self.lang, self.sregistry)

        # Turn init IET into a Callable
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
        parameters = derive_parameters(body)
        efunc = Callable(name, body, 'void', parameters, 'static')

        # Perform initial fetch by the main thread
        iet = List(
            header=c.Comment("Initialize data stream"),
            body=Call(name, parameters)
        )

        return iet, [efunc]

    def _make_prefetchupdate(self, iet, sync_ops, layer):
        body, prefix = prefetchupdate(layer, iet, sync_ops, self.lang, self.sregistry)

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
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
                sync_ops = mapper[t]

                layers = {infer_layer(s.function) for s in sync_ops}
                if len(layers) != 1:
                    raise CompilationError("Unsupported streaming case")
                layer = layers.pop()

                subs[n], v = callbacks[t](subs.get(n, n), sync_ops, layer)
                efuncs.extend(v)

        iet = Transformer(subs).visit(iet)

        return iet, {'efuncs': efuncs}


# Task handlers

layer_host = HostLayer('host')


@singledispatch
def infer_layer(f):
    """
    The layer of the node storage hierarchy in which a Function is found.
    """
    return layer_host


@singledispatch
def withlock(layer, iet, sync_ops, lang, sregistry):
    raise NotImplementedError


@withlock.register(HostLayer)
def _(layer, iet, sync_ops, lang, sregistry):
    name = sregistry.make_name(prefix='qid')
    qid = lang.AsyncQueue(name=name)

    try:
        body = [lang._map_update_host_async(s.target, s.imask, qid)
                for s in sync_ops]
        if lang._map_wait is not None:
            body.append(lang._map_wait(qid))

        body.extend([DummyExpr(s.handle, 1) for s in sync_ops])
        body.append(BlankLine)

        name = 'copy_to_%s' % layer.suffix
    except NotImplementedError:
        # A non-device backend
        body = []
        name = 'copy_from_%s' % layer.suffix

    body.extend(list(iet.body))

    body.append(BlankLine)
    body.extend([DummyExpr(s.handle, 2) for s in sync_ops])

    return body, name


@singledispatch
def fetchupdate(layer, iet, sync_ops, lang, sregistry):
    raise NotImplementedError


@fetchupdate.register(HostLayer)
def _(layer, iet, sync_ops, lang, sregistry):
    body = list(iet.body)
    try:
        body.extend([lang._map_update_device(s.target, s.imask) for s in sync_ops])
        name = 'init_from_%s' % layer.suffix
    except NotImplementedError:
        name = 'init_to_%s' % layer.suffix

    return body, name


@singledispatch
def prefetchupdate(layer, iet, sync_ops, lang, sregistry):
    raise NotImplementedError


@prefetchupdate.register(HostLayer)
def _(layer, iet, sync_ops, lang, sregistry):
    name = sregistry.make_name(prefix='qid')
    qid = lang.AsyncQueue(name=name)

    try:
        body = [lang._map_update_device_async(s.target, s.imask, qid)
                for s in sync_ops]
        if lang._map_wait is not None:
            body.append(lang._map_wait(qid))
        body.append(BlankLine)

        name = 'prefetch_from_%s' % layer.suffix
    except NotImplementedError:
        body = []
        name = 'prefetch_to_%s' % layer.suffix

    body.extend([DummyExpr(s.handle, 2) for s in sync_ops])

    body = iet.body + (BlankLine,) + tuple(body)

    return body, name
