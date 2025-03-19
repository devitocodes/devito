from collections import OrderedDict
from functools import singledispatch

from sympy import Or

from devito.exceptions import CompilationError
from devito.ir.iet import (Call, Callable, List, SyncSpot, FindNodes, Transformer,
                           BlankLine, BusyWait, DummyExpr, AsyncCall, AsyncCallable,
                           make_callable, derive_parameters)
from devito.ir.support import (WaitLock, WithLock, ReleaseLock, InitArray,
                               SyncArray, PrefetchUpdate, SnapOut, SnapIn)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe
from devito.tools import DAG, as_mapper, as_tuple
from devito.types import HostLayer

__init__ = ['Orchestrator']


class Orchestrator:

    """
    Lower the SyncSpot in IET for efficient host-device asynchronous computation.
    """

    langbb = LangBB
    """
    The language used to implement host-device data movements.
    """

    def __init__(self, sregistry=None, **kwargs):
        self.sregistry = sregistry

    def _make_waitlock(self, iet, sync_ops, *args):
        waitloop = List(
            body=BusyWait(Or(*[CondEq(s.handle, 0) for s in sync_ops])),
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet, []

    def _make_releaselock(self, iet, sync_ops, *args):
        pre = []
        pre.append(BusyWait(Or(*[CondNe(s.handle, 2) for s in sync_ops])))
        pre.extend(DummyExpr(s.handle, 0) for s in sync_ops)

        name = self.sregistry.make_name(prefix="release_lock")
        parameters = derive_parameters(pre, ordering='canonical')
        efunc = Callable(name, pre, 'void', parameters, 'static')

        iet = List(body=[Call(name, efunc.parameters)] + [iet])

        return iet, [efunc]

    def _make_withlock(self, iet, sync_ops, layer):
        body, prefix = withlock(layer, iet, sync_ops, self.langbb, self.sregistry)

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
        parameters = derive_parameters(body, ordering='canonical')
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    def _make_callable(self, name, iet, *args):
        name = self.sregistry.make_name(prefix=name)
        efunc = make_callable(name, iet.body)

        iet = Call(name, efunc.parameters)

        return iet, [efunc]

    def _make_initarray(self, iet, *args):
        return self._make_callable('init_array', iet, *args)

    def _make_snapout(self, iet, *args):
        return self._make_callable('write_snapshot', iet, *args)

    def _make_snapin(self, iet, *args):
        return self._make_callable('read_snapshot', iet, *args)

    def _make_syncarray(self, iet, sync_ops, layer):
        try:
            qid = self.sregistry.queue0
        except AttributeError:
            qid = None

        body = list(iet.body)
        try:
            body.extend([self.langbb._map_update_device(s.target, s.imask, qid=qid)
                         for s in sync_ops])
        except NotImplementedError:
            pass
        iet = List(body=body)

        return iet, []

    def _make_prefetchupdate(self, iet, sync_ops, layer):
        body, prefix = prefetchupdate(layer, iet, sync_ops, self.langbb, self.sregistry)

        # Turn `iet` into an AsyncCallable so that subsequent passes know
        # that we're happy for this Callable to be executed asynchronously
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
        parameters = derive_parameters(body, ordering='canonical')
        efunc = AsyncCallable(name, body, parameters=parameters)

        # The corresponding AsyncCall
        iet = AsyncCall(name, efunc.parameters)

        return iet, [efunc]

    @iet_pass
    def process(self, iet):
        sync_spots = FindNodes(SyncSpot).visit(iet)
        if not sync_spots:
            return iet, {}

        # The SyncOps are to be processed in a given order
        callbacks = OrderedDict([
            (WaitLock, self._make_waitlock),
            (WithLock, self._make_withlock),
            (SyncArray, self._make_syncarray),
            (InitArray, self._make_initarray),
            (SnapOut, self._make_snapout),
            (SnapIn, self._make_snapin),
            (PrefetchUpdate, self._make_prefetchupdate),
            (ReleaseLock, self._make_releaselock),
        ])
        key = lambda s: list(callbacks).index(s)

        # The SyncSpots may be nested, so we compute a topological ordering
        # so that they are processed in a bottom-up fashion. This is necessary
        # because e.g. an inner SyncSpot may generate new objects (e.g., a new
        # Queue), which in turn must be visible to the outer SyncSpot to
        # generate the correct parameters list
        efuncs = []
        while True:
            sync_spots = FindNodes(SyncSpot).visit(iet)
            if not sync_spots:
                break

            n0 = ordered(sync_spots).pop(0)
            mapper = as_mapper(n0.sync_ops, lambda i: type(i))

            subs = {}
            for t in sorted(mapper, key=key):
                sync_ops = mapper[t]

                layers = {infer_layer(s.function) for s in sync_ops}
                if len(layers) != 1:
                    raise CompilationError("Unsupported streaming case")
                layer = layers.pop()

                n1, v = callbacks[t](subs.get(n0, n0), sync_ops, layer)

                subs[n0] = n1
                efuncs.extend(v)

            iet = Transformer(subs).visit(iet)

        return iet, {'efuncs': efuncs}


def ordered(sync_spots):
    dag = DAG(nodes=sync_spots)
    for n0 in sync_spots:
        for n1 in as_tuple(FindNodes(SyncSpot).visit(n0.body)):
            dag.add_edge(n1, n0)

    return dag.topological_sort()


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
