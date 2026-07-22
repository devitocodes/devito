from collections import defaultdict, namedtuple
from contextlib import suppress
from functools import cached_property, singledispatch

from sympy import Or

from devito.exceptions import CompilationError
from devito.ir.iet import (
    AsyncCall, AsyncCallable, BlankLine, Block, BusyWait, Call, Callable, Conditional,
    DummyExpr, List, SyncSpot, Transformer, derive_parameters, make_callable
)
from devito.ir.iet.visitors import LazyVisitor
from devito.ir.support import (
    InitArray, PrefetchUpdate, ReleaseLock, SnapIn, SnapOut, SyncArray, WaitLock, WithLock
)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe
from devito.tools import as_mapper
from devito.types import HostLayer

__all__ = ['Orchestrator']


Task = namedtuple('Task', 'spot guard sync_ops')
TaskGroupKey = namedtuple('TaskGroupKey', 'iteration gid optype in_snapshot')


class Orchestrator:

    """
    Lower synchronization nodes for efficient host-device asynchronous computation.
    """

    langbb = LangBB
    """
    The language used to implement host-device data movements.
    """

    def __init__(self, sregistry=None, options=None, **kwargs):
        self.sregistry = sregistry
        self.npthreads = (options or {}).get('npthreads')

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

        if isinstance(iet, SyncSpot) and not iet.body:
            iet = List()

        iet = List(body=[Call(name, efunc.parameters)] + [iet])

        return iet, [efunc]

    def _make_withlock(self, iet, sync_ops, layer, wrap=True):
        return self._make_async_task(withlock, iet, sync_ops, layer, wrap)

    def _make_async_task(self, callback, iet, sync_ops, layer, wrap):
        body, prefix = callback(layer, iet, sync_ops, self.langbb, self.sregistry)

        if not wrap:
            return body, prefix

        return self._make_async_callable(body, prefix)

    def _make_async_callable(self, body, prefix):
        name = self.sregistry.make_name(prefix=prefix)
        body = List(body=body)
        parameters = derive_parameters(body, ordering='canonical')
        efunc = AsyncCallable(name, body, parameters=parameters)

        return AsyncCall(name, efunc.parameters), [efunc]

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
        with suppress(NotImplementedError):
            body.extend([
                self.langbb._map_update_device(s.target, s.imask, qid=qid)
                for s in sync_ops
            ])
        iet = List(body=body)

        return iet, []

    def _make_prefetchupdate(self, iet, sync_ops, layer, wrap=True):
        return self._make_async_task(prefetchupdate, iet, sync_ops, layer, wrap)

    @iet_pass
    def process(self, iet):
        # The SyncOps are to be processed in a given order
        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            SyncArray: self._make_syncarray,
            InitArray: self._make_initarray,
            SnapOut: self._make_snapout,
            SnapIn: self._make_snapin,
            PrefetchUpdate: self._make_prefetchupdate,
            ReleaseLock: self._make_releaselock,
            AsyncCallable: self._make_async_callable
        }

        # Collect the task groups to lower atomically, if any
        task_groups = CollectTasks().visit(iet) if self.npthreads else ()

        # Lower the SyncSpots in a single bottom-up traversal, atomically lowering
        lowerer = LowerSyncSpots(callbacks, task_groups)
        iet = lowerer.visit(iet)

        return iet, {'efuncs': lowerer.efuncs}


class CollectTasks(LazyVisitor):

    """Collect and group compatible asynchronous SyncSpots."""

    def _post_visit(self, ret):
        groups = defaultdict(list)
        anchors = {}
        for key, task, anchor in ret:
            groups[key].append(task)
            # Activate the group after its last task in program order
            anchors[key] = anchor
        return tuple((tasks, anchors[key], key.optype)
                     for key, tasks in groups.items())

    def visit_Iteration(self, o, **kwargs):
        kwargs['iteration'] = o
        yield from self._visit(o.children, **kwargs)

    def visit_Conditional(self, o, **kwargs):
        kwargs['condition'] = o
        yield from self._visit(o.children, **kwargs)

    def visit_SyncSpot(self, o, iteration=None, condition=None,
                       in_snapshot=False):
        if iteration is not None:
            syncs = as_mapper(o.sync_ops, type)

            optypes = set(syncs)
            for optype in (PrefetchUpdate, WithLock):
                if optypes != {optype, ReleaseLock}:
                    continue

                guard = None
                if condition is not None:
                    # Task SyncSpots inherit a guard without an `else` branch
                    # from the originating Cluster
                    assert not condition.else_body
                    guard = condition.condition

                sync_ops = syncs[optype]

                gid, = {i.gid for i in sync_ops}
                if gid is not None:
                    task = Task(o, guard, sync_ops)
                    key = TaskGroupKey(iteration, gid, optype, in_snapshot)
                    yield key, task, condition or o

                break

        if any(isinstance(i, SnapOut) for i in o.sync_ops):
            # Do not mix composite tasks with other compatible groups
            in_snapshot = True

        yield from self._visit(
            o.children, iteration=iteration, condition=condition,
            in_snapshot=in_snapshot
        )


class LowerSyncSpots(Transformer):

    """
    Lower `SyncSpot`s in a single bottom-up traversal.

    Compatible task `SyncSpot`s are lowered atomically into one asynchronous
    callable when their final anchor is reached. All other `SyncSpot`s are
    lowered individually.

    Parameters
    ----------
    callbacks : mapping
        The callbacks used to lower `SyncOp`s and create asynchronous callables.
    task_groups : iterable
        The task groups to lower atomically.
    """

    def __init__(self, callbacks, task_groups):
        super().__init__({})

        self._callbacks = callbacks

        self._task_bodies = {}
        self._anchors = defaultdict(list)
        for tasks, anchor, optype in task_groups:
            self._task_bodies.update((task.spot, None) for task in tasks)
            self._anchors[anchor].append((tasks, optype))

        self._efuncs = []

    @property
    def efuncs(self):
        return self._efuncs

    @cached_property
    def _priority(self):
        return tuple(self._callbacks).index

    def visit_Node(self, o, **kwargs):
        iet = super().visit_Node(o, **kwargs)
        return self._lower_task_groups(o, iet)

    def visit_SyncSpot(self, o, **kwargs):
        body = self._visit(o.body, **kwargs)

        if o in self._task_bodies:
            # Retain the task body until the group's final anchor is visited
            self._task_bodies[o] = body
            releases = tuple(i for i in o.sync_ops
                             if isinstance(i, ReleaseLock))
            iet = self._lower(SyncSpot(releases), releases)
        else:
            iet = self._lower(o._rebuild(body=body), o.sync_ops)

        return self._lower_task_groups(o, iet)

    def _lower(self, iet, sync_ops):
        mapper = as_mapper(sync_ops, type)
        for optype in sorted(mapper, key=self._priority):
            sync_ops = mapper[optype]
            layer = infer_sync_layer(sync_ops)

            iet, efuncs = self._callbacks[optype](iet, sync_ops, layer)
            self._efuncs.extend(efuncs)

        return iet

    def _lower_task_groups(self, o, iet):
        """
        Lower the task groups activated after `o`.

        Examples
        --------
        A group of three guarded tasks is lowered schematically to::

            AsyncCall(
                if guard0: task0
                if guard1: task1
                if guard2: task2
            )

        The call is inserted after `o`; each task retains its own guard.
        """
        calls = []
        for tasks, optype in self._anchors.get(o, ()):
            sync_ops = tuple(op for task in tasks for op in task.sync_ops)
            layer = infer_sync_layer(sync_ops)

            body = []
            for task in tasks:
                task_body, prefix = self._callbacks[optype](
                    List(body=self._task_bodies.pop(task.spot)),
                    task.sync_ops, layer, wrap=False
                )
                if task.guard is None:
                    # Preserve the local scope of an unguarded task body
                    scope = Block(body=task_body)
                else:
                    scope = Conditional(task.guard, task_body)
                body.append(scope)

            call, efuncs = self._callbacks[AsyncCallable](body, prefix)
            calls.append(call)
            self._efuncs.extend(efuncs)

        if calls:
            iet = List(body=(iet, *calls))

        return iet


# Task handlers

layer_host = HostLayer('host')


@singledispatch
def infer_layer(f):
    """
    The layer of the node storage hierarchy in which a Function is found.
    """
    return layer_host


def infer_sync_layer(sync_ops):
    """
    Infer the unique storage layer used by a sequence of SyncOps.
    """
    layers = {infer_layer(i.function) for i in sync_ops}
    if len(layers) != 1:
        found = ', '.join(sorted(str(i) for i in layers)) or 'none'
        raise CompilationError(
            "Expected synchronization operations to use exactly one storage "
            f"layer, but found: {found}"
        )
    return layers.pop()


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

        name = f'copy_to_{layer.suffix}'
    except NotImplementedError:
        # A non-device backend
        body = []
        name = f'copy_from_{layer.suffix}'

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

        name = f'prefetch_from_{layer.suffix}'
    except NotImplementedError:
        body = []
        name = f'prefetch_to_{layer.suffix}'

    body.extend([DummyExpr(s.handle, 2) for s in sync_ops])

    body = iet.body + (BlankLine,) + tuple(body)

    return body, name
