from collections import OrderedDict, defaultdict, namedtuple
from contextlib import suppress
from functools import cached_property, singledispatch

from sympy import Or

from devito.exceptions import CompilationError
from devito.ir.iet import (
    AsyncCall, AsyncCallable, BlankLine, Block, BusyWait, Call, Callable, Conditional,
    DummyExpr, FindNodes, List, SyncSpot, Transformer, derive_parameters, make_callable
)
from devito.ir.iet.visitors import LazyVisitor
from devito.ir.support import (
    InitArray, PrefetchUpdate, ReleaseLock, SnapIn, SnapOut, SyncArray, WaitLock, WithLock
)
from devito.passes.iet.engine import iet_pass
from devito.passes.iet.langbase import LangBB
from devito.symbolics import CondEq, CondNe
from devito.tools import DAG, as_mapper
from devito.types import HostLayer

__all__ = ['Orchestrator']


Task = namedtuple('Task', 'spot guard sync_ops releases')


class SyncSpotRegion(List):

    """A sequence of SyncSpots treated as one unit during orchestration."""

    def __init__(self, body):
        super().__init__(body=body)
        assert self.body and all(isinstance(i, SyncSpot) for i in self.body)

    @property
    def sync_spots(self):
        return self.body

    @cached_property
    def optype(self):
        """
        The type of the synchronization operation in this region.
        """
        optypes = {type(op) for spot in self.sync_spots for op in spot.sync_ops}
        assert len(optypes) == 1, (
            "Expected a SyncSpotRegion to contain exactly one type of "
            "synchronization operation"
        )
        return optypes.pop()


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

    def _fuse_tasks(self, iet):
        """
        Group compatible task SyncSpots into SyncSpotRegions.
        """
        if self.npthreads is None:
            return iet

        insertions = defaultdict(list)
        subs1 = {}

        for tasks, anchor in CollectTasks().visit(iet):
            spots = []
            for task in tasks:
                if task.guard is None:
                    # Preserve a local scope when there is no Conditional
                    scope = Block(body=task.spot.body)
                else:
                    scope = Conditional(task.guard, task.spot.body)
                spots.append(SyncSpot(task.sync_ops, body=scope))

            insertions[anchor].append(SyncSpotRegion(spots))
            subs1.update({task.spot: SyncSpot(task.releases) for task in tasks})

        if not subs1:
            return iet

        # These substitutions cannot be merged because a Transformer does not
        # revisit a replacement, while task spots may be nested below an anchor
        subs0 = {anchor: List(body=(anchor, *regions))
                 for anchor, regions in insertions.items()}

        iet = Transformer(subs0).visit(iet)
        iet = Transformer(subs1).visit(iet)

        return iet

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

    def _make_withlock(self, iet, sync_ops, layer):
        body, prefix = withlock(layer, iet, sync_ops, self.langbb, self.sregistry)

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

    def _make_prefetchupdate(self, iet, sync_ops, layer):
        body, prefix = prefetchupdate(layer, iet, sync_ops, self.langbb, self.sregistry)

        return self._make_async_callable(body, prefix)

    def _make_region(self, iet):
        """Lower a SyncSpotRegion into one asynchronous callable."""
        sync_ops = tuple(op for spot in iet.sync_spots
                         for op in spot.sync_ops)
        callback = {
            PrefetchUpdate: prefetchupdate,
            WithLock: withlock
        }[iet.optype]

        layer = infer_sync_layer(sync_ops)

        body = []
        for spot in iet.sync_spots:
            scope, = spot.body
            if isinstance(scope, Conditional):
                task_body = scope.then_body
            else:
                assert isinstance(scope, Block)
                task_body = scope.body

            task_body, prefix = callback(
                layer, List(body=task_body), spot.sync_ops, self.langbb,
                self.sregistry
            )
            body.append(scope._rebuild(task_body))

        return self._make_async_callable(body, prefix)

    @iet_pass
    def process(self, iet):
        # Group compatible task SyncSpots into SyncSpotRegions, if requested
        # by the user
        iet = self._fuse_tasks(iet)

        # Lower regions first so their member SyncSpots are not processed
        # independently by the generic SyncSpot lowering below
        efuncs = []
        subs = {}
        for region in FindNodes(SyncSpotRegion).visit(iet):
            call, efuncs1 = self._make_region(region)
            subs[region] = call
            efuncs.extend(efuncs1)

        iet = Transformer(subs).visit(iet)

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
        key = tuple(callbacks).index

        # The SyncSpots may be nested, so we compute a topological ordering
        # so that they are processed in a bottom-up fashion. This is necessary
        # because e.g. an inner SyncSpot may generate new objects (e.g., a new
        # Queue), which in turn must be visible to the outer SyncSpot to
        # generate the correct parameters list
        while True:
            sync_spots = FindNodes(SyncSpot).visit(iet)
            if not sync_spots:
                break

            n0 = ordered(sync_spots).pop(0)
            mapper = as_mapper(n0.sync_ops, type)

            subs = {}
            for t in sorted(mapper, key=key):
                sync_ops = mapper[t]

                layer = infer_sync_layer(sync_ops)

                n1, v = callbacks[t](subs.get(n0, n0), sync_ops, layer)

                subs[n0] = n1
                efuncs.extend(v)

            iet = Transformer(subs).visit(iet)

        return iet, {'efuncs': efuncs}


def ordered(sync_spots):
    dag = DAG(nodes=sync_spots)
    for n0 in sync_spots:
        for n1 in FindNodes(SyncSpot).visit(n0.body):
            dag.add_edge(n1, n0)

    return dag.topological_sort()


class CollectTasks(LazyVisitor):

    """Collect and group compatible asynchronous SyncSpots."""

    def _post_visit(self, ret):
        groups = defaultdict(list)
        anchors = {}
        for key, task, anchor in ret:
            groups[key].append(task)
            # Activate the group after its last task in program order
            anchors[key] = anchor
        return tuple((tasks, anchors[key]) for key, tasks in groups.items())

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
                    task = Task(o, guard, sync_ops, syncs[ReleaseLock])
                    key = (iteration, gid, optype, in_snapshot)
                    yield key, task, condition or o

                break

        if any(isinstance(i, SnapOut) for i in o.sync_ops):
            # Do not mix composite tasks with other compatible groups
            in_snapshot = True

        yield from self._visit(
            o.children, iteration=iteration, condition=condition,
            in_snapshot=in_snapshot
        )


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
