from itertools import groupby

from devito.ir.iet import (
    BusyWait, BusyWaitCall, EntryFunction, FindNodes, FindSymbols, Iteration, List,
    MapNodes, Section, TimedList, Transformer, retrieve_iteration_tree
)
from devito.mpi.routines import (
    ComputeCall, HaloUpdateCall, HaloUpdateList, HaloWaitCall, HaloWaitList, MPICall,
    MPIList, RemainderCall
)
from devito.operator.profiling import TimeLoopTimer
from devito.passes.iet.engine import iet_pass
from devito.types import TempArray, TempFunction, Timer

__all__ = ['instrument']


def instrument(graph, **kwargs):
    profiler = kwargs.get('profiler')
    if profiler is None:
        return

    track_subsections(graph, **kwargs)

    region = None
    if profiler._track_time_loop:
        region = time_loop_region(graph.root.body.body)

    fields = profiler.all_sections
    if region is None:
        timer = Timer(profiler.name, fields)
    else:
        timer = TimeLoopTimer(profiler.name, [*fields, 'time_loop'])
        instrument_time_loop(graph, region=region, timer=timer, **kwargs)

    instrument_sections(graph, timer=timer, **kwargs)
    sync_sections(graph, **kwargs)


def time_loop_region(body):
    """Split the host statements at the first and last time loops."""
    body = list(_host_statements(body))
    indices = []
    for n, stmt in enumerate(body):
        trees = retrieve_iteration_tree((stmt,))
        steps = (next((i for i in tree if i.dim.is_Time), None) for tree in trees)
        if any(steps):
            indices.append(n)
    if indices:
        start, stop = indices[0], indices[-1] + 1
        return body[:start], body[start:stop], body[stop:]


def _host_statements(body):
    """Unpack plain Lists, keeping guards and parallel regions intact."""
    for stmt in body:
        if type(stmt) is List and not (stmt.header or stmt.footer or stmt.inline):
            yield from _host_statements(stmt.body)
        else:
            yield stmt


@iet_pass
def instrument_time_loop(iet, region, timer, langbb, **kwargs):
    """
    Time the host code from the first stepping loop through the last.
    On asynchronous backends, wait for device work at both boundaries.

    For example, a CUDA Operator contains::

        setup(); cudaDeviceSynchronize(); START(time_loop)
        for (time = time_m; time <= time_M; ++time) { ... }
        cudaDeviceSynchronize(); STOP(time_loop, timers); teardown();
    """
    if not isinstance(iet, EntryFunction):
        return iet, {}

    before, body, after = region
    wait = langbb.get('device-wait')
    sync = langbb.get('map-wait')
    if wait is not None:
        waits = [wait]
    elif sync is not None:
        waits = [sync(q) for q in FindSymbols().visit(body)
                 if isinstance(q, langbb.AsyncQueue)]
    else:
        waits = []

    timed = TimedList(timer=timer, lname='time_loop', body=body + waits)
    body = List(body=[*before, *waits, timed, *after])
    piet = iet._rebuild(body=iet.body._rebuild(body=body))
    return piet, {'headers': _timer_headers()}


@iet_pass
def track_subsections(iet, **kwargs):
    """
    Add synchronization Sections and sub-Sections to the `profiler`. These include:

        * MPI Calls (e.g., HaloUpdateCall and HaloUpdateWait)
        * Busy-waiting loops and calls (e.g., from host-device orchestration)
        * Multi-pass implementations -- one sub-Section for each pass, within one
          macro Section
    """
    profiler = kwargs['profiler']
    sregistry = kwargs['sregistry']

    name_mapper = {
        HaloUpdateCall: 'haloupdate',
        HaloWaitCall: 'halowait',
        RemainderCall: 'remainder',
        ComputeCall: 'compute',
        HaloUpdateList: 'haloupdate',
        HaloWaitList: 'halowait',
        BusyWait: 'busywait',
        BusyWaitCall: 'busywait'
    }

    verbosity_mapper = {
        0: (),
        1: (MPIList, RemainderCall, BusyWait, BusyWaitCall),
        2: (MPICall, ComputeCall, BusyWait, BusyWaitCall),
    }

    mapper = {}

    # Enable/disable profiling of sub-Sections
    for NodeType in verbosity_mapper[profiler._verbosity]:
        for k, v in MapNodes(Section, NodeType, mode='immediate').visit(iet).items():
            if k is None and not isinstance(iet, EntryFunction):
                # Unsectioned wait helpers are timed at their `BusyWaitCall` sites.
                continue
            for i in v:
                if i in mapper:
                    continue
                name = sregistry.make_name(prefix=name_mapper[i.__class__])
                mapper[i] = Section(name, body=i, is_subsection=k is not None)
                if k is not None:
                    profiler.track_subsection(k.name, name)

    iet = Transformer(mapper).visit(iet)
    if any(not i.is_subsection for i in mapper.values()):
        profiler.analyze(iet)

    # Multi-pass implementations
    mapper = {}

    for i in FindNodes(Iteration).visit(iet):
        for k, g in groupby(i.nodes, key=lambda n: n.is_Section):
            if not k:
                continue

            candidates = []
            for i in g:
                functions = FindSymbols().visit(i)
                if any(isinstance(f, (TempArray, TempFunction)) for f in functions):
                    candidates.append(i)
                else:
                    # They must be consecutive Sections
                    break
            if len(candidates) < 2:
                continue

            name = sregistry.make_name(prefix='multipass')
            body = [i._rebuild(is_subsection=True) for i in candidates]
            section = Section(name, body=body)

            profiler.group_as_subsections(name, [i.name for i in candidates])

            mapper[candidates.pop(0)] = section
            for i in candidates:
                mapper[i] = None

    iet = Transformer(mapper).visit(iet)

    return iet, {}


@iet_pass
def instrument_sections(iet, profiler, timer, **kwargs):
    """
    Instrument the Sections of the input IET based on `profiler.sections`.
    """
    piet = profiler.instrument(iet, timer)

    if piet is iet:
        return piet, {}

    return piet, {'headers': _timer_headers()}


def _timer_headers():
    return [TimedList._start_timer_header(), TimedList._stop_timer_header()]


@iet_pass
def sync_sections(iet, langbb=None, profiler=None, **kwargs):
    """
    Wrap sections within global barriers if deemed necessary by the profiler.
    """
    try:
        sync = langbb['map-wait']
    except (KeyError, NotImplementedError):
        return iet, {}

    if not profiler._supports_async_sections:
        return iet, {}

    sections = FindNodes(Section).visit(iet)
    timed_sections = [n for s in sections for n in s.body if isinstance(n, TimedList)]
    mapper = {}
    for tl in timed_sections:
        symbols = FindSymbols().visit(tl)

        queues = [i for i in symbols if isinstance(i, langbb.AsyncQueue)]
        unnecessary = any(FindNodes((BusyWait, RemainderCall)).visit(tl))
        if queues and not unnecessary:
            waits = tuple(sync(i) for i in queues)
            mapper[tl] = tl._rebuild(body=tl.body + waits)

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}
