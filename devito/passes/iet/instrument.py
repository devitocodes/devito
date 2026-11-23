from itertools import groupby

from devito.ir.iet import (
    BusyWait, BusyWaitCall, EntryFunction, FindNodes, FindSymbols, Iteration, List,
    MapNodes, Section, TimedList, Transformer
)
from devito.mpi.routines import (
    ComputeCall, HaloUpdateCall, HaloUpdateList, HaloWaitCall, HaloWaitList, MPICall,
    MPIList, RemainderCall
)
from devito.passes.iet.engine import iet_pass
from devito.types import TempArray, TempFunction

__all__ = ['instrument']


def instrument(graph, **kwargs):
    profiler = kwargs.get('profiler')
    if profiler is None:
        return

    track_subsections(graph, **kwargs)

    fields = profiler.all_sections
    if profiler._track_time_loop:
        loops = FindNodes(Iteration).visit(graph.root.body.body)
        if any(i.dim.is_Time for i in loops):
            fields.append('time_loop')

    instrument_sections(graph, timer=profiler.make_timer(fields), **kwargs)
    sync_sections(graph, **kwargs)


def time_loop_region(scope):
    """Find the first and last stepping statements in their common host scope."""
    mapper = MapNodes('any', Iteration, mode='groupby').visit(scope.body) or {}
    paths = [(*parents, i) for parents, loops in mapper.items() for i in loops
             if i.dim.is_Time]
    # Only a plain List can be opened without crossing a guard or parallel region.
    for siblings in zip(*paths, strict=False):
        node = siblings[0]
        if (type(node) is List and not (node.header or node.footer) and
                all(i is node for i in siblings)):
            scope = node
            continue
        positions = [scope.body.index(i) for i in siblings]
        return scope, slice(min(positions), max(positions) + 1)


def instrument_time_loop(iet, timer, langbb):
    """
    Time the host stepping region once, including completion of its device work.

    For example, a CUDA Operator contains::

        setup(); cudaDeviceSynchronize(); START(time_loop)
        for (time = time_m; time <= time_M; ++time) { ... }
        cudaDeviceSynchronize(); STOP(time_loop, timers); teardown();
    """
    scope, span = time_loop_region(iet.body)
    body = list(scope.body)
    if (wait := langbb.get('device-wait')) is not None:
        waits = [wait]
    elif sync := langbb.get('map-wait'):
        waits = [sync(q) for q in FindSymbols().visit(body[span])
                 if isinstance(q, langbb.AsyncQueue)]
    else:
        waits = []

    timed = TimedList(timer=timer, lname='time_loop', body=body[span] + waits)
    body[span] = [*waits, timed]
    return Transformer({scope: scope._rebuild(body=body)}).visit(iet)


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
def instrument_sections(iet, profiler, timer, langbb, **kwargs):
    """
    Instrument the Sections of the input IET based on `profiler.sections`.
    """
    piet = iet
    if isinstance(iet, EntryFunction) and 'time_loop' in timer.fields:
        piet = instrument_time_loop(iet, timer, langbb)
    piet = profiler.instrument(piet, timer)

    if piet is iet:
        return piet, {}

    headers = [TimedList._start_timer_header(), TimedList._stop_timer_header()]

    return piet, {'headers': headers}


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

    mapper = {}
    for tl in FindNodes(TimedList).visit(iet):
        if tl.name == 'time_loop':
            continue
        symbols = FindSymbols().visit(tl)

        queues = [i for i in symbols if isinstance(i, langbb.AsyncQueue)]
        unnecessary = any(FindNodes((BusyWait, RemainderCall)).visit(tl))
        if queues and not unnecessary:
            waits = tuple(sync(i) for i in queues)
            mapper[tl] = tl._rebuild(body=tl.body + waits)

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}
