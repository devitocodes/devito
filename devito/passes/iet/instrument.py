from itertools import groupby

from devito.ir.iet import (BusyWait, Iteration, Section, TimedList,
                           FindNodes, FindSymbols, MapNodes, Transformer)
from devito.mpi.routines import (HaloUpdateCall, HaloWaitCall, MPICall, MPIList,
                                 HaloUpdateList, HaloWaitList, RemainderCall,
                                 ComputeCall)
from devito.passes.iet.engine import iet_pass
from devito.types import TempArray, TempFunction, Timer

__all__ = ['instrument']


def instrument(graph, **kwargs):
    profiler = kwargs['profiler']
    if profiler is None:
        return

    track_subsections(graph, **kwargs)

    # Construct a fresh Timer object
    timer = Timer(profiler.name, list(profiler.all_sections))

    instrument_sections(graph, timer=timer, **kwargs)
    sync_sections(graph, **kwargs)


@iet_pass
def track_subsections(iet, **kwargs):
    """
    Add sub-Sections to the `profiler`. Sub-Sections include:

        * MPI Calls (e.g., HaloUpdateCall and HaloUpdateWait)
        * Busy-waiting on While(lock) (e.g., from host-device orchestration)
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
        BusyWait: 'busywait'
    }

    verbosity_mapper = {
        0: (),
        1: (MPIList, RemainderCall, BusyWait),
        2: (MPICall, ComputeCall, BusyWait),
    }

    mapper = {}

    # Enable/disable profiling of sub-Sections
    for NodeType in verbosity_mapper[profiler._verbosity]:
        for k, v in MapNodes(Section, NodeType).visit(iet).items():
            for i in v:
                if i in mapper:
                    continue
                name = sregistry.make_name(prefix=name_mapper[i.__class__])
                mapper[i] = Section(name, body=i, is_subsection=True)
                profiler.track_subsection(k.name, name)

    iet = Transformer(mapper).visit(iet)

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
def instrument_sections(iet, **kwargs):
    """
    Instrument the Sections of the input IET based on `profiler.sections`.
    """
    profiler = kwargs['profiler']
    timer = kwargs['timer']

    piet = profiler.instrument(iet, timer)

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
        symbols = FindSymbols().visit(tl)

        queues = [i for i in symbols if isinstance(i, langbb.AsyncQueue)]
        unnecessary = any(FindNodes(BusyWait).visit(tl))
        if queues and not unnecessary:
            waits = tuple(sync(i) for i in queues)
            mapper[tl] = tl._rebuild(body=tl.body + waits)

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}
