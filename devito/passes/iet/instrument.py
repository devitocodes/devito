from devito.ir.iet import BusyWait, MapNodes, Section, TimedList, Transformer
from devito.mpi.routines import (HaloUpdateCall, HaloWaitCall, MPICall, MPIList,
                                 HaloUpdateList, HaloWaitList, RemainderCall)
from devito.passes.iet.engine import iet_pass
from devito.types import Timer

__all__ = ['instrument']


def instrument(graph, **kwargs):
    track_subsections(graph, **kwargs)

    # Construct a fresh Timer object
    profiler = kwargs['profiler']
    if profiler is None:
        return
    timer = Timer(profiler.name, list(profiler.all_sections))

    instrument_sections(graph, timer=timer, **kwargs)


@iet_pass
def track_subsections(iet, **kwargs):
    """
    Add custom Sections to the `profiler`. Custom Sections include:

        * MPI Calls (e.g., HaloUpdateCall and HaloUpdateWait)
        * Busy-waiting on While(lock) (e.g., from host-device orchestration)
    """
    profiler = kwargs['profiler']
    sregistry = kwargs['sregistry']

    name_mapper = {
        HaloUpdateCall: 'haloupdate',
        HaloWaitCall: 'halowait',
        RemainderCall: 'remainder',
        HaloUpdateList: 'haloupdate',
        HaloWaitList: 'halowait',
        BusyWait: 'busywait'
    }

    mapper = {}

    for NodeType in [MPIList, MPICall, BusyWait]:
        for k, v in MapNodes(Section, NodeType).visit(iet).items():
            for i in v:
                if i in mapper or not any(issubclass(i.__class__, n)
                                          for n in profiler.trackable_subsections):
                    continue
                name = sregistry.make_name(prefix=name_mapper[i.__class__])
                mapper[i] = Section(name, body=i, is_subsection=True)
                profiler.track_subsection(k.name, name)

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
