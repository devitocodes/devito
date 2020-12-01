from devito.ir.iet import TimedList
from devito.passes.iet.engine import iet_pass
from devito.types import Timer

__all__ = ['instrument']


def instrument(graph, **kwargs):
    analyze_sections(graph, **kwargs)

    # Construct a fresh Timer object
    profiler = kwargs['profiler']
    timer = Timer(profiler.name, list(profiler._sections))

    instrument_sections(graph, timer=timer, **kwargs)


@iet_pass
def analyze_sections(iet, **kwargs):
    """
    Analyze the input IET and update `profiler.sections` with all of the
    Sections introduced by the previous passes.
    """
    profiler = kwargs['profiler']

    profiler.analyze(iet)

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

    return piet, {'args': timer, 'headers': headers}
