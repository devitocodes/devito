from devito.ir import AsyncCall, AsyncCallable
from devito.passes.iet.engine import iet_pass

__all__ = ['pthreadify']


def pthreadify(graph):
    track = {}

    lower_async_callables(graph, track=track)
    lower_async_calls(graph, track=track)


@iet_pass
def lower_async_callables(iet, **kwargs):
    if not isinstance(iet, AsyncCallable):
        return iet, {}

    from IPython import embed; embed()
    return iet, {}


@iet_pass
def lower_async_calls(iet, **kwargs):
    return iet, {}
