from devito.ir.iet import (AsyncCallable, AsyncCall, FindNodes, SyncSpot,
                           Transformer, derive_parameters)
from devito.ir.support import WithLock, FetchPrefetch, PrefetchUpdate
from devito.passes.iet.engine import iet_pass

__all__ = ['split_async_compute']


@iet_pass
def split_async_compute(iet, sregistry=None):
    """
    Move asynchronous computation in to Callables.
    This creates a separation between `iet` and routines that ultimately
    are expected to run asynchronously. How the asynchrony is actually
    implemented is a problem for another pass.
    """
    candidates = (WithLock, PrefetchUpdate, FetchPrefetch)

    # At the moment, the asynchronous computation stems only from SyncSpots
    efuncs = []
    mapper = {}
    for n in FindNodes(SyncSpot).visit(iet):
        sync_ops = [s for s in n.sync_ops if isinstance(s, candidates)]
        if not sync_ops:
            continue

        name = sregistry.make_name(prefix='async')
        body = n._rebuild(sync_ops=sync_ops)
        parameters = derive_parameters(body)

        efuncs.append(AsyncCallable(
            name, body, 'void', parameters=parameters, sync_ops=sync_ops
        ))

        async_call = AsyncCall(name, arguments=parameters)
        sync_ops = [s for s in n.sync_ops if s not in sync_ops]
        mapper[n] = n._rebuild(sync_ops=sync_ops, body=async_call)

    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {'efuncs': efuncs}
