"""
Distributed data redistribution engine.

Indexing an MPI-distributed array is treated as a redistribution between layouts.
The engine is built as four pure layers plus a value object:

* :mod:`selection` -- what an index expression means (serial NumPy semantics).
* :mod:`layout`    -- where a global coordinate physically lives.
* :mod:`plan`      -- the rank-to-rank routing, built without communication.
* :mod:`transport` -- the sparse point-to-point exchange (NBX).
* :mod:`exchange`  -- ``Exchange(data, idx).get()/.put(value)``.

Only :class:`Exchange` is needed by ``Data``; the lower layers are independently
testable (in serial, or with toy buffers).
"""

from devito.data.distributed.exchange import Exchange, cached_exchange  # noqa
from devito.data.distributed.layout import Layout  # noqa
from devito.data.distributed.plan import ExchangePlan  # noqa
from devito.data.distributed.selection import Selection  # noqa
from devito.data.distributed.transport import nbx_exchange  # noqa
