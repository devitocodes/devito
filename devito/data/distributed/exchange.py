"""
Exchange: the value object tying a Selection and a Layout into a reusable plan.

`Exchange(data, idx)` builds the routing for one index expression without any
communication. `get` pulls `data[idx]` and `put` assigns `data[idx] =
value`. Because the plan is communication-free to build and independent of the
data *values*, it can be cached and replayed across calls with the same index
(e.g. sparse injection every timestep), so the steady state is pure
point-to-point with no re-planning. `cached_exchange` provides that cache;
`Data` uses it so `data[idx] = value` is automatically plan-cached.
"""

from functools import lru_cache

import numpy as np

from devito.data.distributed.layout import Layout
from devito.data.distributed.plan import ExchangePlan
from devito.data.distributed.selection import Selection
from devito.tools import as_tuple

__all__ = ['Exchange', 'cached_exchange']


class Exchange:

    """
    A reusable redistribution plan for `data[idx]` on distributed `data`.

    Parameters
    ----------
    data : Data
        The MPI-distributed array being indexed.
    idx : index expression
        Any NumPy index (scalars, slices, integer arrays, boolean masks).
    """

    def __init__(self, data, idx):
        # Keep the data reference (not a snapshot): the underlying buffer is
        # stable across `f.data` views, so a cached plan always reads/writes the
        # current values.
        self._data = data
        global_shape = tuple(
            dec.size if dec is not None else size
            for dec, size in zip(data._decomposition, data.shape, strict=True)
        )
        self._layout = Layout(data._distributor, data._decomposition, global_shape)
        self._selection = Selection.from_index(idx, global_shape)
        self._plan = ExchangePlan.build(self._selection, self._layout)

    def get(self):
        """Return `data[idx]` as a NumPy array."""
        return self._plan.get(np.asarray(self._data))

    def put(self, value):
        """Assign `data[idx] = value`."""
        self._plan.put(np.asarray(self._data), value)


class _ExchangeKey:

    """
    Hashable, content-addressed key wrapping `(data, idx)` for plan caching.

    Generates a signature from the `data` identity and the `idx` content, so that
    the same plan is reused across calls with the same index expression, even if
    the `data` object is a different view of the same underlying buffer.
    The signature does not include the `data` values,
    but only the `data` metadata (shape, decomposition, distributor)
    and the `idx` content so that the same plan is reused across calls.
    """

    __slots__ = ('data', 'idx', '_sig')

    def __init__(self, data, idx):
        self.data = data
        self.idx = idx
        self._sig = _signature(self.data, self.idx)

    def __hash__(self):
        return hash(self._sig)

    def __eq__(self, other):
        return self._sig == other._sig


@lru_cache(maxsize=64)
def _build(key):
    return Exchange(key.data, key.idx)


def cached_exchange(data, idx):
    """Return a (cached) `Exchange` for `data[idx]`."""
    return _build(_ExchangeKey(data, idx))


def _signature(data, idx):
    # The decomposition/distributor are keyed by identity: while cached, the key
    # keeps them alive so their ids cannot be recycled, and distinct live objects
    # always have distinct ids.
    sig = [id(data._distributor), id(data._decomposition), tuple(data.shape)]
    for component in as_tuple(idx):
        if isinstance(component, np.ndarray):
            sig.append(('arr', component.shape, component.dtype.str,
                        component.tobytes()))
        elif isinstance(component, slice):
            sig.append(('slc', component.start, component.stop, component.step))
        elif isinstance(component, (list, tuple)):
            arr = np.asarray(component)
            sig.append(('seq', arr.shape, arr.dtype.str, arr.tobytes()))
        else:
            sig.append(('obj', component))
    return tuple(sig)
