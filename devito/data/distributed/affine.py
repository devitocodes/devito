"""
Affine layer: the structured quadrant of the engine, in the `get`/`put` direction.

`ExchangePlan` routes one "point" per distributed coordinate, so a peer's share
of a basic index costs one `int64` of header per element -- more than a
`float32` payload. A basic index needs none of that: every axis is a strided
range, so a peer's share is a *box*, described by one slice per axis whatever
its extent. `AffinePlan` is that description; it moves the same bytes through
the same `sparse_exchange`, with O(ndim) metadata instead of O(npoints).

It applies when every distributed axis carries a scalar or a slice. Advanced
indexing on a distributed axis still belongs to `ExchangePlan`, which is where
per-point routing is actually needed.
"""

import numpy as np

from devito.data.distributed.selection import Affine, IndexScalar
from devito.data.distributed.transport import sparse_exchange

__all__ = ['AffinePlan']


class AffinePlan:

    """
    The rank-to-rank routing of a basic index, as one box per peer.

    Built without communication, like `ExchangePlan`, and driving `get` and
    `put` alike. Use `build`, which returns `None` when the index is not basic
    on every distributed axis.

    Parameters
    ----------
    layout : Layout
        Physical placement of the array being indexed.
    selection : Selection
        Normalized meaning of the index expression.
    peers : dict
        Maps a peer rank to `(place, local)`: where that peer's share sits in
        the result, and the index reaching it in the peer's local array. Both
        are full-rank index tuples.
    """

    def __init__(self, layout, selection, peers):
        self.layout = layout
        self.selection = selection
        self.peers = peers

    @classmethod
    def build(cls, selection, layout):
        """
        Plan `data[idx]` as one box per peer, or `None` if `idx` is not basic.

        Returns
        -------
        AffinePlan or None
            `None` when a distributed axis is advanced-indexed, which the
            caller must route through `ExchangePlan` instead.
        """
        dist = set(layout.distributed_axes)
        if any(a in dist for a in selection.advanced_axes):
            return None
        if any(not isinstance(s, (Affine, IndexScalar))
               for s in selection.selectors):
            return None

        # The result axis each kept selector contributes, in result order
        kept = [v for kind, v in selection.result_dims if kind == 'basic']

        peers = {}
        for coord, rank in layout.coord_to_rank.items():
            boxes = [_axis_box(selection.selectors[a], layout, a, c)
                     for a, c in zip(layout.distributed_axes, coord, strict=True)]
            if any(b is None for b in boxes):
                continue
            share = dict(zip(layout.distributed_axes, boxes, strict=True))

            place = [slice(None)]*len(kept)
            local = [None]*selection.ndim
            for axis, s in enumerate(selection.selectors):
                at, local[axis] = share.get(axis, (slice(None), _whole(s)))
                if isinstance(s, Affine):
                    place[kept.index(axis)] = at
            peers[rank] = (tuple(place), tuple(local))

        return cls(layout, selection, peers)

    @property
    def comm(self):
        return self.layout.distributor.comm

    def get(self, local, root=None):
        """
        Return `data[idx]`, each peer contributing one contiguous box.

        Collective. With `root` given, only that rank builds the result and the
        others return `None`; otherwise every rank gets the full result.
        """
        targets = (range(self.comm.Get_size()) if root is None
                   else (int(root),))
        mine = self.peers.get(self.comm.Get_rank())

        sendbufs = {}
        if mine is not None:
            block = np.ascontiguousarray(local[mine[1]]).reshape(-1)
            sendbufs = {r: block for r in targets}

        recvd = sparse_exchange(self.comm, sendbufs, local.dtype, tag=45)
        if root is not None and self.comm.Get_rank() != int(root):
            return None

        out = np.empty(self.selection.result_shape, dtype=local.dtype)
        for src, buf in recvd.items():
            place, _ = self.peers[src]
            out[place] = buf.reshape(_extent(place, out.shape))
        return out

    def put(self, local, value):
        """
        Assign `data[idx] = value`, each peer receiving only its own box.

        Collective, and `value` must agree on every rank that holds one.
        """
        rank = self.comm.Get_rank()
        mine = self.peers.get(rank)

        sendbufs = {}
        if value is not None:
            value = np.broadcast_to(np.asarray(value, dtype=local.dtype),
                                    self.selection.result_shape)
            sendbufs = {r: np.ascontiguousarray(value[place]).reshape(-1)
                        for r, (place, _) in self.peers.items()}

        recvd = sparse_exchange(self.comm, sendbufs, local.dtype, tag=46)
        if mine is not None and recvd:
            # Whoever holds `value` sends every peer the same bytes, so the
            # first answer is as good as any
            buf = next(iter(recvd.values()))
            local[mine[1]] = buf.reshape(_extent(mine[0],
                                                 self.selection.result_shape))


def _whole(selector):
    """A replicated axis is held in full, so its selector indexes it directly."""
    if isinstance(selector, IndexScalar):
        return selector.index
    return slice(selector.start, selector.stop, selector.step)


def _extent(place, shape):
    """The shape `place` selects out of an array of `shape`."""
    return tuple(len(range(*i.indices(s)))
                 for i, s in zip(place, shape, strict=True))


def _axis_box(selector, layout, axis, sub):
    """
    Sub-rank `sub`'s share of `selector` on `axis`, as `(place, local)`.

    `place` is the stretch of the result axis this sub-rank contributes;
    `local` reaches it inside the sub-rank's own chunk. `None` when the
    sub-rank holds none of the selection.
    """
    _, local_lut, _ = layout.axis_maps(axis)
    dec = layout.decomposition[axis]
    chunk = np.asarray(dec[sub], dtype=np.int64) - (dec.glb_min or 0)
    if chunk.size == 0:
        return None
    lo, hi = int(chunk.min()), int(chunk.max())

    if isinstance(selector, IndexScalar):
        if not lo <= selector.index <= hi:
            return None
        return slice(None), int(local_lut[selector.index])

    start, step, size = selector.start, selector.step, selector.size
    if size == 0:
        return None

    # The result positions k whose global coordinate start + k*step lands in
    # [lo, hi]; the two bounds swap over with the sign of the step
    edges = sorted(((lo - start)/step, (hi - start)/step))
    k0 = max(0, int(np.ceil(edges[0] - 1e-9)))
    k1 = min(size - 1, int(np.floor(edges[1] + 1e-9)))
    if k0 > k1:
        return None

    l0 = int(local_lut[start + k0*step])
    stop = int(local_lut[start + k1*step]) + step
    return slice(k0, k1 + 1), slice(l0, stop if stop >= 0 else None, step)
