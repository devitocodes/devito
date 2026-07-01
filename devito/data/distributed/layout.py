"""
Layout layer: where a global coordinate physically lives.

A `Layout` wraps a distributor and a per-axis decomposition and answers,
for any axis, "which rank owns global index g, and at what local offset". It is
the single bridge between the layout-independent `Selection` and the
physical MPI placement, and it is what makes replicated and distributed axes
look uniform to the planner. All maps are computed locally from replicated
metadata; no communication happens here.
"""

from functools import cached_property

import numpy as np

from devito.tools import prod

__all__ = ['Layout']


class Layout:

    """
    Physical placement of a distributed array's axes.

    Parameters
    ----------
    distributor : Distributor
        Provides the communicator, topology and rank/coord maps.
    decomposition : tuple
        One entry per array axis: a Decomposition for a distributed axis, or
        `None` for a replicated axis.
    global_shape : tuple of int
        The global array shape (full size on every axis).
    """

    def __init__(self, distributor, decomposition, global_shape):
        self.distributor = distributor
        self.decomposition = tuple(decomposition)
        self.global_shape = tuple(global_shape)

    @cached_property
    def distributed_axes(self):
        """The array axes that are MPI-distributed, in increasing order."""
        return tuple(a for a, d in enumerate(self.decomposition) if d is not None)

    @cached_property
    def replicated_axes(self):
        """The array axes that are replicated on every rank."""
        return tuple(a for a, d in enumerate(self.decomposition) if d is None)

    @cached_property
    def replicated_size(self):
        """Product of the full local sizes of the replicated axes."""
        return prod(self.global_shape[a] for a in self.replicated_axes)

    def axis_maps(self, axis):
        """
        Lookup tables for one distributed axis.

        Returns
        -------
        owner : ndarray
            `owner[g]` is the topology sub-rank owning global index `g` along
            this axis (`-1` if out of bounds).
        local : ndarray
            `local[g]` is the offset of `g` within its owner's subdomain.
        sizes : ndarray
            `sizes[i]` is the number of indices owned by sub-rank `i`.
        """
        return self._axis_maps[axis]

    @cached_property
    def _axis_maps(self):
        maps = {}
        for axis in self.distributed_axes:
            dec = self.decomposition[axis]
            gmin = dec.glb_min or 0
            n = self.global_shape[axis]
            owner = np.full(n, -1, dtype=np.int64)
            local = np.full(n, -1, dtype=np.int64)
            sizes = np.zeros(len(dec), dtype=np.int64)
            for i, sub in enumerate(dec):
                pos = np.asarray(sub, dtype=np.int64) - gmin
                in_range = (pos >= 0) & (pos < n)
                owner[pos[in_range]] = i
                local[pos[in_range]] = np.arange(pos.size)[in_range]
                sizes[i] = pos.size
            maps[axis] = (owner, local, sizes)
        return maps

    @cached_property
    def topology_shape(self):
        """Number of sub-ranks along each distributed axis."""
        return tuple(len(self.decomposition[a]) for a in self.distributed_axes)

    @cached_property
    def coord_to_rank(self):
        """Map a topology coordinate tuple (over distributed axes) to a flat rank.

        Grid distributors expose the inverse Cartesian map via `all_coords`;
        single-axis distributors (e.g. sparse) lay ranks out linearly, so the
        sole sub-rank index is the flat rank.
        """
        if hasattr(self.distributor, 'all_coords'):
            return {tuple(int(c) for c in coord): r
                    for r, coord in enumerate(self.distributor.all_coords)}
        return {(r,): r for r in range(self.distributor.nprocs)}
