import numpy as np
from mpi4py import MPI

from devito.tools import Tag, flatten

__all__ = ['Distributor', 'LEFT', 'RIGHT', 'CENTER']


class Distributor(object):

    """
    A class to perform domain decomposition for a set of MPI processes.

    :param shape: The shape of the domain to be decomposed.
    :param dimensions: The :class:`Dimension`s defining the domain.
    :param comm: An MPI communicator.
    """

    def __init__(self, shape, dimensions, input_comm=None):
        self._glb_shape = shape
        self._dimensions = dimensions
        self._input_comm = input_comm or MPI.COMM_WORLD

        # `Compute_dims` sets the dimension sizes to be as close to each other
        # as possible, using an appropriate divisibility algorithm. Thus, in 3D:
        # * topology[0] >= topology[1] >= topology[2]
        # * topology[0] * topology[1] * topology[2] == self._input_comm.size
        topology = MPI.Compute_dims(self._input_comm.size, len(shape))
        # At this point MPI's dimension 0 corresponds to the rightmost element
        # in `topology`. This is in reverse to `shape`'s ordering. Hence, we
        # now restore consistency
        self._topology = tuple(reversed(topology))

        if self._input_comm is MPI.COMM_WORLD:
            # By default, Devito arranges processes into a cartesian topology.
            # MPI works with numbered dimensions and follows the C row-major
            # numbering of the ranks, i.e. in a 2x3 Cartesian topology (0,0)
            # maps to rank 0, (0,1) maps to rank 1, (0,2) maps to rank 2, (1,0)
            # maps to rank 3, and so on.
            self._comm = self._input_comm.Create_cart(self._topology)
        else:
            self._comm = input_comm

        # Perform domain decomposition
        self._glb_numbs = [np.array_split(range(i), j)
                           for i, j in zip(shape, self._topology)]

    @property
    def myrank(self):
        return self._comm.rank

    @property
    def mycoords(self):
        return tuple(self._comm.coords)

    @property
    def nprocs(self):
        return self._comm.size

    @property
    def ndim(self):
        return len(self._glb_shape)

    @property
    def topology(self):
        return self._topology

    @property
    def glb_numb(self):
        """Return the global numbering of this process' domain."""
        assert len(self._comm.coords) == len(self._glb_numbs)
        return tuple(i[j] for i, j in zip(self._glb_numbs, self._comm.coords))

    @property
    def shape(self):
        """Return the shape of this process' domain."""
        return tuple(len(i) for i in self.glb_numb)

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def neighbours(self):
        """
        Return the mapper ``proc -> direction``; ``proc`` is the rank of a
        neighboring process, while ``direction`` tells whether ``proc`` is
        logically at right (value=1) or left (value=-1) of ``self``.
        """
        shifts = {d: self._comm.Shift(i, 1) for i, d in enumerate(self.dimensions)}
        ret = {}
        for d, (src, dest) in shifts.items():
            ret[d] = {}
            ret[d][LEFT] = src if src != -1 else None
            ret[d][RIGHT] = dest if dest != -1 else None
        return ret

    def __repr__(self):
        return "Distributor(nprocs=%d)" % self.nprocs


class RankRelativePosition(Tag):
    pass


LEFT = RankRelativePosition('Left')
RIGHT = RankRelativePosition('Right')
CENTER = RankRelativePosition('Center')
