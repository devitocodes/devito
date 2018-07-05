import numpy as np
from mpi4py import MPI

from devito.tools import Tag, flatten

__all__ = ['Distributor', 'LEFT', 'RIGHT', 'CENTER']


class Distributor(object):

    """
    A class to perform domain decomposition for a set of MPI processes.

    :param shape: The shape of the domain to be decomposed.
    :param comm: An MPI communicator.
    """

    def __init__(self, shape, input_comm=None):
        self._glb_shape = shape
        self._input_comm = input_comm or MPI.COMM_WORLD
        self._topology = MPI.Compute_dims(self._input_comm.size, len(shape))

        if self._input_comm is MPI.COMM_WORLD:
            # By default, Devito arranges processes into a virtual cartesian topology
            self._comm = self._input_comm.Create_cart(self._topology)
        else:
            self._comm = input_comm

        # Perform domain decomposition
        self._glb_numbs = [np.array_split(range(i), j)
                           for i, j in zip(shape, self._topology)]

    @property
    def rank(self):
        return self._comm.rank

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
    def neighbours(self):
        """
        Return the mapper ``proc -> direction``; ``proc`` is the rank of a
        neighboring process, while ``direction`` tells whether ``proc`` is
        logically at right (value=1) or left (value=-1) of ``self``.
        """
        ret = []
        for src, dest in [self._comm.Shift(i, 1) for i in range(self.ndim)]:
            if src != -1:
                ret.append((src, LEFT))
            if dest != -1:
                ret.append((dest, RIGHT))
        return dict(ret)

    def __repr__(self):
        return "Distributor(nprocs=%d)" % self.nprocs


class RankRelativePosition(Tag):
    pass


LEFT = RankRelativePosition('Left')
RIGHT = RankRelativePosition('Right')
CENTER = RankRelativePosition('Center')
