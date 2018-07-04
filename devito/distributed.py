import numpy as np
from mpi4py import MPI

__all__ = ['Distributor']


class Distributor(object):

    """
    A class to perform domain decomposition for a set of MPI processes.

    :param shape: The shape of the domain to be decomposed.
    :param comm: An MPI communicator.
    """

    def __init__(self, shape, input_comm=None):
        self._shape = shape
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

    def __repr__(self):
        return "Distributor(nprocs=%d)" % self.nprocs
