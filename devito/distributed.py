from collections import OrderedDict
import numpy as np
from mpi4py import MPI

__all__ = ['Distributor', 'default_distributor']


class Distributor(object):

    """
    A class to specify domain decomposition strategies for a set of MPI processes.

    :param comm: An MPI communicator.
    """

    def __init__(self, comm=MPI.COMM_WORLD):
        self._comm = comm

    @property
    def rank(self):
        return self._comm.Get_rank()

    @property
    def nprocs(self):
        return self._comm.Get_size()

    def partition(self, dims, shape):
        # TODO: at the moment, partition all space dimensions into equal parts
        numbs = OrderedDict([(d, np.array_split(range(i), self.nprocs))
                              for d, i in zip(dims, shape)])
        shapes = [tuple(len(i) for i in zip(*list(numbs.values())))]
        # TODO: problem is how I'm calculating shapes here
        return numbs, shapes

    def loc_partition(self, dims, shape):
        numbs, shapes = self.partition(dims, shape)
        loc_numb = tuple(zip(*list(numbs.values())))[self.rank]
        print(shapes, self.rank, shapes[0])
        loc_shape = shapes[self.rank]
        return loc_numb, loc_shape

    def __repr__(self):
        return "Distributor(nprocs=%d)" % self.nprocs


default_distributor = Distributor()
