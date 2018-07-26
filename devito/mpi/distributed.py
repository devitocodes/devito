from collections import namedtuple
from ctypes import Structure, c_int, c_void_p, sizeof
from itertools import product

from cached_property import cached_property
from cgen import Struct, Value

import numpy as np
from mpi4py import MPI

from devito.types import LEFT, RIGHT

__all__ = ['Distributor']


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
        self._input_comm = (input_comm or MPI.COMM_WORLD).Clone()

        # `Compute_dims` sets the dimension sizes to be as close to each other
        # as possible, using an appropriate divisibility algorithm. Thus, in 3D:
        # * topology[0] >= topology[1] >= topology[2]
        # * topology[0] * topology[1] * topology[2] == self._input_comm.size
        topology = MPI.Compute_dims(self._input_comm.size, len(shape))
        # At this point MPI's dimension 0 corresponds to the rightmost element
        # in `topology`. This is in reverse to `shape`'s ordering. Hence, we
        # now restore consistency
        self._topology = tuple(reversed(topology))

        if self._input_comm is not input_comm:
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

    def __del__(self):
        self._input_comm.Free()

    @property
    def myrank(self):
        return self._comm.rank

    @property
    def mycoords(self):
        return tuple(self._comm.coords)

    @property
    def comm(self):
        return self._comm

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
        Return the mapper ``proc -> side``; ``proc`` is the rank of a
        neighboring process, while ``side`` tells whether ``proc`` is
        logically at right (value=1) or left (value=-1) of ``self``.
        """
        shifts = {d: self._comm.Shift(i, 1) for i, d in enumerate(self.dimensions)}
        ret = {}
        for d, (src, dest) in shifts.items():
            ret[d] = {}
            ret[d][LEFT] = src
            ret[d][RIGHT] = dest
        return ret

    @cached_property
    def _C_comm(self):
        """
        A :class:`Object` wrapping an MPI communicator.

        Extracted from: ::

            https://github.com/mpi4py/mpi4py/blob/master/demo/wrap-ctypes/helloworld.py
        """
        from devito.types import Object
        if MPI._sizeof(self._comm) == sizeof(c_int):
            ctype = type('MPI_Comm', (c_int,), {})
        else:
            ctype = type('MPI_Comm', (c_void_p,), {})
        comm_ptr = MPI._addressof(self._comm)
        comm_val = ctype.from_address(comm_ptr)
        return Object(name='comm', dtype=ctype, value=comm_val)

    @cached_property
    def _C_neighbours(self):
        """A ctypes Struct to access the neighborhood of a given rank."""
        from devito.types import CompositeObject
        entries = list(product(self.dimensions, [LEFT, RIGHT]))
        fields = [('%s%s' % (d, i), c_int) for d, i in entries]
        obj = CompositeObject('nb', 'neighbours', Structure, fields)
        for d, i in entries:
            setattr(obj.value._obj, '%s%s' % (d, i), self.neighbours[d][i])
        cdef = Struct('neighbours', [Value('int', i) for i, _ in fields])
        CNeighbours = namedtuple('CNeighbours', 'ctype cdef obj')
        return CNeighbours(obj.dtype, cdef, obj)

    def __repr__(self):
        return "Distributor(nprocs=%d)" % self.nprocs
