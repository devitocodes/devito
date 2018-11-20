from collections import namedtuple
from ctypes import Structure, c_int, c_void_p, sizeof
from itertools import product
from math import ceil
from abc import ABC, abstractmethod
import atexit

from cached_property import cached_property
from cgen import Struct, Value

import numpy as np

from devito.data import Decomposition
from devito.parameters import configuration
from devito.types import LEFT, RIGHT, CompositeObject, Object
from devito.tools import EnrichedTuple, as_tuple, is_integer


# Do not prematurely initialize MPI
# This allows launching a Devito program from within another Python program
# that has *already* initialized MPI
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI  # noqa

__all__ = ['Distributor', 'SparseDistributor', 'MPI']


class AbstractDistributor(ABC):

    """
    Decompose a set of :class:`Dimension`s over a set of MPI processes.

    .. note::

        This is an abstract class, which simply defines the interface that
        all subclasses are expected to implement.
    """

    def __init__(self, shape, dimensions):
        self._glb_shape = as_tuple(shape)
        self._dimensions = as_tuple(dimensions)

    def __repr__(self):
        return "%s(nprocs=%d)" % (self.__class__.__name__, self.nprocs)

    @abstractmethod
    def comm(self):
        """The MPI communicator."""
        return

    @abstractmethod
    def myrank(self):
        """The rank of the calling MPI process."""
        return

    @abstractmethod
    def mycoords(self):
        """The coordinates of the calling MPI rank in the Distributor topology."""
        return

    @abstractmethod
    def nprocs(self):
        """A shortcut for the number of processes in the MPI communicator."""
        return

    @property
    def is_parallel(self):
        return self.nprocs > 1

    @cached_property
    def glb_numb(self):
        """The global indices owned by the calling MPI rank."""
        assert len(self.mycoords) == len(self.decomposition)
        glb_numb = [i[j] for i, j in zip(self.decomposition, self.mycoords)]
        return EnrichedTuple(*glb_numb, getters=self.dimensions)

    @cached_property
    def glb_slices(self):
        """The global indices owned by the calling MPI rank, as a mapper from
        :class:`Dimension`s to slices."""
        return {d: slice(min(i), max(i) + 1)
                for d, i in zip(self.dimensions, self.glb_numb)}

    @property
    def glb_shape(self):
        """Shape of the decomposed domain."""
        return EnrichedTuple(*self._glb_shape, getters=self.dimensions)

    @property
    def shape(self):
        """The calling MPI rank's local shape."""
        return tuple(len(i) for i in self.glb_numb)

    @property
    def dimensions(self):
        """The decomposed :class:`Dimension`s."""
        return self._dimensions

    @cached_property
    def decomposition(self):
        """The :class:`Decomposition`s, one for each decomposed :class:`Dimension`."""
        return EnrichedTuple(*self._decomposition, getters=self.dimensions)

    @property
    def ndim(self):
        """Number of decomposed :class:`Dimension`s"""
        return len(self._glb_shape)

    def glb_to_loc(self, dim, *args, strict=True):
        """
        Convert a global index into a relative (default) or absolute local index.

        Parameters
        ----------
        dim : :class:`Dimension`
            The global index Dimension.
        *args
            There are several possibilities, documented in
            :meth:`Decomposition.convert_index`.
        strict : bool, optional
            If False, return args without raising an error if `dim` does not appear
            among the Distributor Dimensions.
        """
        if dim not in self.dimensions:
            if strict is True:
                raise ValueError("`%s` must be one of the Distributor dimensions" % dim)
            else:
                return args[0]
        return self.decomposition[dim].convert_index(*args)


class Distributor(AbstractDistributor):

    """
    Decompose a set of :class:`Dimension`s over a set of MPI processes.

    :param shape: The global shape of the domain to be decomposed.
    :param dimensions: The decomposed :class:`Dimension`s.
    :param comm: An MPI communicator.
    """

    def __init__(self, shape, dimensions, input_comm=None):
        super(Distributor, self).__init__(shape, dimensions)

        if configuration['mpi']:
            # First time we enter here, we make sure MPI is initialized
            if not MPI.Is_initialized():
                MPI.Init()

                # Make sure Finalize will be called upon exit
                def finalize_mpi():
                    MPI.Finalize()
                atexit.register(finalize_mpi)

            self._input_comm = (input_comm or MPI.COMM_WORLD).Clone()

            # `MPI.Compute_dims` sets the dimension sizes to be as close to each other
            # as possible, using an appropriate divisibility algorithm. Thus, in 3D:
            # * topology[0] >= topology[1] >= topology[2]
            # * topology[0] * topology[1] * topology[2] == self._input_comm.size
            # However, `MPI.Compute_dims` is distro-dependent, so we have to enforce
            # some properties through our own wrapper (e.g., OpenMPI v3 does not
            # guarantee that 9 ranks are arranged into a 3x3 grid when shape=(9, 9))
            topology = compute_dims(self._input_comm.size, len(shape))
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
        else:
            self._input_comm = None
            self._comm = MPI.COMM_NULL
            self._topology = tuple(1 for _ in range(len(shape)))

        # The domain decomposition
        self._decomposition = [Decomposition(np.array_split(range(i), j), c)
                               for i, j, c in zip(shape, self.topology, self.mycoords)]

    def __del__(self):
        if self._input_comm is not None:
            self._input_comm.Free()

    @property
    def comm(self):
        return self._comm

    @property
    def myrank(self):
        if self.comm is not MPI.COMM_NULL:
            return self.comm.rank
        else:
            return 0

    @property
    def mycoords(self):
        if self.comm is not MPI.COMM_NULL:
            return tuple(self.comm.coords)
        else:
            return tuple(0 for _ in range(self.ndim))

    @property
    def nprocs(self):
        if self.comm is not MPI.COMM_NULL:
            return self.comm.size
        else:
            return 1

    @property
    def topology(self):
        return self._topology

    @cached_property
    def all_coords(self):
        """Return an iterable containing the coordinates of each MPI rank in
        the decomposed domain. The iterable is order based on the MPI rank."""
        ret = product(*[range(i) for i in self.topology])
        return tuple(sorted(ret, key=lambda i: self.comm.Get_cart_rank(i)))

    @cached_property
    def all_numb(self):
        """Return an iterable containing the global numbering of all MPI ranks."""
        ret = []
        for c in self.all_coords:
            glb_numb = [i[j] for i, j in zip(self.decomposition, c)]
            ret.append(EnrichedTuple(*glb_numb, getters=self.dimensions))
        return tuple(ret)

    @cached_property
    def all_ranges(self):
        """Return an iterable containing the global ranges of each MPI rank."""
        ret = []
        for i in self.all_numb:
            ret.append(EnrichedTuple(*[range(min(j), max(j) + 1) for j in i],
                                     getters=self.dimensions))
        return tuple(ret)

    @cached_property
    def glb_pos_map(self):
        """Return the mapper ``dimension -> side`` telling the position
        of the calling rank in the global grid."""
        ret = {}
        for d, i, s in zip(self.dimensions, self.mycoords, self.topology):
            v = []
            if i == 0:
                v.append(LEFT)
            if i == s - 1:
                v.append(RIGHT)
            ret[d] = tuple(v)
        return ret

    def glb_to_rank(self, index):
        """
        Return the rank owning a given global index.

        :param index: A single domain index, or a list of domain indices. In
                      the latter case, a list of corresponding ranks is returned.
        """
        if isinstance(index, (tuple, list)):
            if len(index) == 0:
                return None
            elif is_integer(index[0]):
                # `index` is a single point
                indices = [index]
            else:
                indices = index
        ret = []
        for i in indices:
            assert len(i) == self.ndim
            found = False
            for r, j in enumerate(self.all_ranges):
                if all(v in j[d] for v, d in zip(i, self.dimensions)):
                    ret.append(r)
                    found = True
                    break
            assert found
        return tuple(ret) if len(indices) > 1 else ret[0]

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
        """An :class:`Object` wrapping an MPI communicator."""
        return MPICommObject(self.comm)

    @cached_property
    def _C_neighbours(self):
        """A :class:`ctypes.Struct` to access the neighborhood of a given rank."""
        entries = list(product(self.dimensions, [LEFT, RIGHT]))
        fields = [('%s%s' % (d, i), c_int) for d, i in entries]
        obj = CompositeObject('nb', 'neighbours', Structure, fields)
        for d, i in entries:
            setattr(obj.value._obj, '%s%s' % (d, i), self.neighbours[d][i])
        cdef = Struct('neighbours', [Value('int', i) for i, _ in fields])
        CNeighbours = namedtuple('CNeighbours', 'ctype cdef obj')
        return CNeighbours(obj.dtype, cdef, obj)


class SparseDistributor(AbstractDistributor):

    """
    Decompose a :class:`Dimension` representing a set of data values
    arbitrarily spread over a cartesian grid.

    :param npoint: The number of sparse data values.
    :param dimension: The decomposed :class:`Dimension`.
    :param distributor: The :class:`Distributor` the SparseDistributor depends on.
    """

    def __init__(self, npoint, dimension, distributor):
        super(SparseDistributor, self).__init__(npoint, dimension)
        self._distributor = distributor

        # The dimension decomposition
        decomposition = SparseDistributor.decompose(npoint, distributor)
        offs = np.concatenate([[0], np.cumsum(decomposition)])
        self._decomposition = [Decomposition([np.arange(offs[i], offs[i+1])
                                              for i in range(self.nprocs)], self.myrank)]

    @classmethod
    def decompose(cls, npoint, distributor):
        """Distribute `npoint` points over `nprocs` MPI ranks."""
        nprocs = distributor.nprocs
        if isinstance(npoint, int):
            # `npoint` is a global count. The `npoint` are evenly distributed
            # across the various MPI ranks. Note that there is nothing smart
            # in the following -- it's entirely possible that the MPI rank 0,
            # which lives at the top-left of a 2D grid, gets some points even
            # though there physically are no points in the top-left region
            if npoint < 0:
                raise ValueError('`npoint` must be >= 0')
            glb_npoint = [npoint // nprocs]*(nprocs - 1)
            glb_npoint.append(npoint // nprocs + npoint % nprocs)
        elif isinstance(npoint, (tuple, list)):
            # The i-th entry in `npoint` tells how many sparse points the
            # i-th MPI rank has
            if len(npoint) != nprocs:
                raise ValueError('The `npoint` tuple must have as many entries as '
                                 'MPI ranks (got `%d`, need `%d`)' % (npoint, nprocs))
            elif any(i < 0 for i in npoint):
                raise ValueError('All entries in `npoint` must be >= 0')
            glb_npoint = npoint
        else:
            raise TypeError('Need `npoint` int or tuple argument')
        return tuple(glb_npoint)

    @property
    def distributor(self):
        return self._distributor

    @property
    def comm(self):
        return self.distributor._comm

    @property
    def myrank(self):
        return self.distributor.myrank

    @property
    def mycoords(self):
        return (self.myrank,)

    @property
    def nprocs(self):
        return self.distributor.nprocs


class MPICommObject(Object):

    name = 'comm'

    # See https://github.com/mpi4py/mpi4py/blob/master/demo/wrap-ctypes/helloworld.py
    if MPI._sizeof(MPI.Comm) == sizeof(c_int):
        dtype = type('MPI_Comm', (c_int,), {})
    else:
        dtype = type('MPI_Comm', (c_void_p,), {})

    def __init__(self, comm=None):
        if comm is None:
            # Should only end up here upon unpickling
            comm = MPI.COMM_WORLD
        comm_ptr = MPI._addressof(comm)
        comm_val = self.dtype.from_address(comm_ptr)
        self.value = comm_val

    # Pickling support
    _pickle_args = []


def compute_dims(nprocs, ndim):
    # We don't do anything clever here. In fact, we do something very basic --
    # we just try to distribute `nprocs` evenly over the number of dimensions,
    # and if we can't we fallback to whatever MPI.Compute_dims gives...
    v = pow(nprocs, 1/ndim)
    if not v.is_integer():
        # Since pow(64, 1/3) == 3.999..4
        v = int(ceil(v))
        if not v**ndim == nprocs:
            # Fallback
            return MPI.Compute_dims(nprocs, ndim)
    else:
        v = int(v)
    return tuple(v for _ in range(ndim))
