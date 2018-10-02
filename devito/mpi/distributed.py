from collections import namedtuple
from ctypes import Structure, c_int, c_void_p, sizeof
from itertools import product
from math import ceil
import atexit

from cached_property import cached_property
from cgen import Struct, Value

import numpy as np

from devito.parameters import configuration
from devito.types import LEFT, RIGHT
from devito.tools import EnrichedTuple, is_integer


# Do not prematurely initialize MPI
# This allows launching a Devito program from within another Python program
# that has *already* initialized MPI
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI  # noqa

__all__ = ['Distributor', 'MPI']


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
        glb_numbs = [np.array_split(range(i), j)
                     for i, j in zip(shape, self._topology)]
        self._glb_numbs = EnrichedTuple(*glb_numbs, getters=self.dimensions)

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
    def ndim(self):
        return len(self._glb_shape)

    @property
    def topology(self):
        return self._topology

    @property
    def all_coords(self):
        """Return an iterable containing the coordinates of each MPI rank in
        the decomposed domain. The iterable is order based on the MPI rank."""
        ret = product(*[range(i) for i in self.topology])
        return tuple(sorted(ret, key=lambda i: self.comm.Get_cart_rank(i)))

    @property
    def all_numb(self):
        """Return an iterable containing the global numbering of each MPI rank."""
        ret = []
        for c in self.all_coords:
            glb_numb = [i[j] for i, j in zip(self._glb_numbs, c)]
            ret.append(EnrichedTuple(*glb_numb, getters=self.dimensions))
        return tuple(ret)

    @property
    def all_ranges(self):
        """Return an iterable containing the global ranges of each MPI rank."""
        ret = []
        for i in self.all_numb:
            ret.append(EnrichedTuple(*[range(min(j), max(j) + 1) for j in i],
                                     getters=self.dimensions))
        return tuple(ret)

    @property
    def glb_shape(self):
        return EnrichedTuple(*self._glb_shape, getters=self.dimensions)

    @property
    def glb_numb(self):
        """Return the global numbering of this process' domain section."""
        assert len(self.mycoords) == len(self._glb_numbs)
        glb_numb = [i[j] for i, j in zip(self._glb_numbs, self.mycoords)]
        return EnrichedTuple(*glb_numb, getters=self.dimensions)

    @property
    def glb_ranges(self):
        """Return the global ranges of this process' domain section."""
        return EnrichedTuple(*[range(min(i), max(i) + 1) for i in self.glb_numb],
                             getters=self.dimensions)

    @property
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

    def glb_to_loc(self, dim, *args):
        """
        glb_to_loc(dim, index)
        glb_to_loc(dim, offset, side)
        glb_to_loc(dim, (min, max))

        Translate global indices into local indices.

        :param dim: The :class:`Dimension` of the provided global indices.
        :param args: There are three possible cases:
                       * ``args`` is a single integer I representing a global index.
                         Then, the corresponding local index is returned if I is
                         owned by ``self``, otherwise None.
                       * ``args`` consists of two items, O and S -- O is the offset
                         the side S along ``dim``. O is therefore an integer, while S
                         is an object of type :class:`DataSide`. Return the offset in
                         the local domain, possibly 0 if the local range does not
                         intersect with the region defined by the global offset.
                       * ``args`` is a tuple (min, max); return a 2-tuple (min',
                         max'), where ``min'`` and ``max'`` can be either None or
                         an integer:
                           - ``min'=None`` means that ``min`` is not owned by
                             ``self``, but it precedes ``self``'s minimum. Likewise,
                             ``max'=None`` means that ``max`` is not owned by
                             ``self``, but it comes after ``self``'s maximum.
                           - If ``min/max=int``, then the integer can represent
                             either the local index corresponding to the
                             ``min/max``, or it could be any random number such that
                             ``max=min-1``, meaning that the input argument does not
                             represent a valid range for ``self``.
        """
        assert dim in self.dimensions
        glb_numb = self.glb_numb[dim]
        if len(args) == 1:
            base = min(glb_numb)
            if isinstance(args[0], int):
                # glb_to_loc(dim, index)
                glb_index = args[0]
                return (glb_index - base) if glb_index in glb_numb else None
            else:
                # glb_to_loc(dim, (min, max))
                glb_min, glb_max = args[0]
                if glb_min is None or glb_min <= base:
                    loc_min = None
                elif glb_min > max(glb_numb):
                    return (-2, -1)
                else:
                    loc_min = glb_min - base
                if glb_max is None or glb_max >= max(glb_numb):
                    loc_max = None
                elif glb_max < base:
                    return (-2, -1)
                else:
                    loc_max = glb_max - base
                return (loc_min, loc_max)
        else:
            # glb_to_loc(dim, offset, side)
            rel_ofs, side = args
            if side is LEFT:
                abs_ofs = min(min(i) for i in self._glb_numbs[dim]) + rel_ofs
                base = min(glb_numb)
                extent = max(glb_numb) - base
                return min(abs_ofs - base, extent) if abs_ofs > base else 0
            else:
                abs_ofs = max(max(i) for i in self._glb_numbs[dim]) - rel_ofs
                base = max(glb_numb)
                extent = base - min(glb_numb)
                return min(base - abs_ofs, extent) if abs_ofs < base else 0

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
