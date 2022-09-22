from ctypes import c_int, c_void_p, sizeof
from itertools import groupby, product
from math import ceil
from abc import ABC, abstractmethod
import atexit

from cached_property import cached_property
import numpy as np
from cgen import Struct, Value

from devito.data import LEFT, CENTER, RIGHT, Decomposition
from devito.parameters import configuration
from devito.tools import EnrichedTuple, as_tuple, ctypes_to_cstr, filter_ordered
from devito.types import CompositeObject, Object


# Do not prematurely initialize MPI
# This allows launching a Devito program from within another Python program
# that has *already* initialized MPI
try:
    import mpi4py
    mpi4py.rc(initialize=False, finalize=False)
    from mpi4py import MPI  # noqa

    # Devito can be used in two main ways: as a software or as a library. In the
    # latter case, it's likely MPI will be initialized and finalized by the
    # overarching application. So we track who initialized MPI to avoid double
    # finalization
    init_by_devito = False

    # From the `atexit` documentation: "At normal program termination [...]
    # all functions registered are in last in, first out order.". So, MPI.Finalize
    # will be called only at the very end and only if necessary, after all cloned
    # communicators will have been freed
    def cleanup():
        global init_by_devito
        if init_by_devito and MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.Finalize()
    atexit.register(cleanup)
except ImportError:
    # Dummy fallback in case mpi4py/MPI aren't available
    class NoneMetaclass(type):
        def __getattr__(self, name):
            return None

    class MPI(object, metaclass=NoneMetaclass):
        @classmethod
        def Is_initialized(cls):
            return False

        def _sizeof(obj):
            return None

        def __getattr__(self, name):
            return None


__all__ = ['Distributor', 'SparseDistributor', 'MPI']


class AbstractDistributor(ABC):

    """
    Decompose a set of Dimensions over a set of MPI processes.

    Notes
    -----
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
        """
        The global indices owned by the calling MPI rank, as a mapper from
        Dimensions to slices.
        """
        return {d: slice(min(i), max(i) + 1) if len(i) > 0 else slice(0, -1)
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
        """The decomposed Dimensions."""
        return self._dimensions

    @cached_property
    def decomposition(self):
        """The Decompositions, one for each decomposed Dimension."""
        return EnrichedTuple(*self._decomposition, getters=self.dimensions)

    @property
    def ndim(self):
        """Number of decomposed Dimensions"""
        return len(self._glb_shape)

    def glb_to_loc(self, dim, *args, strict=True):
        """
        Convert a global index into a relative (default) or absolute local index.

        Parameters
        ----------
        dim : Dimension
            The global index Dimension.
        *args
            There are several possibilities, documented in
            :meth:`Decomposition.index_glb_to_loc`.
        strict : bool, optional
            If False, return args without raising an error if `dim` does not appear
            among the Distributor Dimensions.
        """
        if dim not in self.dimensions:
            if strict is True:
                raise ValueError("`%s` must be one of the Distributor dimensions" % dim)
            else:
                return args[0]
        return self.decomposition[dim].index_glb_to_loc(*args)


class Distributor(AbstractDistributor):

    """
    Decompose a domain over a set of MPI processes.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the domain to be decomposed.
    dimensions : tuple of Dimensions
        The domain Dimensions.
    comm : MPI communicator, optional
        The set of processes over which the domain is distributed. Defaults to
        MPI.COMM_WORLD.
    """

    def __init__(self, shape, dimensions, input_comm=None, topology=None):
        super(Distributor, self).__init__(shape, dimensions)

        if configuration['mpi']:
            # First time we enter here, we make sure MPI is initialized
            if not MPI.Is_initialized():
                MPI.Init()
                global init_by_devito
                init_by_devito = True

            # Note: the cloned communicator doesn't need to be explicitly freed;
            # mpi4py takes care of that when the object gets out of scope
            self._input_comm = (input_comm or MPI.COMM_WORLD).Clone()

            if topology is None:
                # `MPI.Compute_dims` sets the dimension sizes to be as close to each other
                # as possible, using an appropriate divisibility algorithm. Thus, in 3D:
                # * topology[0] >= topology[1] >= topology[2]
                # * topology[0] * topology[1] * topology[2] == self._input_comm.size
                # However, `MPI.Compute_dims` is distro-dependent, so we have to enforce
                # some properties through our own wrapper (e.g., OpenMPI v3 does not
                # guarantee that 9 ranks are arranged into a 3x3 grid when shape=(9, 9))
                self._topology = compute_dims(self._input_comm.size, len(shape))
            else:
                self._topology = topology

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
    def is_boundary_rank(self):
        """ MPI rank interfaces with the boundary of the domain. """
        return any([True if i == 0 or i == j-1 else False for i, j in
                   zip(self.mycoords, self.topology)])

    @cached_property
    def all_coords(self):
        """
        The coordinates of each MPI rank in the decomposed domain, ordered
        based on the MPI rank.
        """
        ret = product(*[range(i) for i in self.topology])
        return tuple(sorted(ret, key=lambda i: self.comm.Get_cart_rank(i)))

    @cached_property
    def all_numb(self):
        """The global numbering of all MPI ranks."""
        ret = []
        for c in self.all_coords:
            glb_numb = [i[j] for i, j in zip(self.decomposition, c)]
            ret.append(EnrichedTuple(*glb_numb, getters=self.dimensions))
        return tuple(ret)

    @cached_property
    def all_ranges(self):
        """The global ranges of all MPI ranks."""
        ret = []
        for i in self.all_numb:
            ret.append(EnrichedTuple(*[range(min(j), max(j) + 1) for j in i],
                                     getters=self.dimensions))
        return tuple(ret)

    @cached_property
    def glb_pos_map(self):
        """
        A mapper ``Dimension -> DataSide`` providing the position of the calling
        MPI rank in the decomposed domain.
        """
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
        The MPI rank owning a given global index.

        Parameters
        ----------
        index : array of ints
            The index, or array of indices, for which the owning MPI rank(s) is
            retrieved.
        """
        assert isinstance(index, np.ndarray)
        if index.shape[0] == 0:
            return None
        elif sum(index.shape) == 1:
            return index

        assert index.shape[1] == self.ndim

        # Add singleton dimension at the end if only single gridpoint is passed
        # instead of support.
        if len(index.shape) == 2:
            index = np.expand_dims(index, axis=2)

        ret = {}
        for r, j in enumerate(self.all_ranges):
            mins = np.array([b[0] for b in j]).reshape(1, -1, 1)
            maxs = np.array([b[-1] for b in j]).reshape(1, -1, 1)
            inds = np.where(((index <= maxs) & (index >= mins)).all(axis=1))
            if inds[0].size == 0:
                continue
            ret[r] = filter_ordered(inds[0])
        return ret

    @property
    def neighborhood(self):
        """
        A mapper ``M`` describing the calling MPI rank's neighborhood in the
        decomposed grid. Let

            * ``d`` be a Dimension -- ``d0, d1, ..., dn`` are the decomposed
              Dimensions.
            * ``s`` be a DataSide -- possible values are ``LEFT, CENTER, RIGHT``,
            * ``p`` be the rank of a neighbour MPI process.

        Then ``M`` can be indexed in two ways:

            * ``M[d] -> (s -> p)``; that is, ``M[d]`` returns a further mapper
              (from DataSide to MPI rank) from which the two adjacent processes
              along ``d`` can be retrieved.
            * ``M[(s0, s1, ..., sn)] -> p``, where ``s0`` is the DataSide along
              ``d0``, ``s1`` the DataSide along ``d1``, and so on. This can be
              useful to retrieve the diagonal neighbours (e.g., ``M[(LEFT, LEFT)]``
              gives the top-left neighbour in a 2D grid).
        """
        # Set up horizontal neighbours
        shifts = {d: self.comm.Shift(i, 1) for i, d in enumerate(self.dimensions)}
        ret = {}
        for d, (src, dest) in shifts.items():
            ret[d] = {}
            ret[d][LEFT] = src
            ret[d][RIGHT] = dest

        # Set up diagonal neighbours
        for i in product([LEFT, CENTER, RIGHT], repeat=self.ndim):
            neighbor = [c + s.val for c, s in zip(self.mycoords, i)]

            if any(c < 0 or c >= s for c, s in zip(neighbor, self.topology)):
                ret[i] = MPI.PROC_NULL
            else:
                ret[i] = self.comm.Get_cart_rank(neighbor)

        return ret

    @cached_property
    def _obj_comm(self):
        """An Object representing the MPI communicator."""
        return MPICommObject(self.comm)

    @cached_property
    def _obj_neighborhood(self):
        """
        A CompositeObject describing the calling MPI rank's neighborhood
        in the decomposed grid.
        """
        return MPINeighborhood(self.neighborhood)

    def _rebuild(self, shape=None, dimensions=None, comm=None):
        return Distributor(shape or self.shape, dimensions or self.dimensions,
                           comm or self.comm)


class SparseDistributor(AbstractDistributor):

    """
    Decompose a Dimension defining a set of sparse data values arbitrarily
    spread within a cartesian grid.

    Parameters
    ----------
    npoint : int
        The number of sparse data values.
    dimensions : tuple of Dimensions
        The decomposed Dimensions.
    distributor : Distributor
        The domain decomposition the SparseDistributor depends on.
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

    __rargs__ = ()

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
        self.comm = comm

    def _arg_values(self, *args, **kwargs):
        grid = kwargs.get('grid', None)
        # Update `comm` based on object attached to `grid`
        if grid is not None:
            return grid.distributor._obj_comm._arg_defaults()
        else:
            return self._arg_defaults()


class MPINeighborhood(CompositeObject):

    __rargs__ = ('neighborhood',)

    def __init__(self, neighborhood):
        self._neighborhood = neighborhood

        self._entries = [i for i in neighborhood if isinstance(i, tuple)]

        fields = [(''.join(j.name[0] for j in i), c_int) for i in self.entries]
        super(MPINeighborhood, self).__init__('nb', 'neighborhood', fields)

    @property
    def entries(self):
        return self._entries

    @property
    def neighborhood(self):
        return self._neighborhood

    @cached_property
    def _C_typedecl(self):
        # Overriding for better code readability
        #
        # Struct neighborhood                 Struct neighborhood
        # {                                   {
        #   int ll;                             int ll, lc, lr;
        #   int lc;                 VS          ...
        #   int lr;                             ...
        #   ...                                 ...
        # }                                   }
        #
        # With this override, we generate the one on the right
        groups = [list(g) for k, g in groupby(self.pfields, key=lambda x: x[0][0])]
        groups = [(j[0], i) for i, j in [zip(*g) for g in groups]]
        return Struct(self.pname, [Value(ctypes_to_cstr(i), ', '.join(j))
                                   for i, j in groups])

    def _arg_defaults(self):
        values = super(MPINeighborhood, self)._arg_defaults()
        for name, i in zip(self.fields, self.entries):
            setattr(values[self.name]._obj, name, self.neighborhood[i])
        return values

    def _arg_values(self, *args, **kwargs):
        grid = kwargs.get('grid', None)
        # Update `nb` based on object attached to `grid`
        if grid is not None:
            return grid.distributor._obj_neighborhood._arg_defaults()
        else:
            return self._arg_defaults()


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
            return tuple(MPI.Compute_dims(nprocs, ndim))
    else:
        v = int(v)
    return tuple(v for _ in range(ndim))
