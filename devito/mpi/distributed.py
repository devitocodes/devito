from abc import ABC, abstractmethod
from ctypes import c_int, c_void_p, sizeof
from itertools import groupby, product
from functools import cached_property

from math import ceil, pow
from sympy import factorint, Interval

import atexit

import numpy as np
from cgen import Struct, Value

from devito.data import LEFT, CENTER, RIGHT, Decomposition
from devito.parameters import configuration
from devito.tools import (EnrichedTuple, as_tuple, ctypes_to_cstr, filter_ordered,
                          frozendict)
from devito.types import CompositeObject, Object, Constant
from devito.types.utils import DimensionTuple


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
        devito_mpi_finalize()
    atexit.register(cleanup)
except ImportError as e:
    # Dummy fallback in case mpi4py/MPI aren't available
    class NoneMetaclass(type):
        def __getattr__(self, name):
            return None

    class MPI(metaclass=NoneMetaclass):
        init_error = e

        @classmethod
        def Init(cls):
            raise cls.init_error

        @classmethod
        def Is_initialized(cls):
            return False

        def _sizeof(obj):
            return None

        def __getattr__(self, name):
            return None


__all__ = ['Distributor', 'SubDistributor', 'SparseDistributor', 'MPI',
           'CustomTopology', 'devito_mpi_init', 'devito_mpi_finalize']


def devito_mpi_init():
    """
    Initialize MPI, if not already initialized.
    """
    if not MPI.Is_initialized():
        try:
            thread_level = mpi4py_thread_levels[mpi4py.rc.thread_level]
        except KeyError:
            assert False

        MPI.Init_thread(thread_level)

        global init_by_devito
        init_by_devito = True

        return True
    return False


def devito_mpi_finalize():
    """
    Finalize MPI, if initialized by Devito.
    """
    global init_by_devito  # noqa: F824
    if init_by_devito and MPI.Is_initialized() and not MPI.Is_finalized():
        MPI.Finalize()


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
        return DimensionTuple(*[len(i) for i in self.glb_numb],
                              getters=self.dimensions)

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
            if strict:
                raise ValueError("`%s` must be one of the Distributor dimensions" % dim)
            else:
                return args[0]
        return self.decomposition[dim].index_glb_to_loc(*args)

    @cached_property
    def loc_empty(self):
        """This rank is empty"""
        return any(d.loc_empty for d in self.decomposition)


class DenseDistributor(AbstractDistributor):

    """
    Decompose a domain over a set of MPI processes.

    Notes
    -----
    This is an abstract class, which defines the interface that
    all subclasses are expected to implement.
    """

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
    def nprocs_local(self):
        if self.comm is not MPI.COMM_NULL:
            local_comm = MPI.Comm.Split_type(self.comm, MPI.COMM_TYPE_SHARED)
            node_size = local_comm.Get_size()
            local_comm.Free()
            return node_size
        else:
            return 1

    @property
    def topology(self):
        return self._topology

    @property
    def topology_logical(self):
        if isinstance(self.topology, CustomTopology):
            return self.topology.logical
        else:
            return None

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
            # In cases where ranks are empty, return a range(0, 0)
            ret.append(EnrichedTuple(*[range(min(j), max(j) + 1) if len(j) > 0
                                       else range(0, 0) for j in i],
                                     getters=self.dimensions))
        return tuple(ret)

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


class Distributor(DenseDistributor):

    """
    Decompose a grid over a set of MPI processes.

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
        super().__init__(shape, dimensions)

        if configuration['mpi']:
            # First time we enter here, we make sure MPI is initialized
            devito_mpi_init()

            # Note: the cloned communicator doesn't need to be explicitly freed;
            # mpi4py takes care of that when the object gets out of scope
            self._input_comm = (input_comm or MPI.COMM_WORLD).Clone()

            if topology is None:
                # `MPI.Compute_dims` sets the dimension sizes to be as close to
                # each other as possible, using an appropriate divisibility
                # algorithm. Thus, in 3D:
                # * topology[0] >= topology[1] >= topology[2]
                # * topology[0] * topology[1] * topology[2] == self._input_comm.size
                # However, `MPI.Compute_dims` is distro-dependent, so we have
                # to enforce some properties through our own wrapper (e.g.,
                # OpenMPI v3 does not guarantee that 9 ranks are arranged into
                # a 3x3 grid when shape=(9, 9))
                self._topology = compute_dims(self._input_comm.size, len(shape))
            else:
                # A custom topology may contain integers or the wildcard '*'
                self._topology = CustomTopology(topology, self._input_comm)

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

    @cached_property
    def is_boundary_rank(self):
        """
        MPI rank interfaces with the boundary of the domain.
        """
        return any([i == 0 or i == j-1 for i, j in
                   zip(self.mycoords, self.topology)])

    @cached_property
    def glb_pos_map(self):
        """
        A mapper `Dimension -> DataSide` providing the position of the calling
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
        A mapper `M` describing the calling MPI rank's neighborhood in the
        decomposed grid. Let

            * `d` be a Dimension -- `d0, d1, ..., dn` are the decomposed
              Dimensions.
            * `s` be a DataSide -- possible values are `LEFT, CENTER, RIGHT`,
            * `p` be the rank of a neighbour MPI process.

        Then `M` can be indexed in two ways:

            * `M[d] -> (s -> p)`; that is, `M[d]` returns a further mapper
              (from DataSide to MPI rank) from which the two adjacent processes
              along `d` can be retrieved.
            * `M[(s0, s1, ..., sn)] -> p`, where `s0` is the DataSide along
              `d0`, `s1` the DataSide along `d1`, and so on. This can be
              useful to retrieve the diagonal neighbours (e.g., `M[(LEFT, LEFT)]`
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

    def _rebuild(self, shape=None, dimensions=None, comm=None):
        return Distributor(shape or self.shape, dimensions or self.dimensions,
                           comm or self.comm)


class SubDistributor(DenseDistributor):

    """
    Decompose a subset of a domain over a set of MPI processes in a manner consistent
    with the parent distributor.

    Parameters
    ----------
    subdomain : SubDomain
        The subdomain to be decomposed.

    Notes
    -----
    This class is used internally by Devito for distributing
    Functions defined on SubDomains. It stores reference to the
    parent Distributor from which it is created.
    """

    def __init__(self, subdomain):
        # Does not keep reference to the SubDomain since SubDomain will point to
        # this SubDistributor and Distributor does not point to the Grid
        super().__init__(subdomain.shape, subdomain.dimensions)

        self._subdomain_name = subdomain.name
        self._dimension_map = frozendict({pd: sd for pd, sd
                                          in zip(subdomain.grid.dimensions,
                                                 subdomain.dimensions)})

        self._parent = subdomain.grid.distributor
        self._comm = self.parent.comm
        self._topology = self.parent.topology

        self.__decomposition_setup__()

    def __decomposition_setup__(self):
        """
        Set up the decomposition, aligned with that of the parent Distributor.
        """
        decompositions = []
        for dec, i in zip(self.parent._decomposition, self.subdomain_interval):
            if i is None:
                decompositions.append(dec)
            else:
                start, end = _interval_bounds(i)
                decompositions.append([d[np.logical_and(d >= start, d <= end)]
                                       for d in dec])

        self._decomposition = [Decomposition(d, c)
                               for d, c in zip(decompositions, self.mycoords)]

    @property
    def parent(self):
        """The parent distributor of this SubDistributor."""
        return self._parent

    @property
    def p(self):
        """Shortcut for `SubDistributor.parent`"""
        return self.parent

    @property
    def par_slices(self):
        """
        The global indices owned by the calling MPI rank, as a mapper from
        Dimensions to slices. Shortcut for `parent.glb_slices`.
        """
        return self.parent.glb_slices

    @property
    def dimension_map(self):
        return self._dimension_map

    @cached_property
    def domain_interval(self):
        """The interval spanned by this MPI rank."""
        return tuple(Interval(self.par_slices[d].start, self.par_slices[d].stop-1)
                     for d in self.p.dimensions)

    @cached_property
    def subdomain_interval(self):
        """The interval spanned by the SubDomain."""
        # Assumes no override of x_m and x_M supplied to operator
        bounds_map = {d.symbolic_min: 0 for d in self.p.dimensions}
        bounds_map.update({d.symbolic_max: s-1 for d, s in zip(self.p.dimensions,
                                                               self.p.glb_shape)})

        sd_interval = []  # The Interval of SubDimension indices
        for d in self.dimensions:
            if d.is_Sub:
                # Need to filter None from thicknesses as used as placeholder
                tkn_map = {tkn: tkn.value for tkn in d.thickness if tkn.value is not None}
                tkn_map.update(bounds_map)
                # Evaluate SubDimension thicknesses and substitute into Interval
                sd_interval.append(d._interval.subs(tkn_map))
            else:
                sd_interval.append(None)
        return tuple(sd_interval)

    @cached_property
    def intervals(self):
        """The interval spanned by the SubDomain in each dimension on this rank."""
        return tuple(d if s is None else d.intersect(s)
                     for d, s in zip(self.domain_interval, self.subdomain_interval))

    @cached_property
    def crosses(self):
        """
        A mapper `M` indicating the sides of this MPI rank crossed by the SubDomain.
        Let

            * `d` be a Dimension -- `d0, d1, ..., dn` are the dimensions of the parent
              distributor.
            * `s` be a DataSide -- possible values are `LEFT, CENTER, RIGHT`,
            * `c` be a bool indicating whether the SubDomain crosses the edge of the
              rank on this side.

        Then `M` can be indexed in two ways:

            * `M[d] -> (s -> c)`; that is, `M[d]` returns a further mapper
              (from DataSide to bool) used to determine whether the SubDomain crosses
              the edge of the rank on this side.
            * `M[(s0, s1, ..., sn)] -> c`, where `s0` is the DataSide along
              `d0`, `s1` the DataSide along `d1`, and so on. This can be
              determine whether the SubDomain crosses the edge of the rank on this side.
        """
        def get_crosses(d, di, si):
            if not d.is_Sub:
                return {LEFT: True, RIGHT: True}

            # SubDomain is either fully contained by or not present on this rank
            if di.issuperset(si) or di.isdisjoint(si):
                return {LEFT: False, RIGHT: False}
            elif d.local:
                raise ValueError("SubDimension %s is local and cannot be"
                                 " decomposed across MPI ranks" % d)
            return {LEFT: si.left < di.left,
                    RIGHT: si.right > di.right}

        crosses = {d: get_crosses(d, di, si) for d, di, si
                   in zip(self.dimensions, self.domain_interval,
                          self.subdomain_interval)}

        for i in product([LEFT, CENTER, RIGHT], repeat=len(self.dimensions)):
            crosses[i] = all(crosses[d][s] for d, s in zip(self.dimensions, i)
                             if s in crosses[d])  # Skip over CENTER

        return frozendict(crosses)

    @cached_property
    def is_boundary_rank(self):
        """
        MPI rank interfaces with the boundary of the subdomain.
        """
        # Note that domain edges may also be boundaries of the subdomain
        grid_boundary = self.parent.is_boundary_rank
        subdomain_boundary = any(not all(self.crosses[d].values())
                                 for d in self.dimensions)
        return grid_boundary or subdomain_boundary

    @property
    def neighborhood(self):
        """
        A mapper `M` describing the calling MPI rank's neighborhood in the
        decomposed grid. Let

            * `d` be a Dimension -- `d0, d1, ..., dn` are the decomposed
              Dimensions.
            * `s` be a DataSide -- possible values are `LEFT, CENTER, RIGHT`,
            * `p` be the rank of a neighbour MPI process.

        Then `M` can be indexed in two ways:

            * `M[d] -> (s -> p)`; that is, `M[d]` returns a further mapper
              (from DataSide to MPI rank) from which the two adjacent processes
              along `d` can be retrieved.
            * `M[(s0, s1, ..., sn)] -> p`, where `s0` is the DataSide along
              `d0`, `s1` the DataSide along `d1`, and so on. This can be
              useful to retrieve the diagonal neighbours (e.g., `M[(LEFT, LEFT)]`
              gives the top-left neighbour in a 2D grid).
        """
        shifts = {d: self.comm.Shift(i, 1) for i, d in enumerate(self.dimensions)}
        ret = {}
        for d, (src, dest) in shifts.items():
            ret[d] = {}
            ret[d][LEFT] = MPI.PROC_NULL if not self.crosses[d][LEFT] else src
            ret[d][RIGHT] = MPI.PROC_NULL if not self.crosses[d][RIGHT] else dest

        # Set up diagonal neighbours
        for i in product([LEFT, CENTER, RIGHT], repeat=self.ndim):
            neighbor = [c + s.val for c, s in zip(self.mycoords, i)]

            if any(c < 0 or c >= s for c, s in zip(neighbor, self.topology)) \
               or not self.crosses[i]:
                ret[i] = MPI.PROC_NULL
            else:
                ret[i] = self.comm.Get_cart_rank(neighbor)

        return ret

    @cached_property
    def rank_populated(self):
        """Constant symbol for a switch indicating that data is allocated on this rank"""
        return Constant(name=f'rank_populated_{self._subdomain_name}', dtype=np.int8,
                        value=int(not(self.loc_empty)))


def _interval_bounds(interval):
    """Extract SubDimension Interval bounds."""
    if interval.is_empty:
        # SubDimension has no indices. Min and max are +ve and -ve inf respectively.
        return np.inf, -np.inf
    elif interval.is_Interval:
        # Interval containing two or more indices. Min and max are ends.
        return interval.start, interval.end
    else:
        # Interval where start == end defaults to FiniteSet
        # Repeat this value for min and max
        return interval.args[0], interval.args[0]


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
        super().__init__(npoint, dimension)
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
        if nprocs == 1:
            return (npoint,)
        elif isinstance(npoint, int):
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

    @cached_property
    def all_ranges(self):
        """The global ranges of all MPI ranks."""
        ret = []
        for i in self.decomposition[0]:
            # i might be empty if there is less receivers than rank such as for a
            # point source
            try:
                ret.append(EnrichedTuple(range(min(i), max(i) + 1),
                                         getters=self.dimensions))
            except ValueError:
                ret.append(EnrichedTuple(range(0, 0), getters=self.dimensions))
        return tuple(ret)

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
        super().__init__('nb', 'neighborhood', fields)

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
        values = super()._arg_defaults()
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


class CustomTopology(tuple):

    """
    The CustomTopology class provides a mechanism to describe parametric domain
    decompositions. It allows users to specify how the dimensions of a domain are
    decomposed into chunks based on certain parameters.

    Examples
    --------
    For example, let's consider a domain with three distributed dimensions: x,
    y, and z, and an MPI communicator with N processes. Here are a few examples
    of CustomTopology:

    With N known, say N=4:
    * `(1, 1, 4)`: the z Dimension is decomposed into 4 chunks
    * `(2, 1, 2)`: the x Dimension is decomposed into 2 chunks and the z Dimension
                   is decomposed into 2 chunks

    With N unknown:
    * `(1, '*', 1)`: the wildcard `'*'` indicates that the runtime should
                     decompose the y Dimension into N chunks
    * `('*', '*', 1)`: the wildcard `'*'` indicates that the runtime should
                       decompose both the x and y Dimensions in `nstars` factors
                       of N, prioritizing the outermost dimension

    Assuming that the number of ranks `N` cannot evenly be decomposed to the
    requested stars=6 we decompose as evenly as possible by prioritising the
    outermost dimension

    For N=3
    * `('*', '*', 1)` gives: (3, 1, 1)
    * `('*', 1, '*')` gives: (3, 1, 1)
    * `(1, '*', '*')` gives: (1, 3, 1)

    For N=6
    * `('*', '*', 1)` gives: (3, 2, 1)
    * `('*', 1, '*')` gives: (3, 1, 2)
    * `(1, '*', '*')` gives: (1, 3, 2)

    For N=8
    * `('*', '*', '*')` gives: (2, 2, 2)
    * `('*', '*', 1)` gives: (4, 2, 1)
    * `('*', 1, '*')` gives: (4, 1, 2)
    * `(1, '*', '*')` gives: (1, 4, 2)

    Notes
    -----
    Users should not directly use the CustomTopology class. It is instantiated
    by the Devito runtime based on user input.
    """

    _shortcuts = {
        'x': ('*', 1, 1),
        'y': (1, '*', 1),
        'z': (1, 1, '*'),
        'xy': ('*', '*', 1),
    }

    def __new__(cls, items, input_comm):
        # Keep track of nstars and already defined decompositions
        nstars = items.count('*')

        # If no stars exist we are ready
        if nstars == 0:
            processed = items
        else:
            # Init decomposition list and track star positions
            processed = [1] * len(items)
            star_pos = []
            for i, item in enumerate(items):
                if isinstance(item, int):
                    processed[i] = item
                else:
                    star_pos.append(i)

            # Compute the remaining procs to be allocated
            alloc_procs = np.prod([i for i in items if i != '*'])
            rem_procs = int(input_comm.size // alloc_procs)

            # List of all factors of rem_procs in decreasing order
            factors = factorint(rem_procs)
            vals = [k for (k, v) in factors.items() for _ in range(v)][::-1]

            # Split in number of stars
            split = np.array_split(vals, nstars)

            # Reduce
            star_vals = [int(np.prod(s)) for s in split]

            # Apply computed star values to the processed
            for index, value in zip(star_pos, star_vals):
                processed[index] = value

        # Final check that topology matches the communicator size
        assert np.prod(processed) == input_comm.size

        obj = super().__new__(cls, processed)
        obj.logical = items

        return obj


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


# Yes, AFAICT, nothing like this is available in mpi4py
mpi4py_thread_levels = {
    'single': MPI.THREAD_SINGLE,
    'funneled': MPI.THREAD_FUNNELED,
    'serialized': MPI.THREAD_SERIALIZED,
    'multiple': MPI.THREAD_MULTIPLE
}
