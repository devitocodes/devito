from collections import OrderedDict
from itertools import product

import sympy
import numpy as np
from cached_property import cached_property

from devito.finite_differences import generate_fd_shortcuts
from devito.finite_differences.elementary import floor
from devito.mpi import MPI, SparseDistributor
from devito.operations import LinearInterpolator, PrecomputedInterpolator
from devito.symbolics import (INT, cast_mapper, indexify,
                              retrieve_function_carriers)
from devito.tools import (ReducerMap, as_tuple, flatten, prod, filter_ordered,
                          memoized_meth, is_integer)
from devito.types.dense import DiscreteFunction, Function, SubFunction
from devito.types.dimension import (Dimension, ConditionalDimension, DefaultDimension,
                                    DynamicDimension)
from devito.types.basic import Symbol
from devito.types.equation import Eq, Inc
from devito.types.utils import IgnoreDimSort


__all__ = ['SparseFunction', 'SparseTimeFunction', 'PrecomputedSparseFunction',
           'PrecomputedSparseTimeFunction', 'MatrixSparseTimeFunction']


class AbstractSparseFunction(DiscreteFunction):

    """
    An abstract class to define behaviours common to all sparse functions.
    """

    _sparse_position = -1
    """Position of sparse index among the function indices."""

    _radius = 0
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ()
    """SubFunctions encapsulated within this AbstractSparseFunction."""

    __rkwargs__ = DiscreteFunction.__rkwargs__ + ('npoint', 'space_order')

    def __init_finalize__(self, *args, **kwargs):
        super(AbstractSparseFunction, self).__init_finalize__(*args, **kwargs)
        self._npoint = kwargs['npoint']
        self._space_order = kwargs.get('space_order', 0)

        # Dynamically add derivative short-cuts
        self._fd = self.__fd_setup__()

    def __fd_setup__(self):
        """
        Dynamically add derivative short-cuts.
        """
        return generate_fd_shortcuts(self.dimensions, self.space_order)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = as_tuple(kwargs.get('dimensions'))
        if not dimensions:
            dimensions = (Dimension(name='p_%s' % kwargs["name"]),)
        return dimensions, dimensions

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        # A Grid must have been provided
        if grid is None:
            raise TypeError('Need `grid` argument')
        shape = kwargs.get('shape')
        npoint = kwargs['npoint']
        if shape is None:
            glb_npoint = SparseDistributor.decompose(npoint, grid.distributor)
            shape = (glb_npoint[grid.distributor.myrank],)
        return shape

    def _halo_exchange(self):
        # no-op for SparseFunctions
        return

    @property
    def npoint(self):
        return self.shape[self._sparse_position]

    @property
    def space_order(self):
        """The space order."""
        return self._space_order

    @property
    def _sparse_dim(self):
        return self.dimensions[self._sparse_position]

    @property
    def gridpoints(self):
        """
        The *reference* grid point corresponding to each sparse point.

        Notes
        -----
        When using MPI, this property refers to the *physically* owned
        sparse points.
        """
        raise NotImplementedError

    def interpolate(self, *args, **kwargs):
        """
        Implement an interpolation operation from the grid onto the given sparse points
        """
        return self.interpolator.interpolate(*args, **kwargs)

    def inject(self, *args, **kwargs):
        """
        Implement an injection operation from a sparse point onto the grid
        """
        return self.interpolator.inject(*args, **kwargs)

    @cached_property
    def _point_support(self):
        return np.array(tuple(product(range(-self._radius + 1, self._radius + 1),
                                      repeat=self.grid.dim)))

    @property
    def _support(self):
        """
        The grid points surrounding each sparse point within the radius of self's
        injection/interpolation operators.
        """
        max_shape = np.array(self.grid.shape).reshape(1, self.grid.dim)
        minmax = lambda arr: np.minimum(max_shape, np.maximum(0, arr))
        return np.stack([minmax(self.gridpoints + s) for s in self._point_support],
                        axis=2)

    @property
    def _dist_datamap(self):
        """
        Mapper ``M : MPI rank -> required sparse data``.
        """
        return self.grid.distributor.glb_to_rank(self._support) or {}

    def _dist_scatter_mask(self, dmap=None):
        """
        A mask to index into ``self.data``, which creates a new data array that
        logically contains N consecutive groups of sparse data values, where N
        is the number of MPI ranks. The i-th group contains the sparse data
        values accessible by the i-th MPI rank.  Thus, sparse data values along
        the boundary of two or more MPI ranks are duplicated.
        """
        dmap = dmap or self._dist_datamap
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        ret = [slice(None) for _ in range(self.ndim)]
        ret[self._sparse_position] = mask
        return tuple(ret)

    def _dist_count(self, dmap=None):
        """
        A 2-tuple of comm-sized iterables, which tells how many sparse points
        is this MPI rank expected to send/receive to/from each other MPI rank.
        """
        dmap = dmap or self._dist_datamap
        comm = self.grid.distributor.comm

        ssparse = np.array([len(dmap.get(i, [])) for i in range(comm.size)], dtype=int)
        rsparse = np.empty(comm.size, dtype=int)
        comm.Alltoall(ssparse, rsparse)

        return ssparse, rsparse

    @cached_property
    def _dist_reorder_mask(self):
        """
        An ordering mask that puts ``self._sparse_position`` at the front.
        """
        ret = (self._sparse_position,)
        ret += tuple(i for i, d in enumerate(self.indices) if d is not self._sparse_dim)
        return ret

    def _dist_alltoall(self, dmap=None):
        """
        The metadata necessary to perform an ``MPI_Alltoallv`` distributing the
        sparse data values across the MPI ranks needing them.
        """
        ssparse, rsparse = self._dist_count(dmap=dmap)

        # Per-rank shape of send/recv data
        sshape = []
        rshape = []
        for s, r in zip(ssparse, rsparse):
            handle = list(self.shape)
            handle[self._sparse_position] = s
            sshape.append(tuple(handle))

            handle = list(self.shape)
            handle[self._sparse_position] = r
            rshape.append(tuple(handle))

        # Per-rank count of send/recv data
        scount = tuple(prod(i) for i in sshape)
        rcount = tuple(prod(i) for i in rshape)

        # Per-rank displacement of send/recv data (it's actually all contiguous,
        # but the Alltoallv needs this information anyway)
        sdisp = np.concatenate([[0], np.cumsum(scount)[:-1]])
        rdisp = np.concatenate([[0], tuple(np.cumsum(rcount))[:-1]])

        # Total shape of send/recv data
        sshape = list(self.shape)
        sshape[self._sparse_position] = sum(ssparse)
        rshape = list(self.shape)
        rshape[self._sparse_position] = sum(rsparse)

        # May have to swap axes, as `MPI_Alltoallv` expects contiguous data, and
        # the sparse dimension may not be the outermost
        sshape = tuple(sshape[i] for i in self._dist_reorder_mask)
        rshape = tuple(rshape[i] for i in self._dist_reorder_mask)

        return sshape, scount, sdisp, rshape, rcount, rdisp

    def _dist_subfunc_alltoall(self, dmap=None):
        """
        The metadata necessary to perform an ``MPI_Alltoallv`` distributing
        self's SubFunction values across the MPI ranks needing them.
        """
        raise NotImplementedError

    def _dist_scatter(self):
        """
        A ``numpy.ndarray`` containing up-to-date data values belonging
        to the calling MPI rank. A data value belongs to a given MPI rank R
        if its coordinates fall within R's local domain.
        """
        raise NotImplementedError

    def _arg_defaults(self, alias=None):
        key = alias or self
        mapper = {self: key}
        mapper.update({getattr(self, i): getattr(key, i) for i in self._sub_functions})
        args = ReducerMap()

        # Add in the sparse data (as well as any SubFunction data) belonging to
        # self's local domain only
        for k, v in self._dist_scatter().items():
            args[mapper[k].name] = v
            for i, s in zip(mapper[k].indices, v.shape):
                args.update(i._arg_defaults(_min=0, size=s))

        return args

    def _eval_at(self, func):
        return self

    def _arg_values(self, **kwargs):
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, AbstractSparseFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self).reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {}
                for k, v in self._dist_scatter(new).items():
                    values[k.name] = v
                    for i, s in zip(k.indices, v.shape):
                        size = s - sum(k._size_nodomain[i])
                        values.update(i._arg_defaults(size=size))
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_apply(self, dataobj, coordsobj, alias=None):
        key = alias if alias is not None else self
        if isinstance(key, AbstractSparseFunction):
            # Gather into `self.data`
            # Coords may be None if the coordinates are not used in the Operator
            if coordsobj is None:
                pass
            elif np.sum([coordsobj._obj.size[i] for i in range(self.ndim)]) > 0:
                coordsobj = self.coordinates._C_as_ndarray(coordsobj)
            key._dist_gather(self._C_as_ndarray(dataobj), coordsobj)
        elif self.grid.distributor.nprocs > 1:
            raise NotImplementedError("Don't know how to gather data from an "
                                      "object of type `%s`" % type(key))


class AbstractSparseTimeFunction(AbstractSparseFunction):

    """
    An abstract class to define behaviours common to all sparse time-varying functions.
    """

    _time_position = 0
    """Position of time index among the function indices."""

    __rkwargs__ = AbstractSparseFunction.__rkwargs__ + ('nt', 'time_order')

    def __init_finalize__(self, *args, **kwargs):
        self._time_dim = self.indices[self._time_position]
        self._time_order = kwargs.get('time_order', 1)
        if not isinstance(self.time_order, int):
            raise ValueError("`time_order` must be int")

        super(AbstractSparseTimeFunction, self).__init_finalize__(*args, **kwargs)

    def __fd_setup__(self):
        """
        Dynamically add derivative short-cuts.
        """
        return generate_fd_shortcuts(self.dimensions, self.space_order,
                                     to=self.time_order)

    @property
    def time_dim(self):
        """The time dimension."""
        return self._time_dim

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = as_tuple(kwargs.get('dimensions'))
        if not dimensions:
            dimensions = (kwargs['grid'].time_dim,
                          Dimension(name='p_%s' % kwargs["name"]))
        return dimensions, dimensions

    @classmethod
    def __shape_setup__(cls, **kwargs):
        shape = kwargs.get('shape')
        if shape is None:
            nt = kwargs.get('nt')
            if not isinstance(nt, int):
                raise TypeError('Need `nt` int argument')
            if nt <= 0:
                raise ValueError('`nt` must be > 0')

            shape = list(AbstractSparseFunction.__shape_setup__(**kwargs))
            shape.insert(cls._time_position, nt)

        return tuple(shape)

    @property
    def nt(self):
        return self.shape[self._time_position]

    @property
    def time_order(self):
        """The time order."""
        return self._time_order

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]


class SparseFunction(AbstractSparseFunction):
    """
    Tensor symbol representing a sparse array in symbolic equations.

    A SparseFunction carries multi-dimensional data that are not aligned with
    the computational grid. As such, each data value is associated some coordinates.
    A SparseFunction provides symbolic interpolation routines to convert between
    Functions and sparse data points. These are based upon standard [bi,tri]linear
    interpolation.

    Parameters
    ----------
    name : str
        Name of the symbol.
    npoint : int
        Number of sparse points.
    grid : Grid
        The computational domain from which the sparse points are sampled.
    coordinates : np.ndarray, optional
        The coordinates of each sparse point.
    space_order : int, optional
        Discretisation order for space derivatives. Defaults to 0.
    shape : tuple of ints, optional
        Shape of the object. Defaults to ``(npoint,)``.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if the SparseFunction
        defines a multi-dimensional tensor.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.

    Examples
    --------

    Creation

    >>> from devito import Grid, SparseFunction
    >>> grid = Grid(shape=(4, 4))
    >>> sf = SparseFunction(name='sf', grid=grid, npoint=2)
    >>> sf
    sf(p_sf)

    Inspection

    >>> sf.data
    Data([0., 0.], dtype=float32)
    >>> sf.coordinates
    sf_coords(p_sf, d)
    >>> sf.coordinates_data
    array([[0., 0.],
           [0., 0.]], dtype=float32)

    Symbolic interpolation routines

    >>> from devito import Function
    >>> f = Function(name='f', grid=grid)
    >>> exprs0 = sf.interpolate(f)
    >>> exprs1 = sf.inject(f, sf)

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    About SparseFunction and MPI. There is a clear difference between:

        * Where the sparse points *physically* live, i.e., on which MPI rank. This
          depends on the user code, particularly on how the data is set up.
        * and which MPI rank *logically* owns a given sparse point. The logical
          ownership depends on where the sparse point is located within ``self.grid``.

    Right before running an Operator (i.e., upon a call to ``op.apply``), a
    SparseFunction "scatters" its physically owned sparse points so that each
    MPI rank gets temporary access to all of its logically owned sparse points.
    A "gather" operation, executed before returning control to user-land,
    updates the physically owned sparse points in ``self.data`` by collecting
    the values computed during ``op.apply`` from different MPI ranks.
    """

    is_SparseFunction = True

    _radius = 1
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ('coordinates',)

    __rkwargs__ = AbstractSparseFunction.__rkwargs__ + ('coordinates_data',)

    def __init_finalize__(self, *args, **kwargs):
        super(SparseFunction, self).__init_finalize__(*args, **kwargs)
        self.interpolator = LinearInterpolator(self)
        # Set up sparse point coordinates
        coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
        if isinstance(coordinates, Function):
            self._coordinates = coordinates
        else:
            dimensions = (self.indices[self._sparse_position], Dimension(name='d'))
            # Only retain the local data region
            if coordinates is not None:
                coordinates = np.array(coordinates)
            self._coordinates = SubFunction(
                name='%s_coords' % self.name, parent=self, dtype=self.dtype,
                dimensions=dimensions, shape=(self.npoint, self.grid.dim),
                space_order=0, initializer=coordinates, alias=self.alias,
                distributor=self._distributor
            )
            if self.npoint == 0:
                # This is a corner case -- we might get here, for example, when
                # running with MPI and some processes get 0-size arrays after
                # domain decomposition. We "touch" the data anyway to avoid the
                # case ``self._data is None``
                self.coordinates.data

    def __distributor_setup__(self, **kwargs):
        """
        A `SparseDistributor` handles the SparseFunction decomposition based on
        physical ownership, and allows to convert between global and local indices.
        """
        return SparseDistributor(kwargs['npoint'], self._sparse_dim,
                                 kwargs['grid'].distributor)

    @property
    def coordinates(self):
        """The SparseFunction coordinates."""
        return self._coordinates

    @property
    def coordinates_data(self):
        return self.coordinates.data.view(np.ndarray)

    @cached_property
    def _point_symbols(self):
        """Symbol for coordinate value in each dimension of the point."""
        return tuple(Symbol(name='p%s' % d, dtype=self.dtype)
                     for d in self.grid.dimensions)

    @cached_property
    def _position_map(self):
        """
        Symbols map for the position of the sparse points relative to the grid
        origin.

        Notes
        -----
        The expression `(coord - origin)/spacing` could also be computed in the
        mathematically equivalent expanded form `coord/spacing -
        origin/spacing`. This particular form is problematic when a sparse
        point is in close proximity of the grid origin, since due to a larger
        machine precision error it may cause a +-1 error in the computation of
        the position. We mitigate this problem by computing the positions
        individually (hence the need for a position map).
        """
        symbols = [Symbol(name='pos%s' % d, dtype=self.dtype)
                   for d in self.grid.dimensions]
        return OrderedDict([(c - o, p) for p, c, o in zip(symbols,
                                                          self._coordinate_symbols,
                                                          self.grid.origin_symbols)])

    @cached_property
    def _point_increments(self):
        """Index increments in each dimension for each point symbol."""
        return tuple(product(range(2), repeat=self.grid.dim))

    @cached_property
    def _coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension."""
        p_dim = self.indices[self._sparse_position]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @cached_property
    def _coordinate_indices(self):
        """Symbol for each grid index according to the coordinates."""
        return tuple([INT(floor((c - o) / i.spacing))
                      for c, o, i in zip(self._coordinate_symbols,
                                         self.grid.origin_symbols,
                                         self.grid.dimensions[:self.grid.dim])])

    def _coordinate_bases(self, field_offset):
        """Symbol for the base coordinates of the reference grid point."""
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i, of in zip(self._coordinate_symbols,
                                                  self.grid.origin_symbols,
                                                  self._coordinate_indices,
                                                  self.grid.dimensions[:self.grid.dim],
                                                  field_offset)])

    @memoized_meth
    def _index_matrix(self, offset):
        # Note about the use of *memoization*
        # Since this method is called by `_interpolation_indices`, using
        # memoization avoids a proliferation of symbolically identical
        # ConditionalDimensions for a given set of indirection indices

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self._coordinate_indices))
                        for inc in self._point_increments]

        # A unique symbol for each indirection index
        indices = filter_ordered(flatten(index_matrix))
        points = OrderedDict([(p, Symbol(name='ii_%s_%d' % (self.name, i)))
                              for i, p in enumerate(indices)])

        return index_matrix, points

    @property
    def gridpoints(self):
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        return (
            np.floor(self.coordinates.data._local - self.grid.origin) / self.grid.spacing
        ).astype(int)

    def guard(self, expr=None, offset=0):
        """
        Generate guarded expressions, that is expressions that are evaluated
        by an Operator only if certain conditions are met.  The introduced
        condition, here, is that all grid points in the support of a sparse
        value must fall within the grid domain (i.e., *not* on the halo).

        Parameters
        ----------
        expr : expr-like, optional
            Input expression, from which the guarded expression is derived.
            If not specified, defaults to ``self``.
        offset : int, optional
            Relax the guard condition by introducing a tolerance offset.
        """
        _, points = self._index_matrix(offset)

        # Guard through ConditionalDimension
        conditions = {}
        for d, idx in zip(self.grid.dimensions, self._coordinate_indices):
            p = points[idx]
            lb = sympy.And(p >= d.symbolic_min - offset, evaluate=False)
            ub = sympy.And(p <= d.symbolic_max + offset, evaluate=False)
            conditions[p] = sympy.And(lb, ub, evaluate=False)
        condition = sympy.And(*conditions.values(), evaluate=False)
        cd = ConditionalDimension(self._sparse_dim.name, self._sparse_dim,
                                  condition=condition, indirect=True)

        if expr is None:
            out = self.indexify().xreplace({self._sparse_dim: cd})
        else:
            functions = {f for f in retrieve_function_carriers(expr)
                         if f.is_SparseFunction}
            out = indexify(expr).xreplace({f._sparse_dim: cd for f in functions})

        # Temporaries for the position
        temps = [Eq(v, k, implicit_dims=self.dimensions)
                 for k, v in self._position_map.items()]
        # Temporaries for the indirection dimensions
        temps.extend([Eq(v, k.subs(self._position_map),
                         implicit_dims=self.dimensions)
                      for k, v in points.items() if v in conditions])

        return out, temps

    @cached_property
    def _decomposition(self):
        mapper = {self._sparse_dim: self._distributor.decomposition[self._sparse_dim]}
        return tuple(mapper.get(d) for d in self.dimensions)

    def _dist_subfunc_alltoall(self, dmap=None):
        dmap = dmap or self._dist_datamap
        ssparse, rsparse = self._dist_count(dmap=dmap)

        # Per-rank shape of send/recv `coordinates`
        sshape = [(i, self.grid.dim) for i in ssparse]
        rshape = [(i, self.grid.dim) for i in rsparse]

        # Per-rank count of send/recv `coordinates`
        scount = [prod(i) for i in sshape]
        rcount = [prod(i) for i in rshape]

        # Per-rank displacement of send/recv `coordinates` (it's actually all
        # contiguous, but the Alltoallv needs this information anyway)
        sdisp = np.concatenate([[0], np.cumsum(scount)[:-1]])
        rdisp = np.concatenate([[0], tuple(np.cumsum(rcount))[:-1]])

        # Total shape of send/recv `coordinates`
        sshape = list(self.coordinates.shape)
        sshape[0] = sum(ssparse)
        rshape = list(self.coordinates.shape)
        rshape[0] = sum(rsparse)

        return sshape, scount, sdisp, rshape, rcount, rdisp

    def _dist_scatter(self, data=None):
        data = data if data is not None else self.data._local
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.coordinates: self.coordinates.data}

        comm = distributor.comm
        mpitype = MPI._typedict[np.dtype(self.dtype).char]

        # Compute dist map only once
        dmap = self._dist_datamap
        mask = self._dist_scatter_mask(dmap=dmap)

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = data[mask]
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))

        # Send out the sparse point values
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_alltoall(dmap=dmap)
        scattered = np.empty(shape=rshape, dtype=self.dtype)
        comm.Alltoallv([data, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        data = scattered

        # Unpack data values so that they follow the expected storage layout
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))

        # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
        coords = self.coordinates.data._local[mask[self._sparse_position]]

        # Send out the sparse point coordinates
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_subfunc_alltoall(dmap=dmap)
        scattered = np.empty(shape=rshape, dtype=self.coordinates.dtype)
        comm.Alltoallv([coords, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        coords = scattered

        # Translate global coordinates into local coordinates
        coords = coords - np.array(self.grid.origin_offset, dtype=self.dtype)

        return {self: data, self.coordinates: coords}

    def _dist_gather(self, data, coords):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        comm = distributor.comm

        # Compute dist map only once
        dmap = self._dist_datamap
        mask = self._dist_scatter_mask(dmap=dmap)

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        # Send back the sparse point values
        sshape, scount, sdisp, _, rcount, rdisp = self._dist_alltoall(dmap=dmap)
        gathered = np.empty(shape=sshape, dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, rcount, rdisp, mpitype],
                       [gathered, scount, sdisp, mpitype])
        # Unpack data values so that they follow the expected storage layout
        gathered = np.ascontiguousarray(np.transpose(gathered, self._dist_reorder_mask))
        self._data[mask] = gathered[:]

        if coords is not None:
            # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
            coords = coords + np.array(self.grid.origin_offset, dtype=self.dtype)
            # Send out the sparse point coordinates
            sshape, scount, sdisp, _, rcount, rdisp = \
                self._dist_subfunc_alltoall(dmap=dmap)
            gathered = np.empty(shape=sshape, dtype=self.coordinates.dtype)
            mpitype = MPI._typedict[np.dtype(self.coordinates.dtype).char]
            comm.Alltoallv([coords, rcount, rdisp, mpitype],
                           [gathered, scount, sdisp, mpitype])
            self._coordinates.data._local[mask[self._sparse_position]] = gathered[:]

        # Note: this method "mirrors" `_dist_scatter`: a sparse point that is sent
        # in `_dist_scatter` is here received; a sparse point that is received in
        # `_dist_scatter` is here sent.


class SparseTimeFunction(AbstractSparseTimeFunction, SparseFunction):
    """
    Tensor symbol representing a space- and time-varying sparse array in symbolic
    equations.

    Like SparseFunction, SparseTimeFunction carries multi-dimensional data that
    are not aligned with the computational grid. As such, each data value is
    associated some coordinates.
    A SparseTimeFunction provides symbolic interpolation routines to convert
    between TimeFunctions and sparse data points. These are based upon standard
    [bi,tri]linear interpolation.

    Parameters
    ----------
    name : str
        Name of the symbol.
    npoint : int
        Number of sparse points.
    nt : int
        Number of timesteps along the time dimension.
    grid : Grid
        The computational domain from which the sparse points are sampled.
    coordinates : np.ndarray, optional
        The coordinates of each sparse point.
    space_order : int, optional
        Discretisation order for space derivatives. Defaults to 0.
    time_order : int, optional
        Discretisation order for time derivatives. Defaults to 1.
    shape : tuple of ints, optional
        Shape of the object. Defaults to ``(nt, npoint)``.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if the SparseFunction
        defines a multi-dimensional tensor.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.

    Examples
    --------

    Creation

    >>> from devito import Grid, SparseTimeFunction
    >>> grid = Grid(shape=(4, 4))
    >>> sf = SparseTimeFunction(name='sf', grid=grid, npoint=2, nt=3)
    >>> sf
    sf(time, p_sf)

    Inspection

    >>> sf.data
    Data([[0., 0.],
          [0., 0.],
          [0., 0.]], dtype=float32)
    >>> sf.coordinates
    sf_coords(p_sf, d)
    >>> sf.coordinates_data
    array([[0., 0.],
           [0., 0.]], dtype=float32)

    Symbolic interpolation routines

    >>> from devito import TimeFunction
    >>> f = TimeFunction(name='f', grid=grid)
    >>> exprs0 = sf.interpolate(f)
    >>> exprs1 = sf.inject(f, sf)

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_SparseTimeFunction = True

    __rkwargs__ = (AbstractSparseTimeFunction.__rkwargs__ +
                   SparseFunction.__rkwargs__)

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        u_t : expr-like, optional
            Time index at which the interpolation is performed.
        p_t : expr-like, optional
            Time index at which the result of the interpolation is stored.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        # Apply optional time symbol substitutions to expr
        subs = {}
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs({time: u_t, t: u_t})

        if p_t is not None:
            subs = {self.time_dim: p_t}

        return super(SparseTimeFunction, self).interpolate(expr, offset=offset,
                                                           increment=increment,
                                                           self_subs=subs)

    def inject(self, field, expr, offset=0, u_t=None, p_t=None, implicit_dims=None):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        u_t : expr-like, optional
            Time index at which the interpolation is performed.
        p_t : expr-like, optional
            Time index at which the result of the interpolation is stored.
        implicit_dims : Dimension or list of Dimension, optional
            An ordered list of Dimensions that do not explicitly appear in the
            injection expression, but that should be honored when constructing
            the operator.
        """
        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs({field.time_dim: u_t})
        if p_t is not None:
            expr = expr.subs({self.time_dim: p_t})

        return super().inject(field, expr, offset=offset, implicit_dims=implicit_dims)


class PrecomputedSparseFunction(AbstractSparseFunction):
    """
    Tensor symbol representing a sparse array in symbolic equations; unlike
    SparseFunction, PrecomputedSparseFunction uses externally-defined data
    for interpolation.

    Parameters
    ----------
    name : str
        Name of the symbol.
    npoint : int
        Number of sparse points.
    grid : Grid
        The computational domain from which the sparse points are sampled.
    r : int
        Number of gridpoints in each dimension to interpolate a single sparse
        point to. E.g. ``r=2`` for linear interpolation.
    gridpoints : np.ndarray, optional
        An array carrying the *reference* grid point corresponding to each sparse point.
        Of all the gridpoints that one sparse point would be interpolated to, this is the
        grid point closest to the origin, i.e. the one with the lowest value of each
        coordinate dimension. Must be a two-dimensional array of shape
        ``(npoint, grid.ndim)``.
    interpolation_coeffs : np.ndarray, optional
        An array containing the coefficient for each of the r^2 (2D) or r^3 (3D)
        gridpoints that each sparse point will be interpolated to. The coefficient is
        split across the n dimensions such that the contribution of the point (i, j, k)
        will be multiplied by ``interpolation_coeffs[..., i]*interpolation_coeffs[...,
        j]*interpolation_coeffs[...,k]``. So for ``r=6``, we will store 18
        coefficients per sparse point (instead of potentially 216).
        Must be a three-dimensional array of shape ``(npoint, grid.ndim, r)``.
    space_order : int, optional
        Discretisation order for space derivatives. Defaults to 0.
    shape : tuple of ints, optional
        Shape of the object. Defaults to ``(npoint,)``.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if the SparseFunction
        defines a multi-dimensional tensor.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_PrecomputedSparseFunction = True

    _sub_functions = ('gridpoints', 'interpolation_coeffs')

    def __init_finalize__(self, *args, **kwargs):
        super(PrecomputedSparseFunction, self).__init_finalize__(*args, **kwargs)

        # Grid points per sparse point (2 in the case of bilinear and trilinear)
        r = kwargs.get('r')
        gridpoints = kwargs.get('gridpoints')
        interpolation_coeffs = kwargs.get('interpolation_coeffs')

        self.interpolator = PrecomputedInterpolator(self, r, gridpoints,
                                                    interpolation_coeffs)

    @property
    def gridpoints(self):
        return self._gridpoints

    @property
    def interpolation_coeffs(self):
        """ The Precomputed interpolation coefficients."""
        return self._interpolation_coeffs

    def _dist_scatter(self, data=None):
        data = data if data is not None else self.data
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.gridpoints: self.gridpoints.data,
                    self._interpolation_coeffs: self._interpolation_coeffs.data}

        raise NotImplementedError

    def _dist_gather(self, data):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        raise NotImplementedError

    def _arg_apply(self, *args, **kwargs):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        raise NotImplementedError


class PrecomputedSparseTimeFunction(AbstractSparseTimeFunction,
                                    PrecomputedSparseFunction):
    """
    Tensor symbol representing a space- and time-varying sparse array in symbolic
    equations; unlike SparseTimeFunction, PrecomputedSparseTimeFunction uses
    externally-defined data for interpolation.

    Parameters
    ----------
    name : str
        Name of the symbol.
    npoint : int
        Number of sparse points.
    grid : Grid
        The computational domain from which the sparse points are sampled.
    r : int
        Number of gridpoints in each dimension to interpolate a single sparse
        point to. E.g. ``r=2`` for linear interpolation.
    gridpoints : np.ndarray, optional
        An array carrying the *reference* grid point corresponding to each sparse point.
        Of all the gridpoints that one sparse point would be interpolated to, this is the
        grid point closest to the origin, i.e. the one with the lowest value of each
        coordinate dimension. Must be a two-dimensional array of shape
        ``(npoint, grid.ndim)``.
    interpolation_coeffs : np.ndarray, optional
        An array containing the coefficient for each of the r^2 (2D) or r^3 (3D)
        gridpoints that each sparse point will be interpolated to. The coefficient is
        split across the n dimensions such that the contribution of the point (i, j, k)
        will be multiplied by ``interpolation_coeffs[..., i]*interpolation_coeffs[...,
        j]*interpolation_coeffs[...,k]``. So for ``r=6``, we will store 18 coefficients
        per sparse point (instead of potentially 216). Must be a three-dimensional array
        of shape ``(npoint, grid.ndim, r)``.
    space_order : int, optional
        Discretisation order for space derivatives. Defaults to 0.
    time_order : int, optional
        Discretisation order for time derivatives. Default to 1.
    shape : tuple of ints, optional
        Shape of the object. Defaults to ``(npoint,)``.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if the SparseFunction
        defines a multi-dimensional tensor.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    initializer : callable or any object exposing the buffer interface, optional
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses ``*args`` to (re-)create the dimension arguments of the symbolic object.
    """

    is_PrecomputedSparseTimeFunction = True

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        u_t : expr-like, optional
            Time index at which the interpolation is performed.
        p_t : expr-like, optional
            Time index at which the result of the interpolation is stored.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        subs = {}
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs({time: u_t, t: u_t})

        if p_t is not None:
            subs = {self.time_dim: p_t}

        return super(PrecomputedSparseTimeFunction, self).interpolate(
            expr, offset=offset, increment=increment, self_subs=subs
        )


class MatrixSparseTimeFunction(AbstractSparseTimeFunction):
    """
    A specialised type of SparseTimeFunction where the interpolation is externally
    defined.  Currently, this means that the (integer) grid points and associated
    coefficients for each sparse point are explicitly provided as separate
    SubFunctions.

    Additionally, this class allows sources and receivers to be constructed
    from multiple locations, each with their own coefficients.  This is to support
    injection and sampling of dipole (and more general) sources and receivers,
    without needing to store multiple versions of the sample data that vary only
    by a scalar constant.

    matrix: scipy.sparse matrix
        A scipy-style sparse matrix with a row for each physical
        point in the grid, and a column for each index into the
        data array.

    r: int or Mapping[Dimension, Optional[int]]
        The number of gridpoints in each dimension used to inject/interpolate
        each physical point.  e.g. bi-/tri-linear interplation would use 2 coefficients
        in each dimension.

        The Mapping version of this parameter allows a different number of grid points
        in each dimension. If a Dimension maps to None, this has a special
        interpretation - sources are not localised to coordinates in that dimension.
        This is loosely equivalent to specifying r[dim] = dim_size, and with all
        gridpoint locations along that dimension equal to zero.

    par_dim: Dimension
        If set, this is the dimension used to split the sources for parallel
        injection. The source injection loop becomes a loop over this spatial
        dimension, and then a loop over sources which touch that spatial
        dimension coordinate. This defaults to grid.dimensions[0], and if specified
        must correspond to one of the grid.dimensions.

    other parameters as per SparseTimeFunction

    Location/coefficient data:
        msf.gridpoints.data[iloc, idim]: int
            integer, position (in global coordinates)
            of the _minimum_ index that location index
            `iloc` is interpolated from / injected into, in dimension `idim`
            where idim is an index into the grid.dimensions

        msf.interpolation_coefficients: Dict[Dimension, np.ndarray]
            For each dimension, there is an array of interpolation coefficients
            for each location `iloc`.

            This array is of shape (nloc, r), and is also available as
                msf.coefficients_x.data[iloc, ir]

            These are the coefficients that are multiplied by sample values
            at the gridpoints in the range:

            [msf.gridpoints.data[iloc, idim], msf.gridpoints.data[iloc, idim] + r)

    NOTE: *** restriction on space order of functions being sampled/injected into

    The halo of the function being interpolated/injected into
    must be larger than r, otherwise out of bounds access may result.

    NOTE: *** explicit scatter/gather semantics

    Before using this in an Operator, msf.manual_scatter() must be called to
    distribute the data.  This only needs to be done once for any number of
    calls to the Operator (e.g. for checkpointing), if the data, gridpoints
    and coefficients have not changed.

    This is true whether or not MPI is being used, and independent of
    the MPI_Size.

    Likewise, after all time steps have been run, data must be collected
    from remote ranks using msf.manual_gather() before relying on any of the
    data from msf.data[:]

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseFunction = True
    is_SparseTimeFunction = True

    _time_position = 0
    """Position of time index among the function indices."""

    # We use DiscreteFunction instead of AbstractSparseTimeFunction
    # because we want to get rid of 'npoint'
    __rkwargs__ = (DiscreteFunction.__rkwargs__ +
                   ('dimensions', 'r', 'matrix', 'nt', 'grid'))

    def __init_finalize__(self, *args, **kwargs):
        # The crucial argument to DugSparseTimeFunction is a sparse
        # matrix mapping a "source" or "receiver" to a set of locations
        self.matrix = kwargs.pop('matrix')

        from devito.data.allocators import default_allocator
        self._allocator = kwargs.get("allocator", default_allocator())

        # Rows are locations, columns are source/receivers
        nloc, npoint = self.matrix.shape

        super().__init_finalize__(
            *args, **kwargs, npoint=npoint)

        # Grid points per sparse point
        r = kwargs.get('r')
        if r is None:
            raise ValueError('MatrixSparseTimeFunction requires parameter `r`')
        if is_integer(r):
            if r <= 0:
                raise ValueError('MatrixSparseTimeFunction requires r > 0')

            # convert to dictionary with same size in all dims
            r = {dim: r for dim in self.grid.dimensions}

        # Validate radius is set correctly for all grid dimensions
        for d in self.grid.dimensions:
            if d not in r:
                raise ValueError("dimension %s not specified in r mapping" % d)
            if r[d] is None:
                continue
            if not is_integer(r[d]) or r[d] <= 0:
                raise ValueError('invalid parameter value r[%s] = %s' % (d, r[d]))

        # TODO is this going to cause some trouble with users of self.r?
        self._radius = r

        # Get the parallelism dimension for injection
        self._par_dim = kwargs.get("par_dim")
        if self._par_dim is not None:
            assert self._par_dim in self.grid.dimensions
        else:
            self._par_dim = self.grid.dimensions[0]

        # This has one value per dimension (e.g. size=3 for 3D)
        # Maybe this should be unique per SparseFunction,
        # but I can't see a need yet.
        ddim = Dimension('d')

        # Sources have their own Dimension
        # As do Locations
        locdim = Dimension('loc_%s' % self.name)

        self._gridpoints = SubFunction(
            name="%s_gridpoints" % self.name,
            dtype=np.int32,
            dimensions=(locdim, ddim),
            shape=(nloc, self.grid.dim),
            allocator=self._allocator,
            space_order=0, parent=self)

        # There is a coefficient array per grid dimension
        # I could pack these into one array but that seems less readable?
        self.interpolation_coefficients = {}
        self.interpolation_coefficients_t_bogus = {}
        self.rdims = []
        for d in self.grid.dimensions:
            if self._radius[d] is not None:
                rdim = DefaultDimension(
                    name='r%s_%s' % (d.name, self.name),
                    default_value=self._radius[d]
                )
                self.rdims.append(rdim)
                coeff_dim = rdim
                coeff_shape = self._radius[d]
            else:
                coeff_dim = d
                coeff_shape = self.grid.dimension_map[d].glb

            self.interpolation_coefficients[d] = SubFunction(
                name="%s_coefficients_%s" % (self.name, d.name),
                dtype=self.dtype,
                dimensions=(locdim, coeff_dim),
                shape=(nloc, coeff_shape),
                allocator=self._allocator,
                space_order=0, parent=self)

            # For the _sub_functions, these must be named attributes of
            # this SparseFunction object
            setattr(
                self, "coefficients_%s" % d.name,
                self.interpolation_coefficients[d])

        # We also need arrays to represent the sparse matrix map
        # The shapes are bogus; these are really only used when
        # constructing the expression,
        # - the mpi logic dynamically constructs arrays to feed to the
        # operator C code.
        self.nnzdim = Dimension('nnz_%s' % self.name)

        # In the non-MPI case, at least, we should fill these in once
        if self.grid.distributor.nprocs == 1:
            m_coo = self.matrix.tocoo(copy=False)
            nnz_size = m_coo.nnz
        else:
            nnz_size = 1

        self._mrow = SubFunction(
            name='mrow_%s' % self.name,
            dtype=np.int32,
            dimensions=(self.nnzdim,),
            shape=(nnz_size,),
            space_order=0,
            parent=self,
            allocator=self._allocator,
        )
        self._mcol = SubFunction(
            name='mcol_%s' % self.name,
            dtype=np.int32,
            dimensions=(self.nnzdim,),
            shape=(nnz_size,),
            space_order=0,
            parent=self,
            allocator=self._allocator,
        )
        self._mval = SubFunction(
            name='mval_%s' % self.name,
            dtype=self.dtype,
            dimensions=(self.nnzdim,),
            shape=(nnz_size,),
            space_order=0,
            parent=self,
            allocator=self._allocator,
        )

        # This loop maintains a map of nnz indices which touch each
        # coordinate of the parallised injection dimension
        # This takes the form of a list of nnz indices, and a start/end
        # position in that list for each index in the parallel dim
        self.par_dim_to_nnz_dim = DynamicDimension('par_dim_to_nnz_%s' % self.name)

        # This map acts as an indirect sort of the sources according to their
        # position along the parallelisation dimension
        self._par_dim_to_nnz_map = SubFunction(
            name='par_dim_to_nnz_map_%s' % self.name,
            dtype=np.int32,
            dimensions=(self.par_dim_to_nnz_dim,),
            # shape is unknown at this stage
            shape=(1,),
            space_order=0,
            parent=self,
        )
        self._par_dim_to_nnz_m = SubFunction(
            name='par_dim_to_nnz_m_%s' % self.name,
            dtype=np.int32,
            dimensions=(self._par_dim,),
            # shape is unknown at this stage
            shape=(1,),
            space_order=0,
            parent=self,
        )
        self._par_dim_to_nnz_M = SubFunction(
            name='par_dim_to_nnz_M_%s' % self.name,
            dtype=np.int32,
            dimensions=(self._par_dim,),
            # shape is unknown at this stage
            shape=(1,),
            space_order=0,
            parent=self,
        )

        if self.grid.distributor.nprocs == 1:
            self._mrow.data[:] = m_coo.row
            self._mcol.data[:] = m_coo.col
            self._mval.data[:] = m_coo.data

        # self._fd = generate_fd_shortcuts(self)

        self.scatter_result = None
        self.scattered_data = None

    def free_data(self):
        # The sympy cache holds the symbol references, but we can break the link
        # between the symbol and the data, thus causing the memory to be freed
        # This renders the object useless
        self._data = None
        self._gridpoints._data = None
        self._mrow._data = None
        self._mcol._data = None
        self._mval._data = None
        for f in self.interpolation_coefficients.values():
            f._data = None

        self.scatter_result = None
        self.scattered_data = None

    @property
    def dt(self):
        t = self.time_dim
        dt = self.time_dim.spacing
        return (-1 * self.subs(t, t - dt) + self.subs(t, t + dt))/(2 * dt)

    @property
    def dt2(self):
        t = self.time_dim
        dt = self.time_dim.spacing
        return (self.subs(t, t - dt) - 2 * self + self.subs(t, t + dt))/(dt*dt)

    @property
    def mrow(self):
        return self._mrow

    @property
    def mcol(self):
        return self._mcol

    @property
    def mval(self):
        return self._mval

    @property
    def par_dim_to_nnz_map(self):
        return self._par_dim_to_nnz_map

    @property
    def par_dim_to_nnz_m(self):
        return self._par_dim_to_nnz_m

    @property
    def par_dim_to_nnz_M(self):
        return self._par_dim_to_nnz_M

    @property
    def _sub_functions(self):
        return ('gridpoints',
                *['coefficients_%s' % d.name for d in self.grid.dimensions],
                'mrow', 'mcol', 'mval', 'par_dim_to_nnz_map',
                'par_dim_to_nnz_m', 'par_dim_to_nnz_M')

    @property
    def r(self):
        return self._radius

    def interpolate(self, expr, offset=0, u_t=None, p_t=None):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        """
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs(t, u_t).subs(time, u_t)

        gridpoints = self._gridpoints.indexed
        mrow = self._mrow.indexed
        mcol = self._mcol.indexed
        mval = self._mval.indexed
        tdim, pdim = self.indices
        locdim, ddim = self._gridpoints.indices
        nnzdim = self.nnzdim

        row = mrow[nnzdim]

        dim_subs = [(pdim, mcol[nnzdim])]
        coeffs = [mval[nnzdim]]
        for i, d in enumerate(self.grid.dimensions):
            _, rd = self.interpolation_coefficients[d].dimensions
            coefficients = self.interpolation_coefficients[d].indexed

            # If radius is set to None, then the coefficient array is
            # actually the full size of the grid dimension itself
            if self._radius[d] is not None:
                dim_subs.append((d, rd + gridpoints[row, i]))
            else:
                assert d is rd

            coeffs.append(coefficients[row, rd])

        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(tdim, p_t)
        lhs = lhs.subs([(pdim, mcol[nnzdim])])
        rhs = prod(coeffs) * expr.subs(dim_subs)

        return [Eq(self, 0), Inc(lhs, rhs)]

    def inject(self, field, expr, offset=0, u_t=None, p_t=None):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        expr = indexify(expr)
        field = indexify(field)

        tdim, pdim = self.indices
        par_dim_to_nnz_dim = self.par_dim_to_nnz_dim
        locdim, ddim = self.gridpoints.indices

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(tdim, p_t)

        gridpoints = self._gridpoints.indexed
        mrow = self._mrow.indexed
        mcol = self._mcol.indexed
        mval = self._mval.indexed
        partonnz = self._par_dim_to_nnz_map.indexed

        nnz_index = partonnz[par_dim_to_nnz_dim]
        row = mrow[nnz_index]
        dim_subs = [(pdim, mcol[nnz_index])]
        coeffs = [mval[nnz_index]]

        # Devito requires a fixed ordering of dimensions across
        # all loops, which means we need to respect that when constructing
        # the loops for this injection.

        # to that end, we keep the pairs (x, rx) (y, ry) together in the
        # ordering.

        par_dim_seen = False
        implicit_dims_for_range = [tdim]
        implicit_dims_for_inject = [tdim]

        for i, d in enumerate(self.grid.dimensions):
            _, rd = self.interpolation_coefficients[d].dimensions
            coefficients = self.interpolation_coefficients[d].indexed

            # There are four cases here.
            if d is self._par_dim:
                if self._radius[d] is None:
                    # If d is the parallelism dimension, AND this dimension is
                    # non-local (i.e. all sources touch all indices, and
                    # gridpoint for this dim is ignored)
                    coeffs.append(coefficients[row, d])
                else:
                    # d is the parallelism dimension, so the index into
                    # the coefficients array is derived from the value of
                    # this dimension minus the gridpoint of the point
                    coeffs.append(coefficients[row, d - gridpoints[row, i]])

                # loop dim here is always d
                implicit_dims_for_range.append(d)
                implicit_dims_for_inject.append(d)
                implicit_dims_for_inject.append(par_dim_to_nnz_dim)
                par_dim_seen = True
            else:
                if self._radius[d] is None:
                    # d is not the parallelism dimension, AND this dimension
                    # is non-local (i.e. all sources touch all indices,
                    # and gridpoint for this dim is ignored)

                    # the loop is therefore over the original dimension d
                    coeffs.append(coefficients[row, d])
                    loop_dim = d
                else:
                    # d is not the parallelism dimension, and it _is_
                    # local. In this case the loop is over the radius dimension
                    # and we need to substitute d with the offset from the
                    # grid point
                    dim_subs.append((d, rd + gridpoints[row, i]))
                    coeffs.append(coefficients[row, rd])
                    loop_dim = rd

                implicit_dims_for_inject.append(loop_dim)
                if not par_dim_seen:
                    implicit_dims_for_range.append(loop_dim)

        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        out = [
            Eq(
                par_dim_to_nnz_dim.symbolic_min,
                self._par_dim_to_nnz_m,
                implicit_dims=tuple(implicit_dims_for_range)
            ),
            Eq(
                par_dim_to_nnz_dim.symbolic_max,
                self._par_dim_to_nnz_M,
                implicit_dims=tuple(implicit_dims_for_range)
            ),
            Inc(
                field,
                rhs.subs(dim_subs),
                implicit_dims=IgnoreDimSort(implicit_dims_for_inject),
            ),
        ]

        return out

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """
        Return the default dimension indices for a given data shape.
        """
        dimensions = kwargs.get('dimensions')
        if dimensions is None:
            dimensions = (kwargs['grid'].time_dim, Dimension(
                name='p_%s' % kwargs["name"]))
        return dimensions, dimensions

    @classmethod
    def __shape_setup__(cls, **kwargs):
        # This happens before __init__, so we have to get 'npoint'
        # from the matrix
        _, npoint = kwargs['matrix'].shape
        return kwargs.get('shape', (kwargs.get('nt'), npoint))

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this function."""
        return tuple([self.name, self.name + "_" + self.gridpoints.name]
                     + ['%s_%s' % (self.name, x.name)
                        for x in self.interpolation_coefficients.values()])

    @property
    def gridpoints(self):
        return self._gridpoints

    def _rank_to_points(self):
        """
        For each rank in self.grid.distributor, return
        a numpy array of int32s for the positions within
        this rank's self.gridpoints/self.interpolation_coefficients (i.e.
        the locdim) which must be injected into that rank.

        Any given location may require injection into several
        ranks, based on the radius of the injection stencil
        and its proximity to a rank boundary.

        It is assumed, for now, that any given location may be
        completely sampled from within one rank - so when
        gathering the data, any point sampled from more than
        one rank may have duplicates discarded.  This implies
        that the radius of the sampling is less than
        the halo size of the Functions being sampled from.
        It also requires that the halos be exchanged before
        interpolation (must verify that this occurs).
        """
        distributor = self.grid.distributor

        # Along each dimension, the coordinate indices are broken into
        # 2*decomposition_size+3 groups, numbered starting at 0

        # Group 2*i contributes only to rank i-1
        # Group 2*i+1 contributes to rank i-1 and rank i

        # Obviously this means groups 0 and 1 are "bad" - they contribute
        #  to points to the left of the domain (rank -1)
        # So is group 2*decomp_size+1 and 2*decomp_size+2
        #  (these contributes to rank "decomp_size")

        # binned_gridpoints will hold which group the particular
        # point is along that decomposed dimension.
        binned_gridpoints = np.empty_like(self._gridpoints.data)
        dim_group_dim_rank = []

        for idim, dim in enumerate(self.grid.dimensions):
            decomp = distributor.decomposition[idim]
            decomp_size = len(decomp)
            dim_breaks = np.empty([2*decomp_size+2], dtype=np.int32)

            dim_r = self.r[dim]
            if dim_r is None:
                # size is the whole grid
                dim_r = self.grid.dimension_map[dim].glb

            # Define the split
            dim_breaks[:-2:2] = [
                decomp_part[0] - self.r + 1 for decomp_part in decomp]
            dim_breaks[-2] = decomp[-1][-1] + 1 - self.r + 1
            dim_breaks[1:-1:2] = [
                decomp_part[0] for decomp_part in decomp]
            dim_breaks[-1] = decomp[-1][-1] + 1

            # Handle the radius is None case by ensuring we treat
            # all grid points in that direction as zero
            gridpoints_dim = self._gridpoints.data[:, idim]
            if self.r[dim] is None:
                gridpoints_dim = np.zeros_like(gridpoints_dim)

            try:
                binned_gridpoints[:, idim] = np.digitize(
                    gridpoints_dim, dim_breaks)
            except ValueError as e:
                raise ValueError(
                    "decomposition failed!  Are some ranks too skinny?"
                ) from e

            this_group_rank_map = {
                0: {None},
                1: {None, 0},
                **{2*i+2: {i} for i in range(decomp_size)},
                **{2*i+2+1: {i, i+1} for i in range(decomp_size-1)},
                2*decomp_size+1: {decomp_size-1, None},
                2*decomp_size+2: {None}}

            dim_group_dim_rank.append(this_group_rank_map)

        # This allows the points to be grouped into non-overlapping sets
        # based on their bin in each dimension.  For each set we build a list
        # of points.
        bins, inverse, counts = np.unique(
            binned_gridpoints,
            return_inverse=True,
            return_counts=True,
            axis=0)

        # inverse is now a "unique bin number" for each point gridpoints
        # we want to turn that into a list of points for each bin
        # so we argsort
        inverse_argsort = np.argsort(inverse).astype(np.int32)
        cumulative_counts = np.cumsum(counts)
        gp_map = {tuple(bi): inverse_argsort[cci-ci:cci]
                  for bi, cci, ci in zip(bins, cumulative_counts, counts)
                  }

        # the result is now going to be a concatenation of these lists
        # for each of the output ranks
        # each bin has a set of ranks -> each rank has a set (possibly empty)
        # of bins

        # For each rank get the per-dimension coordinates
        # TODO maybe we should cache this on the distributor
        dim_ranks_to_glb = {
            tuple(distributor.comm.Get_coords(rank)): rank
            for rank in range(distributor.comm.Get_size())}

        global_rank_to_bins = {}

        from itertools import product
        for bi in bins:
            # This is a list of sets for the dimension-specific rank
            dim_rank_sets = [dgdr[bii]
                             for dgdr, bii in zip(dim_group_dim_rank, bi)]

            # Convert these to an absolute rank
            # This is where we will throw a KeyError if there are points OOB
            for dim_ranks in product(*dim_rank_sets):
                global_rank = dim_ranks_to_glb[tuple(dim_ranks)]
                global_rank_to_bins\
                    .setdefault(global_rank, set())\
                    .add(tuple(bi))

        empty = np.array([], dtype=np.int32)

        return [np.concatenate((
            empty, *[gp_map[bi] for bi in global_rank_to_bins.get(rank, [])]))
            for rank in range(distributor.comm.Get_size())]

    def _build_par_dim_to_nnz(self, active_gp, active_mrow):
        # The case where we parallelise over a non-local index is suboptimal, but
        # supported. In this case, the actual grid point locations are ignored
        # and all points are touched.

        pardim_index = self.grid.dimensions.index(self._par_dim)

        if self._radius[self._par_dim] is None:
            # early exit with degenerate case - no reordering and all coordinate
            # values touch all parts of the array
            nnz_M = active_mrow.size - 1
            return {
                self._par_dim_to_nnz_map: np.arange(active_mrow.size, dtype=np.int32),
                self._par_dim_to_nnz_m: np.zeros(
                    (self.grid.shape_local[pardim_index],), dtype=np.int32
                ),
                self._par_dim_to_nnz_M: np.full(
                    (self.grid.shape_local[pardim_index],), nnz_M, dtype=np.int32
                ),
            }

        # Get the radius along the parallel dimension
        r = self._radius[self._par_dim]

        # now, the parameters can be devito.Data, which doesn't like fancy indexing
        # very much. So, we convert to regular numpy arrays
        active_gp = np.array(active_gp)
        active_mrow = np.array(active_mrow)

        # sort the injected nonzero indices by parallel coordinate
        pardim_coordinates_nnz = active_gp[active_mrow, pardim_index]
        reordering = np.argsort(pardim_coordinates_nnz)
        pardim_reordered = pardim_coordinates_nnz[reordering]

        # now each x coordinate that we inject into has a range
        # of relevant entries in the reordered array

        # we don't worry about MPI here; by the time this function is called,
        # all gridpoints have been renumbered to local offsets

        # this coordinate is touched by any source with gridpoint >= x - r + 1
        # and gridpoint <= x
        all_xs = np.arange(self.grid.shape_local[pardim_index])

        # This should satisfy:
        # x_reordered[i-1] < x - r + 1 <= x_reordered[i]
        reordered_m = np.searchsorted(pardim_reordered, all_xs - r + 1, side='left')
        # x_reordered[i-1] <= x < x_reordered[i]
        reordered_M = np.searchsorted(pardim_reordered, all_xs, side='right') - 1

        # return output suitable for scatter
        return {
            self._par_dim_to_nnz_map: reordering.astype(np.int32),
            self._par_dim_to_nnz_m: reordered_m.astype(np.int32),
            self._par_dim_to_nnz_M: reordered_M.astype(np.int32),
        }

    def manual_scatter(self, *, data_all_zero=False):
        distributor = self.grid.distributor

        if distributor.nprocs == 1:
            self.scattered_data = self.data
            self.scatter_result = {
                self: self.data,
                **{
                    getattr(self, k): getattr(self, k).data for k in self._sub_functions
                },
                self.mrow: self.mrow.data,
                self.mcol: self.mcol.data,
                self.mval: self.mval.data,
                **self._build_par_dim_to_nnz(self.gridpoints.data, self.mrow.data),
            }
            return

        # Generate the matrix arrays
        m_coo = self.matrix.tocoo(copy=False)

        # HACK: for now, only take npoints != 0 on rank 0
        # Broadcast all the data, gridpoints, coefficients to all ranks
        # Each rank then ignores any of the data which isn't in its own
        #  domain.
        if distributor.myrank != 0 and self.npoint != 0:
            raise ValueError("can only accept sources/receivers on rank 0")

        # args[self.mrow.name] = m_coo.row.copy()
        # args[self.mcol.name] = m_coo.col.copy()
        # args[self.mval.name] = m_coo.data.copy()
        # args.update(self.nnzdim._arg_defaults(size=m_coo.nnz))

        # Send out data
        # Send out gridpoints
        # Send out coefficients
        # Send out matrix rows, cols, data
        r_tuple = tuple(self.r[dim] for dim in self.grid.dimensions)

        npoint, nloc, nnz, ndim, r_tuple_bcast, nt = distributor.comm.bcast(
            (self.npoint,
             self._gridpoints.data.shape[0],
             m_coo.nnz,
             self._gridpoints.data.shape[-1],
             r_tuple,
             self.data.shape[self._time_position]), root=0)

        # important that all ranks have the same ndims and same r
        assert r_tuple == r_tuple_bcast
        assert ndim == self._gridpoints.data.shape[-1]

        # handle None radius
        r_tuple_no_none = tuple(
            ri if ri is not None else self.grid.dimension_map[d].glb
            for ri, d in zip(r_tuple, self.grid.dimensions)
        )

        # now all ranks can allocate the buffers to receive into
        if distributor.myrank != 0:
            if data_all_zero:
                scattered_data = np.zeros([nt, npoint], dtype=self.dtype)
            else:
                scattered_data = np.empty([nt, npoint], dtype=self.dtype)
            scattered_gp = np.empty([nloc, ndim], dtype=np.int32)
            scattered_coeffs = [
                np.empty([nloc, r_tuple_no_none[idim]], dtype=self.dtype)
                for idim in range(ndim)
            ]
            scattered_mrow = np.empty([nnz], dtype=np.int32)
            scattered_mcol = np.empty([nnz], dtype=np.int32)
            scattered_mval = np.empty([nnz], dtype=self.dtype)
        else:
            scattered_data = self.data

            # These are copies because we mess with them down below
            scattered_gp = self._gridpoints.data.copy()
            scattered_coeffs = [
                self.interpolation_coefficients[d].data.copy()
                for d in self.grid.dimensions]
            scattered_mrow = m_coo.row.copy()
            scattered_mcol = m_coo.col.copy()
            scattered_mval = m_coo.data.copy()

        if not data_all_zero:
            distributor.comm.Bcast(scattered_data, root=0)
        for arr in [scattered_gp, *scattered_coeffs,
                    scattered_mrow, scattered_mcol, scattered_mval]:
            distributor.comm.Bcast(arr, root=0)

        # now recreate the matrix to only contain points in our
        # local domain.
        # along each dimension, each point is in one of 5 groups
        #  0 - completely to the left
        #  1 - to the left, but the injection stencil touches our domain
        #  2 - completely in our domain
        #  3 - in the domain, but the injection stencil includes points
        #      to the right
        #  4 - completely to the right
        active_mrow = scattered_mrow
        active_mcol = scattered_mcol
        active_mval = scattered_mval

        # first, build a reduced matrix excluding any points outside our domain
        for idim, (dim, mycoord) in enumerate(zip(
                self.grid.dimensions, distributor.mycoords)):
            _left = distributor.decomposition[idim][mycoord][0]
            _right = distributor.decomposition[idim][mycoord][-1] + 1

            this_dim_r = self.r[dim]
            effective_gridpoints = scattered_gp[active_mrow, idim]
            if this_dim_r is None:
                this_dim_r = self.grid.dimension_map[dim].glb
                effective_gridpoints = np.zeros_like(effective_gridpoints)

            # rewrite the matrix to remove the rows in groups 0 and 4
            mask = (
                (effective_gridpoints >= _left - this_dim_r + 1)
                & (effective_gridpoints < _right))

            which = np.nonzero(mask)
            active_mrow = active_mrow[which]
            active_mcol = active_mcol[which]
            active_mval = active_mval[which]

        # then, zero any of the coefficients which refer to points outside our
        # domain.  Do this on all the gridpoints for now, since this is a hack
        # anyway
        for idim, (dim, mycoord) in enumerate(zip(
                self.grid.dimensions, distributor.mycoords)):
            _left = distributor.decomposition[idim][mycoord][0]
            _right = distributor.decomposition[idim][mycoord][-1] + 1

            # points to the left have the first few coeffs zeroed
            this_dim_r = self.r[dim]
            effective_gridpoints = scattered_gp[:, idim]
            if this_dim_r is None:
                this_dim_r = self.grid.dimension_map[dim].glb
                effective_gridpoints = np.zeros_like(effective_gridpoints)

            trim_size = np.clip(_left - effective_gridpoints, 0, this_dim_r)
            for ir in range(this_dim_r):
                # which points need zeroing?
                mask = (trim_size > ir)
                scattered_coeffs[idim][mask, ir] = 0

            # points to the right have the last few coeffs zeroed
            trim_size = np.clip(
                effective_gridpoints - (_right - this_dim_r), 0, this_dim_r)
            for ir in range(this_dim_r):
                # which points need zeroing?
                mask = (trim_size > ir)
                scattered_coeffs[idim][mask, -(ir+1)] = 0

            # finally, we translate to local coordinates
            # no need for this in the broadcasted dimensions
            if self.r[dim] is not None:
                scattered_gp[:, idim] -= _left

        self.scattered_data = scattered_data
        self.scatter_result = {
            self: scattered_data,
            self.gridpoints: scattered_gp,
            **{
                self.interpolation_coefficients[d]: scattered_coeffs[idim]
                for idim, d in enumerate(self.grid.dimensions)
            },
            self.mrow: active_mrow,
            self.mcol: active_mcol,
            self.mval: active_mval,
            **self._build_par_dim_to_nnz(scattered_gp, active_mrow),
        }

    def _dist_scatter(self, data=None):
        assert data is None
        if self.scatter_result is None:
            raise Exception("_dist_scatter called before manual_scatter called")
        return self.scatter_result

    # The implementation in AbstractSparseFunction now relies on us
    # having a .coordinates property, which we don't have.
    def _arg_apply(self, dataobj, alias=None):
        key = alias if alias is not None else self
        if isinstance(key, AbstractSparseFunction):
            # Gather into `self.data`
            key._dist_gather(self._C_as_ndarray(dataobj))
        elif self.grid.distributor.nprocs > 1:
            raise NotImplementedError("Don't know how to gather data from an "
                                      "object of type `%s`" % type(key))

    def manual_gather(self):
        # data, in this case, is set to whatever dist_scatter provided?
        # on rank 0, this is the original data array (hack...)
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        # This relies on all ranks having a copy of all data. Which feels "bad".
        if distributor.myrank != 0:
            distributor.comm.Reduce(
                self.scattered_data,
                None,
                op=MPI.SUM,
                root=0
            )
        else:
            distributor.comm.Reduce(
                MPI.IN_PLACE,
                self.scattered_data,  # Note: on rank 0 data === scattered_data.
                op=MPI.SUM,
                root=0
            )

    def _dist_gather(self, data):
        pass
