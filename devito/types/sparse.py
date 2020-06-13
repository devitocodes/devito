from collections import OrderedDict
from itertools import product

import sympy
import numpy as np
from cached_property import cached_property

from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.mpi import MPI, SparseDistributor
from devito.operations import LinearInterpolator, PrecomputedInterpolator
from devito.symbolics import INT, cast_mapper, indexify, retrieve_function_carriers
from devito.tools import ReducerMap, flatten, prod, filter_ordered, memoized_meth
from devito.types.dense import DiscreteFunction, Function, SubFunction
from devito.types.dimension import Dimension, ConditionalDimension
from devito.types.basic import Symbol, Scalar
from devito.types.equation import Eq

__all__ = ['SparseFunction', 'SparseTimeFunction', 'PrecomputedSparseFunction',
           'PrecomputedSparseTimeFunction']


class AbstractSparseFunction(DiscreteFunction, Differentiable):

    """
    An abstract class to define behaviours common to all sparse functions.
    """

    _sparse_position = -1
    """Position of sparse index among the function indices."""

    _radius = 0
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ()
    """SubFunctions encapsulated within this AbstractSparseFunction."""

    def __init_finalize__(self, *args, **kwargs):
        super(AbstractSparseFunction, self).__init_finalize__(*args, **kwargs)
        self._npoint = kwargs['npoint']
        self._space_order = kwargs.get('space_order', 0)

        # Dynamically add derivative short-cuts
        self._fd = generate_fd_shortcuts(self)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions, dimensions
        else:
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

    @property
    def _support(self):
        """
        The grid points surrounding each sparse point within the radius of self's
        injection/interpolation operators.
        """
        ret = []
        for i in self.gridpoints:
            support = [range(max(0, j - self._radius + 1), min(M, j + self._radius + 1))
                       for j, M in zip(i, self.grid.shape)]
            ret.append(tuple(product(*support)))
        return tuple(ret)

    @property
    def _dist_datamap(self):
        return self._build_dist_datamap(support=self._support)

    @memoized_meth
    def _build_dist_datamap(self, support=None):
        """
        Mapper ``M : MPI rank -> required sparse data``.
        """
        ret = {}
        support = support or self._support
        for i, s in enumerate(support):
            # Sparse point `i` is "required" by the following ranks
            for r in self.grid.distributor.glb_to_rank(s):
                ret.setdefault(r, []).append(i)
        return {k: filter_ordered(v) for k, v in ret.items()}

    @property
    def _dist_scatter_mask(self):
        """
        A mask to index into ``self.data``, which creates a new data array that
        logically contains N consecutive groups of sparse data values, where N
        is the number of MPI ranks. The i-th group contains the sparse data
        values accessible by the i-th MPI rank.  Thus, sparse data values along
        the boundary of two or more MPI ranks are duplicated.
        """
        dmap = self._dist_datamap
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        ret = [slice(None) for i in range(self.ndim)]
        ret[self._sparse_position] = mask
        return tuple(ret)

    @property
    def _dist_subfunc_scatter_mask(self):
        """
        This method is analogous to :meth:`_dist_scatter_mask`, although
        the mask is now suitable to index into self's SubFunctions, rather
        than into ``self.data``.
        """
        return self._dist_scatter_mask[self._sparse_position]

    @property
    def _dist_gather_mask(self):
        """
        A mask to index into the ``data`` received upon returning from
        ``self._dist_alltoall``. This mask creates a new data array in which
        duplicate sparse data values have been discarded. The resulting data
        array can thus be used to populate ``self.data``.
        """
        ret = list(self._dist_scatter_mask)
        mask = ret[self._sparse_position]
        inds = np.unique(mask, return_index=True)[1]
        inds.sort()
        ret[self._sparse_position] = inds.tolist()

        return tuple(ret)

    @property
    def _dist_subfunc_gather_mask(self):
        """
        This method is analogous to :meth:`_dist_subfunc_scatter_mask`, although
        the mask is now suitable to index into self's SubFunctions, rather
        than into ``self.data``.
        """
        return self._dist_gather_mask[self._sparse_position]

    @property
    def _dist_count(self):
        """
        A 2-tuple of comm-sized iterables, which tells how many sparse points
        is this MPI rank expected to send/receive to/from each other MPI rank.
        """
        dmap = self._dist_datamap
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

    @property
    def _dist_alltoall(self):
        """
        The metadata necessary to perform an ``MPI_Alltoallv`` distributing the
        sparse data values across the MPI ranks needing them.
        """
        ssparse, rsparse = self._dist_count

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

    @property
    def _dist_subfunc_alltoall(self):
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

    def _dist_gather(self, data):
        """
        A ``numpy.ndarray`` containing up-to-date data and coordinate values
        suitable for insertion into ``self.data``.
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

        # Add MPI-related data structures
        args.update(self.grid._arg_defaults())

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
                # Add MPI-related data structures
                values.update(self.grid._arg_defaults())
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

    # Pickling support
    _pickle_kwargs = DiscreteFunction._pickle_kwargs + ['npoint', 'space_order']


class AbstractSparseTimeFunction(AbstractSparseFunction):

    """
    An abstract class to define behaviours common to all sparse time-varying functions.
    """

    _time_position = 0
    """Position of time index among the function indices."""

    def __init_finalize__(self, *args, **kwargs):
        self._time_dim = self.indices[self._time_position]
        self._time_order = kwargs.get('time_order', 1)
        if not isinstance(self.time_order, int):
            raise ValueError("`time_order` must be int")

        super(AbstractSparseTimeFunction, self).__init_finalize__(*args, **kwargs)

    @property
    def time_dim(self):
        """The time dimension."""
        return self._time_dim

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions, dimensions
        else:
            dims = (kwargs['grid'].time_dim, Dimension(name='p_%s' % kwargs["name"]))
            return dims, dims

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

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['nt', 'time_order']


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

    def __init_finalize__(self, *args, **kwargs):
        super(SparseFunction, self).__init_finalize__(*args, **kwargs)
        self.interpolator = LinearInterpolator(self)
        # Set up sparse point coordinates
        coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
        if isinstance(coordinates, Function):
            self._coordinates = coordinates
        else:
            dimensions = (self.indices[-1], Dimension(name='d'))
            # Only retain the local data region
            if coordinates is not None:
                coordinates = np.array(coordinates)
            self._coordinates = SubFunction(name='%s_coords' % self.name, parent=self,
                                            dtype=self.dtype, dimensions=dimensions,
                                            shape=(self.npoint, self.grid.dim),
                                            space_order=0, initializer=coordinates,
                                            distributor=self._distributor)
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
        return tuple(Scalar(name='p%s' % d, dtype=self.dtype)
                     for d in self.grid.dimensions)

    @cached_property
    def _point_increments(self):
        """Index increments in each dimension for each point symbol."""
        return tuple(product(range(2), repeat=self.grid.dim))

    @cached_property
    def _coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension."""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @cached_property
    def _coordinate_indices(self):
        """Symbol for each grid index according to the coordinates."""
        indices = self.grid.dimensions
        return tuple([INT(sympy.Function('floor')((c - o) / i.spacing))
                      for c, o, i in zip(self._coordinate_symbols, self.grid.origin,
                                         indices[:self.grid.dim])])

    def _coordinate_bases(self, field_offset):
        """Symbol for the base coordinates of the reference grid point."""
        indices = self.grid.dimensions
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i, of in zip(self._coordinate_symbols,
                                                  self.grid.origin,
                                                  self._coordinate_indices,
                                                  indices[:self.grid.dim],
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
        ret = []
        for coords in self.coordinates.data._local:
            ret.append(tuple(int(np.floor(c - o.data)/i.spacing.data) for c, o, i in
                             zip(coords, self.grid.origin, self.grid.dimensions)))
        return tuple(ret)

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
        cd = ConditionalDimension("%s_g" % self._sparse_dim, self._sparse_dim,
                                  condition=condition)

        if expr is None:
            out = self.indexify().xreplace({self._sparse_dim: cd})
        else:
            functions = {f for f in retrieve_function_carriers(expr)
                         if f.is_SparseFunction}
            out = indexify(expr).xreplace({f._sparse_dim: cd for f in functions})

        # Temporaries for the indirection dimensions
        temps = [Eq(v, k, implicit_dims=self.dimensions)
                 for k, v in points.items() if v in conditions]

        return out, temps

    @cached_property
    def _decomposition(self):
        mapper = {self._sparse_dim: self._distributor.decomposition[self._sparse_dim]}
        return tuple(mapper.get(d) for d in self.dimensions)

    @property
    def _dist_subfunc_alltoall(self):
        ssparse, rsparse = self._dist_count

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

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = data[self._dist_scatter_mask]
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        # Send out the sparse point values
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_alltoall
        scattered = np.empty(shape=rshape, dtype=self.dtype)
        comm.Alltoallv([data, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        data = scattered
        # Unpack data values so that they follow the expected storage layout
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))

        # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
        coords = self.coordinates.data._local[self._dist_subfunc_scatter_mask]
        # Send out the sparse point coordinates
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_subfunc_alltoall
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

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        # Send back the sparse point values
        sshape, scount, sdisp, _, rcount, rdisp = self._dist_alltoall
        gathered = np.empty(shape=sshape, dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, rcount, rdisp, mpitype],
                       [gathered, scount, sdisp, mpitype])
        # Unpack data values so that they follow the expected storage layout
        gathered = np.ascontiguousarray(np.transpose(gathered, self._dist_reorder_mask))
        self._data[:] = gathered[self._dist_gather_mask]

        if coords is not None:
            # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
            coords = coords + np.array(self.grid.origin_offset, dtype=self.dtype)
            # Send out the sparse point coordinates
            sshape, scount, sdisp, _, rcount, rdisp = self._dist_subfunc_alltoall
            gathered = np.empty(shape=sshape, dtype=self.coordinates.dtype)
            mpitype = MPI._typedict[np.dtype(self.coordinates.dtype).char]
            comm.Alltoallv([coords, rcount, rdisp, mpitype],
                           [gathered, scount, sdisp, mpitype])
            self._coordinates.data._local[:] = gathered[self._dist_subfunc_gather_mask]

        # Note: this method "mirrors" `_dist_scatter`: a sparse point that is sent
        # in `_dist_scatter` is here received; a sparse point that is received in
        # `_dist_scatter` is here sent.

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['coordinates_data']


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

    def inject(self, field, expr, offset=0, u_t=None, p_t=None):
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
        """
        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs({field.time_dim: u_t})
        if p_t is not None:
            expr = expr.subs({self.time_dim: p_t})

        return super(SparseTimeFunction, self).inject(field, expr, offset=offset)

    # Pickling support
    _pickle_kwargs = AbstractSparseTimeFunction._pickle_kwargs +\
        SparseFunction._pickle_kwargs


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
