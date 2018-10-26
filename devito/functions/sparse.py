from collections import OrderedDict
from itertools import product

import sympy
import numpy as np
from cached_property import cached_property

from devito.cgen_utils import INT, cast_mapper
from devito.equation import Eq, Inc
from devito.logger import warning
from devito.mpi import MPI
from devito.symbolics import indexify, retrieve_functions
from devito.finite_differences import Differentiable
from devito.functions.dense import Function, GridedFunction, SubFunction
from devito.functions.dimension import Dimension, ConditionalDimension, DefaultDimension
from devito.functions.basic import Symbol, Scalar
from devito.tools import (ReducerMap, as_tuple, flatten, prod,
                          powerset, filter_ordered, memoized_meth)


__all__ = ['SparseFunction', 'SparseTimeFunction', 'PrecomputedSparseFunction',
           'PrecomputedSparseTimeFunction']


class AbstractSparseFunction(GridedFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """

    _sparse_position = -1
    """Position of sparse index among the function indices."""

    _radius = 0
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ()
    """:class:`SubFunction`s encapsulated within this AbstractSparseFunction."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseFunction, self).__init__(*args, **kwargs)

            npoint = kwargs.get('npoint')
            if not isinstance(npoint, int):
                raise TypeError('SparseFunction needs `npoint` int argument')
            if npoint < 0:
                raise ValueError('`npoint` must be >= 0')
            self.npoint = npoint

            # A Grid must have been provided
            if self.grid is None:
                raise TypeError('SparseFunction needs `grid` argument')

            self._space_order = kwargs.get('space_order', 0)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """
        Return the default dimension indices for a given data shape.
        """
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions
        else:
            return (Dimension(name='p_%s' % kwargs["name"]),)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        return kwargs.get('shape', (kwargs.get('npoint'),))

    @property
    def _sparse_dim(self):
        return self.dimensions[self._sparse_position]

    def _is_owned(self, point):
        """Return True if ``point`` is in self's local domain, False otherwise."""
        point = as_tuple(point)
        if len(point) != self.grid.dim:
            raise ValueError("`%s` is an %dD point (expected %dD)"
                             % (point, len(point), self.grid.dim))
        distributor = self.grid.distributor
        return all(distributor.glb_to_loc(d, p) is not None
                   for d, p in zip(self.grid.dimensions, point))

    @property
    def gridpoints(self):
        """
        The *reference* grid point corresponding to each sparse point.
        """
        raise NotImplementedError

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
        return ret

    @property
    def _dist_datamap(self):
        """
        Return a mapper ``M : MPI rank -> required sparse data``.
        """
        ret = {}
        for i, s in enumerate(self._support):
            # Sparse point `i` is "required" by the following ranks
            for r in self.grid.distributor.glb_to_rank(s):
                ret.setdefault(r, []).append(i)
        return {k: filter_ordered(v) for k, v in ret.items()}

    @property
    def _dist_scatter_mask(self):
        """
        Return a mask to index into ``self.data``, which creates a new
        data array that logically contains N consecutive groups of sparse
        data values, where N is the number of MPI ranks. The i-th group
        contains the sparse data values accessible by the i-th MPI rank.
        Thus, sparse data values along the boundary of two or more MPI
        ranks are duplicated.
        """
        dmap = self._dist_datamap
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        ret = [slice(None) for i in range(self.ndim)]
        ret[self._sparse_position] = mask
        return ret

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
        Return a mask to index into the ``data`` received upon returning
        from ``self._dist_alltoall``. This mask creates a new data array
        in which duplicate sparse data values have been discarded. The
        resulting data array can thus be used to populate ``self.data``.
        """
        ret = list(self._dist_scatter_mask)
        mask = ret[self._sparse_position]
        ret[self._sparse_position] = [mask.tolist().index(i)
                                      for i in filter_ordered(mask)]
        return ret

    @property
    def _dist_count(self):
        """
        Return a 2-tuple of comm-sized iterables, which tells how many sparse
        points is this MPI rank expected to send/receive to/from each other
        MPI rank.
        """
        dmap = self._dist_datamap
        comm = self.grid.distributor.comm

        ssparse = np.array([len(dmap.get(i, [])) for i in range(comm.size)], dtype=int)
        rsparse = np.empty(comm.size, dtype=int)
        comm.Alltoall(ssparse, rsparse)

        return ssparse, rsparse

    @property
    def _dist_alltoall(self):
        """
        Return the metadata necessary to perform an ``MPI_Alltoallv`` distributing
        the sparse data values across the MPI ranks needing them.
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
        scount = [prod(i) for i in sshape]
        rcount = [prod(i) for i in rshape]

        # Per-rank displacement of send/recv data (it's actually all contiguous,
        # but the Alltoallv needs this information anyway)
        sdisp = np.concatenate([[0], np.cumsum(scount)[:-1]])
        rdisp = np.concatenate([[0], tuple(np.cumsum(rcount))[:-1]])

        # Total shape of send/recv data
        sshape = list(self.shape)
        sshape[self._sparse_position] = sum(ssparse)
        rshape = list(self.shape)
        rshape[self._sparse_position] = sum(rsparse)

        return sshape, scount, sdisp, rshape, rcount, rdisp

    @property
    def _dist_subfunc_alltoall(self):
        """
        Return the metadata necessary to perform an ``MPI_Alltoallv`` distributing
        self's SubFunction values across the MPI ranks needing them.
        """
        raise NotImplementedError

    def _dist_scatter(self):
        """
        Return a ``numpy.ndarray`` containing up-to-date data values belonging
        to the calling MPI rank. A data value belongs to a given MPI rank R
        if its coordinates fall within R's local domain.
        """
        raise NotImplementedError

    def _dist_gather(self, data):
        """
        Return a ``numpy.ndarray`` containing up-to-date data and coordinate values
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
            for i, s, o in zip(mapper[k].indices, v.shape, k.staggered):
                args.update(i._arg_defaults(start=0, size=s+o))

        # Add MPI-related data structures
        args.update(self.grid._arg_defaults())

        return args

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
                    for i, s, o in zip(k.indices, v.shape, k.staggered):
                        values.update(i._arg_defaults(size=s+o-sum(k._offset_domain[i])))
                # Add MPI-related data structures
                values.update(self.grid._arg_defaults())
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_apply(self, data, alias=None):
        key = alias if alias is not None else self
        if isinstance(key, AbstractSparseFunction):
            key._dist_gather(data)
        elif self.grid.distributor.nprocs > 1:
            raise NotImplementedError("Don't know how to gather data from an "
                                      "object of type `%s`" % type(key))

    # Pickling support
    _pickle_kwargs = GridedFunction._pickle_kwargs + ['npoint', 'space_order']


class AbstractSparseTimeFunction(AbstractSparseFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    time functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """

    _time_position = 0
    """Position of time index among the function indices."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseTimeFunction, self).__init__(*args, **kwargs)

            nt = kwargs.get('nt')
            if not isinstance(nt, int):
                raise TypeError('Sparse TimeFunction needs `nt` int argument')
            if nt <= 0:
                raise ValueError('`nt` must be > 0')
            self.nt = nt

            self.time_dim = self.indices[self._time_position]
            self._time_order = kwargs.get('time_order', 1)
            if not isinstance(self.time_order, int):
                raise ValueError("`time_order` must be int")

    @property
    def time_order(self):
        return self._time_order

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """
        Return the default dimension indices for a given data shape.
        """
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions
        else:
            return (kwargs['grid'].time_dim, Dimension(name='p_%s' % kwargs["name"]))

    @classmethod
    def __shape_setup__(cls, **kwargs):
        return kwargs.get('shape', (kwargs.get('nt'), kwargs.get('npoint'),))

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['nt', 'time_order']


class SparseFunction(AbstractSparseFunction, Differentiable):
    """
    A special :class:`GridedFunction` representing a set of sparse point
    objects that are not aligned with the computational grid.

    A :class:`SparseFunction` provides symbolic interpolation routines
    to convert between grid-aligned :class:`Function` objects and sparse
    data points. These are based upon standard [bi,tri]linear interpolation.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: (Optional) discretisation order for space derivatives.
    :param shape: (Optional) shape of the function. Defaults to ``(npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseFunction = True

    _radius = 1
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ('coordinates',)

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SparseFunction, self).__init__(*args, **kwargs)

            # Set up sparse point coordinates
            coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
            if isinstance(coordinates, Function):
                self._coordinates = coordinates
            else:
                dimensions = (self.indices[-1], Dimension(name='d'))
                self._coordinates = SubFunction(name='%s_coords' % self.name, parent=self,
                                                dtype=self.dtype, dimensions=dimensions,
                                                shape=(self.npoint, self.grid.dim),
                                                space_order=0, initializer=coordinates)
                if self.npoint == 0:
                    # This is a corner case -- we might get here, for example, when
                    # running with MPI and some processes get 0-size arrays after
                    # domain decomposition. We touch the data anyway to avoid the
                    # case ``self._data is None``
                    self.coordinates.data

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def coordinates_data(self):
        return self.coordinates.data

    @property
    def _coefficients(self):
        """Symbolic expression for the coefficients for sparse point
        interpolation according to:
        https://en.wikipedia.org/wiki/Bilinear_interpolation.

        :returns: List of coefficients, eg. [b_11, b_12, b_21, b_22]
        """
        # Grid indices corresponding to the corners of the cell ie x1, y1, z1
        indices1 = tuple(sympy.symbols('%s1' % d) for d in self.grid.dimensions)
        indices2 = tuple(sympy.symbols('%s2' % d) for d in self.grid.dimensions)
        # 1, x1, y1, z1, x1*y1, ...
        indices = list(powerset(indices1))
        indices[0] = (1,)
        point_sym = list(powerset(self._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self._point_increments:
            # substitute x1 by x2 if increment in that dimension
            subs = dict((indices1[d], indices2[d] if i[d] == 1 else indices1[d])
                        for d in range(len(i)))
            A += [[1] + [a.subs(subs) for a in ref_A[1:]]]

        A = sympy.Matrix(A)
        # Coordinate values of the sparse point
        p = sympy.Matrix([[np.prod(ind)] for ind in point_sym])

        # reference cell x1:0, x2:h_x
        left = dict((a, 0) for a in indices1)
        right = dict((b, dim.spacing) for b, dim in zip(indices2, self.grid.dimensions))
        reference_cell = {**left, **right}
        # Substitute in interpolation matrix
        A = A.subs(reference_cell)
        return A.inv().T * p

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

    @cached_property
    def _coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point."""
        indices = self.grid.dimensions
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i in zip(self._coordinate_symbols,
                                              self.grid.origin,
                                              self._coordinate_indices,
                                              indices[:self.grid.dim])])

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
        points = OrderedDict([(p, Symbol(name='ii%d' % i))
                              for i, p in enumerate(indices)])

        return index_matrix, points

    def _interpolation_indices(self, variables, offset=0):
        """Generate interpolation indices for the :class:`GridedFunction`s
        in ``variables``."""
        index_matrix, points = self._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                lb = sympy.And(p >= d.symbolic_start - self._radius, evaluate=False)
                ub = sympy.And(p <= d.symbolic_end + self._radius, evaluate=False)
                condition = sympy.And(lb, ub, evaluate=False)
                mapper[d] = ConditionalDimension(p.name, self._sparse_dim,
                                                 condition=condition, indirect=True)

            # Track Indexed substitutions
            idx_subs.append(OrderedDict([(v, v.subs(mapper)) for v in variables
                                         if v.function is not self]))

        # Equations for the indirection dimensions
        eqns = [Eq(v, k) for k, v in points.items()]
        # Equations (temporaries) for the coefficients
        eqns.extend([Eq(p, c) for p, c in
                     zip(self._point_symbols, self._coordinate_bases)])

        return idx_subs, eqns

    @property
    def gridpoints(self):
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        ret = []
        for coords in self.coordinates.data:
            ret.append(tuple(int(sympy.floor((c - o.data)/i.spacing.data)) for c, o, i in
                             zip(coords, self.grid.origin, self.grid.dimensions)))
        return ret

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param increment: (Optional) if True, perform an increment rather
                          than an assignment. Defaults to False.
        """
        variables = list(retrieve_functions(expr))

        # List of indirection indices for all adjacent grid points
        idx_subs, eqns = self._interpolation_indices(variables, offset)

        # Substitute coordinate base symbols into the coefficients
        args = [expr.subs(v_sub) * b.subs(v_sub)
                for b, v_sub in zip(self._coefficients, idx_subs)]

        # Accumulate point-wise contributions into a temporary
        rhs = Scalar(name='sum', dtype=self.dtype)
        summands = [Eq(rhs, 0.)] + [Inc(rhs, i) for i in args]

        # Write/Incr `self`
        lhs = self.subs(self_subs)
        last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

        return eqns + summands + last

    def inject(self, field, expr, offset=0):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        """

        variables = list(retrieve_functions(expr)) + [field]

        # List of indirection indices for all adjacent grid points
        idx_subs, eqns = self._interpolation_indices(variables, offset)

        # Substitute coordinate base symbols into the coefficients
        eqns.extend([Inc(field.subs(vsub), expr.subs(vsub) * b)
                     for b, vsub in zip(self._coefficients, idx_subs)])

        return eqns

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
        data = data if data is not None else self.data
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.coordinates: self.coordinates.data}

        comm = distributor.comm
        mpitype = MPI._typedict[np.dtype(self.dtype).char]

        # Pack (reordered) data values so that they can be sent out via an Alltoallv
        data = data[self._dist_scatter_mask]
        # Send out the sparse point values
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_alltoall
        scattered = np.empty(shape=rshape, dtype=self.dtype)
        comm.Alltoallv([data, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        data = scattered

        # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
        coords = self.coordinates.data[self._dist_subfunc_scatter_mask]
        # Send out the sparse point coordinates
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_subfunc_alltoall
        scattered = np.empty(shape=rshape, dtype=self.coordinates.dtype)
        comm.Alltoallv([coords, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        coords = scattered

        # Translate global coordinates into local coordinates
        coords = coords - np.array(self.grid.origin_domain, dtype=self.dtype)

        return {self: data, self.coordinates: coords}

    def _dist_gather(self, data):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        comm = distributor.comm

        # Send back the sparse point values
        sshape, scount, sdisp, _, rcount, rdisp = self._dist_alltoall
        gathered = np.empty(shape=sshape, dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, rcount, rdisp, mpitype],
                       [gathered, scount, sdisp, mpitype])
        data = gathered

        # Insert back into `self.data` based on the original (expected) data layout
        self.data[:] = data[self._dist_gather_mask]

        # Note: this method "mirrors" `_dist_scatter`: a sparse point that is sent
        # in `_dist_scatter` is here received; a sparse point that is received in
        # `_dist_scatter` is here sent. However, the `coordinates` SubFunction
        # values are not distributed, as this is a read-only field.

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['coordinates_data']


class SparseTimeFunction(AbstractSparseTimeFunction, SparseFunction):
    """
    A time-dependent :class:`SparseFunction`.

    :param name: Name of the function.
    :param nt: Size of the time dimension for point data.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) Data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseTimeFunction = True

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param increment: (Optional) if True, perform an increment rather
                          than an assignment. Defaults to False.
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
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.time_dim, u_t)
        if p_t is not None:
            expr = expr.subs(self.time_dim, p_t)

        return super(SparseTimeFunction, self).inject(field, expr, offset=offset)

    # Pickling support
    _pickle_kwargs = AbstractSparseTimeFunction._pickle_kwargs +\
        SparseFunction._pickle_kwargs


class PrecomputedSparseFunction(AbstractSparseFunction):
    """
    A specialised type of SparseFunction where the interpolation is externally defined.
    Currently, this means that the grid points and associated coefficients for each
    sparse point is precomputed at the time this object is being created.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param r: The number of gridpoints in each dimension to interpolate a single sparse
              point to. e.g. 2 for linear interpolation.
    :param gridpoints: The *reference* grid point corresponding to each sparse point.
                       Of all the gridpoints that one sparse point would be interpolated
                       to, this is the grid point closest to the origin, i.e. the one
                       with the lowest value of each coordinate dimension. Must be a
                       two-dimensional array of shape [npoint][grid.ndim].
    :param coefficients: An array containing the coefficient for each of the r^2 (2D) or
                         r^3 (3D) gridpoints that each sparsefunction will be interpolated
                         to. The coefficient is split across the n dimensions such that
                         the contribution of the point (i, j, k) will be multiplied by
                         coefficients[..., i]*coefficients[..., j]*coefficients[...,k]. So
                         for r=6, we will store 18 coefficients per sparse point (instead
                         of potentially 216). Shape must be [npoint][grid.ndim][r].
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_PrecomputedSparseFunction = True

    _sub_functions = ('gridpoints', 'coefficients')

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PrecomputedSparseFunction, self).__init__(*args, **kwargs)

            # Grid points per sparse point (2 in the case of bilinear and trilinear)
            r = kwargs.get('r')
            if not isinstance(r, int):
                raise TypeError('Interpolation needs `r` int argument')
            if r <= 0:
                raise ValueError('`r` must be > 0')
            self.r = r

            gridpoints = SubFunction(name="%s_gridpoints" % self.name, dtype=np.int32,
                                     dimensions=(self.indices[-1], Dimension(name='d')),
                                     shape=(self.npoint, self.grid.dim), space_order=0,
                                     parent=self)

            gridpoints_data = kwargs.get('gridpoints', None)
            assert(gridpoints_data is not None)
            gridpoints.data[:] = gridpoints_data[:]
            self._gridpoints = gridpoints

            coefficients = SubFunction(name="%s_coefficients" % self.name,
                                       dimensions=(self.indices[-1], Dimension(name='d'),
                                                   Dimension(name='i')),
                                       shape=(self.npoint, self.grid.dim, self.r),
                                       dtype=self.dtype, space_order=0, parent=self)
            coefficients_data = kwargs.get('coefficients', None)
            assert(coefficients_data is not None)
            coefficients.data[:] = coefficients_data[:]
            self._coefficients = coefficients
            warning("Ensure that the provided coefficient and grid point values are " +
                    "computed on the final grid that will be used for other " +
                    "computations.")

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param increment: (Optional) if True, perform an increment rather
                          than an assignment. Defaults to False.
        """
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs(t, u_t).subs(time, u_t)

        p, _, _ = self.coefficients.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.coefficients[p, i, rd])
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(self.indices[0], p_t)
        rhs = prod(coeffs) * expr.subs(dim_subs)

        return [Eq(lhs, lhs + rhs)]

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

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(self.indices[0], p_t)

        p, _ = self.gridpoints.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.coefficients[p, i, rd])
        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        return [Eq(field, field + rhs.subs(dim_subs))]

    @property
    def gridpoints(self):
        return self._gridpoints

    @property
    def coefficients(self):
        return self._coefficients

    def _dist_scatter(self, data=None):
        data = data if data is not None else self.data
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.gridpoints: self.gridpoints.data,
                    self.coefficients: self.coefficients.data}

        raise NotImplementedError

    def _dist_gather(self, data):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        raise NotImplementedError


class PrecomputedSparseTimeFunction(AbstractSparseTimeFunction,
                                    PrecomputedSparseFunction):
    """
    A specialised type of SparseFunction where the interpolation is externally defined.
    Currently, this means that the grid points and associated coefficients for each
    sparse point is precomputed at the time this object is being created.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param r: The number of gridpoints in each dimension to interpolate a single sparse
              point to. e.g. 2 for linear interpolation.
    :param gridpoints: The *reference* grid point corresponding to each sparse point.
                       Of all the gridpoints that one sparse point would be interpolated
                       to, this is the grid point closest to the origin, i.e. the one
                       with the lowest value of each coordinate dimension. Must be a
                       two-dimensional array of shape [npoint][grid.ndim].
    :param coefficients: An array containing the coefficient for each of the r^2 (2D) or
                         r^3 (3D) gridpoints that each sparsefunction will be interpolated
                         to. The coefficient is split across the n dimensions such that
                         the contribution of the point (i, j, k) will be multiplied by
                         coefficients[..., i]*coefficients[..., j]*coefficients[...,k]. So
                         for r=6, we will store 18 coefficients per sparse point (instead
                         of potentially 216). Shape must be [npoint][grid.ndim][r].
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_PrecomputedSparseTimeFunction = True
