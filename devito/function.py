import numpy as np
import sympy
from sympy.abc import s
from collections import OrderedDict
from functools import partial

from devito.parameters import configuration
from devito.logger import debug, error, warning
from devito.memory import CMemory, first_touch
from devito.cgen_utils import INT, FLOAT
from devito.dimension import d, p, t, time, x, y, z
from devito.arguments import ConstantArgProvider, TensorFunctionArgProvider
from devito.types import SymbolicFunction, AbstractSymbol
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative, generic_derivative,
                                      second_cross_derivative)
from devito.dse.inspection import indexify, retrieve_indexed
from devito.dse.extended_sympy import Eq

__all__ = ['Constant', 'Function', 'TimeFunction', 'SparseFunction',
           'Forward', 'Backward']


class TimeAxis(object):
    """Direction in which to advance the time index on
    :class:`TimeFunction` objects.

    :param axis: Either 'Forward' or 'Backward'
    """

    def __init__(self, axis):
        assert axis in ['Forward', 'Backward']
        self._axis = {'Forward': 1, 'Backward': -1}[axis]

    def __eq__(self, other):
        return self._axis == other._axis

    def __repr__(self):
        return {-1: 'Backward', 1: 'Forward'}[self._axis]


Forward = TimeAxis('Forward')
Backward = TimeAxis('Backward')


class Constant(SymbolicFunction, ConstantArgProvider):

    """
    Data object for constant values.
    """

    is_Constant = True
    is_Scalar = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractSymbol.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.shape = ()
            self.indices = ()
            self.dtype = kwargs.get('dtype', np.float32)
            self._value = kwargs.get('value', 0.)

    @property
    def data(self):
        """The value of the data object, as a scalar (int, float, ...)."""
        return self._value

    @data.setter
    def data(self, val):
        self._value = val


class TensorFunction(SymbolicFunction, TensorFunctionArgProvider):

    """
    Utility class to encapsulate all symbolic :class:`Function` types
    that represent tensor (array) data.
    """

    is_TensorFunction = True
    is_Tensor = True

    @property
    def _mem_external(self):
        """Return True if the associated data was/is/will be allocated directly
        from Python (e.g., via NumPy arrays), False otherwise."""
        return True


class Function(TensorFunction):
    """Data object for spatially varying data acting as a :class:`SymbolicFunction`.

    :param name: Name of the symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param shape: (Optional) shape of the associated data for this symbol.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param space_order: Discretisation order for space derivatives
    :param initializer: Function to initialize the data, optional

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

       :class:`Function` objects are assumed to be constant in time
       and therefore do not support time derivatives. Use
       :class:`TimeFunction` for time-varying grid data.
    """

    is_Function = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.grid = kwargs.get('grid', None)

            if self.grid is None:
                self.shape_domain = kwargs.get('shape', None)
                self.dtype = kwargs.get('dtype', np.float32)
                if self.shape_domain is None:
                    error("Creating a Function requires either 'shape'"
                          "or a 'grid' argument")
                    raise ValueError("Unknown symbol dimensions or shape")
            else:
                self.shape_domain = self.grid.shape_domain
                self.dtype = kwargs.get('dtype', self.grid.dtype)
            self.indices = self._indices(**kwargs)

            self.space_order = kwargs.get('space_order', 1)
            self.initializer = kwargs.get('initializer', None)
            if self.initializer is not None:
                assert(callable(self.initializer))
            self._first_touch = kwargs.get('first_touch', configuration['first_touch'])
            self._data_object = None

            # Dynamically create notational shortcuts for space derivatives
            for dim in self.space_dimensions:
                # First derivative, centred
                dx = partial(first_derivative, order=self.space_order,
                             dim=dim, side=centered)
                setattr(self.__class__, 'd%s' % dim.name,
                        property(dx, 'Return the symbolic expression for '
                                 'the centered first derivative wrt. '
                                 'the %s dimension' % dim.name))

                # First derivative, left
                dxl = partial(first_derivative, order=self.space_order,
                              dim=dim, side=left)
                setattr(self.__class__, 'd%sl' % dim.name,
                        property(dxl, 'Return the symbolic expression for '
                                 'the left-sided first derivative wrt. '
                                 'the %s dimension' % dim.name))

                # First derivative, right
                dxr = partial(first_derivative, order=self.space_order,
                              dim=dim, side=right)
                setattr(self.__class__, 'd%sr' % dim.name,
                        property(dxr, 'Return the symbolic expression for '
                                 'the right-sided first derivative wrt. '
                                 'the %s dimension' % dim.name))

                # Second derivative
                dx2 = partial(generic_derivative, deriv_order=2, dim=dim,
                              fd_order=int(self.space_order / 2))
                setattr(self.__class__, 'd%s2' % dim.name,
                        property(dx2, 'Return the symbolic expression for '
                                 'the second derivative wrt. the '
                                 '%s dimension' % dim.name))

                # Fourth derivative
                dx4 = partial(generic_derivative, deriv_order=4, dim=dim,
                              fd_order=max(int(self.space_order / 2), 2))
                setattr(self.__class__, 'd%s4' % dim.name,
                        property(dx4, 'Return the symbolic expression for '
                                 'the fourth derivative wrt. the '
                                 '%s dimension' % dim.name))

                for dim2 in self.space_dimensions:
                    # First cross derivative
                    dxy = partial(cross_derivative, order=self.space_order,
                                  dims=(dim, dim2))
                    setattr(self.__class__, 'd%s%s' % (dim.name, dim2.name),
                            property(dxy, 'Return the symbolic expression for '
                                     'the first cross derivative wrt. the '
                                     '%s and %s dimensions' %
                                     (dim.name, dim2.name)))

                    # Second cross derivative
                    dx2y2 = partial(second_cross_derivative, dims=(dim, dim2),
                                    order=self.space_order)
                    setattr(self.__class__, 'd%s2%s2' % (dim.name, dim2.name),
                            property(dx2y2, 'Return the symbolic expression for '
                                     'the second cross derivative wrt. the '
                                     '%s and %s dimensions' %
                                     (dim.name, dim2.name)))

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param grid: :class:`Grid` that defines the spatial domain.
        :param dimensions: Optional, list of :class:`Dimension`
                           objects that defines data layout.
        :return: Dimension indices used for each axis.

        ..note::

        Only one of :param grid: or :param dimensions: is required.
        """

        grid = kwargs.get('grid', None)
        dimensions = kwargs.get('dimensions', None)
        if grid is None:
            if dimensions is None:
                error("Creating a Function object requries either "
                      "a 'grid' or the 'dimensions' argument.")
                raise ValueError("Unknown symbol dimensions or shape")
        else:
            if dimensions is not None:
                warning("Creating Function with 'grid' and 'dimensions' "
                        "argument; ignoring the 'dimensions' and using 'grid'.")
            dimensions = grid.dimensions
        return dimensions

    @property
    def shape_data(self):
        """
        Full allocated shape of the data associated with this :class:`Function`.
        """
        return self.shape_domain

    @property
    def shape(self):
        return self.shape_data

    @property
    def space_dimensions(self):
        """Tuple of index dimensions that define physical space."""
        return tuple(d for d in self.indices if d.is_Space)

    def _allocate_memory(self):
        """Allocate memory in terms of numpy ndarrays."""
        debug("Allocating memory for %s (%s)" % (self.name, str(self.shape)))
        self._data_object = CMemory(self.shape, dtype=self.dtype)
        if self._first_touch:
            first_touch(self)
        else:
            self.data.fill(0)

    @property
    def data(self):
        """The value of the data object, as a :class:`numpy.ndarray` storing
        elements in the classical row-major storage layout."""
        if self._data_object is None:
            self._allocate_memory()
        return self._data_object.ndpointer

    def initialize(self):
        """Apply the data initilisation function, if it is not None."""
        if self.initializer is not None:
            self.initializer(self.data)

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative wrt. all spatial dimensions.
        """
        derivs = tuple('d%s2' % d.name for d in self.space_dimensions)

        return sum([getattr(self, d) for d in derivs[:self.dim]])

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian
        wrt. all spatial dimensions.
        """
        order = self.space_order/2
        first = sum([second_derivative(self, dim=d, order=order)
                     for d in self.space_dimensions])
        return sum([second_derivative(first * weight, dim=d, order=order)
                    for d in self.space_dimensions])


class TimeFunction(Function):
    """
    Data object for time-varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param shape: (Optional) shape of the associated data for this symbol.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout in addition to the time dimension.
    :param dtype: (Optional) data type of the buffered data
    :param save: Save the intermediate results to the data buffer. Defaults
                 to `False`, indicating the use of alternating buffers.
    :param time_dim: Size of the time dimension that dictates the leading
                     dimension of the data buffer if :param save: is True.
    :param time_order: Order of the time discretization which affects the
                       final size of the leading time dimension of the
                       data buffer.

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

       The parameter ``shape`` should only define the spatial shape of
       the grid. The temporal dimension will be inserted automatically
       as the leading dimension, according to the ``time_dim``,
       ``time_order`` and whether we want to write intermediate
       timesteps in the buffer. The same is true for explicitly
       provided dimensions, which will be added to the automatically
       derived time dimensions symbol. For example:

       .. code-block:: python

          In []: TimeFunction(name="a", dimensions=(x, y, z))
          Out[]: a(t, x, y, z)

          In []: TimeFunction(name="a", shape=(20, 30))
          Out[]: a(t, x, y)

    """

    is_TimeFunction = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TimeFunction, self).__init__(*args, **kwargs)
            self.time_dim = kwargs.get('time_dim', None)
            self.time_order = kwargs.get('time_order', 1)
            self.save = kwargs.get('save', False)

            if not self.save:
                if self.time_dim is not None:
                    warning('Explicit time dimension size (time_dim) found for '
                            'TimeFunction symbol %s, despite \nusing a buffered time '
                            'dimension (save=False). This value will be ignored!'
                            % self.name)
                self.time_dim = self.time_order + 1
                self.indices[0].modulo = self.time_dim
            else:
                if self.time_dim is None:
                    error('Time dimension (time_dim) is required'
                          'to save intermediate data with save=True')
                    raise ValueError("Unknown time dimensions")

    @property
    def shape_data(self):
        """
        Full allocated shape of the data associated with this :class:`TimeFunction`.
        """
        tsize = self.time_dim if self.save else self.time_order + 1
        return (tsize, ) + self.shape_domain

    def initialize(self):
        if self.initializer is not None:
            self.initializer(self.data)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param dimensions: Optional, list of :class:`Dimension`
                           objects that defines data layout.
        :param shape: Optional, shape of the spatial data to
                      automatically infer dimension symbols.
        :return: Dimension indices used for each axis.
        """
        save = kwargs.get('save', None)
        tidx = time if save else t
        _indices = Function._indices(**kwargs)
        return tuple([tidx] + list(_indices))

    @property
    def dim(self):
        """Returns the spatial dimension of the data object"""
        return len(self.shape[1:])

    @property
    def forward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[0]

        return self.subs(_t, _t + i * s)

    @property
    def backward(self):
        """Symbol for the time-backward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[0]

        return self.subs(_t, _t - i * s)

    @property
    def dt(self):
        """Symbol for the first derivative wrt the time dimension"""
        _t = self.indices[0]
        if self.time_order == 1:
            # This hack is needed for the first-order diffusion test
            indices = [_t, _t + s]
        else:
            width = int(self.time_order / 2)
            indices = [(_t + i * s) for i in range(-width, width + 1)]

        return self.diff(_t).as_finite_difference(indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        _t = self.indices[0]
        width_t = int(self.time_order / 2)
        indt = [(_t + i * s) for i in range(-width_t, width_t + 1)]

        return self.diff(_t, _t).as_finite_difference(indt)


class CompositeFunction(Function):
    """
    Base class for Function classes that have Function children
    """

    is_CompositeFunction = True

    def __init__(self, *args, **kwargs):
        super(CompositeFunction, self).__init__(self, *args, **kwargs)
        self._children = []

    @property
    def children(self):
        return self._children


class SparseFunction(CompositeFunction):
    """
    Data object for sparse point data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param npoint: Number of points to sample
    :param nt: Size of the time dimension for point data
    :param ndim: Dimension of the coordinate data, eg. 2D or 3D
    :param coordinates: Optional coordinate data for the sparse points
    :param dtype: Data type of the buffered data
    """

    is_SparseFunction = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.nt = kwargs.get('nt', 0)
            self.npoint = kwargs.get('npoint')
            self.ndim = kwargs.get('ndim')
            kwargs['shape'] = (self.nt, self.npoint) if self.nt > 0 else (self.npoint, )
            super(SparseFunction, self).__init__(self, *args, **kwargs)

            # Allocate and copy coordinate data
            self.coordinates = Function(name='%s_coords' % self.name,
                                        dimensions=[self.indices[-1], d],
                                        shape=(self.npoint, self.ndim))
            self._children.append(self.coordinates)
            coordinates = kwargs.get('coordinates', None)
            if coordinates is not None:
                self.coordinates.data[:] = coordinates[:]
            self.coordinates_dimensions = kwargs.get('coord_dims', (x, y, z)[:self.ndim])

    def __new__(cls, *args, **kwargs):
        nt = kwargs.get('nt')
        npoint = kwargs.get('npoint')
        kwargs['shape'] = (nt, npoint) if nt > 0 else (npoint, )

        return Function.__new__(cls, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: indices used for axis.
        """
        dimensions = kwargs.get('dimensions', None)
        nt = kwargs.get('nt', 0)
        defaults = [time, p] if nt > 0 else [p]
        return dimensions or defaults

    @property
    def coefficients(self):
        """Symbolic expression for the coefficients for sparse point
        interpolation according to:
        https://en.wikipedia.org/wiki/Bilinear_interpolation.

        :returns: List of coefficients, eg. [b_11, b_12, b_21, b_22]
        """
        # Grid indices corresponding to the corners of the cell
        x1, y1, z1, x2, y2, z2 = sympy.symbols('x1, y1, z1, x2, y2, z2')
        # Coordinate values of the sparse point
        px, py, pz = self.point_symbols
        if self.ndim == 2:
            A = sympy.Matrix([[1, x1, y1, x1*y1],
                              [1, x1, y2, x1*y2],
                              [1, x2, y1, x2*y1],
                              [1, x2, y2, x2*y2]])

            p = sympy.Matrix([[1],
                              [px],
                              [py],
                              [px*py]])

        elif self.ndim == 3:
            A = sympy.Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                              [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                              [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                              [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                              [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                              [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                              [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                              [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])

            p = sympy.Matrix([[1],
                              [px],
                              [py],
                              [pz],
                              [px*py],
                              [px*pz],
                              [py*pz],
                              [px*py*pz]])
        else:
            error('Point interpolation only supported for 2D and 3D')
            raise NotImplementedError('Interpolation coefficients not '
                                      'implemented for %d dimensions.'
                                      % self.ndim)

        # Map to reference cell
        reference_cell = {x1: 0, y1: 0, z1: 0, x2: x.spacing, y2: y.spacing,
                          z2: z.spacing}
        A = A.subs(reference_cell)
        return A.inv().T.dot(p)

    @property
    def point_symbols(self):
        """Symbol for coordinate value in each dimension of the point"""
        return sympy.symbols('px, py, pz')

    @property
    def point_increments(self):
        """Index increments in each dimension for each point symbol"""
        if self.ndim == 2:
            return ((0, 0), (0, 1), (1, 0), (1, 1))
        elif self.ndim == 3:
            return ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
                    (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
        else:
            error('Point interpolation only supported for 2D and 3D')
            raise NotImplementedError('Point increments not defined '
                                      'for %d dimensions.' % self.ndim)

    @property
    def coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension"""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.ndim)])

    @property
    def coordinate_indices(self):
        """Symbol for each grid index according to the coordinates"""
        return tuple([INT(sympy.Function('floor')(c / i.spacing))
                      for c, i in zip(self.coordinate_symbols,
                                      self.coordinates_dimensions)])

    @property
    def coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        return tuple([FLOAT(c - idx * i.spacing)
                      for c, idx, i in zip(self.coordinate_symbols,
                                           self.coordinate_indices,
                                           self.coordinates_dimensions)])

    def interpolate(self, expr, offset=0, **kwargs):
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
        u_t = kwargs.get('u_t', None)
        p_t = kwargs.get('p_t', None)
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            expr = expr.subs(t, u_t).subs(time, u_t)

        variables = list(retrieve_indexed(expr))
        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]
        # Generate index substituions for all grid variables
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.ndim] + idx])
                      for v in variables]
            idx_subs += [OrderedDict(v_subs)]
        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        rhs = sum([expr.subs(vsub) * b.subs(subs)
                   for b, vsub in zip(self.coefficients, idx_subs)])
        rhs += indexify(self) if kwargs.get('cumulative', False) else 0
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(self.indices[0], p_t)
        return [Eq(lhs, rhs)]

    def inject(self, field, expr, offset=0, **kwargs):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        u_t = kwargs.get('u_t', None)
        p_t = kwargs.get('p_t', None)

        expr = indexify(expr)
        field = indexify(field)
        variables = list(retrieve_indexed(expr)) + [field]

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(self.indices[0], p_t)

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]

        # Generate index substituions for all grid variables except
        # the sparse `SparseFunction` types
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.ndim] + idx])
                      for v in variables if not v.base.function.is_SparseFunction]
            idx_subs += [OrderedDict(v_subs)]

        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        return [Eq(field.subs(vsub),
                   field.subs(vsub) + expr.subs(subs).subs(vsub) * b.subs(subs))
                for b, vsub in zip(self.coefficients, idx_subs)]

    def assign(self, field, expr, offset=0, **kwargs):
        """Symbol for assignment of an expression onto a grid
        This function works at the symbolic level and will therefor not perform any
        safety check. Assigned value may be overwritten if the points are
        too close to each other.
        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        u_t = kwargs.get('u_t', None)
        p_t = kwargs.get('p_t', None)

        expr = indexify(expr)
        field = indexify(field)
        variables = list(retrieve_indexed(expr)) + [field]

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(self.indices[0], p_t)

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]

        # Generate index substituions for all grid variables except
        # the sparse `SparseFunction` types
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.ndim] + idx])
                      for v in variables if not v.base.function.is_SparseFunction]
            idx_subs += [OrderedDict(v_subs)]

        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        return [Eq(field.subs(vsub), expr.subs(subs).subs(vsub) * b.subs(subs))
                for b, vsub in zip(self.coefficients, idx_subs)]
