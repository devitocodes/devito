from collections import OrderedDict
from functools import partial

import sympy
import numpy as np
from psutil import virtual_memory

from devito.cgen_utils import INT, cast_mapper
from devito.data import Data, first_touch
from devito.dimension import Dimension, DefaultDimension
from devito.equation import Eq, Inc
from devito.exceptions import InvalidArgument
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative, generic_derivative,
                                      second_cross_derivative)
from devito.logger import debug, error, warning
from devito.parameters import configuration
from devito.symbolics import indexify, retrieve_indexed
from devito.types import AbstractCachedFunction, AbstractCachedSymbol
from devito.tools import ReducerMap, prod

__all__ = ['Constant', 'Function', 'TimeFunction', 'SparseFunction',
           'SparseTimeFunction']


class Constant(AbstractCachedSymbol):

    """
    Symbol representing constant values in symbolic equations.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_Input = True
    is_Constant = True
    is_Scalar = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.dtype = kwargs.get('dtype', np.float32)
            self._value = kwargs.get('value')

    @property
    def data(self):
        """The value of the data object, as a scalar (int, float, ...)."""
        return self._value

    @data.setter
    def data(self, val):
        self._value = val

    @property
    def base(self):
        return self

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this symbol."""
        return (self.name,)

    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.
        """
        key = alias or self
        return {key.name: self.data}

    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, Constant):
                return new._arg_defaults(alias=self)
            else:
                return {self.name: new}
        else:
            return self._arg_defaults()

    def _arg_check(self, args, dspace):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for %s" % self.name)
        key = args[self.name]
        try:
            # Might be a plain number, w/o a dtype field
            if key.dtype != self.dtype:
                warning("Data type %s of runtime value `%s` does not match the "
                        "Constant data type %s" % (key.dtype, self.name, self.dtype))
        except AttributeError:
            pass


class TensorFunction(AbstractCachedFunction):

    """
    Utility class to encapsulate all symbolic types that represent
    tensor (array) data.

    .. note::

        Users should not instantiate this class. Use :class:`Function` or
        :class:`SparseFunction` (or their subclasses) instead.
    """

    # Required by SymPy, otherwise the presence of __getitem__ will make SymPy
    # think that a TensorFunction is actually iterable, thus breaking many of
    # its key routines (e.g., solve)
    _iterable = False

    is_Input = True
    is_TensorFunction = True
    is_Tensor = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            # Staggered mask
            self._staggered = kwargs.get('staggered', tuple(0 for _ in self.indices))
            if len(self.staggered) != len(self.indices):
                raise ValueError("'staggered' needs %s entries for indices %s"
                                 % (len(self.indices), self.indices))

            # Data-related properties
            self.initializer = kwargs.get('initializer')
            if self.initializer is not None:
                assert(callable(self.initializer))
            self._first_touch = kwargs.get('first_touch', configuration['first_touch'])
            self._data = None

    def __getitem__(self, index):
        """Shortcut for ``self.indexed[index]``."""
        return self.indexed[index]

    def _allocate_memory(func):
        """Allocate memory as a :class:`Data`."""
        def wrapper(self):
            if self._data is None:
                debug("Allocating memory for %s%s" % (self.name, self.shape_allocated))
                self._data = Data(self.shape_allocated, self.indices, self.dtype)
                if self._first_touch:
                    first_touch(self)
                if self.initializer is not None:
                    if self._first_touch:
                        warning("`first touch` together with `initializer` causing "
                                "redundant data initialization")
                    try:
                        self.initializer(self._data)
                    except ValueError:
                        # Perhaps user only wants to initialise the physical domain
                        self.initializer(self._data[self._mask_domain])
                else:
                    self._data.fill(0)
            return func(self)
        return wrapper

    @property
    def _data_buffer(self):
        """Reference to the data. Unlike ``data, data_with_halo, data_allocated``,
        this *never* returns a view of the data. This method is for internal use only."""
        return self.data_allocated

    @property
    def _mem_external(self):
        return True

    @property
    def shape(self):
        """
        Shape of the domain associated with this :class:`TensorFunction`.
        The domain constitutes the area of the data written to in a
        stencil update.
        """
        return self.shape_domain

    @property
    def shape_domain(self):
        """
        Shape of the domain associated with this :class:`TensorFunction`.
        The domain constitutes the area of the data written to in a
        stencil update.

        .. note::

            Alias to ``self.shape``.
        """
        return tuple(i - j for i, j in zip(self._shape, self.staggered))

    @property
    def shape_with_halo(self):
        """
        Shape of the domain plus the read-only stencil boundary associated
        with this :class:`Function`.
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_domain, self._halo))

    @property
    def shape_allocated(self):
        """
        Shape of the allocated data associated with this :class:`Function`.
        It includes the domain and halo regions, as well as any additional
        padding outside of the halo.
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_with_halo, self._padding))

    @property
    def data(self):
        """
        The function data values, as a :class:`numpy.ndarray`.

        Elements are stored in row-major format.
        """
        return self.data_domain

    @property
    @_allocate_memory
    def data_domain(self):
        """
        The domain data values.

        Elements are stored in row-major format.

        .. note::

            Alias to ``self.data``.
        """
        return self._data[self._mask_domain]

    @property
    @_allocate_memory
    def data_with_halo(self):
        """
        The domain+halo data values.

        Elements are stored in row-major format.
        """
        return self._data[self._mask_with_halo]

    @property
    @_allocate_memory
    def data_allocated(self):
        """
        The allocated data values, that is domain+halo+padding.

        Elements are stored in row-major format.
        """
        return self._data

    @property
    def space_dimensions(self):
        """Tuple of :class:`Dimension`s that define physical space."""
        return tuple(d for d in self.indices if d.is_Space)

    @property
    def staggered(self):
        return self._staggered

    @property
    def symbolic_shape(self):
        """
        Return the symbolic shape of the object. This includes: ::

            * the padding, halo, and domain regions. While halo and padding are
              known quantities (integers), the domain size is represented by a symbol.
            * the shifting induced by the ``staggered`` mask
        """
        symbolic_shape = super(TensorFunction, self).symbolic_shape
        return tuple(sympy.Add(i, -j, evaluate=False)
                     for i, j in zip(symbolic_shape, self.staggered))

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this function."""
        return (self.name,)

    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param alias: (Optional) name under which to store values.
        """
        key = alias or self
        args = ReducerMap({key.name: self._data_buffer})

        # Collect default dimension arguments from all indices
        for i, s, o, k in zip(self.indices, self.shape, self.staggered, key.indices):
            args.update(i._arg_defaults(start=0, size=s+o, alias=k))
        return args

    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, TensorFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self).reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {self.name: new}
                # Add value overrides for all associated dimensions
                for i, s, o in zip(self.indices, new.shape, self.staggered):
                    values.update(i._arg_defaults(size=s+o-sum(self._offset_domain[i])))
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_check(self, args, intervals):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.

        :raises InvalidArgument: If, given the runtime arguments ``args``, an
                                 out-of-bounds access will be performed.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for `%s`" % self.name)
        key = args[self.name]
        if len(key.shape) != self.ndim:
            raise InvalidArgument("Shape %s of runtime value `%s` does not match "
                                  "dimensions %s" % (key.shape, self.name, self.indices))
        if key.dtype != self.dtype:
            warning("Data type %s of runtime value `%s` does not match the "
                    "Function data type %s" % (key.dtype, self.name, self.dtype))
        for i, s in zip(self.indices, key.shape):
            i._arg_check(args, s, intervals[i])


class Function(TensorFunction):
    """A :class:`TensorFunction` providing operations to express
    finite-difference approximation. A ``Function`` encapsulates
    space-varying data; for time-varying data, use :class:`TimeFunction`.

    :param name: Name of the symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param space_order: Discretisation order for space derivatives. By default,
                        ``space_order`` points are available on both sides of
                        a generic point of interest, including those on the grid
                        border. Sometimes, fewer points may be necessary; in
                        other cases, depending on the PDE being approximated,
                        more points may be necessary. In such cases, one
                        can pass a 3-tuple ``(o, lp, rp)`` instead of a single
                        integer representing the discretization order. Here,
                        ``o`` is the discretization order, while ``lp`` and ``rp``
                        indicate how many points are expected on left (``lp``)
                        and right (``rp``) of a point of interest.
    :param shape: (Optional) shape of the domain region in grid points.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param staggered: (Optional) tuple containing staggering offsets.
    :param padding: (Optional) allocate extra grid points at a space dimension
                    boundary. These may be used for data alignment. Defaults to 0.
                    In alternative to an integer, a tuple, indicating the padding
                    in each dimension, may be passed; in this case, an error is
                    raised if such tuple has fewer entries then the number of space
                    dimensions.
    :param initializer: (Optional) A callable to initialize the data

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

    .. note::

       :class:`Function` objects are assumed to be constant in time
       and therefore do not support time derivatives. Use
       :class:`TimeFunction` for time-varying grid data.

    .. note::

       The tuple :param staggered: contains a ``1`` in each dimension
       entry that should be staggered, and ``0`` otherwise. For example,
       ``staggered=(1, 0, 0)`` entails discretization on horizontal edges,
       ``staggered=(0, 0, 1)`` entails discretization on vertical edges,
       ``staggered=(0, 1, 1)`` entails discretization side facets and
       ``staggered=(1, 1, 1)`` entails discretization on cells.
    """

    is_Function = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(Function, self).__init__(*args, **kwargs)

            # Grid
            self.grid = kwargs.get('grid')

            # Data type (provided or inferred)
            if self.grid is None:
                self.dtype = kwargs.get('dtype', np.float32)
            else:
                self.dtype = kwargs.get('dtype', self.grid.dtype)

            # Halo region
            space_order = kwargs.get('space_order', 1)
            if isinstance(space_order, int):
                self.space_order = space_order
                halo = (space_order, space_order)
            elif isinstance(space_order, tuple) and len(space_order) == 3:
                self.space_order, left_points, right_points = space_order
                halo = (left_points, right_points)
            else:
                raise ValueError("'space_order' must be int or 3-tuple of ints")
            self._halo = tuple(halo if i in self._halo_indices else (0, 0)
                               for i in self.indices)

            # Padding region
            padding = kwargs.get('padding', 0)
            if isinstance(padding, int):
                padding = tuple((padding,)*2 for i in range(self.ndim))
            elif isinstance(padding, tuple) and len(padding) == self.ndim:
                padding = tuple((i,)*2 if isinstance(i, int) else i for i in padding)
            else:
                raise ValueError("'padding' must be int or %d-tuple of ints" % self.ndim)
            self._padding = padding

            # Dynamically add derivative short-cuts
            self._initialize_derivatives()

    def _initialize_derivatives(self):
        """
        Dynamically create notational shortcuts for space derivatives.
        """
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
    def __indices_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
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

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        if grid is None:
            shape = kwargs.get('shape')
            if shape is None:
                raise ValueError("Function needs either 'shape' or 'grid' argument")
        else:
            shape = grid.shape_domain
        return shape

    @property
    def _halo_indices(self):
        """Return the function indices for which a halo region is defined."""
        return self.indices

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative wrt. all spatial dimensions.
        """
        derivs = tuple('d%s2' % d.name for d in self.space_dimensions)

        return sum([getattr(self, d) for d in derivs[:self.ndim]])

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
    A special :class:`Function` encapsulating time-varying data.

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param space_order: Discretisation order for space derivatives. By default,
                        ``space_order`` points are available on both sides of
                        a generic point of interest, including those on the grid
                        border. Sometimes, fewer points may be necessary; in
                        other cases, depending on the PDE being approximated,
                        more points may be necessary. In such cases, one
                        can pass a 3-tuple ``(o, lp, rp)`` instead of a single
                        integer representing the discretization order. Here,
                        ``o`` is the discretization order, while ``lp`` and ``rp``
                        indicate how many points are expected on left (``lp``)
                        and right (``rp``) of a point of interest.
    :param time_order: Discretization order for time derivatives.
    :param shape: (Optional) shape of the domain region in grid points.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data
    :param save: (Optional) Save the intermediate results to the data buffer.
                 Defaults to `None`, indicating the use of alternating buffers.
                 If intermediate results are required, the value of save must be
                 set to the required size of the time dimension.
    :param time_dim: (Optional) The :class:`Dimension` object to use to represent
                     time in this symbol. Defaults to the time dimension provided
                     by the :class:`Grid`.
    :param staggered: (Optional) tuple containing staggering offsets.
    :param padding: (Optional) allocate extra grid points at a space dimension
                    boundary. These may be used for data alignment. Defaults to 0.
                    In alternative to an integer, a tuple, indicating the padding
                    in each dimension, may be passed; in this case, an error is
                    raised if such tuple has fewer entries then the number of
                    space dimensions.
    :param initializer: (Optional) A callable to initialize the data

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.

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

    _time_position = 0
    """Position of time index among the function indices."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TimeFunction, self).__init__(*args, **kwargs)

            # Check we won't allocate too much memory for the system
            available_mem = virtual_memory().available
            if np.dtype(self.dtype).itemsize * self.size > available_mem:
                warning("Trying to allocate more memory for symbol %s " % self.name +
                        "than available on physical device, this will start swapping")

            self.time_dim = kwargs.get('time_dim', self.indices[self._time_position])
            self.time_order = kwargs.get('time_order', 1)
            self.save = type(kwargs.get('save', None) or None)
            if not isinstance(self.time_order, int):
                raise ValueError("'time_order' must be int")

    @classmethod
    def __indices_setup__(cls, **kwargs):
        save = kwargs.get('save')
        grid = kwargs.get('grid')
        time_dim = kwargs.get('time_dim')

        if grid is None:
            error('TimeFunction objects require a grid parameter.')
            raise ValueError('No grid provided for TimeFunction.')

        if time_dim is None:
            time_dim = grid.time_dim if save else grid.stepping_dim
        elif not (isinstance(time_dim, Dimension) and time_dim.is_Time):
            raise ValueError("'time_dim' must be a time dimension")

        indices = list(Function.__indices_setup__(**kwargs))
        indices.insert(cls._time_position, time_dim)

        return tuple(indices)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        save = kwargs.get('save') or None  # Force to None if 0/False/None/...
        shape = kwargs.get('shape')
        time_order = kwargs.get('time_order', 1)

        if grid is None:
            if shape is None:
                raise ValueError("TimeFunction needs either 'shape' or 'grid' argument")
            if save is not None:
                raise ValueError("Ambiguity detected: provide either 'grid' and 'save' "
                                 "or 'shape', where 'shape[0] == save'")
        else:
            if save is not None:
                if not isinstance(save, int):
                    raise ValueError("save must be an int indicating the number of " +
                                     "timesteps to be saved (is %s)" % type(save))
                shape = (save,) + grid.shape_domain
            else:
                shape = (time_order + 1,) + grid.shape_domain
        return shape

    @property
    def _halo_indices(self):
        return tuple(i for i in self.indices if not i.is_Time)

    @property
    def forward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[0]

        return self.subs(_t, _t + i * _t.spacing)

    @property
    def backward(self):
        """Symbol for the time-backward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[0]

        return self.subs(_t, _t - i * _t.spacing)

    @property
    def dt(self):
        """Symbol for the first derivative wrt the time dimension"""
        _t = self.indices[0]
        if self.time_order == 1:
            # This hack is needed for the first-order diffusion test
            indices = [_t, _t + _t.spacing]
        else:
            width = int(self.time_order / 2)
            indices = [(_t + i * _t.spacing) for i in range(-width, width + 1)]

        return self.diff(_t).as_finite_difference(indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        _t = self.indices[0]
        width_t = int(self.time_order / 2)
        indt = [(_t + i * _t.spacing) for i in range(-width_t, width_t + 1)]

        return self.diff(_t, _t).as_finite_difference(indt)


class AbstractSparseFunction(TensorFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """
    # Symbols that are encapsulated within this symbol (e.g. coordinates)
    _child_functions = []

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseFunction, self).__init__(*args, **kwargs)

            npoint = kwargs.get('npoint')
            if not isinstance(npoint, int) and npoint > 0:
                raise ValueError('SparseFunction requires parameter `npoint` (> 0)')
            self.npoint = npoint

            # Grid must be provided
            grid = kwargs.get('grid')
            if kwargs.get('grid') is None:
                raise ValueError('SparseFunction objects require a grid parameter')
            self.grid = grid

            self.dtype = kwargs.get('dtype', self.grid.dtype)
            self.space_order = kwargs.get('space_order', 0)

            # Halo region
            self._halo = tuple((0, 0) for i in range(self.ndim))

            # Padding region
            self._padding = tuple((0, 0) for i in range(self.ndim))

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

    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param alias: (Optional) name under which to store values.
        """
        key = alias or self
        args = super(AbstractSparseFunction, self)._arg_defaults(alias=alias)
        for child_name in self._child_functions:
            child = getattr(self, child_name)
            args.update(child._arg_defaults(alias=getattr(key, child_name)))
        return args

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this function."""
        return tuple([self.name] + [x for x in self._child_functions])


class SparseFunction(AbstractSparseFunction):
    """
    A special :class:`TensorFunction` representing a set of sparse point
    objects that are not aligned with the computational grid.

    A :class:`SparseFunction` provides symbolic interpolation routines
    to convert between grid-aligned :class:`Function` objects and sparse
    data points.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param shape: (Optional) shape of the function. Defaults to ``(npoints,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: Discretisation order for space derivatives.
    :param dtype: Data type of the buffered data.
    :param initializer: (Optional) A callable to initialize the data

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseFunction = True
    _child_functions = ['coordinates']

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SparseFunction, self).__init__(*args, **kwargs)

            # Set up coordinates of sparse points
            coordinates = Function(name='%s_coords' % self.name, dtype=self.dtype,
                                   dimensions=(self.indices[-1], Dimension(name='d')),
                                   shape=(self.npoint, self.grid.dim), space_order=0)
            coordinate_data = kwargs.get('coordinates')
            if coordinate_data is not None:
                coordinates.data[:] = coordinate_data[:]
            self.coordinates = coordinates

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
        if self.grid.dim == 2:
            A = sympy.Matrix([[1, x1, y1, x1*y1],
                              [1, x1, y2, x1*y2],
                              [1, x2, y1, x2*y1],
                              [1, x2, y2, x2*y2]])

            p = sympy.Matrix([[1],
                              [px],
                              [py],
                              [px*py]])

            # Map to reference cell
            x, y = self.grid.dimensions
            reference_cell = {x1: 0, y1: 0, x2: x.spacing, y2: y.spacing}

        elif self.grid.dim == 3:
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

            # Map to reference cell
            x, y, z = self.grid.dimensions
            reference_cell = {x1: 0, y1: 0, z1: 0, x2: x.spacing,
                              y2: y.spacing, z2: z.spacing}
        else:
            raise NotImplementedError('Interpolation coefficients not implemented '
                                      'for %d dimensions.' % self.grid.dim)

        A = A.subs(reference_cell)
        return A.inv().T.dot(p)

    @property
    def point_symbols(self):
        """Symbol for coordinate value in each dimension of the point"""
        return sympy.symbols('px, py, pz')

    @property
    def point_increments(self):
        """Index increments in each dimension for each point symbol"""
        if self.grid.dim == 2:
            return ((0, 0), (0, 1), (1, 0), (1, 1))
        elif self.grid.dim == 3:
            return ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
                    (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
        else:
            raise NotImplementedError('Point increments not defined '
                                      'for %d dimensions.' % self.grid.dim)

    @property
    def coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension"""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @property
    def coordinate_indices(self):
        """Symbol for each grid index according to the coordinates"""
        indices = self.grid.dimensions
        return tuple([INT(sympy.Function('floor')((c - o) / i.spacing))
                      for c, o, i in zip(self.coordinate_symbols, self.grid.origin,
                                         indices[:self.grid.dim])])

    @property
    def coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        indices = self.grid.dimensions
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i in zip(self.coordinate_symbols,
                                              self.grid.origin,
                                              self.coordinate_indices,
                                              indices[:self.grid.dim])])

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, cummulative=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param cummulative: (Optional) If True, perform an increment rather
                            than an assignment. Defaults to False.
        """
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs(t, u_t).subs(time, u_t)

        variables = list(retrieve_indexed(expr))
        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]
        # Generate index substituions for all grid variables
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.grid.dim] + idx])
                      for v in variables]
            idx_subs += [OrderedDict(v_subs)]
        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        rhs = sum([expr.subs(vsub) * b.subs(subs)
                   for b, vsub in zip(self.coefficients, idx_subs)])
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(self.indices[0], p_t)

        rhs = rhs + lhs if cummulative is True else rhs

        return [Eq(lhs, rhs)]

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

        # Generate index substitutions for all grid variables except
        # the `SparseFunction` types
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.grid.dim] + idx])
                      for v in variables if not v.base.function.is_SparseFunction]
            idx_subs += [OrderedDict(v_subs)]

        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        return [Inc(field.subs(vsub),
                    field.subs(vsub) + expr.subs(subs).subs(vsub) * b.subs(subs))
                for b, vsub in zip(self.coefficients, idx_subs)]


class SparseTimeFunction(SparseFunction):
    """
    A time-dependent :class:`SparseFunction`.

    :param name: Name of the function.
    :param nt: Size of the time dimension for point data.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoints,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: (Optional) Discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) Discretisation order for time derivatives.
                       Default to 1.
    :param dtype: (Optional) Data type of the buffered data.
    :param initializer: (Optional) A callable to initialize the data

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseTimeFunction = True

    _time_position = 0
    """Position of time index among the function indices."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SparseTimeFunction, self).__init__(*args, **kwargs)

            nt = kwargs.get('nt')
            if not isinstance(nt, int) and nt > 0:
                raise ValueError('SparseTimeFunction requires int parameter `nt`')
            self.nt = nt

            self.time_dim = self.indices[self._time_position]
            self.time_order = kwargs.get('time_order', 1)
            if not isinstance(self.time_order, int):
                raise ValueError("'time_order' must be int")

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


class PrecomputedSparseFunction(AbstractSparseFunction):
    """
    A specialised type of SparseFunction where the interpolation is externally defined.
    Currently, this means that the grid points and associated coefficients for each
    sparse point is precomputed at the time this object is being created.

    :param name: Name of the function.
    :param nt: Size of the time dimension for point data.
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
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoints,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param space_order: (Optional) Discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) Discretisation order for time derivatives.
                       Default to 1.
    :param dtype: (Optional) Data type of the buffered data.
    :param initializer: (Optional) A callable to initialize the data

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """
    is_PrecomputedSparseFunction = True
    _child_functions = ['gridpoints', 'coefficients']

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PrecomputedSparseFunction, self).__init__(*args, **kwargs)

            # Grid points per sparse point (2 in the case of bilinear and trilinear)
            r = kwargs.get('r')
            if not isinstance(r, int) and r > 0:
                raise ValueError('Interpolation requires parameter `r` (>0)')
            self.r = r

            gridpoints = Function(name="%s_gridpoints" % self.name, dtype=np.int32,
                                  dimensions=(self.indices[-1], Dimension(name='d')),
                                  shape=(self.npoint, self.grid.dim), space_order=0)

            gridpoints_data = kwargs.get('gridpoints', None)
            assert(gridpoints_data is not None)
            gridpoints.data[:] = gridpoints_data[:]
            self.gridpoints = gridpoints

            coefficients = Function(name="%s_coefficients" % self.name, dtype=self.dtype,
                                    dimensions=(self.indices[-1], Dimension(name='d'),
                                                Dimension(name='i')),
                                    shape=(self.npoint, self.grid.dim, self.r),
                                    space_order=0)
            coefficients_data = kwargs.get('coefficients', None)
            assert(coefficients_data is not None)
            coefficients.data[:] = coefficients_data[:]
            self.coefficients = coefficients
            warning("Ensure that the provided coefficient and grid point values are " +
                    "computed on the final grid that will be used for other " +
                    "computations.")

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, cummulative=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param cummulative: (Optional) If True, perform an increment rather
                            than an assignment. Defaults to False.
        """
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs(t, u_t).subs(time, u_t)

        coefficients = self.coefficients.indexed
        gridpoints = self.gridpoints.indexed
        p, _, _ = self.coefficients.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + gridpoints[p, i])))
            coeffs.append(coefficients[p, i, rd])
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

        gridpoints = self.gridpoints.indexed
        coefficients = self.coefficients.indexed

        p, _ = self.gridpoints.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + gridpoints[p, i])))
            coeffs.append(coefficients[p, i, rd])
        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        return [Eq(field, field + rhs.subs(dim_subs))]
