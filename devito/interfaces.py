import weakref

import numpy as np
from sympy import Function, IndexedBase, as_finite_diff, symbols
from sympy.abc import h, s

from devito.dimension import d, p, t, x, y, z
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative)
from devito.logger import debug, error
from devito.memmap_manager import MemmapManager
from devito.memory import CMemory, first_touch

__all__ = ['DenseData', 'TimeData', 'PointData']


# This cache stores a reference to each created data object
# so that we may re-create equivalent symbols during symbolic
# manipulation with the correct shapes, pointers, etc.
_SymbolCache = {}


class CachedSymbol(object):
    """Base class for symbolic objects that caches on the class type."""

    @classmethod
    def _cached(cls):
        """Test if current class is already in the symbol cache."""
        return cls in _SymbolCache

    @classmethod
    def _cache_put(cls, obj):
        """Store given object instance in symbol cache.

        :param obj: Object to be cached.
        """
        _SymbolCache[cls] = weakref.ref(obj)

    @classmethod
    def _symbol_type(cls, name):
        """Create new type instance from cls and inject symbol name"""
        return type(name, (cls, ), dict(cls.__dict__))

    def _cached_init(self):
        """Initialise symbolic object with a cached object state"""
        original = _SymbolCache[self.__class__]
        self.__dict__ = original().__dict__


class SymbolicData(Function, CachedSymbol):
    """Base class for data classes that provides symbolic behaviour.

    :param name: Symbolic name to give to the resulting function. Must
                 be given as keyword argument.
    :param shape: Shape of the underlying spatial data. Must be given
                  as keyword argument.

    This class implements the symbolic behaviour of Devito's data
    objects by inheriting from and mimicking the behaviour of :class
    sympy.Function:. In order to maintain meta information across the
    numerous re-instantiation SymPy performs during symbolic
    manipulation, we inject the symbol name as the class name and
    cache all created objects on that name. This entails that data
    object should implement `__init__` in the following format:

    def __init__(self, \*args, \*\*kwargs):
        if not self._cached():
            ... # Initialise object properties from kwargs

    Note: The parameters :param name: and :param shape: must always be
    present and given as keyword arguments, since SymPy uses `*args`
    to (re-)create the dimension arguments of the symbolic function.
    """

    is_SymbolicData = True
    is_ScalarData = False
    is_TensorData = False
    is_DenseData = False
    is_TimeData = False
    is_Coordinates = False
    is_PointData = False

    def __new__(cls, *args, **kwargs):
        if cls in _SymbolCache:
            newobj = Function.__new__(cls, *args)
            newobj._cached_init()
        else:
            name = kwargs.get('name')
            if len(args) < 1:
                args = cls._indices(**kwargs)

            # Create the new Function object and invoke __init__
            newcls = cls._symbol_type(name)
            options = kwargs.get('options', {})
            newobj = Function.__new__(newcls, *args, **options)
            newobj.__init__(*args, **kwargs)
            # Store new instance in symbol cache
            newcls._cache_put(newobj)
        return newobj

    @classmethod
    def _indices(cls, **kwargs):
        """Abstract class method to determine the default dimension indices.

        :param shape: Given shape of the data.
        :raises NotImplementedError: 'Abstract class `SymbolicData` does not have
        default indices'.
        """
        raise NotImplementedError('Abstract class'
                                  ' `SymbolicData` does not have default indices')


class ScalarData(SymbolicData):
    """Data object representing a scalar.

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param dtype: Data type of the scalar
    :param initializer: Function to initialize the data, optional
    """

    is_ScalarData = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return SymbolicData.__new__(cls, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        return []

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.dtype = kwargs.get('dtype', np.float32)
            self._data = kwargs.get('_data', None)


class TensorData(SymbolicData):
    """Data object representing a tensor."""

    is_TensorData = True


class DenseData(TensorData):
    """Data object for spatially varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param shape: Shape of the spatial data grid
    :param dtype: Data type of the buffered data
    :param space_order: Discretisation order for space derivatives
    :param initializer: Function to initialize the data, optional

    Note: :class:`DenseData` objects are assumed to be constant in time and
    therefore do not support time derivatives. Use :class:`TimeData` for
    time-varying grid data.
    """

    is_DenseData = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.shape = kwargs.get('shape', None)
            if self.shape is None:
                dimensions = kwargs.get('dimensions')
                self.shape = tuple([d.size for d in dimensions])
            self.indices = self._indices(**kwargs)
            self.dtype = kwargs.get('dtype', np.float32)
            self.space_order = kwargs.get('space_order', 1)
            initializer = kwargs.get('initializer', None)
            if initializer is not None:
                assert(callable(initializer))
            self.initializer = initializer
            self._data = kwargs.get('_data', None)
            MemmapManager.setup(self, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape


        :param dimensions: Optional, list of :class:`Dimension`
                           objects that defines data layout.
        :param shape: Optional, shape of the spatial data to
                      automatically infer dimension symbols.
        :return: Dimension indices used for each axis.
        """
        dimensions = kwargs.get('dimensions', None)
        if dimensions is None:
            # Infer dimensions from default and data shape
            if 'shape' not in kwargs:
                error("Creating symbolic data objects requries either"
                      "a 'shape' or 'dimensions' argument")
                raise ValueError("Unknown symbol dimensions or shape")
            _indices = [x, y, z]
            shape = kwargs.get('shape')
            if len(shape) <= 3:
                dimensions = _indices[:len(shape)]
            else:
                dimensions = [symbols("x%d" % i) for i in range(1, len(shape) + 1)]
        return dimensions

    @property
    def dim(self):
        """Returns the spatial dimension of the data object"""
        return len(self.shape)

    @property
    def indexed(self):
        """:return: Base symbol as devito.IndexedData"""
        return IndexedData(self.name, shape=self.shape, function=self)

    def indexify(self):
        """Convert base symbol and dimensions to indexed data accesses

        :return: Index corrosponding to the indices
        """
        indices = [a.subs({h: 1, s: 1}) for a in self.args]
        return self.indexed[indices]

    def _allocate_memory(self):
        """Function to allocate memmory in terms of numpy ndarrays.

        Note: memmap is a subclass of ndarray.
        """
        if self.memmap:
            self._data = np.memmap(filename=self.f, dtype=self.dtype, mode='w+',
                                   shape=self.shape, order='C')
        else:
            debug("Allocating memory for %s (%s)" % (self.name, str(self.shape)))
            self._data_object = CMemory(self.shape, dtype=self.dtype)
            self._data = self._data_object.ndpointer
            first_touch(self)

    @property
    def data(self):
        """Reference to the :class:`numpy.ndarray` containing the data

        :returns: The ndarray containing the data
        """
        if self._data is None:
            self._allocate_memory()

        return self._data

    def initialize(self):
        """Apply the data initilisation function, if it is not None."""
        if self.initializer is not None:
            self.initializer(self.data)
        # Ignore if no initializer exists - assume no initialisation necessary

    @property
    def dx(self):
        """Symbol for the first derivative wrt the x dimension"""
        return first_derivative(self, order=self.space_order, dim=x, side=centered)

    @property
    def dy(self):
        """Symbol for the first derivative wrt the y dimension"""
        return first_derivative(self, order=self.space_order, dim=y, side=centered)

    @property
    def dz(self):
        """Symbol for the first derivative wrt the z dimension"""
        return first_derivative(self, order=self.space_order, dim=z, side=centered)

    @property
    def dxy(self):
        """Symbol for the cross derivative wrt the x and y dimension"""
        return cross_derivative(self, order=self.space_order, dims=(x, y))

    @property
    def dxz(self):
        """Symbol for the cross derivative wrt the x and z dimension"""
        return cross_derivative(self, order=self.space_order, dims=(x, z))

    @property
    def dyz(self):
        """Symbol for the cross derivative wrt the y and z dimension"""
        return cross_derivative(self, order=self.space_order, dims=(y, z))

    @property
    def dxl(self):
        """Symbol for the derivative wrt to x with a left stencil"""
        return first_derivative(self, order=self.space_order, dim=x, side=left)

    @property
    def dxr(self):
        """Symbol for the derivative wrt to x with a right stencil"""
        return first_derivative(self, order=self.space_order, dim=x, side=right)

    @property
    def dyl(self):
        """Symbol for the derivative wrt to y with a left stencil"""
        return first_derivative(self, order=self.space_order, dim=y, side=left)

    @property
    def dyr(self):
        """Symbol for the derivative wrt to y with a right stencil"""
        return first_derivative(self, order=self.space_order, dim=y, side=right)

    @property
    def dzl(self):
        """Symbol for the derivative wrt to z with a left stencil"""
        return first_derivative(self, order=self.space_order, dim=z, side=left)

    @property
    def dzr(self):
        """Symbol for the derivative wrt to z with a right stencil"""
        return first_derivative(self, order=self.space_order, dim=z, side=right)

    @property
    def dx2(self):
        """Symbol for the second derivative wrt the x dimension"""
        width_h = int(self.space_order/2)
        indx = [(x + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(x, x), indx)

    @property
    def dy2(self):
        """Symbol for the second derivative wrt the y dimension"""
        width_h = int(self.space_order/2)
        indy = [(y + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(y, y), indy)

    @property
    def dz2(self):
        """Symbol for the second derivative wrt the z dimension"""
        width_h = int(self.space_order/2)
        indz = [(z + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(z, z), indz)

    @property
    def dx2y2(self):
        """Symbol for the second cross derivative wrt the x,y dimension"""
        return second_derivative(self.dx2, dim=y, order=self.space_order)

    @property
    def dx2z2(self):
        """Symbol for the second cross derivative wrt the x,z dimension"""
        return second_derivative(self.dx2, dim=z, order=self.space_order)

    @property
    def dy2z2(self):
        """Symbol for the second cross derivative wrt the y,z dimension"""
        return second_derivative(self.dy2, dim=z, order=self.space_order)

    @property
    def dx4(self):
        """Symbol for the fourth derivative wrt the x dimension"""
        width_h = max(int(self.space_order / 2), 2)
        indx = [(x + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(x, x, x, x), indx)

    @property
    def dy4(self):
        """Symbol for the fourth derivative wrt the y dimension"""
        width_h = max(int(self.space_order / 2), 2)
        indy = [(y + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(y, y, y, y), indy)

    @property
    def dz4(self):
        """Symbol for the fourth derivative wrt the z dimension"""
        width_h = max(int(self.space_order / 2), 2)
        indz = [(z + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(z, z, z, z), indz)

    @property
    def laplace(self):
        """Symbol for the second derivative wrt all spatial dimensions"""
        derivs = ['dx2', 'dy2', 'dz2']

        return sum([getattr(self, d) for d in derivs[:self.dim]])

    def laplace2(self, weight=1):
        """Symbol for the double laplacian wrt all spatial dimensions"""
        order = self.space_order/2 + self.space_order/2 % 2
        first = sum([second_derivative(self, dim=d,
                                       order=order)
                     for d in self.indices[1:]])
        second = sum([second_derivative(first * weight, dim=d,
                                        order=order)
                      for d in self.indices[1:]])
        return second


class TimeData(DenseData):
    """Data object for time-varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param shape: Shape of the spatial data grid
    :param dtype: Data type of the buffered data
    :param save: Save the intermediate results to the data buffer. Defaults
                 to `False`, indicating the use of alternating buffers.
    :param pad_time: Set to `True` if save is True and you want to initialize
                     the first :obj:`time_order` timesteps.
    :param time_dim: Size of the time dimension that dictates the leading
                     dimension of the data buffer if :param save: is True.
    :param time_order: Order of the time discretization which affects the
                       final size of the leading time dimension of the
                       data buffer.

    Note: The parameter :shape: should only define the spatial shape of the
    grid. The temporal dimension will be inserted automatically as the
    leading dimension, according to the :param time_dim:, :param time_order:
    and whether we want to write intermediate timesteps in the buffer.
    """

    is_TimeData = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TimeData, self).__init__(*args, **kwargs)
            self._full_data = self._data.view() if self._data else None
            time_dim = kwargs.get('time_dim')
            self.time_order = kwargs.get('time_order', 1)
            self.save = kwargs.get('save', False)
            self.pad_time = kwargs.get('pad_time', False)

            if self.save:
                time_dim += self.time_order
            else:
                time_dim = self.time_order + 1
                self.indices[0].modulo = time_dim

            self.shape = (time_dim,) + self.shape

    def initialize(self):
        if self.initializer is not None:
            if self._full_data is None:
                self._allocate_memory()
            self.initializer(self._full_data)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param dimensions: Optional, list of :class:`Dimension`
                           objects that defines data layout.
        :param shape: Optional, shape of the spatial data to
                      automatically infer dimension symbols.
        :return: Dimension indices used for each axis.
        """
        dimensions = kwargs.get('dimensions', None)
        if dimensions is None:
            # Infer dimensions from default and data shape
            _indices = [t, x, y, z]
            shape = kwargs.get('shape')
            dimensions = _indices[:len(shape) + 1]
        return dimensions

    def _allocate_memory(self):
        """function to allocate memmory in terms of numpy ndarrays."""
        super(TimeData, self)._allocate_memory()

        self._full_data = self._data.view()

        if self.pad_time:
            self._data = self._data[self.time_order:, :, :]

    @property
    def dim(self):
        """Returns the spatial dimension of the data object"""
        return len(self.shape[1:])

    @property
    def forward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1

        return self.subs(t, t + i * s)

    @property
    def backward(self):
        """Symbol for the time-backward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1

        return self.subs(t, t - i * s)

    @property
    def dt(self):
        """Symbol for the first derivative wrt the time dimension"""
        if self.time_order == 1:
            # This hack is needed for the first-order diffusion test
            indices = [t, t + s]
        else:
            width = int(self.time_order / 2)
            indices = [(t + i * s) for i in range(-width, width + 1)]

        return as_finite_diff(self.diff(t), indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        width_t = int(self.time_order/2)
        indt = [(t + i * s) for i in range(-width_t, width_t + 1)]

        return as_finite_diff(self.diff(t, t), indt)


class CoordinateData(TensorData):
    """
    Data object for sparse coordinate data that acts as a Function symbol
    """

    is_Coordinates = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.ndim = kwargs.get('ndim')
            self.npoint = kwargs.get('npoint')
            self.shape = (self.npoint, self.ndim)
            self.indices = self._indices(**kwargs)
            self.dtype = kwargs.get('dtype', np.float32)
            self._data_object = CMemory(self.shape, dtype=self.dtype)
            self.data = self._data_object.ndpointer
            first_touch(self)

    def __new__(cls, *args, **kwargs):
        ndim = kwargs.get('ndim')
        npoint = kwargs.get('npoint')
        kwargs['shape'] = (npoint, ndim)
        return SymbolicData.__new__(cls, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: indices used for axis.
        """
        dimensions = kwargs.get('dimensions', None)
        return dimensions or [p, d]

    @property
    def indexed(self):
        """:return: Base symbol as devito.IndexedData"""
        return IndexedData(self.name, shape=self.shape, function=self)


class PointData(DenseData):
    """
    Data object for sparse point data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param npoint: Number of points to sample
    :param coordinates: Coordinates data for the sparse points
    :param nt: Size of the time dimension for point data
    :param dtype: Data type of the buffered data

    Note: This class is expected to eventually evolve into a
    full-fledged sparse data container. For now, the naming and
    symbolic behaviour follows the use in the current problem.
    """

    is_PointData = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.nt = kwargs.get('nt')
            self.npoint = kwargs.get('npoint')
            ndim = kwargs.get('ndim')
            kwargs['shape'] = (self.nt, self.npoint)
            super(PointData, self).__init__(self, *args, **kwargs)
            coordinates = kwargs.get('coordinates')
            self.coordinates = CoordinateData(name='%s_coords' % self.name,
                                              dimensions=[self.indices[1], d],
                                              data=coordinates, ndim=ndim,
                                              nt=self.nt, npoint=self.npoint)
            self.coordinates.data[:] = kwargs.get('coordinates')[:]

    def __new__(cls, *args, **kwargs):
        nt = kwargs.get('nt')
        npoint = kwargs.get('npoint')
        kwargs['shape'] = (nt, npoint)

        return DenseData.__new__(cls, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: indices used for axis.
        """
        dimensions = kwargs.get('dimensions', None)
        return dimensions or [t, p]


class IndexedData(IndexedBase):
    """Wrapper class that inserts a pointer to the symbolic data object"""

    def __new__(cls, label, shape=None, function=None, **kw_args):
        obj = IndexedBase.__new__(cls, label, shape)
        obj.function = function
        return obj

    def func(self, *args):
        obj = super(IndexedData, self).func(*args)
        obj.function = self.function
        return obj
