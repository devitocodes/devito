import weakref

import numpy as np
from sympy import Function, IndexedBase, Symbol, as_finite_diff, symbols
from sympy.abc import h, s

from devito.dimension import t, x, y, z, time
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative)
from devito.logger import debug, error, warning
from devito.memory import CMemory, first_touch
from devito.arguments import (SymbolicDataArgProvider, ScalarFunctionArgProvider,
                              TensorFunctionArgProvider)

__all__ = ['DenseData', 'TimeData', 'Forward', 'Backward']


# This cache stores a reference to each created data object
# so that we may re-create equivalent symbols during symbolic
# manipulation with the correct shapes, pointers, etc.
_SymbolCache = {}


class TimeAxis(object):
    """Direction in which to advance the time index on
    :class:`TimeData` objects.

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


class AbstractSymbol(Function, CachedSymbol):
    """Base class for data classes that provides symbolic behaviour.

    :param name: Symbolic name to give to the resulting function. Must
                 be given as keyword argument.
    :param shape: Shape of the underlying object. Must be given as
                  keyword argument.

    This class implements the behaviour of Devito's symbolic
    objects by inheriting from and mimicking the behaviour of :class
    sympy.Function:. In order to maintain meta information across the
    numerous re-instantiation SymPy performs during symbolic
    manipulation, we inject the symbol name as the class name and
    cache all created objects on that name. This entails that a symbolic
    object should implement `__init__` in the following format:

    def __init__(self, \*args, \*\*kwargs):
        if not self._cached():
            ... # Initialise object properties from kwargs

    Note: The parameters :param name: and :param shape: must always be
    present and given as keyword arguments, since SymPy uses `*args`
    to (re-)create the dimension arguments of the symbolic function.
    """

    is_AbstractSymbol = True
    is_SymbolicFunction = False
    is_SymbolicData = False
    is_ScalarFunction = False
    is_TensorFunction = False
    is_DenseData = False
    is_TimeData = False
    is_CompositeData = False
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

            # All objects cached on the AbstractSymbol /newobj/ keep a reference
            # to /newobj/ through the /function/ field. Thus, all indexified
            # object will point to /newobj/, the "actual Function".
            newobj.function = newobj

            # Store new instance in symbol cache
            newcls._cache_put(newobj)
        return newobj

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices."""
        return []

    @property
    def dim(self):
        """Return the rank of the object."""
        return len(self.shape)

    @property
    def indexed(self):
        """Extract a :class:`IndexedData` object from the current object."""
        return IndexedData(self.name, shape=self.shape, function=self.function)

    @property
    def symbolic_shape(self):
        """
        Return the symbolic shape of the object. For an entry ``E`` in ``self.shape``,
        there are two possibilities: ::

            * ``E`` is already in symbolic form, then simply use ``E``.
            * ``E`` is an integer representing the size along a dimension ``D``,
              then, use a symbolic representation of ``D``.
        """
        sshape = []
        for i, j in zip(self.shape, self.indices):
            try:
                i.is_algebraic_expr()
                sshape.append(i)
            except AttributeError:
                sshape.append(j.symbolic_size)
        return tuple(sshape)

    @property
    def _mem_external(self):
        """Return True if the associated data was/is/will be allocated directly
        from Python (e.g., via NumPy arrays), False otherwise."""
        return False

    @property
    def _mem_stack(self):
        """Return True if the associated data was/is/will be allocated on the stack
        in a C module, False otherwise."""
        return False

    @property
    def _mem_heap(self):
        """Return True if the associated data was/is/will be allocated on the heap
        in a C module, False otherwise."""
        return False

    def indexify(self):
        """Create a :class:`sympy.Indexed` object from the current object."""
        indices = [a.subs({h: 1, s: 1}) for a in self.args]
        if indices:
            return self.indexed[indices]
        else:
            return EmptyIndexed(self.indexed)


class SymbolicFunction(AbstractSymbol):

    """
    A symbolic function object, created and managed directly by Devito.

    Unlike :class:`SymbolicData` objects, the state of a SymbolicFunction
    is mutable.
    """

    is_SymbolicFunction = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractSymbol.__new__(cls, *args, **kwargs)

    def update(self):
        return


class ScalarFunction(SymbolicFunction, ScalarFunctionArgProvider):
    """Symbolic object representing a scalar.

    :param name: Name of the symbol
    :param dtype: Data type of the scalar
    """

    is_ScalarFunction = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.shape = ()
            self.indices = ()
            self.dtype = kwargs.get('dtype', np.float32)

    @property
    def _mem_stack(self):
        """Return True if the associated data should be allocated on the stack
        in a C module, False otherwise."""
        return True

    def update(self, dtype=None):
        self.dtype = dtype or self.dtype


class TensorFunction(SymbolicFunction, TensorFunctionArgProvider):
    """Symbolic object representing a tensor.

    :param name: Name of the symbol
    :param dtype: Data type of the scalar
    :param shape: The shape of the tensor
    :param dimensions: The symbolic dimensions of the tensor.
    :param onstack: Pass True to enforce allocation on stack
    """

    is_TensorFunction = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.name = kwargs.get('name')
            self.shape = kwargs.get('shape')
            self.indices = kwargs.get('dimensions')
            self.dtype = kwargs.get('dtype', np.float32)
            self._onstack = kwargs.get('onstack', False)

    @classmethod
    def _indices(cls, **kwargs):
        return kwargs.get('dimensions')

    @property
    def _mem_stack(self):
        return self._onstack

    @property
    def _mem_heap(self):
        return not self._onstack

    def update(self, dtype=None, shape=None, dimensions=None, onstack=None):
        self.dtype = dtype or self.dtype
        self.shape = shape or self.shape
        self.indices = dimensions or self.indices
        self._onstack = onstack or self._mem_stack


class SymbolicData(AbstractSymbol, SymbolicDataArgProvider):
    """A symbolic object associated with data."""

    is_SymbolicData = True

    @property
    def _mem_external(self):
        return True


class DenseData(SymbolicData):
    """Data object for spatially varying data acting as a :class:`SymbolicData`.

    :param name: Name of the symbol
    :param dtype: Data type of the scalar
    :param shape: The shape of the tensor
    :param dimensions: The symbolic dimensions of the tensor.
    :param space_order: Discretisation order for space derivatives
    :param initializer: Function to initialize the data, optional

    .. note::

       :class:`DenseData` objects are assumed to be constant in time
       and therefore do not support time derivatives. Use
       :class:`TimeData` for time-varying grid data.
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
            self.initializer = kwargs.get('initializer', None)
            if self.initializer is not None:
                assert(callable(self.initializer))
            self.numa = kwargs.get('numa', False)
            self._data_object = None

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

    def _allocate_memory(self):
        """Allocate memory in terms of numpy ndarrays."""
        debug("Allocating memory for %s (%s)" % (self.name, str(self.shape)))
        self._data_object = CMemory(self.shape, dtype=self.dtype)
        if self.numa:
            first_touch(self)
        else:
            self.data.fill(0)

    @property
    def data(self):
        """Reference to the :class:`numpy.ndarray` containing the data

        :returns: The ndarray containing the data
        """
        if self._data_object is None:
            self._allocate_memory()

        return self._data_object.ndpointer

    def initialize(self):
        """Apply the data initilisation function, if it is not None."""
        if self.initializer is not None:
            self.initializer(self.data)

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
    """
    Data object for time-varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param shape: Shape of the spatial data grid
    :param dimensions: The symbolic dimensions of the function in addition
                       to time.
    :param dtype: Data type of the buffered data
    :param save: Save the intermediate results to the data buffer. Defaults
                 to `False`, indicating the use of alternating buffers.
    :param time_dim: Size of the time dimension that dictates the leading
                     dimension of the data buffer if :param save: is True.
    :param time_order: Order of the time discretization which affects the
                       final size of the leading time dimension of the
                       data buffer.

    .. note::

       The parameter ``shape`` should only define the spatial shape of
       the grid. The temporal dimension will be inserted automatically
       as the leading dimension, according to the ``time_dim``,
       ``time_order`` and whether we want to write intermediate
       timesteps in the buffer. The same is true for explicitly
       provided dimensions, which will be added to the automatically
       derived time dimensions symbol. For example:

       .. code-block:: python

          In []: TimeData(name="a", dimensions=(x, y, z))
          Out[]: a(t, x, y, z)

          In []: TimeData(name="a", shape=(20, 30))
          Out[]: a(t, x, y)

    """

    is_TimeData = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TimeData, self).__init__(*args, **kwargs)
            time_dim = kwargs.get('time_dim', None)
            self.time_order = kwargs.get('time_order', 1)
            self.save = kwargs.get('save', False)

            if not self.save:
                if time_dim is not None:
                    warning('Explicit time dimension size (time_dim) found for '
                            'TimeData symbol %s, despite \nusing a buffered time '
                            'dimension (save=False). This value will be ignored!'
                            % self.name)
                time_dim = self.time_order + 1
                self.indices[0].modulo = time_dim
            else:
                if time_dim is None:
                    error('Time dimension (time_dim) is required'
                          'to save intermediate data with save=True')
                    raise ValueError("Unknown time dimensions")
            self.shape = (time_dim,) + self.shape

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
        _indices = DenseData._indices(**kwargs)
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

        return as_finite_diff(self.diff(_t), indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        _t = self.indices[0]
        width_t = int(self.time_order / 2)
        indt = [(_t + i * s) for i in range(-width_t, width_t + 1)]

        return as_finite_diff(self.diff(_t, _t), indt)


class IndexedData(IndexedBase):
    """Wrapper class that inserts a pointer to the symbolic data object"""

    def __new__(cls, label, shape=None, function=None):
        obj = IndexedBase.__new__(cls, label, shape)
        obj.function = function
        return obj

    def func(self, *args):
        obj = super(IndexedData, self).func(*args)
        obj.function = self.function
        return obj


class EmptyIndexed(Symbol):

    """A :class:`sympy.Symbol` capable of mimicking an :class:`sympy.Indexed`"""

    def __new__(cls, base):
        obj = Symbol.__new__(cls, base.label.name)
        obj.base = base
        obj.indices = ()
        obj.function = base.function
        return obj

    def func(self, *args):
        return super(EmptyIndexed, self).func(self.base.func(*self.base.args))


class CompositeData(DenseData):
    """
    Base class for DenseData classes that have DenseData children
    """

    is_CompositeData = True

    def __init__(self, *args, **kwargs):
        super(CompositeData, self).__init__(self, *args, **kwargs)
        self._children = []

    @property
    def children(self):
        return self._children
