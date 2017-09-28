import weakref
import abc

import numpy as np
import sympy
from sympy import Function, IndexedBase, as_finite_diff
from sympy.abc import s

from devito.dimension import t, x, y, z, time, Dimension
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative)
from devito.logger import debug, error, warning
from devito.memory import CMemory, first_touch
from devito.arguments import (ConstantDataArgProvider, TensorDataArgProvider,
                              ScalarFunctionArgProvider, TensorFunctionArgProvider,
                              ObjectArgProvider)
from devito.parameters import configuration

__all__ = ['Symbol', 'Indexed',
           'ConstantData', 'DenseData', 'TimeData',
           'Forward', 'Backward']

configuration.add('first_touch', 0, [0, 1], lambda i: bool(i))

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


class Basic(object):
    """
    Base class for API objects, used to build and run :class:`Operator`s.

    There are two main types of objects: symbolic and generic. Symbolic objects
    may carry data, and are used to build equations. Generic objects may be
    used to represent or pass arbitrary data structures. The following diagram
    outlines the top of this hierarchy.

                                 Basic
                                   |
                    ----------------------------------
                    |                                |
              CachedSymbol                         Object
                    |                       <see Object.__doc__>
             AbstractSymbol
    <see diagram in AbstractSymbol.__doc__>

    All derived :class:`Basic` objects may be emitted through code generation
    to create a just-in-time compilable kernel.
    """

    # Top hierarchy
    is_AbstractSymbol = False
    is_Object = False

    # Symbolic objects created internally by Devito
    is_SymbolicFunction = False
    is_ScalarFunction = False
    is_TensorFunction = False

    # Symbolic objects created by user
    is_SymbolicData = False
    is_ConstantData = False
    is_TensorData = False
    is_DenseData = False
    is_TimeData = False
    is_CompositeData = False
    is_PointData = False

    # Basic symbolic object properties
    is_Scalar = False
    is_Tensor = False

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return


class CachedSymbol(Basic):
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

    This class is the root of the Devito data objects hierarchy, which
    is structured as follows.

                             AbstractSymbol
                                   |
                 -------------------------------------
                 |                                   |
          SymbolicFunction                      SymbolicData
                 |                                   |
          ------------------                 ------------------
          |                |                 |                |
    ScalarFunction  TensorFunction     ConstantData           |
                                                              |
                                                ----------------------------
                                                |             |            |
                                            DenseData      TimeData  CompositeData
                                                                           |
                                                                       PointData

    The key difference between a :class:`SymbolicData` and a :class:`SymbolicFunction`
    is that the former is created directly by the user and employed in some
    computation, while the latter is created and managed internally by Devito.
    """

    is_AbstractSymbol = True

    def __new__(cls, *args, **kwargs):
        if cls in _SymbolCache:
            options = kwargs.get('options', {})
            newobj = Function.__new__(cls, *args, **options)
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

    def indexify(self, indices=None):
        """Create a :class:`sympy.Indexed` object from the current object."""
        if indices is not None:
            return Indexed(self.indexed, *indices)

        subs = dict([(i.spacing, 1) for i in self.indices])
        indices = [a.subs(subs) for a in self.args]
        if indices:
            return Indexed(self.indexed, *indices)
        else:
            return Symbol(self.indexed)


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
    is_Scalar = True

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

    def update(self, dtype=None, **kwargs):
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
    is_Tensor = True

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


class SymbolicData(AbstractSymbol):
    """A symbolic object associated with data.

    Unlike :class:`SymbolicFunction` objects, the structure of a SymbolicData
    is immutable (e.g., shape, dtype, ...). Obviously, the object value (``data``)
    may be altered, either directly by the user or by an :class:`Operator`.
    """

    is_SymbolicData = True

    @property
    def _data_buffer(self):
        """Reference to the actual data. This is *not* a view of the data.
        This method is for internal use only."""
        return self.data

    @abc.abstractproperty
    def data(self):
        """The value of the data object."""
        return


class ConstantData(SymbolicData, ConstantDataArgProvider):

    """
    Data object for constant values.
    """

    is_ConstantData = True
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


class TensorData(SymbolicData, TensorDataArgProvider):

    is_TensorData = True
    is_Tensor = True

    @property
    def _mem_external(self):
        """Return True if the associated data was/is/will be allocated directly
        from Python (e.g., via NumPy arrays), False otherwise."""
        return True


class DenseData(TensorData):
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
            self._first_touch = kwargs.get('first_touch', configuration['first_touch'])
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
            _indices = (x, y, z)
            shape = kwargs.get('shape')
            if len(shape) <= 3:
                dimensions = _indices[:len(shape)]
            else:
                dimensions = [Dimension("x%d" % i) for i in range(1, len(shape) + 1)]
        return dimensions

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
        indx = [(x + i * x.spacing) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(x, x), indx)

    @property
    def dy2(self):
        """Symbol for the second derivative wrt the y dimension"""
        width_h = int(self.space_order/2)
        indy = [(y + i * y.spacing) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(y, y), indy)

    @property
    def dz2(self):
        """Symbol for the second derivative wrt the z dimension"""
        width_h = int(self.space_order/2)
        indz = [(z + i * z.spacing) for i in range(-width_h, width_h + 1)]

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
        indx = [(x + i * x.spacing) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(x, x, x, x), indx)

    @property
    def dy4(self):
        """Symbol for the fourth derivative wrt the y dimension"""
        width_h = max(int(self.space_order / 2), 2)
        indy = [(y + i * y.spacing) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(y, y, y, y), indy)

    @property
    def dz4(self):
        """Symbol for the fourth derivative wrt the z dimension"""
        width_h = max(int(self.space_order / 2), 2)
        indz = [(z + i * z.spacing) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(z, z, z, z), indz)

    @property
    def laplace(self):
        """Symbol for the second derivative wrt all spatial dimensions"""
        derivs = ['dx2', 'dy2', 'dz2']

        return sum([getattr(self, d) for d in derivs[:self.dim]])

    def laplace2(self, weight=1):
        """Symbol for the double laplacian wrt all spatial dimensions"""
        order = self.space_order/2
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


# Objects belonging to the Devito API not involving data, such as data structures
# that need to be passed to external libraries


class Object(ObjectArgProvider):

    """
    Represent a generic pointer object.
    """

    is_Object = True

    def __init__(self, name, dtype, value=None):
        self.name = name
        self.dtype = dtype
        self.value = value

    def __repr__(self):
        return self.name


# Extended SymPy hierarchy follows, for essentially two reasons:
# - To keep track of `function`
# - To override SymPy caching behaviour


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

    def __getitem__(self, indices, **kwargs):
        """
        Return :class:`Indexed`, rather than :class:`sympy.Indexed`.
        """
        indexed = super(IndexedData, self).__getitem__(indices, **kwargs)
        return Indexed(*indexed.args)


class Symbol(sympy.Symbol):

    """A :class:`sympy.Symbol` capable of mimicking an :class:`sympy.Indexed`"""

    def __new__(cls, base):
        obj = sympy.Symbol.__new__(cls, base.label.name)
        obj.base = base
        obj.indices = ()
        obj.function = base.function
        return obj

    def func(self, *args):
        return super(Symbol, self).func(self.base.func(*self.base.args))


class Indexed(sympy.Indexed):

    # The two type flags have changed in upstream sympy as of version 1.1,
    # but the below interpretation is used throughout the DSE and DLE to
    # identify Indexed objects. With the sympy-1.1 changes a new flag
    # obj.is_Indexed was introduced which should be preferred, but the
    # required changes are cumbersome and many...
    is_Symbol = False
    is_Atom = False

    def _hashable_content(self):
        return super(Indexed, self)._hashable_content() + (self.base.function,)
