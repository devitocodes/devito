from __future__ import absolute_import
import weakref
import abc
import gc

from collections import namedtuple
from operator import mul
from functools import reduce

import numpy as np
import sympy

from devito.parameters import configuration
from devito.tools import EnrichedTuple, single_or

__all__ = ['Symbol', 'Indexed']

configuration.add('first_touch', 0, [0, 1], lambda i: bool(i))

# This cache stores a reference to each created data object
# so that we may re-create equivalent symbols during symbolic
# manipulation with the correct shapes, pointers, etc.
_SymbolCache = {}


class Basic(object):
    """
    Base class for API objects, mainly used to build equations.

    There are three relevant sub-types of :class:`Basic`: ::

        * AbstractSymbol: represents a scalar; may carry data; may be used
                          to build equations.
        * AbstractFunction: represents a discrete function as a tensor; may
                            carry data; may be used to build equations.
        * Object: represents a generic object, for example a (pointer to) data
                  structure.

                                        Basic
                                          |
                    ------------------------------------------
                    |                     |                  |
             AbstractSymbol       AbstractFunction        Object

    .. note::

        The :class:`AbstractFunction` sub-hierarchy is mainly implemented in
        :mod:`function.py`.
    """

    # Top hierarchy
    is_AbstractFunction = False
    is_AbstractSymbol = False
    is_Object = False

    # Symbolic objects created internally by Devito
    is_Symbol = False
    is_Array = False

    # Created by the user
    is_Input = False
    # Scalar symbolic objects created by the user
    is_Dimension = False
    is_Constant = False
    # Tensor symbolic objects created by the user
    is_TensorFunction = False
    is_Function = False
    is_TimeFunction = False
    is_SparseTimeFunction = False
    is_SparseFunction = False
    is_PrecomputedSparseFunction = False

    # Basic symbolic object properties
    is_Scalar = False
    is_Tensor = False

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abc.abstractmethod
    def _arg_defaults(self):
        """
        Returns a map of default argument values defined by this symbol.
        """
        raise NotImplementedError('%s does not provide any default arguments' %
                                  self.__class__)

    @abc.abstractmethod
    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        raise NotImplementedError('%s does not provide argument value derivation' %
                                  self.__class__)


class Cached(object):
    """
    Base class for symbolic objects that caches on the class type.

    In order to maintain meta information across the numerous
    re-instantiation SymPy performs during symbolic manipulation, we inject
    the symbol name as the class name and cache all created objects on that
    name. This entails that a symbolic object inheriting from :class:`Cached`
    should implement `__init__` in the following way:

        .. code-block::
            def __init__(self, \*args, \*\*kwargs):
                if not self._cached():
                    ... # Initialise object properties from kwargs
    """

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


class AbstractSymbol(sympy.Symbol, Basic):
    """
    Base class for dimension-free symbols, only cached by SymPy.

    The sub-hierarchy is structured as follows

                             AbstractSymbol
                                   |
                 -------------------------------------
                 |                                   |
        AbstractCachedSymbol                      Dimension
                 |
        --------------------
        |                  |
     Symbol            Constant
        |
     Scalar

    There are three relevant :class:`AbstractSymbol` sub-types: ::

        * Symbol: A generic scalar symbol that can be used to build an equation.
                  It does not carry data. Typically, :class:`Symbol`s are
                  created internally by Devito (e.g., for temporary variables)
        * Constant: A generic scalar symbol that can be used to build an equation.
                    It carries data (a scalar value).
        * Dimension: A problem dimension, used to create an iteration space. It
                     may be used to build equations; typically, it is used as
                     an index for a :class:`Indexed`.
    """

    is_AbstractSymbol = True

    @property
    def indices(self):
        return ()

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def symbolic_shape(self):
        return ()

    @property
    def function(self):
        return self

    def indexify(self):
        return self


class AbstractCachedSymbol(AbstractSymbol, Cached):
    """
    Base class for dimension-free symbols, cached by both Devito and Sympy.

    For more information, refer to the documentation of :class:`AbstractSymbol`.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})
        if cls in _SymbolCache:
            newobj = sympy.Symbol.__new__(cls, *args, **options)
            newobj._cached_init()
        else:
            name = kwargs.get('name')

            # Create the new Function object and invoke __init__
            newcls = cls._symbol_type(name)
            newobj = sympy.Symbol.__new__(newcls, name, *args, **options)

            # Initialization
            newobj.__init__(*args, **kwargs)

            # Store new instance in symbol cache
            newcls._cache_put(newobj)
        return newobj


class Symbol(AbstractCachedSymbol):

    """A :class:`sympy.Symbol` capable of mimicking an :class:`sympy.Indexed`"""

    is_Symbol = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.dtype = kwargs.get('dtype', np.int32)

    @property
    def base(self):
        return self


class Scalar(Symbol):
    """Symbolic object representing a scalar.

    :param name: Name of the symbol
    :param dtype: Data type of the scalar
    """

    is_Scalar = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.dtype = kwargs.get('dtype', np.float32)

    @property
    def _mem_stack(self):
        """Return True if the associated data should be allocated on the stack
        in a C module, False otherwise."""
        return True

    def update(self, dtype=None, **kwargs):
        self.dtype = dtype or self.dtype


class AbstractFunction(sympy.Function, Basic):
    """
    Base class for tensor symbols, only cached by SymPy. It inherits from and
    mimick the behaviour of a :class:`sympy.Function`.

    The sub-hierarchy is structured as follows

                          AbstractFunction
                                 |
                       AbstractCachedFunction
                                 |
               -------------------------------------
               |                                   |
             Array                          TensorFunction
                                                   |
                                     ------------------------------
                                     |                            |
                                  Function                  SparseFunction
                                     |                            |
                                TimeFunction              SparseTimeFunction

    There are five relevant :class:`AbstractFunction` sub-types: ::

        * Array: A function that does not carry data. Usually created by the DSE.
        * Function: A space-varying discrete function, which carries user data.
        * TimeFunction: A time- and space-varying discrete function, which carries
                        user data.
        * SparseFunction: A space-varying discrete function representing "sparse"
                          points, i.e. points that are not aligned with the
                          computational grid.
        * SparseTimeFunction: A time- and space-varying function representing "sparse"
                          points, i.e. points that are not aligned with the
                          computational grid.
    """

    is_AbstractFunction = True


class AbstractCachedFunction(AbstractFunction, Cached):
    """
    Base class for tensor symbols, cached by both Devito and Sympy.

    For more information, refer to the documentation of :class:`AbstractFunction`.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})
        if cls in _SymbolCache:
            newobj = sympy.Function.__new__(cls, *args, **options)
            newobj._cached_init()
        else:
            name = kwargs.get('name')
            indices = cls.__indices_setup__(**kwargs)

            # Create the new Function object and invoke __init__
            newcls = cls._symbol_type(name)
            newobj = sympy.Function.__new__(newcls, *indices, **options)

            # Initialization
            newobj._name = name
            newobj._indices = indices
            newobj._shape = cls.__shape_setup__(**kwargs)
            newobj.__init__(*args, **kwargs)

            # All objects cached on the AbstractFunction /newobj/ keep a reference
            # to /newobj/ through the /function/ field. Thus, all indexified
            # object will point to /newobj/, the "actual Function".
            newobj.function = newobj

            # Store new instance in symbol cache
            newcls._cache_put(newobj)
        return newobj

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """Extract the function indices from ``kwargs``."""
        return ()

    @classmethod
    def __shape_setup__(cls, **kwargs):
        """Extract the function shape from ``kwargs``."""
        return ()

    @property
    def name(self):
        """Return the name of the function."""
        return self._name

    @property
    def indices(self):
        """Return the indices (aka dimensions) of the function."""
        return self._indices

    @property
    def dimensions(self):
        """Tuple of :class:`Dimension`s representing the function indices."""
        return self.indices

    @property
    def shape(self):
        """Return the shape of the function."""
        return self._shape

    @property
    def ndim(self):
        """Return the rank of the function."""
        return len(self.indices)

    @property
    def symbolic_shape(self):
        """
        Return the symbolic shape of the object. This includes the padding,
        halo, and domain regions. While halo and padding are known quantities
        (integers), the domain size is represented by a symbol.
        """
        halo_sizes = [sympy.Add(*i, evaluate=False) for i in self._extent_halo]
        padding_sizes = [sympy.Add(*i, evaluate=False) for i in self._extent_padding]
        domain_sizes = [i.symbolic_size for i in self.indices]
        return tuple(sympy.Add(i, j, k, evaluate=False)
                     for i, j, k in zip(domain_sizes, halo_sizes, padding_sizes))

    def indexify(self, indices=None):
        """Create a :class:`sympy.Indexed` object from the current object."""
        if indices is not None:
            return Indexed(self.indexed, *indices)

        subs = dict([(i.spacing, 1) for i in self.indices])
        indices = [a.subs(subs) for a in self.args]
        return Indexed(self.indexed, *indices)

    @property
    def indexed(self):
        """Extract a :class:`IndexedData` object from the current object."""
        return IndexedData(self.name, shape=self.shape, function=self.function)

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

    @property
    def size(self):
        """Return the number of elements this function is expected to store in memory.
           Note that this would need to be combined with self.dtype to give the actual
           size in bytes
        """
        return reduce(mul, self.shape)

    @property
    def _offset_domain(self):
        """
        The number of grid points between the first (last) allocated element
        (possibly in the halo/padding region) and the first (last) domain element,
        for each dimension.
        """
        left = tuple(np.add(self._extent_halo.left, self._extent_padding.left))
        right = tuple(np.add(self._extent_halo.right, self._extent_padding.right))

        Offset = namedtuple('Offset', 'left right')
        offsets = tuple(Offset(i, j) for i, j in np.add(self._halo, self._padding))

        return EnrichedTuple(*offsets, getters=self.dimensions, left=left, right=right)

    @property
    def _offset_halo(self):
        """
        The number of grid points between the first (last) allocated element
        (possibly in the halo/padding region) and the first (last) halo element,
        for each dimension.
        """
        left = self._extent_padding.left
        right = self._extent_padding.right

        Offset = namedtuple('Offset', 'left right')
        offsets = tuple(Offset(i, j) for i, j in self._padding)

        return EnrichedTuple(*offsets, getters=self.dimensions, left=left, right=right)

    @property
    def _extent_halo(self):
        """
        The number of grid points in the halo region.
        """
        left = tuple(zip(*self._halo))[0]
        right = tuple(zip(*self._halo))[1]

        Extent = namedtuple('Extent', 'left right')
        extents = tuple(Extent(i, j) for i, j in self._halo)

        return EnrichedTuple(*extents, getters=self.dimensions, left=left, right=right)

    @property
    def _extent_padding(self):
        """
        The number of grid points in the padding region.
        """
        left = tuple(zip(*self._padding))[0]
        right = tuple(zip(*self._padding))[1]

        Extent = namedtuple('Extent', 'left right')
        extents = tuple(Extent(i, j) for i, j in self._padding)

        return EnrichedTuple(*extents, getters=self.dimensions, left=left, right=right)

    @property
    def _mask_domain(self):
        """A mask to access the domain region of the allocated data."""
        return tuple(slice(i, -j) if j != 0 else slice(i, None)
                     for i, j in self._offset_domain)

    @property
    def _mask_with_halo(self):
        """A mask to access the domain+halo region of the allocated data."""
        return tuple(slice(i, -j) if j != 0 else slice(i, None)
                     for i, j in self._offset_halo)


class Array(AbstractCachedFunction):
    """A symbolic function object, created and managed directly by Devito..

    :param name: Name of the object.
    :param dtype: Data type of the object.
    :param shape: The shape of the object.
    :param dimensions: The symbolic dimensions of the object.
    :param halo: The halo region of the object, expressed as an iterable
                 ``[(dim1_left_halo, dim1_right_halo), (dim2_left_halo, ...)]``
    :param padding: The padding region of the object, expressed as an iterable
                    ``[(dim1_left_pad, dim1_right_pad), (dim2_left_pad, ...)]``
    :param external: Pass True if there is no need to allocate storage
    :param onstack: Pass True to enforce allocation on the stack
    :param onheap: Pass True to enforce allocation on the heap
    """

    is_Array = True
    is_Tensor = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractCachedFunction.__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.dtype = kwargs.get('dtype', np.float32)

            self._halo = kwargs.get('halo', tuple((0, 0) for i in range(self.ndim)))
            self._padding = kwargs.get('padding', tuple((0, 0) for i in range(self.ndim)))

            self._external = bool(kwargs.get('external', False))
            self._onstack = bool(kwargs.get('onstack', False))
            self._onheap = bool(kwargs.get('onheap', True))

            # The memory scope of an Array must be well-defined
            assert single_or([self._external, self._onstack, self._onheap])

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return tuple(kwargs.get('dimensions'))

    @classmethod
    def __shape_setup__(cls, **kwargs):
        return tuple(kwargs.get('shape'))

    @property
    def _mem_external(self):
        return self._external

    @property
    def _mem_stack(self):
        return self._onstack

    @property
    def _mem_heap(self):
        return self._onheap

    def update(self, dtype=None, shape=None, dimensions=None, halo=None, padding=None,
               onstack=None, onheap=None, external=None):
        self.dtype = dtype or self.dtype
        self._shape = shape or self.shape
        self._indices = dimensions or self.indices
        self._halo = halo or self._halo
        self._padding = padding or self._padding

        if any(i is not None for i in [external, onstack, onheap]):
            self._external = bool(external)
            self._onstack = bool(onstack)
            self._onheap = bool(onheap)
            assert single_or([self._external, self._onstack, self._onheap])


# Objects belonging to the Devito API not involving data, such as data structures
# that need to be passed to external libraries


class Object(Basic):

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

    def _arg_defaults(self):
        if callable(self.value):
            return {self.name: self.value()}
        else:
            return {self.name: self.value}

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            return {self.name: kwargs.pop(self.name)}
        else:
            return {}


# Extended SymPy hierarchy follows, for essentially two reasons:
# - To keep track of `function`
# - To override SymPy caching behaviour


class IndexedData(sympy.IndexedBase):
    """Wrapper class that inserts a pointer to the symbolic data object"""

    def __new__(cls, label, shape=None, function=None):
        obj = sympy.IndexedBase.__new__(cls, label, shape)
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

    @property
    def function(self):
        return self.base.function

    @property
    def dtype(self):
        return self.function.dtype


# Utilities


class CacheManager(object):

    """
    Drop unreferenced objects from the SymPy and Devito caches. The associated
    data is lost (and thus memory is freed).
    """

    @classmethod
    def clear(cls):
        sympy.cache.clear_cache()
        gc.collect()
        for key, val in list(_SymbolCache.items()):
            if val() is None:
                del _SymbolCache[key]
