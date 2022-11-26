import abc
from collections import namedtuple
from ctypes import POINTER, _Pointer
from functools import reduce
from operator import mul

import numpy as np
import sympy
from sympy.core.assumptions import _assume_rules
from cached_property import cached_property

from devito.data import default_allocator
from devito.tools import (Pickable, as_tuple, ctypes_to_cstr, dtype_to_ctype,
                          frozendict, memoized_meth, sympy_mutex)
from devito.types.args import ArgProvider
from devito.types.caching import Cached, Uncached
from devito.types.lazy import Evaluable
from devito.types.utils import DimensionTuple

__all__ = ['Symbol', 'Scalar', 'Indexed', 'IndexedData', 'DeviceMap']


Size = namedtuple('Size', 'left right')
Offset = namedtuple('Offset', 'left right')


class CodeSymbol(object):

    """
    Abstract base class for objects representing symbols in the generated code.

    The _C_* properties describe the object in C-land. For example its name and
    its type.

    The _mem_* properties describe the object memory allocation strategy. There
    are three axes, with a few possible values each:

        * "liveness": `_mem_external`, `_mem_internal_eager`, `_mem_internal_lazy`
        * "space": `_mem_local`, `_mem_mapped`, `_mem_host`
        * "scope": `_mem_stack`, `_mem_heap`, `_mem_constant`, `_mem_shared`

    For example, an object that is `<_mem_internal_lazy, _mem_local, _mem_heap>`
    is allocated within the Operator entry point, on either the host or device
    memory (but not both), and on the heap. Refer to the __doc__ of the single
    _mem_* properties for more info. Obviously, not all triplets make sense
    for a given architecture.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @property
    @abc.abstractmethod
    def dtype(self):
        """
        The data type of the object in the generated code, represented as a
        Python class:

            * `numpy.dtype`: basic data types. For example, `np.float64 -> double`.
            * `ctypes`: composite objects (e.g., structs), foreign types.
        """
        return

    @abc.abstractproperty
    def _C_name(self):
        """
        The name of the object in the generated code.

        Returns
        -------
        str
        """
        return

    @property
    def _C_typedata(self):
        """
        The type of the object in the generated code as a `str`.
        """
        _type = self._C_ctype
        while issubclass(_type, _Pointer):
            _type = _type._type_
        return ctypes_to_cstr(_type)

    @abc.abstractproperty
    def _C_ctype(self):
        """
        The type of the object in the generated code as a `ctypes` class.
        """
        return

    @property
    def _C_symbol(self):
        """
        The entry symbol. This may or may not coincide with the symbol used
        to construct symbolic expressions.

        Returns
        -------
        Basic
        """
        return self

    @property
    def _mem_external(self):
        """
        True if the associated data is allocated and freed in Python, False otherwise.
        """
        return False

    @property
    def _mem_internal_eager(self):
        """
        True if the associated data is allocated and freed inside the first
        Callable in which the symbol appears as a free variable.
        """
        return False

    @property
    def _mem_internal_lazy(self):
        """
        True if the associated data is allocated and freed at the level of
        the Operator entry point.
        """
        return False

    @property
    def _mem_local(self):
        """
        True if the associated data is allocated in the underlying platform's
        local memory space, False otherwise.

        The local memory space is:

            * the host DRAM if platform=CPU
            * the device DRAM if platform=GPU
        """
        return False

    @property
    def _mem_mapped(self):
        """
        True if the associated data is allocated in the underlying platform's
        local memory space and subsequently mapped to the underlying platform's
        remote memory space, False otherwise.

        The local memory space is:

            * the host DRAM if platform=CPU
            * the device DRAM if platform=GPU

        The remote memory space is:

            * the host DRAM if platform=GPU
            * the device DRAM if platform=CPU
        """
        return False

    @property
    def _mem_host(self):
        """
        True if the associated data is systematically allocated in the host DRAM.
        """
        return False

    @property
    def _mem_stack(self):
        """
        True if the associated data should be allocated on the stack, False otherwise.
        """
        return False

    @property
    def _mem_heap(self):
        """
        True if the associated data gets allocated on the heap, False otherwise.
        """
        return False

    @property
    def _mem_constant(self):
        """
        True if the associated data gets allocated in global constant memory,
        False otherwise.
        """
        return False

    @property
    def _mem_shared(self):
        """
        True if the associated data gets allocated in so called shared memory,
        False otherwise.
        """
        return False


class Basic(CodeSymbol):

    """
    Abstract base class for objects to construct symbolic expressions.

    Four relevant types inherit from this class:

        * AbstractSymbol: represents a scalar; may carry data; may be used
                          to build equations.
        * AbstractFunction: represents a discrete R^n -> R function; may
                            carry data; may be used to build equations.
        * AbstractTensor: represents a discrete 2nd order tensor or vector:
                          R^n -> R^(nd x nd) tensor (nd dimensions),
                          R^n -> R^nd vector (nd dimensions),
                          may carry data; may be used to build equations.
        * AbstractObject: represents a generic object, for example a (pointer
                          to) data structure.

                                            Basic
                                              |
              --------------------------------------------------------------
              |                     |                  |                   |
        AbstractSymbol      AbstractFunction      AbstractTensor      AbstractObject

    All these subtypes must implement a number of methods/properties to enable
    code generation via the Devito compiler. These methods/properties are
    easily recognizable as their name starts with _C_.

    Notes
    -----
    The AbstractFunction sub-hierarchy is implemented in :mod:`dense.py`.
    The AbstractTensor sub-hierarchy is implemented in :mod:`tensor.py`.
    """

    # Top hierarchy
    is_AbstractFunction = False
    is_AbstractObject = False

    # Symbolic objects created internally by Devito
    is_Symbol = False
    is_ArrayBasic = False
    is_Array = False
    is_PointerArray = False
    is_ObjectArray = False
    is_Bundle = False
    is_Object = False
    is_LocalObject = False

    # Created by the user
    is_Input = False

    # Scalar symbolic objects created by the user
    is_Dimension = False
    is_Constant = False

    # Tensor symbolic objects created by the user
    is_DiscreteFunction = False
    is_Function = False
    is_TimeFunction = False
    is_TempFunction = False
    is_SparseTimeFunction = False
    is_SparseFunction = False

    # Time dependence
    is_TimeDependent = False

    # Some other properties
    is_PerfKnob = False  # Does it impact the Operator performance?

    @property
    def bound_symbols(self):
        """
        Unlike SymPy, we systematically define `bound_symbols` on all of
        the API and internal objects that may be used to construct an
        Operator.
        """
        return set()


class AbstractSymbol(sympy.Symbol, Basic, Pickable, Evaluable):

    """
    Base class for scalar symbols.

    The hierarchy is structured as follows

                             AbstractSymbol
                                   |
                 -------------------------------------
                 |                                   |
             DataSymbol                            Symbol
                 |                                   |
         ----------------                   -------------------
         |              |                   |                 |
      Constant   DefaultDimension         Scalar          Dimension
                                                    <:mod:`dimension.py`>

    All symbols can be used to build equations. However, while DataSymbol
    carries data, Symbol is a pure symbolic object.

    Constant, DefaultDimension, and Dimension (and most of its subclasses) are
    part of the user API; Scalar, instead, is only used internally by Devito.

    DefaultDimension and Dimension define a problem dimension (in other words,
    an "iteration space"). They can be used to index into Functions. For more
    information, refer to :mod:`dimension.py`.
    """

    is_AbstractSymbol = True
    is_Symbol = True

    # SymPy default assumptions
    is_real = True
    is_imaginary = False
    is_commutative = True

    __rkwargs__ = ('name', 'dtype', 'is_const')

    @classmethod
    def _filter_assumptions(cls, **kwargs):
        """Extract sympy.Symbol-specific kwargs."""
        assumptions = {}
        # pop predefined assumptions
        for key in ('real', 'imaginary', 'commutative'):
            kwargs.pop(key, None)
        # extract sympy.Symbol-specific kwargs
        for i in list(kwargs):
            if i in _assume_rules.defined_facts:
                assumptions[i] = kwargs.pop(i)
        return assumptions, kwargs

    def __new__(cls, *args, **kwargs):
        name = kwargs.get('name') or args[0]
        assumptions, kwargs = cls._filter_assumptions(**kwargs)

        # Create the new Symbol
        # Note: use __xnew__ to bypass sympy caching
        newobj = sympy.Symbol.__xnew__(cls, name, **assumptions)

        # Initialization
        newobj._dtype = cls.__dtype_setup__(**kwargs)
        newobj.__init_finalize__(*args, **kwargs)

        return newobj

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        """Extract the object data type from ``kwargs``."""
        return kwargs.get('dtype', np.int32)

    def __init__(self, *args, **kwargs):
        # no-op, the true init is performed by __init_finalize__
        pass

    def __init_finalize__(self, *args, **kwargs):
        self._is_const = kwargs.get('is_const', False)

    @property
    def dtype(self):
        return self._dtype

    @property
    def indices(self):
        return ()

    @property
    def dimensions(self):
        return self.indices

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
    def base(self):
        return self

    @property
    def function(self):
        return self

    def _evaluate(self, **kwargs):
        return self

    def indexify(self, indices=None):
        return self

    @property
    def is_const(self):
        """
        True if the symbol value cannot be modified within an Operator (and thus
        its value is provided by the user directly from Python-land), False otherwise.
        """
        return self._is_const

    @property
    def _C_name(self):
        return self.name

    @property
    def _C_ctype(self):
        return dtype_to_ctype(self.dtype)

    def _subs(self, old, new, **hints):
        """
        This stub allows sympy.Basic.subs to operate on an expression
        involving devito Scalars.  Ordinarily the comparisons between
        devito subclasses of sympy types are quite strict.
        """
        try:
            if old.name == self.name:
                return new
        except AttributeError:
            pass

        return self

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__

    def __getnewargs_ex__(self):
        args, kwargs = Pickable.__getnewargs_ex__(self)
        kwargs.update(self.assumptions0)
        return args, kwargs


class Symbol(AbstractSymbol, Cached):

    """
    A scalar symbol, cached by both Devito and SymPy, which does not carry
    any data.

    Notes
    -----
    A Symbol may not be in the SymPy cache, but still be present in the
    Devito cache. This is because SymPy caches operations, rather than
    actual objects.
    """

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        args = list(args)
        key = {}

        # The base type is necessary, otherwise two objects such as
        # `Scalar(name='s')` and `Dimension(name='s')` would have the same key
        key['cls'] = cls

        # The name is always present, and added as if it were an arg
        key['name'] = kwargs.pop('name', None) or args.pop(0)

        # From the args
        key['args'] = tuple(args)

        # From the kwargs
        key.update(kwargs)

        return frozendict(key)

    def __new__(cls, *args, **kwargs):
        assumptions, kwargs = cls._filter_assumptions(**kwargs)
        key = cls._cache_key(*args, **{**assumptions, **kwargs})
        obj = cls._cache_get(key)

        if obj is not None:
            return obj

        # Not in cache. Create a new Symbol via sympy.Symbol
        args = list(args)
        name = kwargs.pop('name', None) or args.pop(0)

        # Note: use __xnew__ to bypass sympy caching
        newobj = sympy.Symbol.__xnew__(cls, name, **assumptions)

        # Initialization
        newobj._dtype = cls.__dtype_setup__(**kwargs)
        newobj.__init_finalize__(name, *args, **kwargs)

        # Store new instance in symbol cache
        Cached.__init__(newobj, key)

        return newobj

    __hash__ = Cached.__hash__


class DataSymbol(AbstractSymbol, Uncached):

    """
    A unique scalar symbol that carries data.
    """

    def __new__(cls, *args, **kwargs):
        # Create a new Symbol via sympy.Symbol
        name = kwargs.get('name') or args[0]
        assumptions, kwargs = cls._filter_assumptions(**kwargs)

        # Note: use __xnew__ to bypass sympy caching
        newobj = sympy.Symbol.__xnew__(cls, name, **assumptions)

        # Initialization
        newobj._dtype = cls.__dtype_setup__(**kwargs)
        newobj.__init_finalize__(*args, **kwargs)

        return newobj

    __hash__ = Uncached.__hash__


class Scalar(Symbol, ArgProvider):

    """
    Like a Symbol, but in addition it can pass runtime values to an Operator.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    is_const : bool, optional
        True if the symbol value cannot be modified within an Operator,
        False otherwise. Defaults to False.
    **assumptions
        Any SymPy assumptions, such as ``nonnegative=True``. Refer to the
        SymPy documentation for more information.
    """

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def default_value(self):
        return None

    @property
    def _arg_names(self):
        return (self.name,)

    def _arg_defaults(self, **kwargs):
        if self.default_value is None:
            # It is possible that the Scalar value is provided indirectly
            # through a wrapper object (e.g., a Dimension spacing `h_x` gets its
            # value via a Grid object)
            return {}
        else:
            return {self.name: self.default_value}

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            return {self.name: kwargs.pop(self.name)}
        else:
            return self._arg_defaults()


class AbstractTensor(sympy.ImmutableDenseMatrix, Basic, Pickable, Evaluable):

    """
    Base class for vector and tensor valued functions. It inherits from and
    mimicks the behavior of a sympy.ImmutableDenseMatrix.


    The sub-hierachy is as follows

                         AbstractTensor
                                |
                          TensorFunction
                                |
                 ---------------------------------
                 |                               |
          VectorFunction                 TensorTimeFunction
                        \-------\                |
                                 \------- VectorTimeFunction

    There are four relevant AbstractTensor sub-types: ::

        * TensorFunction: A space-varying tensor valued function.
        * VectorFunction: A space-varying vector valued function.
        * TensorTimeFunction: A time-space-varying tensor valued function.
        * VectorTimeFunction: A time-space-varying vector valued function.
    """

    # SymPy attributes
    is_MatrixLike = True
    is_Matrix = True

    # Devito attributes
    is_AbstractTensor = True
    is_TensorValued = True
    is_VectorValued = False

    @classmethod
    def _new(cls, *args, **kwargs):
        if args:
            try:
                # Constructor if input is (rows, cols, lambda)
                newobj = super(AbstractTensor, cls)._new(*args)
            except ValueError:
                # Constructor if input is list of list as (row, cols, list_of_list)
                # doesn't work as it expects a flattened.
                newobj = super(AbstractTensor, cls)._new(args[2])

            # Filter grid and dimensions
            grid, dimensions = newobj._infer_dims()
            if grid is None and dimensions is None:
                return sympy.ImmutableDenseMatrix(*args)
            # Initialized with constructed object
            newobj.__init_finalize__(newobj.rows, newobj.cols, newobj.flat(),
                                     grid=grid, dimensions=dimensions)
        else:
            # Initialize components and create new Matrix from standard
            # Devito inputs
            comps = cls.__subfunc_setup__(*args, **kwargs)
            newobj = super(AbstractTensor, cls)._new(comps)
            newobj.__init_finalize__(*args, **kwargs)

        return newobj

    @classmethod
    def _fromrep(cls, rep):
        """
        This the new constructor mechanism for matrices in sympy 1.9.
        Standard new object go through `_new` but arithmetic operations directly use
        the representation based one.
        This class method is only accessible from an existing AbstractTensor
        that contains a grid or dimensions.
        """
        newobj = super(AbstractTensor, cls)._fromrep(rep)
        grid, dimensions = newobj._infer_dims()
        try:
            # This is needed when `_fromrep` is called directly in 1.9
            # for example with mul.
            newobj.__init_finalize__(newobj.rows, newobj.cols, newobj.flat(),
                                     grid=grid, dimensions=dimensions)
        except TypeError:
            # We can end up here when `_fromrep` is called through the default _new
            # when input `comps` don't have grid or dimensions. For example
            # `test_non_devito_tens` in `test_tensor.py`.
            pass
        return newobj

    def _infer_dims(self):
        grids = {getattr(c, 'grid', None) for c in self.flat()} - {None}
        dimensions = {d for c in self.flat()
                      for d in getattr(c, 'dimensions', ())} - {None}
        # If none of the components are devito objects, returns a sympy Matrix
        if len(grids) == 0 and len(dimensions) == 0:
            return None, None
        elif len(grids) > 0:
            dimensions = None
            assert len(grids) == 1
            grid = grids.pop()
        else:
            grid = None
            dimensions = tuple(dimensions)

        return grid, dimensions

    def flat(self):
        try:
            return super().flat()
        except AttributeError:
            return self._mat

    def __init_finalize__(self, *args, **kwargs):
        pass

    __hash__ = sympy.ImmutableDenseMatrix.__hash__

    def doit(self, **hint):
        return self

    def _eval_matrix_mul(self, other):
        """
        Copy paste from sympy to avoid explicit call to sympy.Add
        TODO: fix inside sympy
        """
        other_len = other.rows*other.cols
        new_len = self.rows*other.cols
        new_mat = [self.zero]*new_len

        # If we multiply an n x 0 with a 0 x m, the
        # expected behavior is to produce an n x m matrix of zeros
        if self.cols != 0 and other.rows != 0:
            self_cols = self.cols
            mat = self.flat()
            try:
                other_mat = other.flat()
            except AttributeError:
                other_mat = other._mat
            for i in range(new_len):
                row, col = i // other.cols, i % other.cols
                row_indices = range(self_cols*row, self_cols*(row+1))
                col_indices = range(col, other_len, other.cols)
                vec = [mat[a]*other_mat[b] for a, b in zip(row_indices, col_indices)]
                new_mat[i] = sum(vec)

        # Get new class and return product
        newcls = self.classof_prod(other, new_mat)
        return newcls._new(self.rows, other.cols, new_mat, copy=False)

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """Setup each component of the tensor as a Devito type."""
        return []


class AbstractFunction(sympy.Function, Basic, Cached, Pickable, Evaluable):

    """
    Base class for tensor symbols, cached by both SymPy and Devito. It inherits
    from and mimicks the behaviour of a sympy.Function.

    The hierarchy is structured as follows

                         AbstractFunction
                                |
                 ---------------------------------
                 |                               |
         DiscreteFunction                      Array
                 |
         ----------------------------------------
         |                                      |
         |                           AbstractSparseFunction
         |                                      |
         |               -----------------------------------------------------
         |               |                      |                            |
      Function     SparseFunction   AbstractSparseTimeFunction  PrecomputedSparseFunction
         |               |                      |                            |
         |               |   ------------------------------------     --------
         |               |   |                                  |     |
    TimeFunction  SparseTimeFunction                 PrecomputedSparseTimeFunction

    There are five relevant AbstractFunction sub-types: ::

        * Array: A function that does not carry data.
        * Function: A space-varying discrete function, which carries user data.
        * TimeFunction: A time- and space-varying discrete function, which carries
                        user data.
        * SparseFunction: A space-varying discrete function representing "sparse"
                          points, i.e. points that are not aligned with the
                          computational grid.
        * SparseTimeFunction: A time- and space-varying function representing "sparse"
                          points, i.e. points that are not aligned with the
                          computational grid.
        * PrecomputedSparseFunction: A SparseFunction that uses a custom interpolation
                                     scheme, instead of linear interpolators.
        * PrecomputedSparseTimeFunction: A SparseTimeFunction that uses a custom
                                         interpolation scheme, instead of linear
                                         interpolators.

    """

    # SymPy attributes, explicitly say these are not Matrices
    is_MatrixLike = False
    is_Matrix = False

    is_AbstractFunction = True

    # SymPy default assumptions
    is_real = True
    is_imaginary = False
    is_commutative = True

    # Devito default assumptions
    is_regular = True
    """
    True if data and iteration points are aligned. Cases where they won't be
    aligned (currently unsupported): Functions defined on SubDomains; compressed
    Functions; etc.
    """

    is_compact = True
    """
    True if data is allocated as a single, contiguous chunk of memory.
    """

    __rkwargs__ = ('name', 'dtype', 'halo', 'padding', 'alias')

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        return cls, args

    @classmethod
    def _cache_get(cls, *args, **kwargs):
        # Is the object already in cache (e.g., f(x), f(x+1)) ?
        key = cls._cache_key(*args, **kwargs)
        obj = super()._cache_get(key)
        if obj is not None:
            return obj

        # Does the base object exist at least (e.g. f(x))?
        obj = super()._cache_get(cls)
        if obj is not None:
            options = kwargs.get('options', {'evaluate': False})
            with sympy_mutex:
                newobj = sympy.Function.__new__(cls, *args, **options)
            newobj.__init_cached__(obj)
            Cached.__init__(newobj, key)
            return newobj

        # Not in cache
        return None

    def __new__(cls, *args, **kwargs):
        # Only perform a construction if the object isn't in cache
        obj = cls._cache_get(*args, **kwargs)
        if obj is not None:
            return obj

        options = kwargs.get('options', {'evaluate': False})

        # Preprocess arguments
        args, kwargs = cls.__args_setup__(*args, **kwargs)

        # Not in cache. Create a new Function via sympy.Function
        name = kwargs.get('name')
        dimensions, indices = cls.__indices_setup__(**kwargs)

        # Create new, unique type instance from cls and the symbol name
        newcls = type(name, (cls,), dict(cls.__dict__))

        # Create the new Function object and invoke __init__
        with sympy_mutex:
            newobj = sympy.Function.__new__(newcls, *indices, **options)

        # Initialization. The following attributes must be available
        # when executing __init_finalize__
        newobj._name = name
        newobj._dimensions = dimensions
        newobj._shape = cls.__shape_setup__(**kwargs)
        newobj._dtype = cls.__dtype_setup__(**kwargs)
        newobj.__init_finalize__(*args, **kwargs)

        # All objects cached on the AbstractFunction `newobj` keep a reference
        # to `newobj` through the `function` field. Thus, all indexified
        # object will point to `newobj`, the "actual Function".
        newobj.function = newobj

        # Store new instance in symbol cache
        key = (newcls, indices)
        Cached.__init__(newobj, key, newcls)

        return newobj

    def __init__(self, *args, **kwargs):
        # no-op, the true init is performed by __init_finalize__
        pass

    def __init_finalize__(self, *args, **kwargs):
        # Setup halo and padding regions
        self._is_halo_dirty = False
        self._halo = self.__halo_setup__(**kwargs)
        self._padding = self.__padding_setup__(**kwargs)

        # Symbol properties
        # "Aliasing" another DiscreteFunction means that `self` logically
        # represents another object. For example, `self` might be used as the
        # formal parameter of a routine generated by the compiler, where the
        # routines is applied to several actual DiscreteFunctions
        self._alias = kwargs.get('alias', False)

    __hash__ = Cached.__hash__

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        """
        Preprocess *args and **kwargs before object initialization.

        Notes
        -----
        This stub is invoked only if a look up in the cache fails.
        """
        return args, kwargs

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """Extract the object indices from ``kwargs``."""
        return (), ()

    @classmethod
    def __shape_setup__(cls, **kwargs):
        """Extract the object shape from ``kwargs``."""
        return ()

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        """Extract the object data type from ``kwargs``."""
        return None

    def __halo_setup__(self, **kwargs):
        halo = tuple(kwargs.get('halo', [(0, 0) for i in range(self.ndim)]))
        return DimensionTuple(*halo, getters=self.dimensions)

    def __padding_setup__(self, **kwargs):
        padding = tuple(kwargs.get('padding', [(0, 0) for i in range(self.ndim)]))
        return DimensionTuple(*padding, getters=self.dimensions)

    @cached_property
    def _honors_autopadding(self):
        """
        True if the actual padding is greater or equal than whatever autopadding
        would produce, False otherwise.
        """
        autopadding = self.__padding_setup__(autopadding=True)
        return all(l0 >= l1 and r0 >= r1
                   for (l0, r0), (l1, r1) in zip(self.padding, autopadding))

    @property
    def name(self):
        """The name of the object."""
        return self._name

    @property
    def indices(self):
        """The indices of the object."""
        return DimensionTuple(*self.args, getters=self.dimensions)

    @property
    def indices_ref(self):
        """The reference indices of the object (indices at first creation)."""
        return DimensionTuple(*self.function.indices, getters=self.dimensions)

    @property
    def origin(self):
        """
        Origin of the AbstractFunction in term of Dimension
        f(x) : origin = 0
        f(x + hx/2) : origin = hx/2
        """
        return tuple(r - d for d, r in zip(self.dimensions, self.indices_ref))

    @property
    def dimensions(self):
        """Tuple of Dimensions representing the object indices."""
        return self._dimensions

    @property
    def _eval_deriv(self):
        return self

    @property
    def _is_on_grid(self):
        """
        Check whether the object is on the grid and requires averaging.
        For example, if the original non-staggered function is f(x)
        then f(x) is on the grid and f(x + h_x/2) is off the grid.
        """
        for i, j, d in zip(self.indices, self.indices_ref, self.dimensions):
            # Two indices are aligned if they differ by an Integer*spacing.
            v = i - j
            try:
                if int(v/d.spacing) != v/d.spacing:
                    return False
            except TypeError:
                return False
        return True

    def _evaluate(self, **kwargs):
        # Average values if at a location not on the Function's grid
        if self._is_on_grid:
            return self

        weight = 1.0
        avg_list = [self]
        is_averaged = False
        for i, ir, d in zip(self.indices, self.indices_ref, self.dimensions):
            off = (i - ir)/d.spacing
            if not isinstance(off, sympy.Number) or int(off) == off:
                pass
            else:
                weight *= 1/2
                is_averaged = True
                avg_list = [(a.xreplace({i: i - d.spacing/2}) +
                             a.xreplace({i: i + d.spacing/2})) for a in avg_list]

        if not is_averaged:
            return self
        return weight * sum(avg_list)

    @property
    def shape(self):
        """The shape of the object."""
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        """The rank of the object."""
        return len(self.indices)

    @property
    def symbolic_shape(self):
        """
        The symbolic shape of the object. This includes the domain, halo, and
        padding regions. While halo and padding are known quantities (integers),
        the domain size is given as a symbol.
        """
        halo = [sympy.Add(*i, evaluate=False) for i in self._size_halo]
        padding = [sympy.Add(*i, evaluate=False) for i in self._size_padding]
        domain = [i.symbolic_size for i in self.dimensions]
        ret = tuple(sympy.Add(i, j, k)
                    for i, j, k in zip(domain, halo, padding))
        return DimensionTuple(*ret, getters=self.dimensions)

    @property
    def symbolic_size(self):
        return np.prod(self.symbolic_shape)

    @property
    def symbolic_nbytes(self):
        # TODO: one day we'll have to fix the types/ vs extended_sympy/ thing
        from devito.symbolics import SizeOf  # noqa
        return self.symbolic_size*SizeOf(self.indexed._C_typedata)

    @cached_property
    def bound_symbols(self):
        return ({self._C_symbol, self.indexed} |
                set().union(*[d.bound_symbols for d in self.dimensions]))

    @cached_property
    def indexed(self):
        """The wrapped IndexedData object."""
        return IndexedData(self.name, shape=self._shape, function=self.function)

    @cached_property
    def dmap(self):
        """
        A symbolic pointer to the device data map. If there's no such a map,
        return None.
        """
        if self._mem_mapped:
            return DeviceMap('d_%s' % self.name, shape=self._shape,
                             function=self.function)
        elif self._mem_local:
            return self.indexed
        else:
            return None

    @property
    def size(self):
        """
        The number of elements this object is expected to store in memory.
        Note that this would need to be combined with self.dtype to give the actual
        size in bytes.
        """
        return reduce(mul, self.shape_allocated)

    @property
    def nbytes(self):
        return self.size*self._dtype().itemsize

    @property
    def halo(self):
        return self._halo

    @property
    def padding(self):
        return self._padding

    @property
    def is_const(self):
        return False

    @property
    def alias(self):
        return self._alias

    @property
    def _C_name(self):
        return "%s_vec" % self.name

    @cached_property
    def _C_symbol(self):
        return BoundSymbol(name=self._C_name, dtype=self.dtype, function=self.function)

    @property
    def _mem_mapped(self):
        return not self._mem_local

    def _make_pointer(self):
        """Generate a symbolic pointer to self."""
        raise NotImplementedError

    @cached_property
    def _size_domain(self):
        """Number of points in the domain region."""
        return DimensionTuple(*self.shape, getters=self.dimensions)

    @cached_property
    def _size_halo(self):
        """Number of points in the halo region."""
        left = tuple(zip(*self._halo))[0]
        right = tuple(zip(*self._halo))[1]

        sizes = tuple(Size(i, j) for i, j in self._halo)

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _size_owned(self):
        """Number of points in the owned region."""
        left = tuple(self._size_halo.right)
        right = tuple(self._size_halo.left)

        sizes = tuple(Size(i.right, i.left) for i in self._size_halo)

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _size_padding(self):
        """Number of points in the padding region."""
        left = tuple(zip(*self._padding))[0]
        right = tuple(zip(*self._padding))[1]

        sizes = tuple(Size(i, j) for i, j in self._padding)

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _size_nopad(self):
        """Number of points in the domain+halo region."""
        sizes = tuple(i+sum(j) for i, j in zip(self._size_domain, self._size_halo))
        return DimensionTuple(*sizes, getters=self.dimensions)

    @cached_property
    def _size_nodomain(self):
        """Number of points in the padding+halo region."""
        left = tuple(i for i, _ in np.add(self._halo, self._padding))
        right = tuple(i for _, i in np.add(self._halo, self._padding))

        sizes = tuple(Size(i, j) for i, j in np.add(self._halo, self._padding))

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _offset_domain(self):
        """Number of points before the first domain element."""
        offsets = tuple(np.add(self._size_padding.left, self._size_halo.left))
        return DimensionTuple(*offsets, getters=self.dimensions)

    @cached_property
    def _offset_halo(self):
        """Number of points before the first and last halo elements."""
        left = tuple(self._size_padding.left)
        right = tuple(np.add(np.add(left, self._size_halo.left), self._size_domain))

        offsets = tuple(Offset(i, j) for i, j in zip(left, right))

        return DimensionTuple(*offsets, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _offset_owned(self):
        """Number of points before the first and last owned elements."""
        left = tuple(self._offset_domain)
        right = tuple(np.add(self._offset_halo.left, self._size_domain))

        offsets = tuple(Offset(i, j) for i, j in zip(left, right))

        return DimensionTuple(*offsets, getters=self.dimensions, left=left, right=right)

    @property
    def _data_alignment(self):
        """
        The base virtual address of the data carried by the object is a multiple
        of the alignment.
        """
        return default_allocator().guaranteed_alignment

    def indexify(self, indices=None, subs=None):
        """Create a types.Indexed from the current object."""
        if indices is not None:
            return Indexed(self.indexed, *indices)

        # Substitution for each index (spacing only used in own dimension)
        subs = subs or {}
        subs = [{**{d.spacing: 1, -d.spacing: -1}, **subs} for d in self.dimensions]

        # Indices after substitutions
        indices = [sympy.sympify((a - o).xreplace(s)) for a, o, s in
                   zip(self.args, self.origin, subs)]
        indices = [i.xreplace({k: sympy.Integer(k) for k in i.atoms(sympy.Float)})
                   for i in indices]
        return self.indexed[indices]

    def __getitem__(self, index):
        """Shortcut for ``self.indexed[index]``."""
        return self.indexed[index]

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__

    # Reconstruction support
    @property
    def _rcls(self):
        return self.__class__.__base__


# Extended SymPy hierarchy follows, for essentially two reasons:
# - To keep track of `function`
# - To override SymPy caching behaviour


class IndexedBase(sympy.IndexedBase, Basic, Pickable):

    """
    Wrapper class that inserts a pointer to the symbolic data object.
    """

    __rargs__ = ('label', 'shape')
    __rkwargs__ = ('function',)

    def __new__(cls, label, shape, function=None):
        # Make sure `label` is a devito.Symbol, not a sympy.Symbol
        if isinstance(label, str):
            label = Symbol(name=label, dtype=None)
        with sympy_mutex:
            obj = sympy.IndexedBase.__new__(cls, label, shape)
        obj.function = function
        return obj

    func = Pickable._rebuild

    def __getitem__(self, indices, **kwargs):
        """Produce a types.Indexed, rather than a sympy.Indexed."""
        return Indexed(self, *as_tuple(indices))

    def _hashable_content(self):
        return super()._hashable_content() + (self.function,)

    @property
    def _C_name(self):
        return self.name

    @cached_property
    def _C_ctype(self):
        try:
            return POINTER(dtype_to_ctype(self.dtype))
        except TypeError:
            # `dtype` is a ctypes-derived type!
            return self.dtype

    @property
    def base(self):
        return self

    @property
    def indices(self):
        return ()

    @property
    def dtype(self):
        return self.function.dtype

    @cached_property
    def free_symbols(self):
        ret = {self}
        for i in self.indices:
            try:
                ret.update(i.free_symbols)
            except AttributeError:
                pass
        return ret

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class IndexedData(IndexedBase):
    pass


class DeviceMap(IndexedBase):
    pass


class BoundSymbol(AbstractSymbol):

    """
    Wrapper class for Symbols that are bound to a symbolic data object.

    Notes
    -----
    By deliberately inheriting from AbstractSymbol, a BoundSymbol won't be
    in the devito cache. This will avoid cycling references in the cache
    (e.g., an entry for a Function `u(x)` and an entry for `u._C_symbol` with
    the latter's key including `u(x)`). This is totally fine. The BoundSymbol
    is tied to a specific Function; once the Function gets out of scope, the
    BoundSymbol will also become a garbage collector candidate.
    """

    def __new__(cls, *args, function=None, **kwargs):
        obj = AbstractSymbol.__new__(cls, *args, **kwargs)
        obj._function = function
        return obj

    @property
    def function(self):
        return self._function

    def _hashable_content(self):
        return super()._hashable_content() + (self.function,)

    @property
    def _C_ctype(self):
        return self.function._C_ctype


class Indexed(sympy.Indexed):

    # The two type flags have changed in upstream sympy as of version 1.1,
    # but the below interpretation is used throughout the compiler to
    # identify Indexed objects. With the sympy-1.1 changes a new flag
    # obj.is_Indexed was introduced which should be preferred, but the
    # required changes are cumbersome and many...
    is_Symbol = False
    is_Atom = False

    is_Dimension = False

    @memoized_meth
    def __str__(self):
        return super().__str__()

    def _hashable_content(self):
        return super(Indexed, self)._hashable_content() + (self.base.function,)

    @cached_property
    def indices(self):
        return DimensionTuple(*super().indices, getters=self.function.dimensions)

    @property
    def function(self):
        return self.base.function

    @property
    def dtype(self):
        return self.function.dtype

    @property
    def name(self):
        return self.base.name

    @property
    def origin(self):
        return self.function.origin

    @cached_property
    def free_symbols(self):
        # Make it cached, since it's relatively expensive and called often
        ret = super(Indexed, self).free_symbols
        # Get rid of the IndexedBase label this Indexed stems from
        # as in Devito we can't have it floating around in Eq's
        ret.discard(self.base.label)
        return ret

    def compare(self, other):
        """
        Override `sympy.Basic.compare` to honor Devito's canonical ordering
        of arguments.
        In SymPy:

            f[x+1] < f[x+2] < ... < f[x+9] < f[x]

        While in Devito we pretend

            f[x] < f[x+1] < f[x+2] < ... < f[x+9]

        That is the arguments need to be ordered monothonically based on the indices
        so that the symbolic trees of two derivative expressions can be compared
        argument-wise.
        """
        if (self.__class__ != other.__class__) or (self.function is not other.function):
            return super().compare(other)
        for l, r in zip(self.indices, other.indices):
            try:
                c = int(sympy.sign(l - r))
            except TypeError:
                # E.g., `l=x+1` and `r=y` or `r=sqrt(x)`
                c = l.compare(r)
            if c:
                return c
        return 0
