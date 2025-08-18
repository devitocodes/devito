import abc
import inspect
from collections import namedtuple
from ctypes import POINTER, _Pointer, c_char_p, c_char, Structure
from functools import reduce, cached_property
from operator import mul

import numpy as np
import sympy

from sympy.core.assumptions import _assume_rules
from sympy.core.decorators import call_highest_priority

from devito.data import default_allocator
from devito.parameters import configuration
from devito.tools import (Pickable, as_tuple, dtype_to_ctype,
                          frozendict, memoized_meth, sympy_mutex, CustomDtype)
from devito.types.args import ArgProvider
from devito.types.caching import Cached, Uncached
from devito.types.lazy import Evaluable
from devito.types.utils import DimensionTuple

__all__ = ['Symbol', 'Scalar', 'Indexed', 'IndexedData', 'DeviceMap',
           'IrregularFunctionInterface']


Size = namedtuple('Size', 'left right')
Offset = namedtuple('Offset', 'left right')


class CodeSymbol:

    """
    Abstract base class for objects representing symbols in the generated code.

    The _C_* properties describe the object in C-land. For example its name and
    its type.

    The _mem_* properties describe the object memory allocation strategy. There
    are three axes, with a few possible values each:

        * "liveness": `_mem_external`, `_mem_internal_eager`, `_mem_internal_lazy`
        * "space": `_mem_local`, `_mem_mapped`, `_mem_host`
        * "scope": `_mem_stack`, `_mem_heap`, `_mem_global`, `_mem_shared`,
                   `_mem_shared_remote`, `_mem_constant`, `_mem_registers`,
                   `_mem_rvalue`

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

    @property
    @abc.abstractmethod
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
        The type of the object's data in the generated code.
        """
        _type = self._C_ctype
        if isinstance(_type, CustomDtype):
            return _type

        while issubclass(_type, _Pointer):
            _type = _type._type_

        # `ctypes` treats C strings specially
        if _type is c_char_p:
            _type = c_char

        try:
            # We have internal types such as c_complex that are
            # Structure too but should be treated as plain c_type
            _type._base_dtype
        except AttributeError:
            if issubclass(_type, Structure):
                _type = f'struct {_type.__name__}'

        return _type

    @property
    @abc.abstractmethod
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
        True if the associated data is allocated on the stack, False otherwise.
        """
        return False

    @property
    def _mem_heap(self):
        """
        True if the associated data is allocated on the heap, False otherwise.
        """
        return False

    @property
    def _mem_global(self):
        """
        True if the symbol is globally scoped, False otherwise.
        """
        return self._mem_constant

    @property
    def _mem_constant(self):
        """
        True if the associated data is allocated in global constant memory,
        False otherwise. This is a special case of `_mem_global`.
        """
        return False

    @property
    def _mem_shared(self):
        """
        True if the associated data is allocated in so called shared memory,
        False otherwise.
        """
        return False

    @property
    def _mem_shared_remote(self):
        """
        True if the associated data is allocated in so called remote shared
        memory, False otherwise.
        """
        return False

    @property
    def _mem_registers(self):
        """
        True if the associated data is allocated in registers, False otherwise.
        """
        return False

    @property
    def _mem_rvalue(self):
        """
        True if the associated data is allocated in a temporary (or "transient")
        variable, such as rvalues in CXX, False otherwise.
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
    is_AbstractTensor = False
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
    is_LocalType = False

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
    def base(self):
        return self

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
    is_commutative = True

    __rkwargs__ = ('name', 'dtype', 'is_const')

    @classmethod
    def _filter_assumptions(cls, **kwargs):
        """Extract sympy.Symbol-specific kwargs."""
        assumptions = {}
        # Pop predefined assumptions
        for key in ('real', 'imaginary', 'commutative'):
            kwargs.pop(key, None)

        # Extract sympy.Symbol-specific kwargs
        for i in list(kwargs):
            if i in _assume_rules.defined_facts:
                assumptions[i] = kwargs.pop(i)

        return assumptions, kwargs

    @staticmethod
    def __xnew__(cls, name, **assumptions):
        # Create the new Symbol
        # Note: use __xnew__ to bypass sympy caching
        newobj = sympy.Symbol.__xnew__(cls, name, **assumptions)

        assumptions = newobj._assumptions.copy()
        for key in ('real', 'imaginary', 'complex'):
            assumptions.pop(key, None)
        newobj._assumptions = assumptions

        return newobj

    def __new__(cls, *args, **kwargs):
        name = kwargs.get('name') or args[0]
        assumptions, kwargs = cls._filter_assumptions(**kwargs)

        newobj = cls.__xnew__(cls, name, **assumptions)

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

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.dtype is other.dtype and
                self.is_const == other.is_const and
                super().__eq__(other))

    __hash__ = sympy.Symbol.__hash__

    def _hashable_content(self):
        return super()._hashable_content() + (self.dtype, self.is_const)

    @property
    def dtype(self):
        return self._dtype

    def _eval_is_real(self):
        return not self.is_imaginary

    def _eval_is_imaginary(self):
        try:
            return np.iscomplexobj(self.dtype(0))
        except TypeError:
            # Non-callabale dtype, likely non-numpy
            # Assuming it's not complex
            return False

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

        # Any missing __rkwargs__ along with their default values
        params = inspect.signature(cls.__init_finalize__).parameters
        missing = [i for i in cls.__rkwargs__ if i in set(params).difference(key)]
        key.update({i: params[i].default for i in missing})

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
        newobj = cls.__xnew__(cls, name, **assumptions)

        # Initialization
        newobj._dtype = cls.__dtype_setup__(**kwargs)
        newobj.__init_finalize__(name, *args, **kwargs)

        # Store new instance in symbol cache
        Cached.__init__(newobj, key)

        return newobj

    __hash__ = Cached.__hash__


class DataSymbol(AbstractSymbol, Uncached, ArgProvider):

    """
    A unique scalar symbol that carries data.
    """
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
            return self._arg_defaults(**kwargs)


class AbstractFunction(sympy.Function, Basic, Pickable, Evaluable):

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

        * Array: A compiler-generated object that does not carry data.
        * Function: A space-varying discrete function, which carries user data.
        * TimeFunction: A time- and space-varying discrete function, which carries
                        user data.
        * SparseFunction: A space-varying discrete function representing "sparse"
                          points, i.e. points that are not aligned with the
                          computational grid.
        * SparseTimeFunction: A time- and space-varying function representing
                              "sparse" points, i.e. points that are not aligned
                              with the computational grid.
        * PrecomputedSparseFunction: A SparseFunction that uses a custom
                                     interpolation scheme, instead of linear
                                     interpolators.
        * PrecomputedSparseTimeFunction: A SparseTimeFunction that uses a custom
                                         interpolation scheme, instead of linear
                                         interpolators.
    """

    # SymPy attributes, explicitly say these are not Matrices
    is_MatrixLike = False
    is_Matrix = False

    is_AbstractFunction = True

    # SymPy default assumptions
    is_commutative = True

    # Devito default assumptions
    is_regular = True
    """
    True if alignment between iteration and data points is affine. Examples of cases
    where this would be False include types such as compressed Functions, etc.
    """

    is_autopaddable = False
    """
    True if the Function can be padded automatically by the Devito runtime,
    thus increasing its size, False otherwise. Note that this property has no
    effect if autopadding is disabled, which is the default behavior.
    """

    __rkwargs__ = ('name', 'dtype', 'grid', 'halo', 'padding', 'ghost',
                   'alias', 'space', 'function', 'is_transient', 'avg_mode')

    __properties__ = ('is_const', 'is_transient')

    def __new__(cls, *args, **kwargs):
        # Preprocess arguments
        args, kwargs = cls.__args_setup__(*args, **kwargs)

        # Extract the `indices`, as perhaps they're explicitly provided
        dimensions, indices = cls.__indices_setup__(*args, **kwargs)

        # If it's an alias or simply has a different name, ignore `function`.
        # These cases imply the construction of a new AbstractFunction off
        # an existing one! This relieves the pressure on the caller by not
        # requiring `function=None` explicitly at rebuild
        name = kwargs.get('name')
        alias = kwargs.get('alias')
        function = kwargs.get('function')
        if alias is True or (function and function.name != name):
            function = kwargs['function'] = None

        # If same name/indices and `function` isn't None, then it's
        # definitely a reconstruction
        if function is not None and \
           function.name == name and \
           function.indices == indices:
            # Special case: a syntactically identical alias of `function`, so
            # let's just return `function` itself
            return function

        # If dimensions have been replaced, then it is necessary to set `function`
        # to None. It may also be necessary to remove halo and padding so that
        # they are rebuilt with the new dimensions
        if function is not None and function.dimensions != dimensions:
            function = kwargs['function'] = None
            for i in ('halo', 'padding'):
                if len(kwargs[i]) != len(dimensions):
                    kwargs.pop(i)
                else:
                    # Downcast from DimensionTuple so that the new `dimensions`
                    # are used down the line
                    kwargs[i] = tuple(kwargs[i])

        with sympy_mutex:
            # Go straight through Basic, thus bypassing caching and machinery
            # in sympy.Application/Function that isn't really necessary
            # AbstractFunctions are unique by construction!
            newobj = sympy.Basic.__new__(cls, *sympy.sympify(indices))

        # Initialization. The following attributes must be available
        # when executing __init_finalize__
        newobj._name = name
        newobj._dimensions = dimensions
        newobj._shape = cls.__shape_setup__(**kwargs)
        newobj._dtype = cls.__dtype_setup__(**kwargs)

        # All objects created off an existing AbstractFunction `f` (e.g.,
        # via .func, or .subs, such as `f(x + 1)`) keep a reference to `f`
        # through the `function` field
        newobj.function = function or newobj

        newobj.__init_finalize__(*args, **kwargs)

        return newobj

    def __init__(self, *args, **kwargs):
        # no-op, the true init is performed by __init_finalize__
        pass

    def __str__(self):
        return f"{self.name}({', '.join(str(i) for i in self.indices)})"

    __repr__ = __str__

    def _sympystr(self, printer, **kwargs):
        return str(self)

    _latex = _sympystr
    _eval_is_real = AbstractSymbol._eval_is_real
    _eval_is_imaginary = AbstractSymbol._eval_is_imaginary

    def _pretty(self, printer, **kwargs):
        return printer._print_Function(self, func_name=self.name)

    def __eq__(self, other):
        try:
            return (self.function is other.function and
                    self.indices == other.indices and
                    other.is_AbstractFunction)
        except AttributeError:
            # `other` not even an AbstractFunction
            return False

    __hash__ = sympy.Function.__hash__

    def _hashable_content(self):
        return super()._hashable_content() + (id(self.function), self.indices)

    @sympy.cacheit
    def sort_key(self, order=None):
        # Ensure that `f(x)` appears before `g(x)`
        # With the legacy caching framework this wasn't necessary because
        # the function name was already encoded in the class_key
        class_key, args, exp, coeff = super().sort_key(order=order)
        args = (len(args[1]) + 1, (self.name,) + args[1])
        return class_key, args, exp, coeff

    def __init_finalize__(self, *args, **kwargs):
        # A `Distributor` to handle domain decomposition
        self._distributor = self.__distributor_setup__(**kwargs)

        # Setup halo, padding, and ghost regions
        self._is_halo_dirty = False
        self._halo = self.__halo_setup__(**kwargs)
        self._padding = self.__padding_setup__(**kwargs)
        self._ghost = self.__ghost_setup__(**kwargs)

        # There may or may not be a `Grid`
        self._grid = kwargs.get('grid')

        # Symbol properties

        # "Aliasing" another AbstractFunction means that `self` logically
        # represents another object. For example, `self` might be used as the
        # formal parameter of a routine generated by the compiler, where the
        # routines is applied to several actual DiscreteFunctions
        self._alias = kwargs.get('alias', False)

        # The memory space of the AbstractFunction
        # See `_mem_{local,mapped,host}.__doc__` for more info
        self._space = kwargs.get('space', 'mapped')
        assert self._space in ['local', 'mapped', 'host']

        # If True, the AbstractFunction is treated by the compiler as a "transient
        # field", meaning that its content cannot be accessed by the user in
        # Python-land. This allows the compiler/run-time to apply certain
        # optimizations, such as avoiding memory copies across different Operator
        # executions
        self._is_transient = kwargs.get('is_transient', False)

        # Averaging mode for off the grid evaluation
        self._avg_mode = kwargs.get('avg_mode', 'arithmetic')
        if self._avg_mode not in ['arithmetic', 'harmonic']:
            raise ValueError("Invalid averaging mode_mode %s, accepted values are"
                             " arithmetic or harmonic" % self._avg_mode)

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
    def __indices_setup__(cls, *args, **kwargs):
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
        halo = tuple(kwargs.get('halo', ((0, 0),)*self.ndim))
        return DimensionTuple(*halo, getters=self.dimensions)

    def __padding_setup__(self, **kwargs):
        padding = tuple(kwargs.get('padding', ((0, 0),)*self.ndim))
        return DimensionTuple(*padding, getters=self.dimensions)

    @cached_property
    def __padding_dtype__(self):
        v = configuration['autopadding']
        if not self.is_autopaddable or not v:
            return None
        try:
            if issubclass(v, np.number):
                return v
        except TypeError:
            return np.float32

    def __padding_setup_smart__(self, **kwargs):
        nopadding = ((0, 0),)*self.ndim

        if not self.__padding_dtype__:
            return nopadding

        # The padded Dimension
        if not self.space_dimensions:
            return nopadding
        d = self.space_dimensions[-1]

        mmts = configuration['platform'].max_mem_trans_size(self.__padding_dtype__)
        remainder = self._size_nopad[d] % mmts
        if remainder == 0:
            # Already a multiple of `mmts`, no need to pad
            return nopadding

        dpadding = (0, (mmts - remainder))
        padding = [(0, 0)]*self.ndim
        padding[self.dimensions.index(d)] = dpadding

        return tuple(padding)

    def __ghost_setup__(self, **kwargs):
        return (0, 0)

    def __distributor_setup__(self, **kwargs):
        # There may or may not be a `Distributor`. In the latter case, the
        # AbstractFunction is to be considered "local" to each MPI rank
        try:
            return kwargs.get('grid').distributor
        except AttributeError:
            return kwargs.get('distributor')

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
        return DimensionTuple(*(r - d + o for d, r, o
                                in zip(self.dimensions, self.indices_ref,
                                       self._offset_subdomain)),
                              getters=self.dimensions)

    @property
    def dimensions(self):
        """Tuple of Dimensions representing the object indices."""
        return self._dimensions

    @cached_property
    def space_dimensions(self):
        """Tuple of Dimensions defining the physical space."""
        return tuple(d for d in self.dimensions if d.is_Space)

    @cached_property
    def root_dimensions(self):
        """Tuple of root Dimensions of the physical space Dimensions."""
        return tuple(d.root for d in self.space_dimensions)

    @property
    def base(self):
        return self.indexed

    @property
    def c0(self):
        """
        `self`'s first component if `self` is a tensor, otherwise just `self`.
        """
        return self

    @property
    def _eval_deriv(self):
        return self

    @property
    def _grid_map(self):
        """
        Mapper of off-grid interpolation points indices for each dimension.
        """
        mapper = {}
        for i, j, d in zip(self.indices, self.indices_ref, self.dimensions):
            # Two indices are aligned if they differ by an Integer*spacing.
            v = (i - j)/d.spacing
            try:
                if not isinstance(v, sympy.Number) or int(v) == v:
                    continue
                # Skip if index is just a Symbol or integer
                elif (i.is_Symbol and not i.has(d)) or i.is_Integer:
                    continue
                else:
                    mapper.update({d: i})
            except (AttributeError, TypeError):
                mapper.update({d: i})
        return mapper

    def _evaluate(self, **kwargs):
        """
        Evaluate off the grid with 2nd order interpolation.
        Directly available through zeroth order derivative of the base object
        i.e f(x + a) = f(x).diff(x, deriv_order=0, fd_order=2, x0={x: x + a})

        This allow to evaluate off grid points as EvalDerivative that are better
        for the compiler.
        """
        # Average values if at a location not on the Function's grid
        if not self._grid_map:
            return self

        # Base function
        if self._avg_mode == 'harmonic':
            retval = 1 / self.function
        else:
            retval = self.function
        # Apply interpolation from inner most dim
        for d, i in self._grid_map.items():
            retval = retval.diff(d, deriv_order=0, fd_order=2, x0={d: i})

        # Evaluate. Since we used `self.function` it will be on the grid when evaluate
        # is called again within FD
        if self._avg_mode == 'harmonic':
            from devito.finite_differences.differentiable import SafeInv
            retval = SafeInv(retval.evaluate, self.function)
        else:
            retval = retval.evaluate

        return retval

    @property
    def shape(self):
        """The shape of the object."""
        return self._shape

    @property
    def grid(self):
        """The Grid on which the discretization occurred."""
        return self._grid

    @property
    def _is_on_subdomain(self):
        """True if defined on a SubDomain"""
        return self.grid and self.grid.is_SubDomain

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
            return DeviceMap(f'd_{self.name}', shape=self._shape,
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
    def ghost(self):
        return self._ghost

    @property
    def is_const(self):
        return False

    @property
    def is_transient(self):
        return self._is_transient

    @property
    def is_persistent(self):
        """
        True if the AbstractFunction is persistent, i.e., its data is guaranteed
        to exist across multiple Operator invocations, False otherwise.
        By default, transient AbstractFunctions are not persistent. However,
        subclasses may override this behavior.
        """
        return not self.is_transient

    @cached_property
    def properties(self):
        return frozendict([(i, getattr(self, i)) for i in self.__properties__])

    @property
    def avg_mode(self):
        return self._avg_mode

    @property
    def alias(self):
        return self._alias

    @property
    def space(self):
        return self._space

    @property
    def _C_name(self):
        return f"{self.name}_vec"

    @cached_property
    def _C_symbol(self):
        return BoundSymbol(name=self._C_name, dtype=self.dtype, function=self.function)

    @property
    def _mem_local(self):
        return self._space == 'local'

    @property
    def _mem_mapped(self):
        return self._space == 'mapped'

    @property
    def _mem_host(self):
        return self._space == 'host'

    @cached_property
    def _signature(self):
        """
        The signature of an AbstractFunction is the set of fields that
        makes it "compatible" with another AbstractFunction. The fact that
        two AbstractFunctions are compatible may be exploited by the compiler
        to generate smarter code
        """
        ret = [type(self), self.indices]
        attrs = set(self.__rkwargs__) - {'name', 'function'}
        ret.extend(getattr(self, i) for i in attrs)
        return frozenset(ret)

    def _make_pointer(self):
        """Generate a symbolic pointer to self."""
        raise NotImplementedError

    @cached_property
    def _dist_dimensions(self):
        """The Dimensions decomposed for distributed-parallelism."""
        if self._distributor is None:
            return ()
        else:
            return tuple(d for d in self.dimensions if d in self._distributor.dimensions)

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
    def _size_ghost(self):
        """
        Number of points in the ghost region, that is the two areas before
        and after the allocated data.
        """
        return Size(*self._ghost)

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

    @cached_property
    def _offset_subdomain(self):
        """Offset of subdomain indices versus the global index."""
        # If defined on a SubDomain, then need to offset indices accordingly
        if not self._is_on_subdomain:
            return DimensionTuple(*[0 for _ in self.dimensions], getters=self.dimensions)
        # Symbolic offsets to avoid potential issues with user overrides
        offsets = []
        for d in self.dimensions:
            if d.is_Sub:
                l_tkn, r_tkn = d.tkns
                if l_tkn.value is None:
                    # Right subdimension
                    offsets.append(-r_tkn + d.symbolic_max + 1)
                elif r_tkn.value is None:
                    # Left subdimension
                    offsets.append(0)
                else:
                    # Middle subdimension
                    offsets.append(l_tkn)
            else:
                offsets.append(0)
        return DimensionTuple(*offsets, getters=self.dimensions)

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
        indices = []
        for a, d, o, s in zip(self.args, self.dimensions, self.origin, subs):
            if a.is_Function and len(a.args) == 1:
                # E.g. Abs(expr)
                arg = a.args[0]
                func = a.func
            else:
                arg = a
                func = lambda x: x
            if d in arg.free_symbols:
                # Shift by origin d -> d - o.
                indices.append(func(sympy.sympify(arg.subs(d, d - o).xreplace(s))))
            else:
                # Dimension has been removed, e.g. u[10], plain shift by origin
                indices.append(func(sympy.sympify(arg - o).xreplace(s)))

        indices = [i.xreplace({k: sympy.Integer(k) for k in i.atoms(sympy.Float)})
                   for i in indices]

        return self.indexed[indices]

    def __getitem__(self, index):
        """Shortcut for ``self.indexed[index]``."""
        return self.indexed[index]

    # Reconstruction support
    func = Pickable._rebuild

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__

    def __getnewargs_ex__(self):
        args, kwargs = super().__getnewargs_ex__()

        # Since f(x) stashes a reference to itself f(x), we must drop it here
        # or we'll end up with infinite recursion while attempting to serialize
        # the object. Upon unpickling, we'll then get `function=None`, which is
        # perfectly fine based on how `__new__`, and in particular the
        # initialization of the `.function` attribute, works
        if self is kwargs.get('function'):
            kwargs.pop('function')

        return args, kwargs


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
                        \\-------\\              |
                                 \\------- VectorTimeFunction

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

    __rkwargs__ = AbstractFunction.__rkwargs__

    @classmethod
    def _new(cls, *args, **kwargs):
        if args:
            try:
                # Constructor if input is (rows, cols, lambda)
                newobj = super()._new(*args)
            except ValueError:
                # Constructor if input is list of list as (row, cols, list_of_list)
                # doesn't work as it expects a flattened.
                newobj = super()._new(args[2])

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
            newobj = super()._new(comps)
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
        newobj = super()._fromrep(rep)
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

    @classmethod
    def __subfunc_setup__(cls, *args, **kwargs):
        """Setup each component of the tensor as a Devito type."""
        return []

    @classmethod
    def _sympify(self, arg):
        # This is used internally by sympy to process arguments at rebuilt. And since
        # some of our properties are non-sympyfiable we need to have a fallback
        try:
            return super()._sympify(arg)
        except sympy.SympifyError:
            return arg

    @property
    def grid(self):
        """
        A Tensor is expected to have all its components defined over the same grid
        """
        grids = {getattr(c, 'grid', None) for c in self.flat()} - {None}
        if len(grids) == 0:
            return None
        assert len(grids) == 1
        return grids.pop()

    @property
    def name(self):
        for c in self.values():
            try:
                return c.name.split('_')[0]
            except AttributeError:
                # `c` is not a devito object
                pass
        # If we end up here, then we have no devito objects
        # in the matrix, so we ust return the class name
        return self.__class__.__name__

    def _rebuild(self, *args, **kwargs):
        # Plain `func` call (row, col, comps)
        if not kwargs.keys() & self.__rkwargs__:
            if len(args) != 3:
                raise ValueError("Invalid number of arguments, expected nrow, ncol, "
                                 "list of components")
            return self._new(*args, **kwargs)
        # We need to rebuild the components with the new name then
        # rebuild the matrix
        newname = kwargs.pop('name', self.name)
        comps = [f.func(*args, name=f.name.replace(self.name, newname), **kwargs)
                 for f in self.flat()]
        # Rebuild the matrix with the new components
        return self._new(comps)

    func = _rebuild

    def _infer_dims(self):
        grids = {getattr(c, 'grid', None) for c in self.flat()} - {None}
        grids = {g.root for g in grids}
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
        self._name = kwargs.get('name', None)

    __hash__ = sympy.ImmutableDenseMatrix.__hash__

    def doit(self, **hint):
        return self

    def transpose(self, inner=True):
        new = super().transpose()
        if inner:
            return new.applyfunc(lambda x: getattr(x, 'T', x))
        return new

    def adjoint(self, inner=True):
        # Real valued adjoint is transpose
        return self.transpose(inner=inner)

    @call_highest_priority('__radd__')
    def __add__(self, other):
        try:
            # Most case support sympy add
            tsum = super().__add__(other)
        except TypeError:
            # Sympy doesn't support add with scalars
            tsum = self.applyfunc(lambda x: x + other)

        # As of sympy 1.13, super does not throw an exception but
        # only returns NotImplemented for some internal dispatch.
        if tsum is NotImplemented:
            return self.applyfunc(lambda x: x + other)

        return tsum

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
        newcls = self.classof_prod(other, other.cols)
        return newcls._new(self.rows, other.cols, new_mat, copy=False)


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

    @sympy.cacheit
    def sort_key(self, order=None):
        class_key, args, exp, coeff = super().sort_key(order=order)
        args = (self.function.class_key(), *args)
        return class_key, args, exp, coeff

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

    __rkwargs__ = AbstractSymbol.__rkwargs__ + ('function',)

    def __new__(cls, *args, function=None, **kwargs):
        obj = AbstractSymbol.__new__(cls, *args, **kwargs)
        obj._function = function
        return obj

    @property
    def function(self):
        return self._function

    @property
    def dimensions(self):
        return self.function.dimensions

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
        return super()._hashable_content() + (self.base.function,)

    @cached_property
    def indices(self):
        return DimensionTuple(*super().indices, getters=self.function.dimensions)

    @cached_property
    def dimensions(self):
        return self.function.dimensions

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
        ret = super().free_symbols
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

    def _subs(self, old, new, **hints):
        # Wrap in a try to make sure no substitution happens when
        # old is an Indexed as only checkink `old is new` would lead to
        # incorrect substitution of `old.base` by `new`
        try:
            if old.is_Indexed:
                if old.base == self.base and old.indices == self.indices:
                    return new
                else:
                    return self
        except AttributeError:
            pass
        return super()._subs(old, new, **hints)


class IrregularFunctionInterface:

    """
    A common interface for all irregular AbstractFunctions.
    """

    is_regular = False

    @property
    def nbytes_max(self):
        raise NotImplementedError


class LocalType(Basic):
    """
    This is the abstract base class for local types, which are
    generated by the compiler in C rather than in Python.

    Notes
    -----
    Subclasses should setup `_liveness`.
    """

    is_LocalType = True

    @property
    def liveness(self):
        return self._liveness

    @property
    def _mem_internal_eager(self):
        return self._liveness == 'eager'

    @property
    def _mem_internal_lazy(self):
        return self._liveness == 'lazy'

    """
    A modifier added to the subclass C declaration when it appears
    in a function signature. For example, a subclass might define `_C_modifier = '&'`
    to impose pass-by-reference semantics.
    """
    _C_modifier = None
