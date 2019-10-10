from cached_property import cached_property
import sympy
import numpy as np
import ctypes
from functools import reduce
from operator import mul

from devito.exceptions import InvalidArgument
from devito.logger import yask as log, yask_warning as warning
from devito.tools import Signer, memoized_meth, dtype_to_ctype
import devito.types.basic as basic
import devito.types.caching as caching
import devito.types.constant as constant
import devito.types.dense as dense
import devito.types.grid as grid

from devito.yask.data import Data, DataScalar
from devito.yask.utils import namespace
from devito.yask.wrappers import contexts

__all__ = ['Constant', 'Function', 'TimeFunction', 'Grid', 'CacheManager',
           'YaskVarObject', 'YaskSolnObject']


class Constant(constant.Constant):

    from_YASK = True

    def __init_finalize__(self, *args, **kwargs):
        value = kwargs.pop('value', 0.)
        super(Constant, self).__init_finalize__(*args, value=DataScalar(value), **kwargs)

    @property
    def data(self):
        return self._value

    @data.setter
    def data(self, val):
        self._value = DataScalar(val)

    def _arg_defaults(self, alias=None):
        args = super(Constant, self)._arg_defaults(alias=alias)

        key = alias or self
        args[namespace['code-var-name'](key.name)] = None

        return args

    def _arg_values(self, **kwargs):
        values = super(Constant, self)._arg_values(**kwargs)

        # Necessary when there's a scalar (i.e., non-Constant) override
        values[namespace['code-var-name'](self.name)] = None

        return values


class Function(dense.Function, Signer):

    from_YASK = True

    def __new__(cls, *args, **kwargs):
        if cls._cached():
            newobj = sympy.Function.__new__(cls, *args, **kwargs.get('options', {}))
            newobj._cached_init()
        else:
            # If a Function has no SpaceDimension, than for sure it won't be
            # used by YASK. We then return a devito.Function, which employs
            # a standard row-major format for data values
            indices = cls.__indices_setup__(**kwargs)
            klass = cls if any(i.is_Space for i in indices) else cls.__base__
            newobj = cls.__base__.__new__(klass, *args, **kwargs)
        return newobj

    def __padding_setup__(self, **kwargs):
        # YASK calculates the padding, so we bypass the dense.Function's autopadding
        return tuple((0, 0) for i in range(self.ndim))

    def _allocate_memory(func):
        """Allocate memory in terms of YASK vars."""
        def wrapper(self):
            if self._data is None:
                log("Allocating memory for %s%s" % (self.name, self.shape_allocated))

                # Free memory carried by stale symbolic objects
                # TODO: see issue #944
                # CacheManager.clear(dump_contexts=False, force=False)

                # Fetch the appropriate context
                context = contexts.fetch(self.dimensions, self.dtype)

                # Create a YASK var; this allocates memory
                var = context.make_var(self)

                # `self._padding` must be updated as (from the YASK docs):
                # "The value may be slightly larger [...] due to rounding"
                padding = []
                for i in self.dimensions:
                    if i.is_Space:
                        padding.append((var.get_left_extra_pad_size(i.name),
                                        var.get_right_extra_pad_size(i.name)))
                    else:
                        # time and misc dimensions
                        padding.append((0, 0))
                self._padding = tuple(padding)
                del self.shape_allocated  # Invalidate cached_property

                self._data = Data(var, self.shape_allocated, self.indices, self.dtype)
                self._data.reset()
            return func(self)
        return wrapper

    def __del__(self):
        if self._data is not None:
            self._data.release_storage()

    @property
    @_allocate_memory
    def _data_buffer(self):
        num_elements = self._data.var.get_num_storage_elements()
        shape = self.shape_allocated
        ctype_1d = dtype_to_ctype(self.dtype) * reduce(mul, shape)

        if num_elements != reduce(mul, shape):
            warning("num_storage_elements(%d) != reduce(mul, %s)",
                    num_elements, str(shape))

        buf = ctypes.cast(
            int(self._data.var.get_raw_storage_buffer()),
            ctypes.POINTER(ctype_1d)).contents

        return np.frombuffer(buf, dtype=self.dtype).reshape(shape)

    @property
    def data(self):
        """
        The domain data values, as a Data.

        The returned object, which behaves as a `numpy.ndarray`, provides
        a *view* of the actual data, in row-major format. Internally, the data is
        stored in whatever layout adopted by YASK.

        Any read/write from/to the returned Data should be performed
        assuming a row-major storage layout; behind the scenes, these accesses
        are automatically translated into whatever YASK expects, in order to pick
        the intended values.

        Abstracting away the internal storage layout adopted by YASK guarantees
        that user code works independently of the chosen Devito backend. This may
        introduce a little performance penalty when accessing data w.r.t. the
        default Devito backend. Such penalty should however be easily amortizable,
        as the time spent in running Operators is expected to be vastly greater
        than any user-level data manipulation.

        For further information, refer to ``Data.__doc__``.
        """
        return self.data_domain

    @cached_property
    @_allocate_memory
    def data_domain(self):
        """
        Notes
        -----
        Alias to ``self.data``.
        """
        return Data(self._data.var, self.shape, self.indices, self.dtype,
                    offset=self._offset_domain)

    @cached_property
    @_allocate_memory
    def data_with_halo(self):
        return Data(self._data.var, self.shape_with_halo, self.indices, self.dtype,
                    offset=self._offset_halo.left)

    @cached_property
    @_allocate_memory
    def _data_allocated(self):
        return Data(self._data.var, self.shape_allocated, self.indices, self.dtype)

    def _arg_defaults(self, alias=None):
        args = super(Function, self)._arg_defaults(alias=alias)

        key = alias or self
        args[namespace['code-var-name'](key.name)] = self.data.rawpointer

        return args

    def _signature_items(self):
        return (self.name,) + tuple(i.name for i in self.indices)


class TimeFunction(dense.TimeFunction, Function):

    from_YASK = True

    @classmethod
    def __indices_setup__(cls, **kwargs):
        indices = list(dense.TimeFunction.__indices_setup__(**kwargs))
        # Never use a SteppingDimension in the yask backend: it is simply
        # unnecessary
        if indices[cls._time_position].is_Stepping:
            indices[cls._time_position] = indices[cls._time_position].root
        return tuple(indices)

    @memoized_meth
    def _arg_defaults(self, alias=None):
        args = super(TimeFunction, self)._arg_defaults(alias=alias)
        # This is a little hack: a TimeFunction originally meant to be accessed
        # via modulo buffered iteration should never impose anything on the time
        # dimension
        if self._time_buffering:
            args.pop(self.time_dim.max_name)
            args.pop(self.time_dim.size_name)
        return args

    def _arg_check(self, args, intervals):
        if self._time_buffering:
            # Using a TimeDimension in place of a SteppingDimension, so we
            # should silence any errors due to assuming OOB accesses
            try:
                super(TimeFunction, self)._arg_check(args, intervals)
            except InvalidArgument:
                for i, s in zip(self.indices, args[self.name].shape):
                    size = np.inf if i.is_Time else s
                    i._arg_check(args, size, intervals[i])
        else:
            super(TimeFunction, self)._arg_check(args, intervals)


class Grid(grid.Grid):

    def __init__(self, *args, **kwargs):
        super(Grid, self).__init__(*args, **kwargs)

        # Initialize a new YaskContext for this Grid
        contexts.putdefault(self)

    @property
    def _const(self):
        return Constant

    def _make_stepping_dim(self, time_dim, **kwargs):
        # In the `yask` backend, the stepping dimension is an alias of the
        # time dimension
        return time_dim

    def __setstate__(self, state):
        super(Grid, self).__setstate__(state)
        # A new context is created, as the unpickled Dimensions are new objects
        contexts.putdefault(self)


basic.Basic.from_YASK = False
basic.Basic.is_YaskVarObject = False
basic.Array.from_YASK = True


class YaskVarObject(basic.Object):

    is_YaskVarObject = True

    dtype = namespace['type-var']
    value = None

    def __init__(self, mapped_function_name):
        self.mapped_function_name = mapped_function_name
        self.name = namespace['code-var-name'](mapped_function_name)

    def _arg_values(self, args=None, **kwargs):
        # The C-pointer to a YASK var is provided directly by Function/TimeFunction
        return {}

    # Pickling support
    _pickle_args = ['mapped_function_name']
    _pickle_kwargs = []


class YaskSolnObject(basic.Object):

    dtype = namespace['type-solution']
    value = None

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = []


class CacheManager(caching.CacheManager):

    @classmethod
    def clear(cls, dump_contexts=True, force=True):
        log("Dumping contexts and symbol caches")
        if dump_contexts:
            contexts.dump()
        super(CacheManager, cls).clear(force=force)
