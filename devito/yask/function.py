from cached_property import cached_property
import ctypes
import numpy as np

import devito.function as function
from devito.logger import yask as log
from devito.tools import numpy_to_ctypes

from devito.yask import namespace
from devito.yask.data import Data, DataScalar
from devito.yask.wrappers import contexts

__all__ = ['Constant', 'Function', 'TimeFunction']


class Constant(function.Constant):

    from_YASK = True

    def __init__(self, *args, **kwargs):
        value = kwargs.pop('value', 0.)
        super(Constant, self).__init__(*args, value=DataScalar(value), **kwargs)

    @function.Constant.data.setter
    def data(self, val):
        self._value = DataScalar(val)

    def _arg_defaults(self, alias=None):
        args = super(Constant, self)._arg_defaults(alias=alias)

        key = alias or self
        args[namespace['code-grid-name'](key.name)] = None

        return args

    def _arg_values(self, **kwargs):
        values = super(Constant, self)._arg_values(**kwargs)

        # Necessary when there's a scalar (i.e., non-Constant) override
        values[namespace['code-grid-name'](self.name)] = None

        return values


class Function(function.Function):

    from_YASK = True

    def _allocate_memory(func):
        """Allocate memory in terms of YASK grids."""
        def wrapper(self):
            if self._data is None:
                log("Allocating memory for %s%s" % (self.name, self.shape_allocated))

                # Fetch the appropriate context
                context = contexts.fetch(self.grid, self.dtype)

                # TODO : the following will fail if not using a SteppingDimension,
                # eg with save=True one gets /time/ instead /t/
                grid = context.make_grid(self)

                # /self._padding/ must be updated as (from the YASK docs):
                # "The value may be slightly larger [...] due to rounding"
                padding = [0 if i.is_Time else grid.get_extra_pad_size(i.name)
                           for i in self.indices]
                self._padding = tuple((i,)*2 for i in padding)

                self._data = Data(grid, self.shape_allocated, self.indices, self.dtype)
                self._data.reset()
            return func(self)
        return wrapper

    def __del__(self):
        if self._data is not None:
            self._data.release_storage()

    @property
    @_allocate_memory
    def _data_buffer(self):
        ctype = numpy_to_ctypes(self.dtype)
        cpointer = ctypes.cast(int(self._data.grid.get_raw_storage_buffer()),
                               ctypes.POINTER(ctype))
        ndpointer = np.ctypeslib.ndpointer(dtype=self.dtype, shape=self.shape_allocated)
        casted = ctypes.cast(cpointer, ndpointer)
        ndarray = np.ctypeslib.as_array(casted, shape=self.shape_allocated)
        return ndarray

    @property
    def data(self):
        """
        The domain data values, as a :class:`Data`.

        The returned object, which behaves as a :class:`numpy.ndarray`, provides
        a *view* of the actual data, in row-major format. Internally, the data is
        stored in whatever layout adopted by YASK.

        Any read/write from/to the returned :class:`Data` should be performed
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
        .. note::

            Alias to ``self.data``.
        """
        return Data(self._data.grid, self.shape, self.indices, self.dtype,
                    offset=self._offset_domain.left)

    @cached_property
    @_allocate_memory
    def data_with_halo(self):
        return Data(self._data.grid, self.shape_with_halo, self.indices, self.dtype,
                    offset=self._offset_halo.left)

    @cached_property
    @_allocate_memory
    def data_allocated(self):
        return Data(self._data.grid, self.shape_allocated, self.indices, self.dtype)

    def initialize(self):
        raise NotImplementedError

    def _arg_defaults(self, alias=None):
        args = super(Function, self)._arg_defaults(alias=alias)

        key = alias or self
        args[namespace['code-grid-name'](key.name)] = self.data.rawpointer

        return args


class TimeFunction(function.TimeFunction, Function):

    from_YASK = True
