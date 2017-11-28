import devito.types as types
import devito.function as function

from devito.yask.wrappers import YaskGridConst, contexts

__all__ = ['Constant', 'Function', 'TimeFunction']


types.Basic.from_YASK = False

types.Array.from_YASK = True


class Constant(function.Constant):

    from_YASK = True

    def __init__(self, *args, **kwargs):
        value = kwargs.pop('value', 0.)
        super(Constant, self).__init__(*args, value=YaskGridConst(value), **kwargs)

    @function.Constant.data.setter
    def data(self, val):
        self._value = YaskGridConst(val)


class Function(function.Function):

    from_YASK = True

    def _allocate_memory(self):
        """Allocate memory in terms of YASK grids."""

        # TODO: YASK assumes that self.shape == "domain_size" (in YASK jargon)
        # This is exactly how Devito will work too once the Grid abstraction lands

        # Fetch the appropriate context
        context = contexts.fetch(self.grid, self.dtype)

        # TODO : the following will fail if not using a SteppingDimension,
        # eg with save=True one gets /time/ instead /t/
        self._data_object = context.make_grid(self)

    @property
    def _data_buffer(self):
        return super(Function, self).data

    @property
    def data(self):
        """
        The value of the data object, as a :class:`YaskGrid`.

        The returned object, which behaves as a :class:`numpy.ndarray`, provides
        a *view* of the actual data, in row-major format. Internally, the data is
        stored in whatever layout adopted by YASK.

        Any read/write from/to the returned :class:`YaskGrid` should be performed
        assuming a row-major storage layout; behind the scenes, these accesses
        are automatically translated into whatever YASK expects, in order to pick
        the intended values.

        Abstracting away the internal storage layout adopted by YASK guarantees
        that user code works independently of the chosen Devito backend. This may
        introduce a little performance penalty when accessing data w.r.t. the
        default Devito backend. Such penalty should however be easily amortizable,
        as the time spent in running Operators is expected to be vastly greater
        than any user-level data manipulation.
        """
        super(Function, self).data
        return self._data_object

    def initialize(self):
        raise NotImplementedError


class TimeFunction(function.TimeFunction, Function):

    from_YASK = True
