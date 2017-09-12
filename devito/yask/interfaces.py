import devito.interfaces as interfaces
from devito.logger import yask as log

from devito.yask import exit
from devito.yask.wrappers import yask_context

__all__ = ['ConstantData', 'DenseData', 'TimeData']


interfaces.Basic.from_YASK = False


class ConstantData(interfaces.ConstantData):

    from_YASK = True


class DenseData(interfaces.DenseData):

    from_YASK = True

    def _allocate_memory(self):
        """Allocate memory in terms of YASK grids."""

        # Fetch the appropriate context
        context = yask_context(self.indices, self.shape, self.dtype, self.space_order)

        # Only create a YaskGrid if the requested grid is dense
        dimensions = tuple(i.name for i in self.indices)
        # TODO : following check fails if not using BufferedDimension ('time' != 't')
        if dimensions in [context.dimensions, context.space_dimensions]:
            self._data_object = context.make_grid(self.name, dimensions, self.shape,
                                                  self.space_order, self.dtype)
        else:
            exit("Couldn't allocate YaskGrid.")

    @property
    def _data_buffer(self):
        return super(DenseData, self).data

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
        super(DenseData, self).data
        return self._data_object

    def initialize(self):
        raise NotImplementedError


class TimeData(interfaces.TimeData, DenseData):

    from_YASK = True
