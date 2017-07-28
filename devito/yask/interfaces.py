from collections import OrderedDict
import sys

import numpy as np

import devito.interfaces as interfaces
from devito.logger import yask as log
from devito.tools import as_tuple

from devito.yask.kernel import yask_context
from devito.yask.utils import convert_multislice

__all__ = ['DenseData', 'TimeData']


class DenseData(interfaces.DenseData):

    def _allocate_memory(self):
        """Allocate memory in terms of Yask grids."""
        log("Allocating YaskGrid for %s (%s)" % (self.name, str(self.shape)))

        # TODO: a context, eventually, should be independent of the space_order,
        # but we currently have to pass it because we have no other way of determining
        # the solution "minimum padding". This means that all grids in a solution
        # must have the same halo extent (equivalently, in Devito jargon:
        # "all data objects in an equation must have the same space order").
        context = yask_context(self.indices, self.shape, self.dtype, self.space_order)

        # Only create a YaskGrid if the requested grid is a dense one
        dimensions = tuple(i.name for i in self.indices)
        # TODO : following check fails if not using BufferedDimension ('time' != 't')
        if dimensions in [context.dimensions, context.space_dimensions]:
            grid = context.add_grid(self.name, dimensions)
            dim_shape = OrderedDict(zip(dimensions, self.shape))
            self._data_object = YaskGrid(grid, dim_shape, self.dtype)
        else:
            log("Failed. Reverting to plain allocation...")
            super(DenseData, self)._allocate_memory()

    def initialize(self):
        raise NotImplementedError


class TimeData(interfaces.TimeData, DenseData):
    pass


class YaskGrid(object):

    """
    A ``YaskGrid`` wraps a YASK grid.

    An implementation of an array that behaves similarly to a ``numpy.ndarray``,
    suitable for the YASK storage layout.

    Subclassing ``numpy.ndarray`` would have led to shadow data copies, because
    of the different storage layout.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __init__(self, grid, dim_shape, dtype, buffer=None):
        """
        Initialize a new :class:`YaskGrid`.
        """
        self.grid = grid
        self.dim_shape = dim_shape
        self.dtype = dtype

        # Always init the grid, at least with 0.0
        self[:] = 0.0 if buffer is None else buffer

    def __getitem__(self, index):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(index, self.grid, self.dim_shape)
        if not shape:
            log("YaskGrid: Getting single entry")
            assert start == stop
            out = self.grid.get_element(*start)
        else:
            log("YaskGrid: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.grid.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        # TODO: ATM, no MPI support.
        start, stop, shape = convert_multislice(index, self.grid, self.dim_shape, 'set')
        if all(i == 1 for i in shape):
            log("YaskGrid: Setting single entry")
            assert start == stop
            self.grid.set_element(val, *start)
        elif isinstance(val, np.ndarray):
            log("YaskGrid: Setting full-array/block via index [%s]" % str(index))
            self.grid.set_elements_in_slice(val, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.dim_shape.values())):
            log("YaskGrid: Setting full-array to given scalar via single grid sweep")
            self.grid.set_all_elements_same(val)
        else:
            log("YaskGrid: Setting block to given scalar via index [%s]" % str(index))
            self.grid.set_elements_in_slice_same(val, start, stop, True)

    def __getslice__(self, start, stop):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        return self.__getitem__(slice(start, stop))

    def __setslice__(self, start, stop, val):
        if stop == sys.maxint:
            # Emulate default NumPy behaviour
            stop = None
        self.__setitem__(slice(start, stop), val)

    def __repr__(self):
        return repr(self[:])

    def __meta_binop(op):
        # Used to build all binary operations such as __eq__, __add__, etc.
        # These all boil down to calling the numpy equivalents
        def f(self, other):
            return getattr(self[:], op)(other)
        return f
    __eq__ = __meta_binop('__eq__')
    __ne__ = __meta_binop('__ne__')
    __le__ = __meta_binop('__le__')
    __lt__ = __meta_binop('__lt__')
    __ge__ = __meta_binop('__ge__')
    __gt__ = __meta_binop('__gt__')
    __add__ = __meta_binop('__add__')
    __radd__ = __meta_binop('__add__')
    __sub__ = __meta_binop('__sub__')
    __rsub__ = __meta_binop('__sub__')
    __mul__ = __meta_binop('__mul__')
    __rmul__ = __meta_binop('__mul__')
    __div__ = __meta_binop('__div__')
    __rdiv__ = __meta_binop('__div__')
    __truediv__ = __meta_binop('__truediv__')
    __rtruediv__ = __meta_binop('__truediv__')
    __mod__ = __meta_binop('__mod__')
    __rmod__ = __meta_binop('__mod__')

    @property
    def name(self):
        return self.grid.get_name()

    @property
    def ndpointer(self):
        # TODO: see corresponding comment in interfaces.py about CMemory
        return self

    def view(self):
        """
        View of the YASK grid in standard (i.e., Devito) row-major layout.
        """
        return self[:]
