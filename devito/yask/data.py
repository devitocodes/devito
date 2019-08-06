import ctypes
import sys

import numpy as np

from devito.logger import yask as log
from devito.tools import as_tuple

from devito.yask.utils import namespace


class Data(object):

    """
    A view of a YASK var.

    A ``Data`` implements a subset of the ``numpy.ndarray`` API. The subset
    of API implemented should suffice to transition between Devito backends
    w/o changes to the user code.

    From the user level, YASK vars can only be accessed through views.
    Subclassing ``numpy.ndarray`` is theoretically possible, but it's in
    practice very difficult and inefficient, as multiple data copies would
    have to be maintained. This is because YASK's storage layout is different
    than Devito's, and in particular different than what users want to see
    (a very standard row-majot format).

    The storage layout of a YASK var on any given grid dim looks as follows: ::

    ------------------------------------------------------------------------------------
    | left_extra_padding | left_halo |              | right_halo | right_extra_padding |
    ----------------------------------    domain    ------------------------------------
    |            left_padding        |              |       right_padding              |
    ------------------------------------------------------------------------------------
    |                                   allocation                                     |
    ------------------------------------------------------------------------------------

    Parameters
    ----------
    var : YASK.var
        The YASK var for which a view is produced.
    shape : tuple of ints
        Shape of the data view in grid points.
    dimensions : tuple of Dimension
        The Dimensions of `var` along which the view is produced.
    dtype : data-type, optional
        The data type of the raw data.
    offset : tuple of ints, optional
        The offset of the data view from the first allocated item in each Dimension.

    Notes
    -----
    This type supports logical indexing over modulo buffered dimensions.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __init__(self, var, shape, dimensions, dtype, offset=None):
        assert len(shape) == len(dimensions)
        self.var = var
        self.dimensions = dimensions
        self.shape = shape
        self.dtype = dtype

        self._modulo = tuple(True if i.is_Stepping else False for i in dimensions)

        offset = offset or tuple(0 for _ in dimensions)
        assert len(offset) == len(dimensions)
        self._offset = [(self.get_first_rank_alloc_index(i.name)+j) if i.is_Space else 0
                        for i, j in zip(dimensions, offset)]

    def __getitem__(self, index):
        start, stop, shape = self._convert_index(index)
        if not shape:
            log("Data: Getting single entry %s" % str(start))
            assert start == stop
            out = self.var.get_element(start)
        else:
            log("Data: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.var.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        start, stop, shape = self._convert_index(index, 'set')
        if all(i == 1 for i in shape):
            log("Data: Setting single entry %s" % str(start))
            assert start == stop
            self.var.set_element(val, start)
        elif isinstance(val, np.ndarray):
            log("Data: Setting full-array/block via index [%s]" % str(index))
            if val.shape == shape:
                self.var.set_elements_in_slice(val, start, stop)
            elif len(val.shape) > len(shape):
                raise ValueError("Data: could not broadcast input array from shape "
                                 "%s into shape %s" % (val.shape, shape))
            else:
                # Emulate NumPy broadcasting
                broadcasted = np.empty(shape=shape, dtype=val.dtype)
                broadcasted[:] = val
                self.var.set_elements_in_slice(broadcasted, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.shape)):
            log("Data: Setting full-array to given scalar via single var sweep")
            self.var.set_all_elements_same(val)
        else:
            log("Data: Setting block to given scalar via index [%s]" % str(index))
            self.var.set_elements_in_slice_same(val, start, stop, True)

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

    def _convert_index(self, index, mode='get'):
        """
        Convert an ``index`` into a format suitable for YASK's get_elements_{...}
        and set_elements_{...} routines.

        ``index`` can be of any type out of the types supported by NumPy's
        ``ndarray.__getitem__`` and ``ndarray.__setitem__``.

        In particular, an ``index`` is either a single element or an iterable of
        elements. An element can be a slice object, an integer index, or a tuple
        of integer indices.

        In the general case in which ``index`` is an iterable, each element in
        the iterable corresponds to a dimension in ``shape``. In this case, an element
        can be either a slice or an integer, but not a tuple of integers.

        If ``index`` is a single element,  then it is interpreted as follows: ::

            * slice object: the slice spans the whole shape;
            * single integer: shape is one-dimensional, and the index represents
              a specific entry;
            * a tuple of integers: it must be ``len(index) == len(shape)``,
              and each entry in ``index`` corresponds to a specific entry in a
              dimension in ``shape``.

        The returned value is a 3-tuple ``(starts, ends, shapes)``, where ``starts,
        ends, shapes`` are lists of length ``len(shape)``. By taking ``starts[i]`` and
        `` ends[i]``, one gets the start and end points of the section of elements to
        be accessed along dimension ``i``; ``shapes[i]`` gives the size of the section.
        """

        # Note: the '-1' below are because YASK uses '<=', rather than '<', to check
        # bounds when iterating over var dimensions

        assert mode in ['get', 'set']
        index = as_tuple(index)

        # Index conversion
        cstart = []
        cstop = []
        cshape = []
        for i, size, use_modulo in zip(index, self.shape, self._modulo):
            if isinstance(i, type(np.newaxis)):
                raise NotImplementedError("Unsupported introduction of np.newaxis")
            elif isinstance(i, (np.ndarray, tuple, list)):
                raise NotImplementedError("Unsupported numpy advanced indexing")
            elif isinstance(i, slice):
                if i.step is not None:
                    raise NotImplementedError("Unsupported stepping != 1.")
                if i.start is None:
                    start = 0
                elif i.start < 0:
                    start = size + i.start
                else:
                    start = i.start
                if i.stop is None:
                    stop = size - 1
                elif i.stop < 0:
                    stop = size + (i.stop - 1)
                else:
                    stop = i.stop - 1
                shape = stop - start + 1
            else:
                if i < 0:
                    start = size + i
                    stop = size + i
                else:
                    start = i
                    stop = i
                shape = 1 if mode == 'set' else None
            # Apply logical indexing
            if use_modulo is True:
                start %= size
                stop %= size
            # Finally append the converted index
            cstart.append(start)
            cstop.append(stop)
            if shape is not None:
                cshape.append(shape)

        # Remainder (e.g., requesting A[1] and A has shape (3,3))
        nremainder = len(self.shape) - len(index)
        cstart.extend([0]*nremainder)
        cstop.extend([self.shape[len(index) + j] - 1 for j in range(nremainder)])
        cshape.extend([self.shape[len(index) + j] for j in range(nremainder)])

        assert len(self.shape) == len(cstart) == len(cstop) == len(self._offset)

        # Shift by the specified offsets
        cstart = [int(j + i) for i, j in zip(self._offset, cstart)]
        cstop = [int(j + i) for i, j in zip(self._offset, cstop)]

        return tuple(cstart), tuple(cstop), tuple(cshape)

    def _give_storage(self, target):
        """
        Replace meta-data and storage of ``target`` with those of ``self.var``.
        Any subsequent API call via either handle will get data from and/or
        alter the shared var.
        """
        assert self.var.is_storage_allocated()
        target.fuse_vars(self.var)
        assert target.is_storage_allocated()
        assert self.var.is_storage_allocated()

    def __getattr__(self, name):
        """Proxy to yk::var methods."""
        return getattr(self.var, name)

    def __repr__(self):
        return repr(self[:])

    def __meta_op__(op, reverse=False):
        # Used to build all binary operations such as __eq__, __add__, etc.
        # These all boil down to calling the numpy equivalents
        def f(self, other):
            o1, o2 = (self[:], other) if reverse is False else (other, self[:])
            return getattr(o1, op)(o2)
        return f
    __eq__ = __meta_op__('__eq__')
    __ne__ = __meta_op__('__ne__')
    __le__ = __meta_op__('__le__')
    __lt__ = __meta_op__('__lt__')
    __ge__ = __meta_op__('__ge__')
    __gt__ = __meta_op__('__gt__')
    __add__ = __meta_op__('__add__')
    __radd__ = __meta_op__('__add__')
    __sub__ = __meta_op__('__sub__')
    __rsub__ = __meta_op__('__sub__', True)
    __mul__ = __meta_op__('__mul__')
    __rmul__ = __meta_op__('__mul__', True)
    __div__ = __meta_op__('__div__')
    __rdiv__ = __meta_op__('__div__', True)
    __truediv__ = __meta_op__('__truediv__')
    __rtruediv__ = __meta_op__('__truediv__', True)
    __mod__ = __meta_op__('__mod__')
    __rmod__ = __meta_op__('__mod__', True)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def rawpointer(self):
        return ctypes.cast(int(self.var), namespace['type-var'])

    def reset(self):
        """
        Set all var entries to 0.
        """
        self[:] = 0.0

    def view(self, *args):
        """
        View of the YASK var in standard (i.e., Devito) row-major layout,
        returned as a `numpy.ndarray`.
        """
        return self[:]


class DataScalar(np.float):

    """A YASK var wrapper for scalar values."""

    def _give_storage(self, target):
        if not target.is_storage_allocated():
            target.alloc_storage()
        target.set_element(float(self.real), [])

    @property
    def rawpointer(self):
        return None
