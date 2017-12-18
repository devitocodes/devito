import sys

import numpy as np

from devito.logger import yask as log
from devito.tools import as_tuple

from devito.yask.utils import rawpointer


class Data(object):

    """
    A ``Data`` wraps a YASK grid.

    A ``Data`` implements a subset of the ``numpy.ndarray`` API. The subset of
    API implemented should suffice to transition between Devito backends w/o changes
    to the user code. It was not possible to subclass ``numpy.ndarray``, as this
    would have led to shadow data copies, since YASK employs a storage layout
    different than what users expect (row-major).

    The storage layout of a YASK grid looks as follows: ::

        --------------------------------------------------------------
        | extra_padding | halo |              | halo | extra_padding |
        ------------------------    domain    ------------------------
        |       padding        |              |       padding        |
        --------------------------------------------------------------
        |                         allocation                         |
        --------------------------------------------------------------

    :param grid: The YASK yk::grid that will be wrapped. Data storage will be
                 allocated if not yet allocated.
    :param shape: Shape of the domain region in grid points.
    :param dimensions: A tuple of :class:`Dimension`s, representing the dimensions
                       of the ``Data``.
    :param halo: An integer indicating the extent of the halo region.
    :param dtype: A ``numpy.dtype`` for the raw data.

    .. note::

        This type supports logical indexing over modulo buffered dimensions.
    """

    # Force __rOP__ methods (OP={add,mul,...) to get arrays, not scalars, for efficiency
    __array_priority__ = 1000

    def __init__(self, grid, shape, dimensions, halo, dtype):
        self.grid = grid
        self.dimensions = dimensions
        self.dtype = dtype

        self._modulo = tuple(i.modulo if i.is_Stepping else None for i in dimensions)

        # TODO: the initialization below will (slightly) change after the
        # domain-allocation switch will have happened in Devito. E.g.,
        # currently shape == domain

        if not grid.is_storage_allocated():
            # Set up halo sizes
            for i, j in zip(dimensions, shape):
                if i.is_Time:
                    assert self.grid.is_dim_used(i.name)
                    assert self.grid.get_alloc_size(i.name) == j
                else:
                    # Note, from the YASK docs:
                    # "If the halo is set to a value larger than the padding size,
                    # the padding size will be automatically increase to accomodate it."
                    self.grid.set_halo_size(i.name, halo)

            # Allocate memory
            self.grid.alloc_storage()

            self._halo = []
            self._ofs = [0 if i.is_Time else self.get_first_rank_domain_index(i.name)
                         for i in dimensions]
            self._shape = shape

            # `self` will actually act as a view of `self.base`
            self.base = Data(grid, shape, dimensions, 0, dtype)

            # Initialize memory to 0
            self.reset()
        else:
            self._halo = [0 if i.is_Time else self.get_halo_size(i.name)
                          for i in dimensions]
            self._ofs = [0 if i.is_Time else (self.get_first_rank_domain_index(i.name)-j)
                         for i, j in zip(dimensions, self._halo)]
            self._shape = [i + 2*j for i, j in zip(shape, self._halo)]

            # Like numpy.ndarray, `base = None` indicates that this is the real
            # array, i.e., it's not a view
            self.base = None

    def __del__(self):
        if self.base is None:
            self.grid.release_storage()

    def __getitem__(self, index):
        start, stop, shape = self._convert_index(index)
        if not shape:
            log("Data: Getting single entry %s" % str(start))
            assert start == stop
            out = self.grid.get_element(start)
        else:
            log("Data: Getting full-array/block via index [%s]" % str(index))
            out = np.empty(shape, self.dtype, 'C')
            self.grid.get_elements_in_slice(out.data, start, stop)
        return out

    def __setitem__(self, index, val):
        start, stop, shape = self._convert_index(index, 'set')
        if all(i == 1 for i in shape):
            log("Data: Setting single entry %s" % str(start))
            assert start == stop
            self.grid.set_element(val, start)
        elif isinstance(val, np.ndarray):
            log("Data: Setting full-array/block via index [%s]" % str(index))
            self.grid.set_elements_in_slice(val, start, stop)
        elif all(i == j-1 for i, j in zip(shape, self.shape)):
            log("Data: Setting full-array to given scalar via single grid sweep")
            self.grid.set_all_elements_same(val)
        else:
            log("Data: Setting block to given scalar via index [%s]" % str(index))
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
        # bounds when iterating over grid dimensions

        assert mode in ['get', 'set']
        index = as_tuple(index)

        # Index conversion
        cstart = []
        cstop = []
        cshape = []
        for i, size, mod in zip(index, self.shape, self._modulo):
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
                    stop = size + i.stop
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
            if mod is not None:
                start %= mod
                stop %= mod
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

        assert len(self.shape) == len(cstart) == len(cstop) == len(self._ofs)

        # Shift by the specified offsets
        cstart = [int(j + i) for i, j in zip(self._ofs, cstart)]
        cstop = [int(j + i) for i, j in zip(self._ofs, cstop)]

        return cstart, cstop, cshape

    def _give_storage(self, target):
        """
        Share self's storage with ``target``.
        """
        for i in self.dimensions:
            if i.is_Time:
                target.set_alloc_size(i.name, self.get_alloc_size(i.name))
            else:
                target.set_halo_size(i.name, self.get_halo_size(i.name))
        target.share_storage(self.grid)

    def __getattr__(self, name):
        """Proxy to yk::grid methods."""
        return getattr(self.grid, name)

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
    def with_halo(self):
        if self.base is None:
            raise ValueError("Cannot access the halo of a non-view Data")
        return self.base

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def rawpointer(self):
        return rawpointer(self.grid)

    def reset(self):
        """
        Set all grid entries to 0.
        """
        self[:] = 0.0

    def view(self):
        """
        View of the YASK grid in standard (i.e., Devito) row-major layout,
        returned as a :class:`numpy.ndarray`.
        """
        return self[:]


class DataScalar(np.float64):

    """A YASK grid wrapper for scalar values."""

    def _give_storage(self, target):
        if not target.is_storage_allocated():
            target.alloc_storage()
        target.set_element(float(self.real), [])

    @property
    def rawpointer(self):
        return None
