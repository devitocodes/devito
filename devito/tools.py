import ctypes
from collections import Callable, Iterable, OrderedDict

import numpy as np


def as_tuple(item, type=None, length=None):
    """
    Force item to a tuple.

    Partly extracted from: https://github.com/OP2/PyOP2/.
    """
    # Empty list if we get passed None
    if item is None:
        t = ()
    elif isinstance(item, str):
        t = (item,)
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except (TypeError, NotImplementedError):
            t = (item,) * (length or 1)
    if length and not len(t) == length:
        raise ValueError("Tuple needs to be of length %d" % length)
    if type and not all(isinstance(i, type) for i in t):
        raise TypeError("Items need to be of type %s" % type)
    return t


def flatten(l):
    """Flatten a hierarchy of nested lists into a plain list."""
    newlist = []
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            for sub in flatten(el):
                newlist.append(sub)
        else:
            newlist.append(el)
    return newlist


def filter_ordered(elements, key=None):
    """Filter elements in a list while preserving order.

    :param key: Optional conversion key used during equality comparison.
    """
    seen = set()
    if key is None:
        key = lambda x: x
    return [e for e in elements if not (key(e) in seen or seen.add(key(e)))]


def filter_sorted(elements, key=None):
    """Filter elements in a list and sort them by key"""
    return sorted(filter_ordered(elements, key=key), key=key)


def convert_dtype_to_ctype(dtype):
    """Maps numpy types to C types.

    :param dtype: A Python numpy type of int32, float32, int64 or float64
    :returns: Corresponding C type
    """
    conversion_dict = {np.int32: ctypes.c_int, np.float32: ctypes.c_float,
                       np.int64: ctypes.c_int64, np.float64: ctypes.c_double}

    return conversion_dict[dtype]


def aligned(a, alignment=16):
    """Function to align the memmory

    :param a: The given memory
    :param alignment: Granularity of alignment, 16 bytes by default
    :returns: Reference to the start of the aligned memory
    """
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)

    assert (aa.ctypes.data % alignment) == 0

    return aa


def pprint(node, verbose=True):
    """
    Shortcut to pretty print Iteration/Expression trees.
    """
    from devito.visitors import printAST
    print(printAST(node, verbose))


class DefaultOrderedDict(OrderedDict):
    # Source: http://stackoverflow.com/a/6190500/562769
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)
