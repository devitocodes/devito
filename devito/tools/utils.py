from abc import ABC
import os
import ctypes
from collections import Callable, Iterable, OrderedDict, Mapping
from functools import reduce
from itertools import chain, combinations, product, zip_longest
from operator import attrgetter, mul

import numpy as np

from multidict import MultiDict

from devito.logger import error

__all__ = ['prod', 'as_tuple', 'is_integer', 'generator', 'grouper', 'split',
           'roundm', 'powerset', 'invert', 'flatten', 'single_or', 'filter_ordered',
           'filter_sorted', 'build_dependence_lists', 'toposort', 'numpy_to_ctypes',
           'ctypes_to_C', 'ctypes_pointer', 'pprint', 'DefaultOrderedDict',
           'change_directory', 'Bunch',
           'EnrichedTuple', 'ReducerMap', 'Tag', 'sweep']


def prod(iterable):
    return reduce(mul, iterable, 1)


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


def is_integer(value):
    """
    A thorough instance comparison for all integer types.
    """
    return isinstance(value, int) or isinstance(value, np.integer)


def generator():
    """
    Return a function ``f`` that generates integer numbers starting at 0
    with stepping 1.
    """
    def f():
        ret = f.counter
        f.counter += 1
        return ret
    f.counter = 0
    return f


def grouper(iterable, n):
    """Split an interable into groups of size n, plus a reminder"""
    args = [iter(iterable)] * n
    return ([e for e in t if e is not None] for t in zip_longest(*args))


def split(iterable, f):
    """Split an iterable ``I`` into two iterables ``I1`` and ``I2`` of the
    same type as ``I``. ``I1`` contains all elements ``e`` in ``I`` for
    which ``f(e)`` returns True; ``I2`` is the complement of ``I1``."""
    i1 = type(iterable)(i for i in iterable if f(i))
    i2 = type(iterable)(i for i in iterable if not f(i))
    return i1, i2


def roundm(x, y):
    """Return x rounded up to the closest multiple of y."""
    return x if x % y == 0 else x + y - x % y


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def invert(mapper):
    """Invert a dict of lists preserving the order."""
    inverse = OrderedDict()
    for k, v in mapper.items():
        for i in v:
            inverse[i] = k
    return inverse


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


def single_or(l):
    """Return True iff only one item is different than ``None``, False otherwise.
    Note that this is not a XOR function, according to the truth table of the XOR
    boolean function with n > 2 inputs. Hence the name ``single_or``."""
    # No
    i = iter(l)
    return any(i) and not any(i)


def filter_ordered(elements, key=None):
    """Filter elements in a list while preserving order.

    :param key: Optional conversion key used during equality comparison.
    """
    seen = set()
    if key is None:
        key = lambda x: x
    return [e for e in elements if not (key(e) in seen or seen.add(key(e)))]


def filter_sorted(elements, key=None):
    """Filter elements in a list and sort them by key. The default key is
    ``operator.attrgetter('name')``."""
    if key is None:
        key = attrgetter('name')
    return sorted(filter_ordered(elements, key=key), key=key)


def build_dependence_lists(elements):
    """
    Given an iterable of dependences, return the dependence lists as a
    mapper suitable for graph-like algorithms. A dependence is an iterable of
    elements ``[a, b, c, ...]``, meaning that ``a`` preceeds ``b`` and ``c``,
    ``b`` preceeds ``c``, and so on.
    """
    mapper = OrderedDict()
    for element in elements:
        for idx, i0 in enumerate(element):
            v = mapper.setdefault(i0, set())
            for i1 in element[idx + 1:]:
                v.add(i1)
    return mapper


def toposort(data):
    """
    Given items that depend on other items, a topological sort arranges items in
    order that no one item precedes an item it depends on.

    ``data`` captures the various dependencies. It may be:

        * A dictionary whose keys are items and whose values are a set of
          dependent items. The dictionary may contain self-dependencies
          (which are ignored), and dependent items that are not also
          dict keys.
        * An iterable of dependences as expected by :func:`build_dependence_lists`.

    Readapted from: ::

        http://code.activestate.com/recipes/577413/
    """
    if not isinstance(data, Mapping):
        assert isinstance(data, Iterable)
        data = build_dependence_lists(data)

    processed = []

    if not data:
        return processed

    # Do not transform `data` in place
    mapper = OrderedDict([(k, set(v)) for k, v in data.items()])

    # Ignore self dependencies
    for k, v in mapper.items():
        v.discard(k)

    # Perform the topological sorting
    extra_items_in_deps = reduce(set.union, mapper.values()) - set(mapper)
    mapper.update(OrderedDict([(item, set()) for item in extra_items_in_deps]))
    while True:
        ordered = set(item for item, dep in mapper.items() if not dep)
        if not ordered:
            break
        processed = filter_sorted(ordered) + processed
        mapper = OrderedDict([(item, (dep - ordered)) for item, dep in mapper.items()
                              if item not in ordered])
    if len(processed) != len(set(flatten(data) + flatten(data.values()))):
        raise ValueError("A cyclic dependency exists amongst %r" % data)
    return processed


def numpy_to_ctypes(dtype):
    """Map numpy types to ctypes types."""
    return {np.int32: ctypes.c_int,
            np.float32: ctypes.c_float,
            np.int64: ctypes.c_int64,
            np.float64: ctypes.c_double}[dtype]


def ctypes_to_C(ctype):
    """Map ctypes types to C types."""
    if issubclass(ctype, ctypes.Structure):
        return 'struct %s' % ctype.__name__
    elif issubclass(ctype, ctypes.Union):
        return 'union %s' % ctype.__name__
    elif ctype.__name__.startswith('c_'):
        # FIXME: Is there a better way of extracting the C typename ?
        # Here, we're following the ctypes convention that each basic type has
        # the format c_X_p, where X is the C typename, for instance `int` or `float`.
        return ctype.__name__[2:-2]
    else:
        raise TypeError('Unrecognised %s during converstion to C type' % str(ctype))


def ctypes_pointer(name):
    """Create a ctypes type representing a C pointer to a custom data type ``name``."""
    return type("c_%s_p" % name, (ctypes.c_void_p,), {})


def pprint(node, verbose=True):
    """
    Shortcut to pretty print Iteration/Expression trees.
    """
    from devito.ir.iet import printAST
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


class change_directory(object):
    """
    Context manager for changing the current working directory.

    Adapted from: ::

        https://stackoverflow.com/questions/431684/how-do-i-cd-in-python/
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def sweep(parameters, keys=None):
    """
    Generator to create a parameter sweep from a dictionary of values
    or value lists.
    """
    keys = keys or parameters.keys()
    sweep_values = [parameters[key] for key in keys]
    # Ensure all values are iterables to make sweeping safe
    sweep_values = [[v] if isinstance(v, str) or not isinstance(v, Iterable) else v
                    for v in sweep_values]
    for vals in product(*sweep_values):
        yield dict(zip(keys, vals))


class Bunch(object):
    """
    Bind together an arbitrary number of generic items. This is a mutable
    alternative to a ``namedtuple``.

    From: ::

        http://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of\
        -a-bunch-of-named/?in=user-97991
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class EnrichedTuple(tuple):
    """
    A tuple with an arbitrary number of additional attributes.
    """
    def __new__(cls, *items, getters=None, **kwargs):
        obj = super(EnrichedTuple, cls).__new__(cls, items)
        obj.__dict__.update(kwargs)
        obj._getters = dict(zip(getters or [], items))
        return obj

    def __getitem__(self, key):
        if isinstance(key, int):
            return super(EnrichedTuple, self).__getitem__(key)
        else:
            return self._getters[key]


class ReducerMap(MultiDict):
    """
    Specialised :class:`MultiDict` object that maps a single key to a
    list of potential values and provides a reduction method for
    retrieval.
    """

    def update(self, values):
        """
        Update internal mapping with standard dictionary semantics.
        """
        if isinstance(values, Mapping):
            self.extend(values)
        elif isinstance(values, Iterable) and not isinstance(values, str):
            for v in values:
                self.extend(v)
        else:
            self.extend(values)

    def unique(self, key):
        """
        Returns a unique value for a given key, if such a value
        exists, and raises a ``ValueError`` if it does not.

        :param key: Key for which to retrieve a unique value
        """
        candidates = self.getall(key)

        def compare_to_first(v):
            first = candidates[0]
            if isinstance(first, np.ndarray) or isinstance(v, np.ndarray):
                return (first == v).all()
            else:
                return first == v

        if len(candidates) == 1:
            return candidates[0]
        elif all(map(compare_to_first, candidates)):
            return candidates[0]
        else:
            error("Unable to find unique value for key %s, candidates: %s" %
                  (key, candidates))
            raise ValueError('Inconsistent values for key reduction')

    def reduce(self, key, op=None):
        """
        Returns a reduction of all candidate values for a given key.

        :param key: Key for which to retrieve candidate values
        :param op: Operator for reduction among candidate values.
                   If not provided, a unique value will be returned,
                   or a ``ValueError`` raised if no unique value exists.
        """
        if op is None:
            # Return a unique value if it exists
            return self.unique(key)
        else:
            return reduce(op, self.getall(key))

    def reduce_all(self):
        """
        Returns a dictionary with reduced/unique values for all keys.
        """
        return {k: self.reduce(key=k) for k in self}


class Tag(ABC):

    """
    An abstract class to define categories of object decorators.

    .. note::

        This class must be subclassed for each new category.
    """

    _repr = 'AbstractTag'

    _KNOWN = []

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

        self._KNOWN.append(self)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.name, self.val))

    def __str__(self):
        return self.name if self.val is None else '%s%s' % (self.name, str(self.val))

    def __repr__(self):
        if self.val is None:
            return "%s: %s" % (self._repr, self.name)
        else:
            return "%s: %s[%s]" % (self._repr, self.name, str(self.val))
