import ctypes
from collections import Iterable, OrderedDict
from functools import reduce
from itertools import chain, combinations, product, zip_longest
from operator import attrgetter, mul

import numpy as np
import sympy

__all__ = ['prod', 'as_tuple', 'is_integer', 'generator', 'grouper', 'split', 'roundm',
           'powerset', 'invert', 'flatten', 'single_or', 'filter_ordered', 'as_mapper',
           'filter_sorted', 'ctypes_to_C', 'ctypes_pointer', 'pprint', 'sweep']


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


def as_mapper(iterable, key=None):
    """
    Rearrange an iterable into a dictionary of lists in which keys are
    produced by the function ``key``.
    """
    if key is None:
        key = lambda i: i
    mapper = {}
    for i in iterable:
        mapper.setdefault(key(i), []).append(i)
    return mapper


def is_integer(value):
    """
    A thorough instance comparison for all integer types.
    """
    return isinstance(value, (int, np.integer, sympy.Integer))


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


def ctypes_to_C(ctype):
    """Map ctypes types to C types."""
    if issubclass(ctype, ctypes.Structure):
        return 'struct %s' % ctype.__name__
    elif issubclass(ctype, ctypes.Union):
        return 'union %s' % ctype.__name__
    elif issubclass(ctype, ctypes._Pointer):
        return '%s*' % ctypes_to_C(ctype._type_)
    elif ctype.__name__.startswith('c_'):
        # A primitive datatype
        # FIXME: Is there a better way of extracting the C typename ?
        # Here, we're following the ctypes convention that each basic type has
        # the format c_X_p, where X is the C typename, for instance `int` or `float`.
        return ctype.__name__[2:-2]
    else:
        # A custom datatype (e.g., a typedef-ed pointer to struct)
        return ctype.__name__


def ctypes_pointer(name):
    """Create a ctypes type representing a C pointer to a custom data type ``name``."""
    return type("c_%s_p" % name, (ctypes.c_void_p,), {})


def pprint(node, verbose=True):
    """
    Shortcut to pretty print Iteration/Expression trees.
    """
    from devito.ir.iet import printAST
    print(printAST(node, verbose))


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
