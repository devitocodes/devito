import numpy as np
import os
import ctypes
from collections import Callable, Iterable, OrderedDict, Hashable
from functools import partial, wraps
from itertools import product
from subprocess import PIPE, Popen
import cpuinfo
try:
    from itertools import izip_longest as zip_longest
except ImportError:
    # Python3.5 compatibility
    from itertools import zip_longest

from devito.parameters import configuration

__all__ = ['memoized', 'infer_cpu', 'sweep', 'silencio']


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


def grouper(iterable, n):
    """Split an interable into groups of size n, plus a reminder"""
    args = [iter(iterable)] * n
    return ([e for e in t if e is not None] for t in zip_longest(*args))


def roundm(x, y):
    """Return x rounded up to the closest multiple of y."""
    return x if x % y == 0 else x + y - x % y


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
    """Filter elements in a list and sort them by key"""
    return sorted(filter_ordered(elements, key=key), key=key)


def partial_order(elements):
    """Compute a partial order for the items in ``elements``. If a partial order
    cannot be established, return the empty list. If multiple partial orderings are
    possible, determinism in ensured."""
    elements = [i for i in elements if i]

    # Compute items dependencies
    mapper = OrderedDict()
    for i in elements:
        shifted = list(i)
        last = shifted.pop()
        for j, k in zip(shifted, i[1:]):
            handle = mapper.setdefault(j, [])
            if k not in handle:
                handle.append(k)
        mapper.setdefault(last, [])

    # In a partially ordered set, there can be no cyclic dependencies amongst
    # items, so there must always be at least one root.
    roots = OrderedDict([(k, v) for k, v in mapper.items()
                         if k not in flatten(mapper.values())])

    # Start by queuing up roots
    ordering = []
    queue = []
    for k, v in roots.items():
        ordering.append(k)
        queue += [(i, k) for i in v]

    # Compute the partial orders from roots
    while queue:
        item, prev = queue.pop(0)
        if item not in ordering:
            ordering.append(item)
        queue = [(i, item) for i in mapper[item]] + queue

    return ordering


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


class memoized(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    Adapted from: ::

        https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)


default_isa = 'cpp'
default_platform = 'intel64'


def infer_cpu():
    """
    Detect the highest Instruction Set Architecture and the platform
    codename using cpu flags and/or leveraging other tools. Return default
    values if the detection procedure was unsuccesful.
    """
    info = cpuinfo.get_cpu_info()
    # ISA
    isa = default_isa
    for i in reversed(configuration._accepted['isa']):
        if i in info['flags']:
            isa = i
            break
    # Platform
    try:
        # First, try leveraging `gcc`
        p1 = Popen(['gcc', '-march=native', '-Q', '--help=target'], stdout=PIPE)
        p2 = Popen(['grep', 'march'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        output, _ = p2.communicate()
        platform = output.decode("utf-8").split()[1]
    except:
        # Then, try infer from the brand name, otherwise fallback to default
        try:
            mapper = {'v3': 'haswell', 'v4': 'broadwell', 'v5': 'skylake'}
            cpu_iteration = info['brand'].split()[4]
            platform = mapper[cpu_iteration]
        except:
            platform = None
    # Is it a known platform?
    if platform not in configuration._accepted['platform']:
        platform = default_platform
    return isa, platform


class silencio(object):
    """
    Decorator to temporarily change log levels.
    """

    def __init__(self, log_level='WARNING'):
        self.log_level = log_level

    def __call__(self, func, *args, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previous = configuration['log_level']
            configuration['log_level'] = self.log_level
            result = func(*args, **kwargs)
            configuration['log_level'] = previous
            return result
        return wrapper


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
