from abc import ABC
import os
import inspect
import ctypes
from collections import Callable, Iterable, OrderedDict, Hashable, Mapping
from decorator import decorator
from functools import partial, wraps, reduce
from itertools import chain, combinations, product, zip_longest
from operator import attrgetter, mul
from subprocess import DEVNULL, PIPE, Popen, CalledProcessError, check_output

import cpuinfo
import numpy as np
from distutils import version

from multidict import MultiDict

from devito.logger import error
from devito.parameters import configuration

__all__ = ['memoized_func', 'memoized_meth', 'infer_cpu', 'sweep', 'silencio']


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


class memoized_func(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated). This decorator may also be used on class methods,
    but it will cache at the class level; to cache at the instance level,
    use ``memoized_meth``.

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


class memoized_meth(object):
    """
    Decorator. Cache the return value of a class method.

    Unlike ``memoized_func``, the return value of a given method invocation
    will be cached on the instance whose method was invoked. All arguments
    passed to a method decorated with memoize must be hashable.

    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method: ::

        class Obj(object):
            @memoize
            def add_to(self, arg):
                return self + arg
        Obj.add_to(1) # not enough arguments
        Obj.add_to(1, 2) # returns 3, result is not cached

    Adapted from: ::

        code.activestate.com/recipes/577452-a-memoize-decorator-for-instance-methods/
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)

    def __call__(self, *args, **kw):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args)
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


def infer_cpu():
    """
    Detect the highest Instruction Set Architecture and the platform
    codename using cpu flags and/or leveraging other tools. Return default
    values if the detection procedure was unsuccesful.
    """
    info = cpuinfo.get_cpu_info()
    # ISA
    isa = configuration._defaults['isa']
    for i in reversed(configuration._accepted['isa']):
        if any(j.startswith(i) for j in info['flags']):
            # Using `startswith`, rather than `==`, as a flag such as 'avx512'
            # appears as 'avx512f, avx512cd, ...'
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
        # Full list of possible /platform/ values at this point at:
        # https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
        platform = {'sandybridge': 'snb', 'ivybridge': 'ivb', 'haswell': 'hsw',
                    'broadwell': 'bdw', 'skylake': 'skx', 'knl': 'knl'}[platform]
    except:
        # Then, try infer from the brand name, otherwise fallback to default
        try:
            platform = info['brand'].split()[4]
            platform = {'v2': 'ivb', 'v3': 'hsw', 'v4': 'bdw', 'v5': 'skx'}[platform]
        except:
            platform = None
    # Is it a known platform?
    if platform not in configuration._accepted['platform']:
        platform = configuration._defaults['platform']
    return isa, platform


def sniff_compiler_version(cc):
    """
    Try to detect the compiler version.

    Adapted from: ::

        https://github.com/OP2/PyOP2/
    """
    try:
        ver = check_output([cc, "--version"]).decode("utf-8")
    except (CalledProcessError, UnicodeDecodeError):
        return version.LooseVersion("unknown")

    if ver.startswith("gcc"):
        compiler = "gcc"
    elif ver.startswith("clang"):
        compiler = "clang"
    elif ver.startswith("Apple LLVM"):
        compiler = "clang"
    elif ver.startswith("icc"):
        compiler = "icc"
    else:
        compiler = "unknown"

    ver = version.LooseVersion("unknown")
    if compiler in ["gcc", "icc"]:
        try:
            # gcc-7 series only spits out patch level on dumpfullversion.
            ver = check_output([cc, "-dumpfullversion"], stderr=DEVNULL).decode("utf-8")
            ver = version.StrictVersion(ver.strip())
        except CalledProcessError:
            try:
                ver = check_output([cc, "-dumpversion"], stderr=DEVNULL).decode("utf-8")
                ver = version.StrictVersion(ver.strip())
            except (CalledProcessError, UnicodeDecodeError):
                pass
        except UnicodeDecodeError:
            pass

    # Pure integer versions (e.g., ggc5, rather than gcc5.0) need special handling
    try:
        ver = version.StrictVersion(float(ver))
    except TypeError:
        pass

    return ver


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


class GenericVisitor(object):

    """
    A generic visitor.

    To define handlers, subclasses should define :data:`visit_Foo`
    methods for each class :data:`Foo` they want to handle.
    If a specific method for a class :data:`Foo` is not found, the MRO
    of the class is walked in order until a matching method is found.

    The method signature is:

        .. code-block::
           def visit_Foo(self, o, [*args, **kwargs]):
               pass

    The handler is responsible for visiting the children (if any) of
    the node :data:`o`.  :data:`*args` and :data:`**kwargs` may be
    used to pass information up and down the call stack.  You can also
    pass named keyword arguments, e.g.:

        .. code-block::
           def visit_Foo(self, o, parent=None, *args, **kwargs):
               pass
    """

    def __init__(self):
        handlers = {}
        # visit methods are spelt visit_Foo.
        prefix = "visit_"
        # Inspect the methods on this instance to find out which
        # handlers are defined.
        for (name, meth) in inspect.getmembers(self, predicate=inspect.ismethod):
            if not name.startswith(prefix):
                continue
            # Check the argument specification
            # Valid options are:
            #    visit_Foo(self, o, [*args, **kwargs])
            argspec = inspect.getargspec(meth)
            if len(argspec.args) < 2:
                raise RuntimeError("Visit method signature must be "
                                   "visit_Foo(self, o, [*args, **kwargs])")
            handlers[name[len(prefix):]] = meth
        self._handlers = handlers

    """
    :attr:`default_args`. A dict of default keyword arguments for the visitor.
    These are not used by default in :meth:`visit`, however, a caller may pass
    them explicitly to :meth:`visit` by accessing :attr:`default_args`.
    For example::

        .. code-block::
           v = FooVisitor()
           v.visit(node, **v.default_args)
    """
    default_args = {}

    @classmethod
    def default_retval(cls):
        """A method that returns an object to use to populate return values.

        If your visitor combines values in a tree-walk, it may be useful to
        provide a object to combine the results into. :meth:`default_retval`
        may be defined by the visitor to be called to provide an empty object
        of appropriate type.
        """
        return None

    def lookup_method(self, instance):
        """Look up a handler method for a visitee.

        :param instance: The instance to look up a method for.
        """
        cls = instance.__class__
        try:
            # Do we have a method handler defined for this type name
            return self._handlers[cls.__name__]
        except KeyError:
            # No, walk the MRO.
            for klass in cls.mro()[1:]:
                entry = self._handlers.get(klass.__name__)
                if entry:
                    # Save it on this type name for faster lookup next time
                    self._handlers[cls.__name__] = entry
                    return entry
        raise RuntimeError("No handler found for class %s", cls.__name__)

    def visit(self, o, *args, **kwargs):
        """Apply this :class:`Visitor` to an AST.

            :param o: The :class:`Node` to visit.
            :param args: Optional arguments to pass to the visit methods.
            :param kwargs: Optional keyword arguments to pass to the visit methods.
        """
        meth = self.lookup_method(o)
        return meth(o, *args, **kwargs)

    def visit_object(self, o, **kwargs):
        return self.default_retval()


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


# Method/function arguments validation

class validate_base(object):

    """Decorator to validate arguments

    Formal parameters that don't exist in the definition of the function
    being decorated as well as actual arguments not being present when
    the validation is called are silently ignored.

    Readapted from: ::

        https://github.com/OP2/PyOP2/
    """

    def __init__(self, *checks):
        self._checks = checks

    def __call__(self, f):
        def wrapper(f, *args, **kwargs):
            if configuration["develop-mode"]:
                self.nargs = f.__code__.co_argcount
                self.defaults = f.__defaults__ or ()
                self.varnames = f.__code__.co_varnames
                self.file = f.__code__.co_filename
                self.line = f.__code__.co_firstlineno + 1
                self.check_args(args, kwargs)
            return f(*args, **kwargs)
        return decorator(wrapper, f)

    def check_args(self, args, kwargs):
        for argname, argcond, exception in self._checks:
            # If the argument argname is not present in the decorated function
            # silently ignore it
            try:
                i = self.varnames.index(argname)
            except ValueError:
                # No formal parameter argname
                continue
            # Try the argument by keyword first, and by position second.
            # If the argument isn't given, silently ignore it.
            try:
                arg = kwargs.get(argname)
                arg = arg or args[i]
            except IndexError:
                # No actual parameter argname
                continue
            # If the argument has a default value, also accept that (since the
            # constructor will be able to deal with that)
            default_index = i - self.nargs + len(self.defaults)
            if default_index >= 0 and arg == self.defaults[default_index]:
                continue
            self.check_arg(arg, argcond, exception)


class validate_type(validate_base):

    """
    Decorator to validate argument types

    The decorator expects one or more arguments, which are 3-tuples of
    (name, type, exception), where name is the argument name in the
    function being decorated, type is the argument type to be validated
    and exception is the exception type to be raised if validation fails.

    Readapted from: ::

        https://github.com/OP2/PyOP2/
    """

    def __init__(self, *checks):
        processed = []
        for i in checks:
            try:
                argname, argtype = i
                processed.append((argname, argtype, TypeError))
            except ValueError:
                processed.append(i)
        super(validate_type, self).__init__(*processed)

    def check_arg(self, arg, argtype, exception):
        if not isinstance(arg, argtype):
            raise exception("%s:%d Parameter %s must be of type %r"
                            % (self.file, self.line, arg, argtype))
