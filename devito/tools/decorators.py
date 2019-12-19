from collections import OrderedDict
from decorator import decorator
from functools import partial
from threading import get_ident
from time import time

__all__ = ['validate_type', 'timed_pass', 'timed_region']


class validate_base(object):

    """
    Decorator to validate arguments.

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
            from devito.parameters import configuration
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
    Decorator to validate argument types.

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


class timed_pass(object):

    """
    A decorator to record the timing of functions or methods.
    """

    timings = {}
    """
    A ``thread_id -> timings`` mapper. The only reason we key the timings by
    thread id is to support multi-threaded codes that want to have separate
    Python threads compiling multiple Operators in parallel.
    """

    def __init__(self, func, name=None):
        self.func = func
        self.name = name

    @classmethod
    def is_enabled(cls):
        return isinstance(cls.timings.get(get_ident()), dict)

    def __call__(self, *args, **kwargs):
        timings = timed_pass.timings.get(get_ident())
        if not isinstance(timings, dict):
            raise ValueError("Attempting to use `timed_pass` outside a `timed_region`")
        tic = time()
        retval = self.func(*args, **kwargs)
        toc = time()
        if self.name is not None:
            key = self.name
        else:
            key = self.func.__name__
        if key in timings:
            timings[key] += toc - tic
        else:
            timings[key] = toc - tic
        return retval

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)


class timed_region(object):

    """
    A context manager for code regions in which the `timed_pass` decorator is used.
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if isinstance(timed_pass.timings.get(get_ident()), dict):
            raise ValueError("Cannot nest `timed_region`")
        self.timings = OrderedDict()
        timed_pass.timings[get_ident()] = self.timings
        self.tic = time()
        return self

    def __exit__(self, *args):
        self.timings[self.name] = time() - self.tic
        del timed_pass.timings[get_ident()]
