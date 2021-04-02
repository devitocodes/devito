from collections import OrderedDict, defaultdict
from functools import partial
from threading import get_ident
from time import time

__all__ = ['timed_pass', 'timed_region']


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

    stack = defaultdict(list)
    """
    A ``thread_id -> stack`` mapper, to keep track of nested `timed_pass`.
    """

    def __new__(cls, *args, name=None):
        if args:
            # The typical use case:
            #
            # @timed_pass
            # def foo(...)
            if len(args) == 1:
                func, = args
            elif len(args) == 2:
                assert name is None
                func, name = args
            else:
                assert False
            obj = object.__new__(cls)
            obj.__init__(func, name)
            return obj
        else:
            # Handle the case:
            #
            # @timed_pass(name='X')
            # def foo(...)
            def wrapper(func):
                return timed_pass(func, name)
            return wrapper

    def __init__(self, func, name=None):
        self.func = func
        self.name = name

    @classmethod
    def is_enabled(cls):
        return isinstance(cls.timings.get(get_ident()), dict)

    def __call__(self, *args, **kwargs):
        tid = get_ident()

        timings = timed_pass.timings.get(tid)
        if not isinstance(timings, dict):
            raise ValueError("Attempting to use `timed_pass` outside a `timed_region`")

        if self.name is not None:
            frame = self.name
        else:
            frame = self.func.__name__

        stack = timed_pass.stack[tid]
        stack.append(frame)

        tic = time()
        retval = self.func(*args, **kwargs)
        toc = time()

        for f in stack:
            timings = timings.setdefault(f, {})
        if 'total' in timings:
            timings['total'] += toc - tic
        else:
            timings['total'] = toc - tic

        stack.pop()

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
        try:
            # Necessary clean up should one be constructing an Operator within
            # a try-except, with the Operator construction failing
            del timed_pass.stack[get_ident()]
        except KeyError:
            # Typically we end up here
            pass
