from collections import Iterable, OrderedDict

from devito.symbolics import retrieve_terminals, uxreplace
from devito.tools import flatten, timed_pass
from devito.types import Dimension, Symbol

__all__ = ['cluster_pass', 'makeit_ssa', 'make_is_time_invariant']


class cluster_pass(object):

    def __new__(cls, *args, mode='dense'):
        if args:
            if len(args) == 1:
                func, = args
            elif len(args) == 2:
                func, mode = args
            else:
                assert False
            obj = object.__new__(cls)
            obj.__init__(func, mode)
            return obj
        else:
            def wrapper(func):
                return cluster_pass(func, mode)
            return wrapper

    def __init__(self, func, mode='dense'):
        self.func = func

        if mode == 'dense':
            self.cond = lambda c: c.is_dense
        elif mode == 'sparse':
            self.cond = lambda c: not c.is_dense
        else:
            self.cond = lambda c: True

    def __call__(self, *args):
        if timed_pass.is_enabled():
            maybe_timed = lambda *_args: timed_pass(self.func, self.func.__name__)(*_args)
        else:
            maybe_timed = lambda *_args: self.func(*_args)
        args = list(args)
        maybe_clusters = args.pop(0)
        if isinstance(maybe_clusters, Iterable):
            # Instance method
            processed = [maybe_timed(c, *args) if self.cond(c) else c
                         for c in maybe_clusters]
        else:
            # Pure function
            self = maybe_clusters
            clusters = args.pop(0)
            processed = [maybe_timed(self, c, *args) if self.cond(c) else c
                         for c in clusters]
        return flatten(processed)


def makeit_ssa(exprs):
    """
    Convert an iterable of Eqs into Static Single Assignment (SSA) form.
    """
    # Identify recurring LHSs
    seen = {}
    for i, e in enumerate(exprs):
        seen.setdefault(e.lhs, []).append(i)
    # Optimization: don't waste time reconstructing stuff if already in SSA form
    if all(len(i) == 1 for i in seen.values()):
        return exprs
    # SSA conversion
    c = 0
    mapper = {}
    processed = []
    for i, e in enumerate(exprs):
        where = seen[e.lhs]
        rhs = uxreplace(e.rhs, mapper)
        if len(where) > 1:
            needssa = e.is_Scalar or where[-1] != i
            lhs = Symbol(name='ssa%d' % c, dtype=e.dtype) if needssa else e.lhs
            if e.is_Increment:
                # Turn AugmentedAssignment into Assignment
                processed.append(e.func(lhs, mapper[e.lhs] + rhs, is_Increment=False))
            else:
                processed.append(e.func(lhs, rhs))
            mapper[e.lhs] = lhs
            c += 1
        else:
            processed.append(e.func(e.lhs, rhs))
    return processed


def make_is_time_invariant(context):
    """
    Given an ordered list of expressions, returns a callable that finds out whether
    a given expression is time invariant or not.
    """
    dimensions = set().union(*[e.dimensions for e in context])
    if len([i for i in dimensions if i.is_Time]) == 0:
        # No concept of time in the provided set of expressions
        return lambda i: False

    mapper = OrderedDict([(i.lhs, i) for i in makeit_ssa(context)])

    def is_time_invariant(mapper, expr):
        if any(isinstance(i, Dimension) and i.is_Time for i in expr.free_symbols):
            return False

        queue = [expr.rhs if expr.is_Equality else expr]
        seen = set()
        while queue:
            item = queue.pop()
            nodes = set()
            for i in retrieve_terminals(item):
                if i in seen:
                    # Already inspected, nothing more can be inferred
                    continue
                elif any(isinstance(j, Dimension) and j.is_Time for j in i.free_symbols):
                    # Definitely not time-invariant
                    return False
                elif i in mapper:
                    # Go on with the search
                    nodes.add(i)
                elif isinstance(i, Dimension):
                    # Go on with the search, as `i` is not a time dimension
                    pass
                elif not i.function.is_DiscreteFunction:
                    # It didn't come from the outside and it's not in `mapper`, so
                    # cannot determine if time-invariant; assume time-varying then
                    return False
                seen.add(i)
            queue.extend([mapper[i].rhs for i in nodes])
        return True

    callback = lambda i: is_time_invariant(mapper, i)

    return callback
