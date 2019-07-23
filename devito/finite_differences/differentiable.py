from collections import ChainMap

import sympy
from sympy import S

from sympy.functions.elementary.integers import floor
from sympy.core.evalf import evalf_table
from sympy.core.decorators import call_highest_priority

from cached_property import cached_property

from devito.tools import Evaluable, filter_ordered, flatten
from devito.logger import warning

__all__ = ['Differentiable']


class Differentiable(sympy.Expr, Evaluable):

    """
    A Differentiable is an algebric expression involving Functions, which can
    be derived w.r.t. one or more Dimensions.
    """

    # Set the operator priority higher than SymPy (10.0) to force the overridden
    # operators to be used
    _op_priority = sympy.Expr._op_priority + 1.

    _state = ('space_order', 'time_order', 'indices')

    @cached_property
    def _functions(self):
        return frozenset().union(*[i._functions for i in self._args_diff])

    @cached_property
    def _args_diff(self):
        ret = [i for i in self.args if isinstance(i, Differentiable)]
        ret.extend([i.function for i in self.args if i.is_Indexed])
        return tuple(ret)

    @cached_property
    def space_order(self):
        # Default 100 is for "infinitely" differentiable
        return min([getattr(i, 'space_order', 100) or 100 for i in self._args_diff],
                   default=100)

    @cached_property
    def time_order(self):
        # Default 100 is for "infinitely" differentiable
        return min([getattr(i, 'time_order', 100) or 100 for i in self._args_diff],
                   default=100)

    @cached_property
    def is_TimeDependent(self):
        # Default False, True if anything is time dependant in the expression
        return any(getattr(i, 'is_TimeDependent', False) for i in self._args_diff)

    @cached_property
    def is_VectorValued(self):
        # Default False, True if anything is time dependant in the expression
        return any(getattr(i, 'is_VectorValued', False) for i in self._args_diff)

    @cached_property
    def is_TensorValued(self):
        # Default False, True if anything is time dependant in the expression
        return any(getattr(i, 'is_TensorValued', False) for i in self._args_diff)

    @cached_property
    def is_Function(self):
        # Default False, True if anything is time dependant in the expression
        return any(getattr(i, 'is_Function', False) for i in self._args_diff)

    @cached_property
    def grid(self):
        # Default False, True if anything is time dependant in the expression
        grids = [getattr(i, 'grid', None) for i in self._args_diff]
        grid = set(grids)
        grid.discard(None)
        if len(grid) > 1:
            warning("Expression contains multiple grids, returning first found")
        return list(grid)[0]

    @cached_property
    def indices(self):
        return tuple(filter_ordered(flatten(getattr(i, 'indices', ())
                                            for i in self._args_diff)))

    @cached_property
    def dimensions(self):
        return tuple(filter_ordered(flatten(getattr(i, 'dimensions', ())
                                            for i in self._args_diff)))

    @cached_property
    def staggered(self):
        return tuple(filter_ordered(flatten(getattr(i, 'staggered', ())
                                            for i in self._args_diff)))

    @cached_property
    def is_Staggered(self):
        return any([getattr(i, 'is_Staggered', False) for i in self._args_diff])

    @cached_property
    def _fd(self):
        return dict(ChainMap(*[getattr(i, '_fd', {}) for i in self._args_diff]))

    @cached_property
    def _symbolic_functions(self):
        return frozenset([i for i in self._functions if i.coefficients == 'symbolic'])

    @cached_property
    def _uses_symbolic_coefficients(self):
        return bool(self._symbolic_functions)

    def eval_at(self, var):
        if not var.is_Staggered:
            return self
        # print([(a, type(a), getattr(a, 'eval_at', lambda x: a)(var)) for a in self.args])
        return self.func(*[getattr(a, 'eval_at', lambda x: a)(var) for a in self.args])

    def __hash__(self):
        return super(Differentiable, self).__hash__()

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        if name in self._fd:
            return self._fd[name][0](self)
        raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

    # Override SymPy arithmetic operators
    def __add__(self, other):
        return Add(self, other)

    def __iadd__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Add(self, -other)

    def __isub__(self, other):
        return Add(self, -other)

    def __rsub__(self, other):
        return Add(other, -self)

    def __mul__(self, other):
        return Mul(self, other)

    def __imul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    def __div__(self, other):
        return Mul(self, Pow(other, sympy.S.NegativeOne))

    def __rdiv__(self, other):
        return Mul(other, Pow(self, sympy.S.NegativeOne))

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        return floor(self / other)

    def __rfloordiv__(self, other):
        return floor(other / self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __neg__(self):
        return Mul(sympy.S.NegativeOne, self)

    def __eq__(self, other):
        return super(Differentiable, self).__eq__(other) and\
            all(getattr(self, i, None) == getattr(other, i, None) for i in self._state)

    def index(self, dim):
        inds = [self.dimensions[i] for i, d in enumerate(self.dimensions) if d == dim]
        return inds[0]

    @property
    def name(self):
        return "".join(f.name for f in self._functions)

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative w.r.t all spatial Dimensions.
        """
        space_dims = [d for d in self.dimensions if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return Add(*[getattr(self, d) for d in derivs])

    @property
    def div(self):
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s' % d.name for d in space_dims)
        return Add(*[getattr(self, d) for d in derivs])

    @property
    def grad(self):
        from devito.types.tensor import VectorFunction, VectorTimeFunction
        comps = [getattr(self, 'd%s' % d.name) for d in self.dimensions if d.is_Space]
        vec_func = VectorTimeFunction if self.is_TimeDependent else VectorFunction
        return vec_func(name='grad_%s' % self.name, time_order=self.time_order,
                        space_order=self.space_order, components=comps, grid=self.grid)

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian w.r.t.
        all spatial Dimensions.
        """
        space_dims = [d for d in self.dimensions if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return Add(*[getattr(self.laplace * weight, d) for d in derivs])

    def diff(self, *symbols, **assumptions):
        """
        Like ``sympy.diff``, but return a ``devito.Derivative`` instead of a
        ``sympy.Derivative``.
        """
        from devito.finite_differences.derivative import Derivative
        return Derivative(self, *symbols, **assumptions)

    def _has(self, pattern):
        """
        Unlike generic SymPy use cases, in Devito the majority of calls to `_has`
        occur through the finite difference routines passing `sympy.core.symbol.Symbol`
        as `pattern`. Since the generic `_has` can be prohibitively expensive,
        we here quickly handle this special case, while using the superclass' `_has`
        as fallback.
        """
        if isinstance(pattern, type) and issubclass(pattern, sympy.Symbol):
            # Symbols (and subclasses) are the leaves of an expression, and they
            # are promptly available via `free_symbols`. So this is super quick
            return any(isinstance(i, pattern) for i in self.free_symbols)
        return super(Differentiable, self)._has(pattern)


class Add(sympy.Add, Differentiable):
    pass


class Mul(sympy.Mul, Differentiable):
    pass


class Pow(sympy.Pow, Differentiable):
    def __new__(cls, *args, **kwargs):
        obj = sympy.Pow.__new__(cls, *args, **kwargs)
        return obj


class Mod(sympy.Mod, Differentiable):
    def __new__(cls, *args, **kwargs):
        obj = sympy.Mod.__new__(cls, *args, **kwargs)
        return obj


# Make sure `sympy.evalf` knows how to evaluate the inherited classes
# Without these, `evalf` would rely on a much slower, much more generic, and
# thus much more time-inefficient fallback routine. This would hit us
# pretty badly when taking derivatives (see `finite_difference.py`), where
# `evalf` is used systematically
evalf_table[Add] = evalf_table[sympy.Add]
evalf_table[Mul] = evalf_table[sympy.Mul]
evalf_table[Pow] = evalf_table[sympy.Pow]


# Monkey-patch sympy.Mul/sympy.Add/sympy.Pow/...'s __new__ so that we can
# return a devito.Mul/devito.Add/devito.Pow if any of the arguments is
# of type Differentiable
def __new__(cls, *args, **options):
    if cls in __new__.table and any(isinstance(i, Differentiable) for i in args):
        return __new__.__real_new__(__new__.table[cls], *args, **options)
    else:
        return __new__.__real_new__(cls, *args, **options)
__new__.table = {getattr(sympy, i.__name__): i for i in [Add, Mul, Pow, Mod]}  # noqa
__new__.__real_new__ = sympy.Basic.__new__
sympy.Basic.__new__ = __new__
