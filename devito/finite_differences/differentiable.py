from collections import ChainMap

import sympy
from sympy.functions.elementary.integers import floor
from sympy.core.evalf import evalf_table

from cached_property import cached_property

from devito.tools import filter_ordered, flatten

__all__ = ['Differentiable']


class Differentiable(sympy.Expr):
    """
    This class represents Devito differentiable objects such as functions,
    sum of functions, product of function, or any FD approximation. Differentiable
    objects provide FD shortcuts to easily compute FD approximations.
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
    def indices(self):
        return tuple(filter_ordered(flatten(getattr(i, 'indices', ())
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

    def __hash__(self):
        return super(Differentiable, self).__hash__()

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        .. note::

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

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative wrt. all spatial dimensions.
        """
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return Add(*[getattr(self, d) for d in derivs])

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian
        wrt. all spatial dimensions.
        """
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return sum([getattr(self.laplace * weight, d) for d in derivs])

    def collect(self, *args, **kwargs):
        return make_diff(super(Differentiable, self).collect(*args, **kwargs))


def make_diff(expr):
    """
    Recursively reconstruct ``expr`` using objects of type Differentiable, such
    as Differentiable.Add/Differentiable.Mul/Differentiable.Pow/... replacing any
    sympy.Add/sympy.Mul/sympy.Pow/...

    Notes
    -----
    Using this function is quite horrible, but it's the only work around
    to inherent limitations within SymPy.
    """
    if expr.is_Function or expr.is_Atom:
        return expr
    elif type(expr) in classmap:
        return classmap[type(expr)](*[make_diff(i) for i in expr.args], evaluate=False)
    else:
        return expr.func(*[make_diff(i) for i in expr.args], evaluate=False)


class Add(sympy.Add, Differentiable):

    def __new__(cls, *args, **kwargs):
        obj = sympy.Add.__new__(cls, *args, **kwargs)

        # `(f + f)` is evaluated as `2*f`, with `*` being a sympy.Mul.
        # Here we make sure to return our own Mul.
        if obj.is_Mul:
            obj = Mul(*obj.args)

        return obj


class Mul(sympy.Mul, Differentiable):

    def __new__(cls, *args, **kwargs):
        obj = sympy.Mul.__new__(cls, *args, **kwargs)

        # `(f + g)*2` is evaluated as `2*f + 2*g`, with `+` being a sympy.Add.
        # Here we make sure to return our own Add.
        if obj.is_Add:
            obj = Add(*obj.args)

        # `(f * f)` is evaluated as `f**2`, with `**` being a sympy.Pow.
        # Here we make sure to return our own Pow.
        if obj.is_Pow:
            obj = Pow(*obj.args)

        return obj


class Pow(sympy.Pow, Differentiable):
    pass


class Mod(sympy.Mod, Differentiable):
    pass


classmap = {}
classmap[sympy.Add] = Add
classmap[sympy.Mul] = Mul
classmap[sympy.Pow] = Pow
classmap[sympy.Mod] = Mod

# Make sure `sympy.evalf` knows how to evaluate the inherited classes
# Without these, `evalf` would rely on a much slower, much more generic, and
# thus much more time-inefficient fallback routine. This would hit us
# pretty badly when taking derivatives (see `finite_difference.py`), where
# `evalf` is used systematically
evalf_table[Add] = evalf_table[sympy.Add]
evalf_table[Mul] = evalf_table[sympy.Mul]
evalf_table[Pow] = evalf_table[sympy.Pow]
