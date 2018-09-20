from collections import ChainMap

import sympy
from sympy.functions.elementary.integers import floor
import numpy as np
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

    _state = ('space_order', 'time_order', 'dtype', 'indices', 'grid')

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
    def grid(self):
        ret = {getattr(i, 'grid', None) for i in self._args_diff}
        ret = {i for i in ret if i is not None}
        if len(ret) == 1:
            return ret.pop()
        elif len(ret) > 1:
            raise ValueError("Found multiple grids `%s` in `%s`" % (ret, self))
        else:
            return None

    @cached_property
    def dtype(self):
        dtypes = filter_ordered(getattr(i, 'dtype', None) for i in self._args_diff)
        dtypes = {i for i in dtypes if i is not None}
        fdtypes = [i for i in dtypes if np.issubdtype(i, np.floating)]
        if len(dtypes) == 0:
            return None
        elif len(dtypes) == 1:
            return dtypes.pop()
        elif len(fdtypes) > 1:
            raise ValueError("Illegal mixed floating point arithmetic in `%s`" % self)
        elif len(fdtypes) == 1:
            # Floating point arithmetic "wins" over integer arithmetic
            return fdtypes.pop()
        else:
            raise ValueError("Illegal arithmetic in `%s` [mixed integer?]" % self)

    @cached_property
    def _fd(self):
        return dict(ChainMap(*[getattr(i, '_fd', {}) for i in self._args_diff]))

    def __hash__(self):
        return super(Differentiable, self).__hash__()

    def __getattr__(self, name):
        """
        __getattr__ has two cases: ::

            * Fetch a "conventional" property using the standard '__getattribute__', or
            * Call a dynamically created FD shortcut, stored as a partial object in
              ``self._fd``..
        """
        if name == '_fd':
            raise AttributeError
        elif self.__dict__.get('_fd'):
            if name in self._fd:
                # self._fd[name] = (property, description), calls self._fd[name][0]
                return self._fd[name][0](self)
            else:
                return self.__getattribute__(name)
        else:
            return self.__getattribute__(name)

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
        return sum([getattr(self, d) for d in derivs])

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian
        wrt. all spatial dimensions.
        """
        space_dims = [d for d in self.indices if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return sum([getattr(self.laplace * weight, d) for d in derivs])


class Add(sympy.Add, Differentiable):

    def __new__(cls, *args, **kwargs):
        return sympy.Add.__new__(cls, *args, **kwargs)


class Mul(sympy.Mul, Differentiable):

    def __new__(cls, *args, **kwargs):
        return sympy.Mul.__new__(cls, *args, **kwargs)


class Pow(sympy.Pow, Differentiable):

    def __new__(cls, *args, **kwargs):
        return sympy.Pow.__new__(cls, *args, **kwargs)


class Mod(sympy.Mod, Differentiable):

    def __new__(cls, *args, **kwargs):
        return sympy.Mod.__new__(cls, *args, **kwargs)
