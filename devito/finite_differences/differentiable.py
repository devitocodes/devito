import sympy
from sympy.functions.elementary.integers import floor
import numpy as np

from devito.tools import filter_ordered

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

    _state = ('space_order', 'time_order', 'staggered', 'dtype', 'indices', 'grid')

    def __new_diff__(cls, obj, *args):
        for k, v in merge_fd_properties(*args).items():
            setattr(obj, "_%s" % k, v)

    # FD properties
    # TODO: Since Function (rightfully) inherits from Differentiable, some
    # of these properties below can be removed from Function... eg, space_order,
    # time_order, staggered, ...
    @property
    def space_order(self):
        return self._space_order

    @property
    def time_order(self):
        return self._time_order

    @property
    def staggered(self):
        return self._staggered

    @property
    def indices(self):
        return self._indices

    @property
    def fd(self):
        return self._fd

    @property
    def grid(self):
        return self._grid

    @property
    def dtype(self):
        return self._dtype

    def __hash__(self):
        return super(Differentiable, self).__hash__()

    def __getattr__(self, name):
        """
        __getattr__ has two cases: ::

            * Fetch a "conventional" property using the standard '__getattribute__', or
            * Call a dynamically created FD shortcut, stored as a partial object in
              ``self._fd``..
        """
        if name == 'fd' or name == '_fd':
            raise AttributeError
        elif self.__dict__.get('_fd'):
            if name in self.fd:
                # self.fd[name] = (property, description), calls self.fd[name][0]
                return self.fd[name][0](self)
            else:
                return self.__getattribute__(name)
        else:
            return self.__getattribute__(name)

    # Override SymPy arithmetic operators
    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Add(self, -other)

    def __rsub__(self, other):
        return Add(other, -self)

    def __mul__(self, other):
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

    def __new__(cls, *args):
        obj = sympy.Add.__new__(cls, *args)
        Differentiable.__new_diff__(cls, obj, *args)
        return obj


class Mul(sympy.Mul, Differentiable):

    def __new__(cls, *args):
        obj = sympy.Mul.__new__(cls, *args)
        Differentiable.__new_diff__(cls, obj, *args)
        return obj


class Pow(sympy.Pow, Differentiable):

    def __new__(cls, *args):
        obj = sympy.Pow.__new__(cls, *args)
        Differentiable.__new_diff__(cls, obj, *args)
        return obj


class Mod(sympy.Mod, Differentiable):

    def __new__(cls, *args):
        obj = sympy.Mod.__new__(cls, *args)
        Differentiable.__new_diff__(cls, obj, *args)
        return obj


def merge_fd_properties(f0, f1):
    """
    Combine the FD properties of two :class:`Differentiable` objects into a
    single dictionary.

    .. note::

        ``space_order`` and ``time_order`` default to 100 as "infinitely"
        differentiable; if not provided, it is assumed that the Differentiable
        object is independent of space or time.
    """
    # TODO: Generalize to combine from an arbitrary number of expressions,
    # ie allow to pass in ``*function`` rather than just ``f0`` and ``f1``
    merged = {}
    merged["space_order"] = np.min([getattr(f0, 'space_order', 100) or 100,
                                    getattr(f1, 'space_order', 100)])
    merged["time_order"] = np.min([getattr(f0, 'time_order', 100) or 100,
                                   getattr(f1, 'time_order', 100)])
    merged["indices"] = tuple(filter_ordered(getattr(f0, 'indices', ()) +
                                             getattr(f1, 'indices', ())))
    merged["fd"] = dict(getattr(f0, 'fd', {}), **getattr(f1, 'fd', {}))
    # TODO: Assert staddered and dtype are identical here?
    merged["staggered"] = getattr(f0, 'staggered', getattr(f1, 'staggered', ()))
    merged["dtype"] = getattr(f0, 'dtype', getattr(f1, 'dtype', None))
    # TODO: Adding grid, is this OK? shold we assert they are identical too?
    merged["grid"] = getattr(f0, 'grid', getattr(f1, 'grid', None))
    return merged
