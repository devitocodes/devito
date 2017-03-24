from __future__ import absolute_import

from functools import reduce
from operator import mul

from sympy import finite_diff_weights, symbols

from devito.dimension import x, y
from devito.logger import error

__all__ = ['first_derivative', 'second_derivative', 'cross_derivative',
           'left', 'right', 'centered']

# Default spacing symbol
h = symbols('h')


class Transpose(object):
    """Class that defines if the derivative is itself or adjoint (transpose).
    This only matter for odd order derivatives that requires
    a minus sign for the transpose."""
    def __init__(self, transpose):
        self._transpose = transpose

    def __eq__(self, other):
        return self._transpose == other._transpose

    def __repr__(self):
        return {1: 'direct', -1: 'transpose'}[self._transpose]


direct = Transpose(1)
transpose = Transpose(-1)


class Side(object):
    """Class encapsulating the side of the shift for derivatives."""

    def __init__(self, side):
        self._side = side

    def __eq__(self, other):
        return self._side == other._side

    def __repr__(self):
        return {-1: 'left', 0: 'centered', 1: 'right'}[self._side]

    def adjoint(self, matvec):
        if matvec == direct:
            return self
        else:
            if self == centered:
                return centered
            elif self == right:
                return left
            elif self == left:
                return right
            else:
                error("Unsupported side value")


left = Side(-1)
right = Side(1)
centered = Side(0)


def second_derivative(*args, **kwargs):
    """Derives second derivative for a product of given functions.

    :param \*args: All positional arguments must be fully qualified
       function objects, eg. `f(x, y)` or `g(t, x, y, z)`.
    :param dim: Symbol defininf the dimension wrt. which to
       differentiate, eg. `x`, `y`, `z` or `t`.
    :param diff: Finite Difference symbol to insert, default `h`.
    :param order: Discretisation order of the stencil to create.
    :returns: The second derivative

    Example: Deriving the second derivative of f(x, y)*g(x, y) wrt. x via
       ``second_derivative(f(x, y), g(x, y), order=2, dim=x)``
       results in ``(-2.0*f(x, y)*g(x, y) + 1.0*f(-h + x, y)*g(-h + x, y) +
       1.0*f(h + x, y)*g(h + x, y)) / h**2``.
    """
    order = kwargs.get('order', 2)
    dim = kwargs.get('dim', x)
    diff = kwargs.get('diff', h)

    ind = [(dim + i * diff) for i in range(-int(order / 2),
                                           int(order / 2) + 1)]

    coeffs = finite_diff_weights(2, ind, dim)[-1][-1]
    deriv = 0

    for i in range(0, len(ind)):
            var = [a.subs({dim: ind[i]}) for a in args]
            deriv += coeffs[i] * reduce(mul, var, 1)
    return deriv


def cross_derivative(*args, **kwargs):
    """Derives cross derivative for a product of given functions.

    :param \*args: All positional arguments must be fully qualified
       function objects, eg. `f(x, y)` or `g(t, x, y, z)`.
    :param dims: 2-tuple of symbols defining the dimension wrt. which
       to differentiate, eg. `x`, `y`, `z` or `t`.
    :param diff: Finite Difference symbol to insert, default `h`.
    :returns: The cross derivative

    Example: Deriving the cross-derivative of f(x, y)*g(x, y) wrt. x and y via:
       ``cross_derivative(f(x, y), g(x, y), dims=(x, y))``
       results in:
       ``0.5*(-2.0*f(x, y)*g(x, y) + f(x, -h + y)*g(x, -h + y) +``
       ``f(x, h + y)*g(x, h + y) + f(-h + x, y)*g(-h + x, y) -``
       ``f(-h + x, h + y)*g(-h + x, h + y) + f(h + x, y)*g(h + x, y) -``
       ``f(h + x, -h + y)*g(h + x, -h + y)) / h**2``
    """
    dims = kwargs.get('dims', (x, y))
    diff = kwargs.get('diff', h)
    order = kwargs.get('order', 1)

    assert(isinstance(dims, tuple) and len(dims) == 2)
    deriv = 0

    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    ind1r = [(dims[0] + i * diff)
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind2r = [(dims[1] + i * diff)
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind1l = [(dims[0] - i * diff)
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind2l = [(dims[1] - i * diff)
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]

    # Finite difference weights from Taylor approximation with this positions
    c1 = finite_diff_weights(1, ind1r, dims[0])
    c1 = c1[-1][-1]
    c2 = finite_diff_weights(1, ind1l, dims[0])
    c2 = c2[-1][-1]

    # Diagonal elements
    for i in range(0, len(ind1r)):
        for j in range(0, len(ind2r)):
            var1 = [a.subs({dims[0]: ind1r[i], dims[1]: ind2r[j]}) for a in args]
            var2 = [a.subs({dims[0]: ind1l[i], dims[1]: ind2l[j]}) for a in args]
            deriv += (.5 * c1[i] * c1[j] * reduce(mul, var1, 1) +
                      .5 * c2[-(j+1)] * c2[-(i+1)] * reduce(mul, var2, 1))

    return -deriv


def first_derivative(*args, **kwargs):
    """Derives first derivative for a product of given functions.

    :param \*args: All positional arguments must be fully qualified
       function objects, eg. `f(x, y)` or `g(t, x, y, z)`.
    :param dims: symbol defining the dimension wrt. which
       to differentiate, eg. `x`, `y`, `z` or `t`.
    :param diff: Finite Difference symbol to insert, default `h`.
    :param side: Side of the shift for the first derivatives.
    :returns: The first derivative

    Example: Deriving the first-derivative of f(x)*g(x) wrt. x via:
       ``cross_derivative(f(x), g(x), dim=x, side=1, order=1)``
       results in:
       ``*(-f(x)*g(x) + f(x + h)*g(x + h) ) / h``
    """
    dim = kwargs.get('dim', x)
    diff = kwargs.get('diff', h)
    order = int(kwargs.get('order', 1))
    matvec = kwargs.get('matvec', direct)
    side = kwargs.get('side', centered).adjoint(matvec)
    deriv = 0
    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    if side == right:
        ind = [(dim + i * diff) for i in range(-int(order / 2) + 1 - (order % 2),
                                               int((order + 1) / 2) + 2 - (order % 2))]
    elif side == left:
        ind = [(dim - i * diff) for i in range(-int(order / 2) + 1 - (order % 2),
                                               int((order + 1) / 2) + 2 - (order % 2))]
    else:
        ind = [(dim + i * diff) for i in range(-int(order / 2),
                                               int((order + 1) / 2) + 1)]
    # Finite difference weights from Taylor approximation with this positions
    c = finite_diff_weights(1, ind, dim)
    c = c[-1][-1]

    # Loop through positions
    for i in range(0, len(ind)):
            var = [a.subs({dim: ind[i]}) for a in args]
            deriv += c[i] * reduce(mul, var, 1)
    return matvec._transpose*deriv
