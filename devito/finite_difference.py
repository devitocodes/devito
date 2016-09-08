from __future__ import absolute_import

from functools import reduce
from operator import mul

from sympy import finite_diff_weights, symbols

from devito.dimension import x, y

__all__ = ['second_derivative', 'cross_derivative']


# Explicitly derived Finite Difference coefficients for
# symmetric derivative stencils.
fd_coefficients = {
    16: [-3.054844, 1.777778, -0.311111, 0.075421, -0.017677, 0.003481, -0.000518,
         0.000051, -0.000002],
    14: [-3.023594, 1.750000, -0.291667, 0.064815, -0.013258, 0.002121, -0.000227,
         0.000012],
    12: [-2.982778, 1.714286, -0.267857, 0.052910, -0.008929, 0.001039, -0.000060],
    10: [-2.927222, 1.666667, -0.238095, 0.039683, -0.004960, 0.000317],
    8: [-2.847222, 1.600000, -0.200000, 0.025397, -0.001786],
    6: [-2.722222, 1.500000, -0.150000, 0.01111],
    4: [-2.500000, 1.333333, -0.08333],
    2: [-2., 1.],
}


# Default spacing symbol
h = symbols('h')


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

    assert(order in fd_coefficients)

    coeffs = fd_coefficients[order]
    deriv = coeffs[0] * reduce(mul, args, 1)

    for i in range(1, int(order / 2) + 1):
        aux1 = [a.subs(dim, dim + i * diff) for a in args]
        aux2 = [a.subs(dim, dim - i * diff) for a in args]
        deriv += coeffs[i] * (reduce(mul, aux1, 1) + reduce(mul, aux2, 1))

    return (1 / diff**2) * deriv


def cross_derivative(*args, **kwargs):
    """Derives corss derivative for a product of given functions.

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
    """Derives corss derivative for a product of given functions.

    :param \*args: All positional arguments must be fully qualified
       function objects, eg. `f(x, y)` or `g(t, x, y, z)`.
    :param dims: 2-tuple of symbols defining the dimension wrt. which
       to differentiate, eg. `x`, `y`, `z` or `t`.
    :param diff: Finite Difference symbol to insert, default `h`.
    :returns: The cross derivative

    Example: Deriving the first-derivative of f(x)*g(x) wrt. x via:
       ``cross_derivative(f(x), g(x), dim=x, side=1, order=1)``
       results in:
       ``*(-f(x)*g(x) + f(x + h)*g(x + h) ) / h``
    """
    dim = kwargs.get('dim', x)
    diff = kwargs.get('diff', h)
    order = kwargs.get('order', 1)
    side = kwargs.get('side', "centered")
    deriv = 0
    sign = 1
    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    if side == "right":
        ind = [(dim + i * diff) for i in range(-int(order / 2) + 1 - (order % 2),
                                               int((order + 1) / 2) + 2 - (order % 2))]
    elif side == "left":
        ind = [(dim - i * diff) for i in range(-int(order / 2) + 1 - (order % 2),
                                               int((order + 1) / 2) + 2 - (order % 2))]
        sign = -1
    else:
        ind = [(dim + i * diff) for i in range(-int(order / 2),
                                               int((order + 1) / 2) + 1)]
        side = 1
    # Finite difference weights from Taylor approximation with this positions
    c = finite_diff_weights(1, ind, dim)
    c = c[-1][-1]

    # Diagonal elements
    for i in range(0, len(ind)):
            var = [a.subs({dim: ind[i]}) for a in args]
            deriv += c[i] * reduce(mul, var, 1)
    return -sign*deriv
