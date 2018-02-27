from __future__ import absolute_import

from functools import reduce
from operator import mul

from sympy import finite_diff_weights

from devito.logger import error
__all__ = ['first_derivative', 'second_derivative', 'cross_derivative',
           'generic_derivative', 'second_cross_derivative',
           'sparse_cross_derivative', 'sparse_generic_derivative',
           'left', 'right', 'centered', 'sparse_fd_list']


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
    dim = kwargs.get('dim')
    diff = kwargs.get('diff', dim.spacing)

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
    dims = kwargs.get('dims')
    diff = kwargs.get('diff', (dims[0].spacing, dims[1].spacing))
    order = kwargs.get('order', 1)

    assert(isinstance(dims, tuple) and len(dims) == 2)
    deriv = 0

    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    ind1r = [(dims[0] + i * diff[0])
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind2r = [(dims[1] + i * diff[1])
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind1l = [(dims[0] - i * diff[0])
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]
    ind2l = [(dims[1] - i * diff[1])
             for i in range(-int(order / 2) + 1 - (order < 4),
                            int((order + 1) / 2) + 2 - (order < 4))]

    # Finite difference weights from Taylor approximation with this positions
    c11 = finite_diff_weights(1, ind1r, dims[0])[-1][-1]
    c21 = finite_diff_weights(1, ind1l, dims[0])[-1][-1]
    c12 = finite_diff_weights(1, ind2r, dims[1])[-1][-1]
    c22 = finite_diff_weights(1, ind2l, dims[1])[-1][-1]

    # Diagonal elements
    for i in range(0, len(ind1r)):
        for j in range(0, len(ind2r)):
            var1 = [a.subs({dims[0]: ind1r[i], dims[1]: ind2r[j]}) for a in args]
            var2 = [a.subs({dims[0]: ind1l[i], dims[1]: ind2l[j]}) for a in args]
            deriv += (.5 * c11[i] * c12[j] * reduce(mul, var1, 1) +
                      .5 * c21[-(j+1)] * c22[-(i+1)] * reduce(mul, var2, 1))

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
    dim = kwargs.get('dim')
    diff = kwargs.get('diff', dim.spacing)
    order = int(kwargs.get('order', 1))
    matvec = kwargs.get('matvec', direct)
    side = kwargs.get('side', centered).adjoint(matvec)
    deriv = 0
    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    if order == 1:
        ind = [dim, dim - diff] if side == left else [dim, dim + diff]
    elif side == right:
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


def generic_derivative(function, deriv_order, dim, fd_order):
    """
    Create generic arbitrary order derivative expression from a
    single :class:`Function` object. This methods is essentially a
    dedicated wrapper around SymPy's `as_finite_diff` utility for
    :class:`devito.Function` objects.

    :param function: The symbol representing a function.
    :param deriv_order: Derivative order, eg. 2 for a second derivative.
    :param dim: The dimension for which to take the derivative.
    :param fd_order: Order of the coefficient discretization and thus
                     the width of the resulting stencil expression.
    """

    deriv = function.diff(*(tuple(dim for _ in range(deriv_order))))
    indices = [(dim + i * dim.spacing) for i in range(-fd_order, fd_order + 1)]
    return deriv.as_finite_difference(indices)


def second_cross_derivative(function, dims, order):
    """
    Create a second order order cross derivative for a given function.

    :param function: The symbol representing a function.
    :param dims: Dimensions for which to take the derivative.
    :param order: Discretisation order of the stencil to create.
    """
    first = second_derivative(function, dim=dims[0], width=order)
    return second_derivative(first, dim=dims[1], order=order)


def sparse_generic_derivative(func, deriv_order, dim, fd_order):
    """
    First order spatial derivative of a SparseFunction. Consitent with Dipole sources
    in geopphysics, the finite-differences are computed on 2 times finer grid (h_x/2).
    This does not return a finite-difference
    expression but a list of (weight, offset) to construct the interpolation at
    the finite-difference locations.
    e.g, for space_order = 1 it returns in 3D
    [(1/h_x, (-h_x/2, 0, 0) ), (0, (0, 0, 0)), (-1/h_x, (h_x/2, 0, 0))]

    These derivatives are to be used in combination with 'inject'.
    """
    # Check number of dimension and which dimension to set offset to
    if hasattr(func, 'space_dimensions'):
        spc_dims = func.space_dimensions
    else:
        spc_dims = [f.space_dimensions for f in func.args
                    if hasattr(f, 'space_dimensions')][0]
    offsets = lambda x, y: dict([(d, 0) if d != x else (x, y)
                                 for d in spc_dims])
    # Computes fd offsets and cofficients
    indices = [(dim + i * dim.spacing/2) for i in range(-fd_order, fd_order + 1)]
    coeffs = finite_diff_weights(deriv_order, indices, dim)[-1][-1]
    coeffs = [c.subs({dim: 0}) for c in coeffs[::-1]]
    indices = [offsets(dim, i.subs({dim: 0})) for i in indices]
    # Build fd List
    if isinstance(func, sparse_fd_list):
        fd = [(w*f[0], sum_dict(pos, f[1])) for w, pos in zip(coeffs, indices)
              for f in func]
    else:
        fd = sparse_fd_list([(w*func, pos) for w, pos in zip(coeffs, indices)])
    return sparse_fd_list([f for f in fd if f[0] != 0])


def sparse_cross_derivative(func, dims, fd_order):
    """
    Derives cross derivative for a product of given sparse functions. This does not
    return a finite-difference expression but a list of (weight, offset) to construct
    the interpolation at the finite-difference locations.

    :param dims: 2-tuple of symbols defining the dimension wrt. which
       to differentiate, eg. `x`, `y`, `z`.
    :returns: The cross derivative
    """
    # Check number of dimension and which dimension to set offset to
    if hasattr(func, 'space_dimensions'):
        spc_dims = func.space_dimensions
    else:
        spc_dims = [f.space_dimensions for f in func.args
                    if hasattr(f, 'space_dimensions')][0]
    offsets = lambda x, y: dict([(d, 0) for d in spc_dims if d not in x] +
                                [(xx, yy) for xx, yy in zip(x, y)])
    # First dimension
    indices1 = [(dims[0] + i * dims[0].spacing/2) for i in range(-fd_order, fd_order + 1)]
    coeffs1 = finite_diff_weights(1, indices1, dims[0])[-1][-1]
    coeffs1 = [c.subs({dims[0]: 0}) for c in coeffs1[::-1]]
    indices1 = [i.subs({dims[0]: 0}) for i in indices1]
    # Second dimension
    indices2 = [(dims[1] + i * dims[1].spacing/2) for i in range(-fd_order, fd_order + 1)]
    coeffs2 = finite_diff_weights(1, indices2, dims[1])[-1][-1]
    coeffs2 = [c.subs({dims[1]: 0}) for c in coeffs2[::-1]]
    indices2 = [i.subs({dims[1]: 0}) for i in indices2]
    if isinstance(func, sparse_fd_list):
        cross = [(w1*w2*f[0], sum_dict(f[1], offsets(dims, (h1, h2))))
                 for (w2, h2) in zip(coeffs2, indices2)
                 for (w1, h1) in zip(coeffs1, indices1) for f in func]
    else:
        cross = [[(w1*w2*func, offsets(dims, (h1, h2)))
                 for (w2, h2) in zip(coeffs2, indices2)]
                 for (w1, h1) in zip(coeffs1, indices1)]
    return sparse_fd_list([f for f in cross if f[0] != 0])


class sparse_fd_list(list):
    """
    A list of tuples (weight, position) for sparse finite differences
    """
    @property
    def space_dimensions(self):
        return tuple(self[0][1].keys())

    def __add__(self, other):
        if isinstance(other, sparse_fd_list):
            return sparse_fd_list(super(sparse_fd_list, self).__add__(other))
        else:
            default_offset = dict((k, 0) for k in self[0][1].keys())
            new = sparse_fd_list([(o, default_offset) for o in other.args])
            return sparse_fd_list(super(sparse_fd_list, self).__add__(new))

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, sparse_fd_list):
            return sparse_fd_list(super(sparse_fd_list, self).__add__(other))
        else:
            default_offset = dict((k, 0) for k in self[0][1].keys())
            new = sparse_fd_list([(o, default_offset) for o in other.args])
            return sparse_fd_list(super(sparse_fd_list, self).__add__(new))

    def __iadd__(self, other):
        if isinstance(other, sparse_fd_list):
            return sparse_fd_list(super(sparse_fd_list, self).__iadd__(other))
        else:
            default_offset = dict((k, 0) for k in self[0][1].keys())
            new = sparse_fd_list([(o, default_offset) for o in other.args])
            return sparse_fd_list(super(sparse_fd_list, self).__iadd__(new))

    def __mul__(self, other_list):
        return sparse_fd_list([(i[0] * other_list, i[1]) for i in self])

    def __rmul__(self, constant):
        return sparse_fd_list([(i[0] * constant, i[1]) for i in self])

    def __floordiv__(self, constant):
        return sparse_fd_list([(i[0] / constant, i[1]) for i in self])

    def __truediv__(self, constant):
        return sparse_fd_list([(i[0] / constant, i[1]) for i in self])


def sum_dict(d1, d2):
    """
    Sum two dictionaries with overlaping or common keys
    """
    d3 = dict()
    for k, v in d1.items():
        d3[k] = v
    for k, v in d2.items():
        d3[k] += v
    return d3
