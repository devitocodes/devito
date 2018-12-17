from functools import partial, wraps

from sympy import S, finite_diff_weights

from devito.finite_differences import Differentiable
from devito.tools import Tag

__all__ = ['first_derivative', 'second_derivative', 'cross_derivative',
           'generic_derivative', 'left', 'right', 'centered', 'transpose']

# Number of digits for FD coefficients to avoid roundup errors and non-deeterministic
# code generation
_PRECISION = 9


class Transpose(Tag):
    """
    Utility class to change the sign of a derivative. This is only needed
    for odd order derivatives, which require a minus sign for the transpose.
    """
    pass


direct = Transpose('direct', 1)
transpose = Transpose('transpose', -1)


class Side(Tag):
    """
    Class encapsulating the side of the shift for derivatives.
    """

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
                raise ValueError("Unsupported side value")


left = Side('left', -1)
right = Side('right', 1)
centered = Side('centered', 0)


def check_input(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr.is_Number:
            return S.Zero
        elif not isinstance(expr, Differentiable):
            raise ValueError("`%s` must be of type Differentiable (found `%s`)"
                             % (expr, type(expr)))
        else:
            return func(expr, *args, **kwargs)
    return wrapper


@check_input
def first_derivative(expr, **kwargs):
    """
    First-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the first-order derivative is produced.
    **kwargs
        - ``dim``: Dimension w.r.t. which to differentiate.
        - ``diff``: Finite-difference symbol, defaults to `h`.
        - ``order``: Discretisation order of the derivative.

    Returns
    -------
    expr-like
        The first-order derivative of ``expr``

    Examples
    --------
    >>> from devito import Function, Grid, first_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid)
    >>> g = Function(name='g', grid=grid)
    >>> first_derivative(f*g, dim=x)
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x

    This is also more easily obtainable via:

    >>> (f*g).dx
    -f(x, y)*g(x, y)/h_x + f(x + h_x, y)*g(x + h_x, y)/h_x
    """
    dim = kwargs.get('dim')
    diff = kwargs.get('diff', dim.spacing)
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
    all_dims = tuple(set((dim,) + tuple([i for i in expr.indices if i.root == dim])))
    # Loop through positions
    for i in range(0, len(ind)):
            subs = dict([(d, ind[i].subs({dim: d})) for d in all_dims])
            deriv += expr.subs(subs) * c[i]
    return (matvec.val*deriv).evalf(_PRECISION)


@check_input
def second_derivative(expr, dim, fd_order, stagger=None):
    """
    Second-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.

    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    stagger : Side, optional
        Shift of the finite-difference approximation.

    Returns
    -------
    expr-like
        The derivative of ``expr`` of order ``deriv-order``.

    Examples
    --------
    >>> from devito import Function, Grid, second_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, _ = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> second_derivative(f*g, order=2, dim=x)
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2

    This is also more easily obtainable via:

    >>> (f*g).dx2
    -2.0*f(x, y)*g(x, y)/h_x**2 + f(x - h_x, y)*g(x - h_x, y)/h_x**2 +\
 f(x + h_x, y)*g(x + h_x, y)/h_x**2
    """

    return generic_derivative(expr, 2, dim, fd_order, stagger=None)


@check_input
def cross_derivative(expr, **kwargs):
    """
    Shifted cross derivative of a given expression. This generates the cross derivatives
    shifted from the center point to avoid odd-even coupling.
    For a conventional cross derivative use `generic_cross_derivative`

    Parameters
    ----------
    expr : expr-like
        Expression for which the cross derivative is produced.
    **kwargs
        - ``dims``: 2-tuple Dimensions w.r.t. which to differentiate.
        - ``diff``: Finite-difference symbol, defaults to `h`.
        - ``order``: 2-tuple representing the discretisation order of the
                     derivative in the two Dimensions.

    Returns
    -------
    expr-like
        The cross derivative of ``expr``

    Examples
    --------
    >>> from devito import Function, Grid, cross_derivative
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> f = Function(name='f', grid=grid, space_order=2)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> # cross_derivative(f*g, dims=(x, y))
    TODO

    This is also more easily obtainable via:

    >>> # (f*g).dxdy
    TODO
    """

    dims = kwargs.get('dims')
    diff = kwargs.get('diff', (dims[0].spacing, dims[1].spacing))
    order = kwargs.get('order', (1, 1))

    assert(isinstance(dims, tuple) and len(dims) == 2)
    deriv = 0

    # Stencil positions for non-symmetric cross-derivatives with symmetric averaging
    ind1r = [(dims[0] + i * diff[0])
             for i in range(-int(order[0] / 2) + 1 - (order[0] < 4),
                            int((order[0] + 1) / 2) + 2 - (order[0] < 4))]
    ind2r = [(dims[1] + i * diff[1])
             for i in range(-int(order[1] / 2) + 1 - (order[1] < 4),
                            int((order[1] + 1) / 2) + 2 - (order[1] < 4))]
    ind1l = [(dims[0] - i * diff[0])
             for i in range(-int(order[0] / 2) + 1 - (order[0] < 4),
                            int((order[0] + 1) / 2) + 2 - (order[0] < 4))]
    ind2l = [(dims[1] - i * diff[1])
             for i in range(-int(order[1] / 2) + 1 - (order[1] < 4),
                            int((order[1] + 1) / 2) + 2 - (order[1] < 4))]

    # Finite difference weights from Taylor approximation with this positions
    c11 = finite_diff_weights(1, ind1r, dims[0])[-1][-1]
    c21 = finite_diff_weights(1, ind1l, dims[0])[-1][-1]
    c12 = finite_diff_weights(1, ind2r, dims[1])[-1][-1]
    c22 = finite_diff_weights(1, ind2l, dims[1])[-1][-1]
    all_dims1 = tuple(set((dims[0], ) +
                      tuple([i for i in expr.indices if i.root == dims[0]])))
    all_dims2 = tuple(set((dims[1], ) +
                      tuple([i for i in expr.indices if i.root == dims[1]])))
    # Diagonal elements
    for i in range(0, len(ind1r)):
        for j in range(0, len(ind2r)):
            subs1 = dict([(d1, ind1r[i].subs({dims[0]: d1})) +
                          (d2, ind2r[i].subs({dims[1]: d2}))
                          for (d1, d2) in zip(all_dims1, all_dims2)])
            subs2 = dict([(d1, ind1l[i].subs({dims[0]: d1})) +
                          (d2, ind2l[i].subs({dims[1]: d2}))
                          for (d1, d2) in zip(all_dims1, all_dims2)])
            var1 = expr.subs(subs1)
            var2 = expr.subs(subs2)
            deriv += .5 * (c11[i] * c12[j] * var1 + c21[-(j+1)] * c22[-(i+1)] * var2)
    return -deriv.evalf(_PRECISION)


@check_input
def generic_cross_derivative(expr, dims, fd_order, deriv_order, stagger=(None, None)):
    """
    Arbitrary-order cross derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the cross derivative is produced.
    dims : tuple
        Dimensions w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    deriv_order : int
        Derivative order, e.g. 2 for a second-order derivative.
    """
    
    for d, fd, dim, s in zip(deriv_order, fd_order, dims, stagger):
        first = generic_derivative(first, deriv_order=d, fd_order=fd, dim=dim, stagger=s)

    return first

@check_input
def generic_derivative(expr, deriv_order, dim, fd_order, stagger=None):
    """
    Arbitrary-order derivative of a given expression.

    Parameters
    ----------
    expr : expr-like
        Expression for which the derivative is produced.
    deriv_order : int
        Derivative order, e.g. 2 for a second-order derivative.
    dim : Dimension
        The Dimension w.r.t. which to differentiate.
    fd_order : int
        Coefficient discretization order. Note: this impacts the width of
        the resulting stencil.
    stagger : Side, optional
        Shift of the finite-difference approximation.

    Returns
    -------
    expr-like
        The derivative of ``expr`` of order ``deriv-order``.
    """

    diff = dim.spacing

    if stagger == left or not expr.is_Staggered:
        off = -.5
    elif stagger == right:
        off = .5
    else:
        off = 0

    if expr.is_Staggered:
        indices = list(set([(dim + int(i+.5+off) * dim.spacing)
                            for i in range(-fd_order//2, fd_order//2)]))
        x0 = (dim + off*diff)
        if fd_order < 2:
            indices = [dim + diff, dim] if stagger == right else [dim - diff, dim]

    else:
        indices = [(dim + i * dim.spacing) for i in range(-fd_order//2, fd_order//2 + 1)]
        x0 = dim
        if fd_order < 2:
            indices = [dim, dim + diff]

    c = finite_diff_weights(deriv_order, indices, x0)[-1][-1]

    deriv = 0
    all_dims = tuple(set((dim, ) +
                     tuple([i for i in expr.indices if i.root == dim])))
    for i in range(0, len(indices)):
            subs = dict([(d, indices[i].subs({dim: d})) for d in all_dims])
            deriv += expr.subs(subs) * c[i]

    return deriv.evalf(_PRECISION)


def generate_fd_shortcuts(function):
    """Create all legal finite-difference derivatives for the given Function."""
    dimensions = function.indices
    space_fd_order = function.space_order
    time_fd_order = function.time_order if function.is_TimeFunction else 0

    deriv_function = generic_derivative
    c_deriv_function = generic_cross_derivative

    side = dict()
    for (d, s) in zip(dimensions, function.staggered):
        if s == 0:
            side[d] = left
        elif s == 1:
            side[d] = right
        else:
            side[d] = centered

    derivatives = dict()
    done = []
    for d in dimensions:
        # Dimension is treated, remove from list
        done += [d]
        other_dims = tuple(i for i in dimensions if i not in done)
        # Dimension name and corresponding FD order
        dim_order = time_fd_order if d.is_Time else space_fd_order
        name = 't' if d.is_Time else d.root.name
        # All possible derivatives go up to the dimension FD order
        for o in range(1, dim_order + 1):
            deriv = partial(deriv_function, deriv_order=o, dim=d,
                            fd_order=dim_order, stagger=side[d])
            name_fd = 'd%s%d' % (name, o) if o > 1 else 'd%s' % name
            desciption = 'derivative of order %d w.r.t dimension %s' % (o, d)

            derivatives[name_fd] = (deriv, desciption)
            # Cross derivatives with the other dimension
            # Skip already done dimensions a dxdy is the same as dydx
            for d2 in other_dims:
                dim_order2 = time_fd_order if d2.is_Time else space_fd_order
                name2 = 't' if d2.is_Time else d2.root.name
                for o2 in range(1, dim_order2 + 1):
                    deriv = partial(c_deriv_function, deriv_order=(o, o2), dims=(d, d2),
                                    fd_order=(dim_order, dim_order2),
                                    stagger=(side[d], side[d2]))
                    name_fd2 = 'd%s%d' % (name, o) if o > 1 else 'd%s' % name
                    name_fd2 += 'd%s%d' % (name2, o2) if o2 > 1 else 'd%s' % name2
                    desciption = 'derivative of order (%d, %d) ' % (o, o2)
                    desciption += 'w.r.t dimension (%s, %s) ' % (d, d2)
                    derivatives[name_fd2] = (deriv, desciption)

    # Add non-conventional, non-centered first-order FDs
    for d in dimensions:
        name = 't' if d.is_Time else d.root.name
        if function.is_Staggered:
            # Add centered first derivatives if staggered
            deriv = partial(deriv_function, deriv_order=1, dim=d,
                            fd_order=dim_order, stagger=centered)
            name_fd = 'd%sc' % name
            desciption = 'centered derivative staggered w.r.t dimension %s' % d

            derivatives[name_fd] = (deriv, desciption)
        else:
            # Left
            dim_order = time_fd_order if d.is_Time else space_fd_order
            deriv = partial(first_derivative, order=dim_order, dim=d, side=left)
            name_fd = 'd%sl' % name
            desciption = 'left first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)
            # Right
            deriv = partial(first_derivative, order=dim_order, dim=d, side=right)
            name_fd = 'd%sr' % name
            desciption = 'right first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)

    return derivatives
