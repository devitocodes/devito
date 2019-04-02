import sympy

from functools import partial

from devito.finite_differences.tools import left, right, centered


def generate_fd_shortcuts(function):
    """Create all legal finite-difference derivatives for the given Function."""
    dimensions = function.indices
    space_fd_order = function.space_order
    time_fd_order = function.time_order if (function.is_TimeFunction or
                                            function.is_SparseTimeFunction) else 0

    from devito.finite_differences.derivative import Derivative

    def deriv_function(expr, deriv_order, dims, fd_order, side=centered, **kwargs):
        if isinstance(dims, (tuple, sympy.Tuple)):
            return Derivative(expr, *dims, deriv_order=deriv_order, fd_order=fd_order,
                              side=side, **kwargs)
        return Derivative(expr, dims, deriv_order=deriv_order, fd_order=fd_order,
                          side=side, **kwargs)

    side = form_side(dimensions, function)

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
            deriv = partial(deriv_function, deriv_order=o, dims=d,
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
                    deriv = partial(deriv_function, deriv_order=(o, o2), dims=(d, d2),
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
            stagg = dict()
            stagg[d] = centered
            deriv = partial(deriv_function, deriv_order=1, dims=d,
                            fd_order=dim_order, stagger=stagg)
            name_fd = 'd%sc' % name
            desciption = 'centered derivative staggered w.r.t dimension %s' % d

            derivatives[name_fd] = (deriv, desciption)
        else:
            # Left
            dim_order = time_fd_order if d.is_Time else space_fd_order
            deriv = partial(deriv_function, deriv_order=1,
                            dims=d, fd_order=dim_order, side=left)
            name_fd = 'd%sl' % name
            desciption = 'left first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)
            # Right
            deriv = partial(deriv_function, deriv_order=1,
                            dims=d, fd_order=dim_order, side=right)
            name_fd = 'd%sr' % name
            desciption = 'right first order derivative w.r.t dimension %s' % d
            derivatives[name_fd] = (deriv, desciption)

    return derivatives


def symbolic_weights(function, deriv_order, indices, dim):
    return [function._coeff_symbol(indices[j], deriv_order, function, dim)
            for j in range(0, len(indices))]


def generate_indices(func, dim, diff, order, stagger=None, side=None):

    # Check if called from first_derivative()
    if bool(side):
        if side == right:
            ind = [(dim+i*diff) for i in range(-int(order/2)+1-(order % 2),
                                               int((order+1)/2)+2-(order % 2))]
        elif side == left:
            ind = [(dim-i*diff) for i in range(-int(order/2)+1-(order % 2),
                                               int((order+1)/2)+2-(order % 2))]
        else:
            ind = [(dim+i*diff) for i in range(-int(order/2),
                                               int((order+1)/2)+1)]
        x0 = None
    else:
        if func.is_Staggered:
            if stagger == left:
                off = -.5
            elif stagger == right:
                off = .5
            else:
                off = 0
            ind = list(set([(dim + int(i+.5+off) * dim.spacing)
                            for i in range(-order//2, order//2)]))
            x0 = (dim + off*diff)
            if order < 2:
                ind = [dim + diff, dim] if stagger == right else [dim - diff, dim]

        else:
            ind = [(dim + i*dim.spacing) for i in range(-order//2, order//2 + 1)]
            x0 = dim
            if order < 2:
                ind = [dim, dim + diff]
    return ind, x0


def form_side(dimensions, function):
    side = dict()
    for (d, s) in zip(dimensions, function.staggered):
        if s == 0:
            side[d] = left
        elif s == 1:
            side[d] = right
        else:
            side[d] = centered
    return side
