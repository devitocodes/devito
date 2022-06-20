from functools import wraps, partial
from itertools import product

import numpy as np
from sympy import S, finite_diff_weights, cacheit, sympify

from devito.tools import Tag, as_tuple
from devito.types.dimension import StencilDimension


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
        if matvec is direct:
            return self
        else:
            if self is centered:
                return centered
            elif self is right:
                return left
            elif self is left:
                return right
            else:
                raise ValueError("Unsupported side value")


left = Side('left', -1)
right = Side('right', 1)
centered = Side('centered', 0)


def check_input(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        try:
            return S.Zero if expr.is_Number else func(expr, *args, **kwargs)
        except AttributeError:
            raise ValueError("'%s' must be of type Differentiable, not %s"
                             % (expr, type(expr)))
    return wrapper


def check_symbolic(func):
    @wraps(func)
    def wrapper(expr, *args, **kwargs):
        if expr._uses_symbolic_coefficients:
            expr_dict = expr.as_coefficients_dict()
            if any(len(expr_dict) > 1 for item in expr_dict):
                raise NotImplementedError("Applying the chain rule to functions "
                                          "with symbolic coefficients is not currently "
                                          "supported")
        kwargs['symbolic'] = expr._uses_symbolic_coefficients
        return func(expr, *args, **kwargs)
    return wrapper


def dim_with_order(dims, orders):
    """
    Create all possible derivative order for each dims
    for example dim_with_order((x, y), 1) outputs:
    [(1, 0), (0, 1), (1, 1)]
    """
    ndim = len(dims)
    max_order = np.min([6, np.max(orders)])
    # Get all combinations and remove (0, 0, 0)
    all_comb = tuple(product(range(max_order+1), repeat=ndim))[1:]
    # Only keep the one with each dimension maximum order
    all_comb = [c for c in all_comb if all(c[k] <= orders[k] for k in range(ndim))]
    return all_comb


def deriv_name(dims, orders):
    name = []
    for d, o in zip(dims, orders):
        name_dim = 't' if d.is_Time else d.root.name
        name.append('d%s%s' % (name_dim, o) if o > 1 else 'd%s' % name_dim)

    return ''.join(name)


def generate_fd_shortcuts(dims, so, to=0):
    """Create all legal finite-difference derivatives for the given Function."""
    orders = tuple(to if i.is_Time else so for i in dims)

    from devito.finite_differences.derivative import Derivative

    def diff_f(expr, deriv_order, dims, fd_order, side=None, **kwargs):
        return Derivative(expr, *as_tuple(dims), deriv_order=deriv_order,
                          fd_order=fd_order, side=side, **kwargs)

    all_combs = dim_with_order(dims, orders)

    derivatives = {}

    # All conventional FD shortcuts
    for o in all_combs:
        fd_dims = tuple(d for d, o_d in zip(dims, o) if o_d > 0)
        d_orders = tuple(o_d for d, o_d in zip(dims, o) if o_d > 0)
        fd_orders = tuple(to if d.is_Time else so for d in fd_dims)
        deriv = partial(diff_f, deriv_order=d_orders, dims=fd_dims, fd_order=fd_orders)
        name_fd = deriv_name(fd_dims, d_orders)
        dname = (d.root.name for d in fd_dims)
        desciption = 'derivative of order %s w.r.t dimension %s' % (d_orders, dname)
        derivatives[name_fd] = (deriv, desciption)

    # Add non-conventional, non-centered first-order FDs
    for d, o in zip(dims, orders):
        name = 't' if d.is_Time else d.root.name
        # Add centered first derivatives
        deriv = partial(diff_f, deriv_order=1, dims=d, fd_order=o, side=centered)
        name_fd = 'd%sc' % name
        desciption = 'centered derivative staggered w.r.t dimension %s' % d.name
        derivatives[name_fd] = (deriv, desciption)
        # Left
        deriv = partial(diff_f, deriv_order=1, dims=d, fd_order=o, side=left)
        name_fd = 'd%sl' % name
        desciption = 'left first order derivative w.r.t dimension %s' % d.name
        derivatives[name_fd] = (deriv, desciption)
        # Right
        deriv = partial(diff_f, deriv_order=1, dims=d, fd_order=o, side=right)
        name_fd = 'd%sr' % name
        desciption = 'right first order derivative w.r.t dimension %s' % d.name
        derivatives[name_fd] = (deriv, desciption)

    return derivatives


class IndexSet(tuple):

    """
    The points of a finite-difference expansion.
    """

    def __new__(cls, dim, indices=None, expr=None, fd=None):
        assert indices is not None or expr is not None

        if fd is None:
            try:
                v = {d for d in expr.free_symbols if isinstance(d, StencilDimension)}
                assert len(v) == 1
                fd = v.pop()
            except AttributeError:
                pass

        if not indices:
            assert expr is not None
            indices = [expr.subs(fd, i) for i in fd.range]

        obj = super().__new__(cls, indices)
        obj.dim = dim
        obj.expr = expr
        obj.free_dim = fd

        return obj

    def __repr__(self):
        return "IndexSet(%s)" % ", ".join(str(i) for i in self)

    @property
    def spacing(self):
        return self.dim.spacing

    def scale(self, v):
        """
        Construct a new IndexSet with all indices scaled by `v`.
        """
        mapper = {self.spacing: v*self.spacing}

        indices = []
        for i in self:
            try:
                iloc = i.xreplace(mapper)
            except AttributeError:
                # Pure number -> sympify
                iloc = sympify(i).xreplace(mapper)
            indices.append(iloc)

        try:
            expr = self.expr.xreplace(mapper)
        except AttributeError:
            expr = None

        return IndexSet(self.dim, indices, expr=expr, fd=self.free_dim)

    def shift(self, v):
        """
        Construct a new IndexSet with all indices shifted by `v`.
        """
        indices = [i + v for i in self]

        try:
            expr = self.expr + v
        except TypeError:
            expr = None

        return IndexSet(self.dim, indices, expr=expr, fd=self.free_dim)


def make_stencil_dimension(expr, _min, _max):
    """
    Create a StencilDimension for `expr` with unique name.
    """
    n = len(expr.find(StencilDimension))
    return StencilDimension(name='i%d' % n, _min=_min, _max=_max)


def symbolic_weights(function, deriv_order, indices, dim):
    return [function._coeff_symbol(indices[j], deriv_order, function, dim)
            for j in range(0, len(indices))]


@cacheit
def numeric_weights(deriv_order, indices, x0):
    return finite_diff_weights(deriv_order, indices, x0)[-1][-1]


def generate_indices(expr, dim, order, side=None, matvec=None, x0=None):
    """
    Indices for the finite-difference scheme.

    Parameters
    ----------
    expr : expr-like
        Expression that is differentiated.
    dim : Dimension
        Dimensions w.r.t which the derivative is taken.
    order : int
        Order of the finite-difference scheme.
    side : Side, optional
        Side of the scheme (centered, left, right).
    matvec : Transpose, optional
        Forward (matvec=direct) or transpose (matvec=transpose) mode of the
        finite difference. Defaults to `direct`.
    x0 : dict of {Dimension: Dimension or Expr or Number}, optional
        Origin of the scheme, ie. `x`, `x + .5 * x.spacing`, ...

    Returns
    -------
    An IndexSet, representing an ordered list of indices.
    """
    if expr.is_Staggered and not dim.is_Time:
        x0, indices = generate_indices_staggered(expr, dim, order, side=side, x0=x0)
    else:
        x0 = (x0 or {dim: dim}).get(dim, dim)
        # Check if called from first_derivative()
        indices = generate_indices_cartesian(expr, dim, order, side, x0)

    assert isinstance(indices, IndexSet)

    return indices, x0


def generate_indices_cartesian(expr, dim, order, side, x0):
    """
    Indices for the finite-difference scheme on a cartesian grid.

    Parameters
    ----------
    expr : expr-like
        Expression that is differentiated.
    dim : Dimension
        Dimensions w.r.t which the derivative is taken.
    order : int
        Order of the finite-difference scheme.
    side : Side
        Side of the scheme (centered, left, right).
    x0 : dict of {Dimension: Dimension or expr-like or Number}
        Origin of the scheme, ie. `x`, `x + .5 * x.spacing`, ...

    Returns
    -------
    An IndexSet, representing an ordered list of indices.
    """
    shift = 0
    # Shift if `x0` is not on the grid
    offset_c = 0 if sympify(x0).is_Integer else (dim - x0)/dim.spacing
    offset_c = np.sign(offset_c) * (offset_c % 1)
    offset = offset_c * dim.spacing
    # Spacing
    diff = dim.spacing
    if side in [left, right]:
        shift = 1
        diff *= side.val
    # Indices
    if order < 2:
        indices = [x0, x0 + diff] if offset == 0 else [x0 - offset, x0 + offset]
        return IndexSet(dim, indices)
    else:
        # Left and right max offsets for indices
        o_min = -order//2 + int(np.ceil(-offset_c))
        o_max = order//2 - int(np.ceil(offset_c))

        d = make_stencil_dimension(expr, o_min, o_max)
        iexpr = x0 + (d + shift) * diff + offset
        return IndexSet(dim, expr=iexpr)


def generate_indices_staggered(expr, dim, order, side=None, x0=None):
    """
    Indices for the finite-difference scheme on a staggered grid.

    Parameters
    ----------
    expr : expr-like
        Expression that is differentiated.
    dim : Dimension
        Dimensions w.r.t which the derivative is taken.
    order : int
        Order of the finite-difference scheme.
    side : Side, optional
        Side of the scheme (centered, left, right).
    x0 : dict of {Dimension: Dimension or expr-like or Number}, optional
        Origin of the scheme, ie. `x`, `x + .5 * x.spacing`, ...

    Returns
    -------
    An IndexSet, representing an ordered list of indices.
    """
    diff = dim.spacing
    start = (x0 or {}).get(dim) or expr.indices_ref[dim]
    try:
        ind0 = expr.indices_ref[dim]
    except AttributeError:
        ind0 = start
    if start != ind0:
        if order < 2:
            indices = [start - diff/2, start + diff/2]
            indices = IndexSet(dim, indices)
        else:
            o_min = -order//2+1
            o_max = order//2

            d = make_stencil_dimension(expr, o_min, o_max)
            iexpr = start - diff/2 + d * diff
            indices = IndexSet(dim, expr=iexpr)
    else:
        if order < 2:
            indices = [start, start - diff]
            indices = IndexSet(dim, indices)
        else:
            o_min = -order//2
            o_max = order//2

            d = make_stencil_dimension(expr, o_min, o_max)
            iexpr = start + d * diff
            indices = IndexSet(dim, expr=iexpr)

    return start, indices


def make_shift_x0(shift, ndim):
    """
    Returns a callable that calculates a shifted origin for each derivative
    of an operation derivatives scheme (given by ndim) given a shift object
    which can be a None, a float or a tuple with shape equal to ndim
    """
    if shift is None:
        return lambda s, d, i, j: None
    elif isinstance(shift, float):
        return lambda s, d, i, j: d + s * d.spacing
    elif type(shift) is tuple and np.shape(shift) == ndim:
        if len(ndim) == 1:
            return lambda s, d, i, j: d + s[j] * d.spacing
        elif len(ndim) == 2:
            return lambda s, d, i, j: d + s[i][j] * d.spacing
        else:
            raise ValueError("ndim length must be equal to 1 or 2")
    raise ValueError("shift parameter must be one of the following options: "
                     "None, float or tuple with shape equal to %s" % (ndim,))
