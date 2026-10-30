"""
Derivatives w.r.t. the physical coordinates of a CurvilinearGrid.

On a CurvilinearGrid, a derivative along a space Dimension is the derivative
w.r.t. the physical coordinate `x_i` of that axis. It is evaluated by the chain
rule, with the metric terms `d xi_j/d x_i` of the map where the derivative is
evaluated,

    d/dx_i = sum_j (d xi_j/d x_i) d/d xi_j,

    d^2/(dx_i dx_l) = sum_jk (d xi_j/d x_i)(d xi_k/d x_l) d^2/(d xi_j d xi_k)
                      + sum_j (d^2 xi_j/(d x_i d x_l)) d/d xi_j,

and derivatives of higher order nest these. The derivatives w.r.t. the
computational coordinates `xi_j` are those of a Grid. On a flat map, only the
term along the derivative's own Dimension remains, with the stencil the user
asked for, so the derivative is that of a Grid.

The transpose `.T` of a derivative `D` is its transpose in the inner product
weighted by the Jacobian `J`, `J^-1 D^T J`: for `D = sum_j c_j D_j`,

    D.T v = J^-1 sum_j D_j^T (J c_j v),

with `J` and `c_j` where the values of `v` are, and the leading `J^-1` where
the transpose writes its values.
"""

from collections import Counter
from itertools import product

import sympy

from .differentiable import Add, Mul
from .tools import centered, transpose

__all__ = ['curvilinear_grid', 'physical_derivative']


def curvilinear_grid(deriv):
    """
    The CurvilinearGrid w.r.t. whose physical coordinates `deriv` is taken.

    Returns None for a derivative w.r.t. computational coordinates, of an
    expression on another grid, or along no space Dimension of the grid.
    """
    # The argument of a Derivative may also be a Dimension, which has no grid
    grid = getattr(deriv.expr, 'grid', None)
    if deriv.computational or grid is None or not grid.root.is_Curvilinear:
        return None

    grid = grid.root
    if not any(d.root in grid.dimensions for d in deriv.dims):
        return None
    return grid


def physical_derivative(deriv, expr, grid, **kwargs):
    """
    Evaluate a derivative w.r.t. the physical coordinates of a CurvilinearGrid.

    Parameters
    ----------
    deriv : Derivative
        The derivative.
    expr : expr-like
        Its argument.
    grid : CurvilinearGrid
        The grid of `expr`.
    **kwargs
        The options of the evaluation, e.g. `expand`.

    Returns
    -------
    expr-like
        The finite-difference approximation of `deriv`.
    """
    if deriv.method == 'RSFD' and deriv.transpose == transpose:
        # Devito transposes a rotated stencil at its own `x0`, for Functions at
        # the nodes, while the transposes of the chain rule are mirrored about
        # the node; the two do not combine
        raise NotImplementedError(f"The transpose of {deriv}, a rotated staggered "
                                  "derivative, is not supported on a CurvilinearGrid")

    # Differentiation is linear: the terms of `expr` with values at different
    # points are differentiated one by one, each with its own metric terms
    parts = _parts(expr, grid)
    if len(parts) > 1:
        return Add(*[physical_derivative(deriv, p, grid, **kwargs) for p in parts])

    factors = _factors(deriv, grid)
    if len(factors) > 1:
        res = _nested(deriv, expr, factors)._evaluate(**kwargs)
    else:
        res = _chain_rule(deriv, expr, grid, factors[0], **kwargs)

    for e in deriv._ppsubs:
        res = res.xreplace(e)
    return res


def _factors(deriv, grid):
    """
    Split `deriv` into the derivatives of order at most 2 in space it is the
    product of, its factors, innermost first.

    E.g. `u.dx2dy` is `d/dy (d^2/dx^2 u)` and `u.dt.dx` is `d/dx (d/dt u)`.
    The derivatives along Dimensions other than space, e.g. time, come first,
    then the space derivatives two at a time, each pair a second derivative
    such as `d^2/(dx dy)`, and the last one alone if their number is odd.
    Each factor is a tuple of `(Dimension, order)`; a derivative of order 0,
    an interpolation, is a single empty factor.
    """
    space = [d for d, o in zip(deriv.dims, deriv.deriv_order, strict=True)
             if d.root in grid.dimensions for _ in range(o)]
    others = tuple((d, o) for d, o in zip(deriv.dims, deriv.deriv_order, strict=True)
                   if d.root not in grid.dimensions and o)

    factors = [others] if others else []
    for start in range(0, len(space), 2):
        counts = Counter(space[start:start + 2])
        factors.append(tuple(counts.items()))
    return factors or [()]


def _nested(deriv, expr, factors):
    """
    `deriv` of `expr` as nested Derivatives, one per factor (see `_factors`).

    A transpose nests them in reverse order, as the transpose of a product
    reverses it. The `x0` along a Dimension goes to the outermost factor along
    it, the inner ones being evaluated where their argument is.
    """
    if deriv.transpose == transpose:
        factors = factors[::-1]

    outermost = {d: k for k, factor in enumerate(factors) for d, _ in factor}
    for k, factor in enumerate(factors):
        dims = tuple(d for d, _ in factor)
        expr = deriv._rebuild(expr, *dims, deriv_order=tuple(o for _, o in factor),
                              fd_order=tuple(deriv.fd_order[d] for d in dims),
                              x0={d: v for d, v in deriv.x0.items()
                                  if outermost.get(d) == k},
                              offsets=deriv.offsets, subs=())
    return expr


def _chain_rule(deriv, expr, grid, factor, **kwargs):
    """
    A derivative of order at most 2 in space, `factor` (see `_factors`), of
    `expr`, by the chain rule.

    It is `sum_k c_k D_k expr`, with the computational derivatives `D_k` and
    their metric coefficients `c_k` where the derivative is evaluated, or, for
    a transpose, `J^-1 sum_k D_k^T (J c_k expr)` (see the module docstring).
    """
    dims = _space_dimensions(expr, grid)

    # Three points along each Dimension: where Devito indexes `expr`, `source`,
    # where its values are, `values`, e.g. at `x + h_x/2` for a derivative to
    # the half points indexed at `x`, and where the derivative is evaluated in
    # the frame of `source`, `target`
    source = {d: _index(expr, d) for d in dims}
    values = {d: _position(expr, d) for d in dims}
    target = {d: deriv.x0.get(d, source[d]) for d in dims}

    transposed = deriv.transpose == transpose
    if transposed:
        # The transpose reads `expr` where its values are, and writes its own
        # values where Devito's transposed stencils write them (see `_written`)
        at = values
        written = {d: _written(deriv, d, target[d], values[d], source[d]) for d in dims}
        # A transposed stencil at `x0` writes at `2 d - x0 + values - source`,
        # so the stencils of all the terms, along any Dimension, write at
        # `written` when taken at this `x0`. Along the Dimensions `deriv`
        # moves `expr` along, it is `target`
        x0 = {d: 2*d - written[d] + values[d] - source[d] for d in dims}
    else:
        at = {d: target[d] + values[d] - source[d] for d in dims}
        x0 = target
    # Along the other Dimensions, e.g. time, the stencils are those of a Grid
    x0 = {**{d: v for d, v in deriv.x0.items() if d not in dims}, **x0}

    res = []
    for coefficient, term_dims, orders in _terms(grid, dims, factor, at):
        if coefficient == 0:
            continue
        # The term of the factor itself, the only one left on a flat map
        own = dict(zip(term_dims, orders, strict=True)) == dict(factor)
        if transposed:
            weight = grid.jacobian_at(_root(at))*coefficient
            stencil = _computational(deriv, weight*expr, term_dims, orders, x0,
                                     source, factor, own)
            res.append(stencil._evaluate(**kwargs))
        else:
            stencil = _computational(deriv, expr, term_dims, orders, x0, source,
                                     factor, own)
            res.append(coefficient*stencil._evaluate(**kwargs))

    if transposed:
        return grid.inverse_jacobian_at(_root(written))*Add(*res)
    return Add(*res)


def _terms(grid, dims, factor, at):
    """
    The terms of the chain rule of `factor`, with its metric coefficients at
    the point `at`.

    Each term is a tuple `(coefficient, dims, orders)`, the coefficient of the
    derivative of `orders` along the computational Dimensions `dims`. A factor
    along no space Dimension, e.g. along time or an interpolation, is a single
    term of coefficient 1.
    """
    space = [d for d, o in factor for _ in range(o) if d in dims]
    others = tuple((d, o) for d, o in factor if d not in dims)
    if not space:
        return [(1, tuple(d for d, _ in others), tuple(o for _, o in others))]

    roots = {d.root: d for d in dims}
    metrics = grid.metrics_at(_root(at))
    if len(space) == 1:
        physical, = space
        return [(metrics[(physical.root, j)], (roots[j],), (1,)) for j in roots]

    # A second derivative along `xi` and `xl`: the products of the metric
    # terms, and the second derivatives of the map
    xi, xl = space[0].root, space[1].root
    curvatures = grid.curvatures_at(_root(at))
    terms = [(curvatures[(xi, xl, j)], (roots[j],), (1,)) for j in roots]
    for j, k in product(roots, repeat=2):
        coefficient = metrics[(xi, j)]*metrics[(xl, k)]
        if j == k:
            terms.append((coefficient, (roots[j],), (2,)))
        else:
            terms.append((coefficient, (roots[j], roots[k]), (1, 1)))
    return terms


def _computational(deriv, expr, dims, orders, x0, source, factor, own):
    """
    One computational derivative of the chain rule: the derivative of `orders`
    along `dims` of `expr` at `x0`.

    Along the space Dimensions it does not differentiate along where `x0` is
    not `source`, where Devito indexes `expr`, `expr` is first interpolated to
    `x0`. A transpose differentiates first, then interpolates, both
    transposed. With `halo=0`, the first of the two reads `expr` as zero
    outside the SubDomain of the equation. The term of the factor itself,
    `own`, keeps the stencil of `deriv`; the others are centered, of the order
    of the derivatives of `factor`.
    """
    moved = tuple(d for d in source if d not in dims and x0[d] != source[d])
    if deriv.method == 'RSFD':
        # A rotated staggered derivative is evaluated at `x0` along every
        # Dimension, so it interpolates nothing
        moved = ()

    # The order of the interpolation along each moved Dimension:
    # - for a derivative of order 0 along it, its own, as the user asked;
    # - for the term of the factor, that of a Grid for an `x0`, so that a flat
    #   map gives the derivative of a Grid;
    # - for the other terms, which vanish on a flat map, that of the stencils
    fd_order = max((deriv.fd_order[d] for d, _ in factor), default=max(deriv.fd_order))
    asked = {d: o for d, n, o in zip(deriv.dims, deriv.deriv_order, deriv.fd_order,
                                     strict=True) if n == 0}
    default = getattr(expr, 'interp_order', 2) if own else fd_order
    interp = {d: asked.get(d, default) for d in moved}

    # The interpolation and the derivative, along different Dimensions,
    # commute
    if deriv.transpose == transpose:
        inner = _derivative(deriv, expr, dims, orders, x0, fd_order, own)
        return _interpolation(deriv, inner, interp, x0, masked=not dims)
    inner = _interpolation(deriv, expr, interp, x0)
    return _derivative(deriv, inner, dims, orders, x0, fd_order, own,
                       masked=not interp)


def _derivative(deriv, expr, dims, orders, x0, fd_order, own, masked=True):
    """
    The derivative of `orders` along the computational `dims` of `expr` at
    `x0`: with the stencil of `deriv` if it is the term of the factor itself,
    `own`, else centered and of order `fd_order`. Unless `masked`, it reads
    `expr` as it is, even with `halo=0` (see `_computational`).
    """
    if not dims:
        return expr

    offsets = {d: v for d, v in deriv.offsets.items() if d in dims}
    # A rotated staggered derivative is evaluated at `x0` along every Dimension
    at = x0 if deriv.method == 'RSFD' else {d: x0[d] for d in dims if d in x0}
    settings = _settings(deriv, masked)
    if own:
        return deriv._rebuild(expr, *dims, deriv_order=orders,
                              fd_order=tuple(deriv.fd_order[d] for d in dims), x0=at,
                              offsets=offsets, subs=(), **settings)

    return type(deriv)(expr, *zip(dims, orders, strict=True),
                       fd_order=(fd_order,)*len(dims), x0=at, offsets=offsets,
                       side=centered, method=deriv.method, **settings)


def _interpolation(deriv, expr, orders, x0, masked=True):
    """
    `expr` interpolated to `x0` along the computational Dimensions of `orders`.
    Unless `masked`, it reads `expr` as it is, even with `halo=0` (see
    `_computational`).
    """
    if not orders:
        return expr

    return type(deriv)(expr, *((d, 0) for d in orders), fd_order=tuple(orders.values()),
                       x0={d: x0[d] for d in orders}, **_settings(deriv, masked))


def _settings(deriv, masked=True):
    """
    The settings of `deriv` that the computational derivatives of its chain
    rule share with it: its `halo` and SubDomain only if `masked`.
    """
    return {'computational': True, 'transpose': deriv.transpose,
            'halo': deriv.halo if masked else None,
            'subdomain': deriv.subdomain if masked else None}


def _parts(expr, grid):
    """
    `expr` as the sum of parts whose values are each at one point: its
    products distributed over the sums they hold, and its terms gathered by
    the point of their values. A single part if they all share it.
    """
    dims = _space_dimensions(expr, grid)
    if len({_location(t, dims) for t in _terms_of(expr)}) <= 1:
        return [expr]

    groups = {}
    for term in _distributed(expr):
        groups.setdefault(_location(term, dims), []).append(term)
    return [Add(*g) for g in groups.values()]


def _terms_of(expr):
    """The terms of `expr` if it is a sum, else `expr`."""
    return expr.args if expr.is_Add else (expr,)


def _distributed(expr):
    """The terms of `expr`, its products distributed over the sums they hold."""
    if expr.is_Add:
        return [t for a in expr.args for t in _distributed(a)]
    if expr.is_Mul and any(a.is_Add for a in expr.args):
        factors = [_distributed(a) if a.is_Add else [a] for a in expr.args]
        return [Mul(*terms) for terms in product(*factors)]
    return [expr]


def _location(expr, dims):
    """The point where the values of `expr` are, along each of `dims`."""
    return tuple(_position(expr, d) for d in dims)


def _space_dimensions(expr, grid):
    """
    The Dimensions `expr` is indexed by along each space Dimension of `grid`,
    e.g. the SubDimensions of a Function on a SubDomain.
    """
    indexed = {d.root: d for d in getattr(expr, 'dimensions', ()) if d.is_Space}
    return tuple(indexed.get(d, d) for d in grid.dimensions)


def _root(location):
    """`location` along the root Dimensions, e.g. `x` for a SubDimension `ix`."""
    return {d.root: sympy.sympify(v).subs(d, d.root) for d, v in location.items()}


def _index(expr, dim):
    """
    Where Devito indexes `expr` along `dim`, the point its stencils are taken
    from; the node for an expression that is not indexed along `dim`.
    """
    indices = getattr(expr, 'indices_ref', None)
    if indices is None or dim not in indices.getters:
        return dim
    return indices[dim]


def _written(deriv, dim, target, values, source):
    """
    Where the transpose `deriv` writes its values along `dim`.

    Devito mirrors the points of a transposed stencil about the node. Along a
    Dimension `deriv` differentiates along, or interpolates to another point
    along, its values are then at the mirror of its `x0`, `target`, about the
    node, shifted as its argument is: `2 dim - target + values - source`,
    `values` being where the values of its argument are and `source` where
    Devito indexes it. Along the other Dimensions they stay at `values`.
    """
    orders = dict(zip(deriv.dims, deriv.deriv_order, strict=True))
    if orders.get(dim, 0) > 0 or target != source:
        return 2*dim - target + values - source
    return values


def _position(expr, dim):
    """
    Where the values of `expr` are along `dim`.

    A Function's values are where it is indexed. A derivative is indexed as its
    argument but its values are at its `x0`, in the frame of its argument: e.g.
    `u.dx(x0=x + h_x/2)` is indexed at `x` and has its values at `x + h_x/2`.
    A transpose has its values where it writes them (see `_written`). An
    expression holding derivatives has their values if they are at one point,
    and otherwise, as on a Grid, is taken where Devito indexes it.
    """
    if getattr(expr, 'is_Derivative', False):
        index = _index(expr.expr, dim)
        target = expr.x0.get(dim, index)
        if expr.transpose == transpose:
            return _written(expr, dim, target, _position(expr.expr, dim), index)
        return target + _position(expr.expr, dim) - index

    held = {_position(a, dim) for a in getattr(expr, 'args', ())
            if a.has(sympy.Derivative)}
    if len(held) == 1:
        return held.pop()
    return _index(expr, dim)
