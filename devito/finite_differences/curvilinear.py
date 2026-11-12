"""
Derivatives on a CurvilinearGrid.

A derivative along a Dimension of a CurvilinearGrid is a derivative w.r.t. a
physical coordinate. Devito computes it with the chain rule. It takes
derivatives along the computational coordinates, which are ordinary finite
differences, and multiplies them by the metric terms of the map:

    d/dx_i = sum_j (d xi_j/d x_i) d/d xi_j,

    d^2/(dx_i dx_l) = sum_jk (d xi_j/d x_i)(d xi_k/d x_l) d^2/(d xi_j d xi_k)
                      + sum_j (d^2 xi_j/(d x_i d x_l)) d/d xi_j.

Higher derivatives nest these formulas. On a flat map, every term but one is
zero, and the result is the derivative of a Grid.

The transpose `.T` is the adjoint in the inner product weighted by the Jacobian
`J`. For `D = sum_j c_j D_j`, it is `D.T v = J^-1 sum_j D_j^T (J c_j v)`.
"""

from collections import Counter
from itertools import product

import sympy

from .differentiable import Add, Mul
from .tools import centered, transpose

__all__ = ['physical_grid', 'physical_derivative']


def physical_grid(deriv):
    """
    Return the CurvilinearGrid if `deriv` needs the chain rule, else None.

    A derivative needs the chain rule if it is taken along a Dimension of a
    CurvilinearGrid. A derivative along a computational Dimension does not.
    """
    # The argument of a Derivative can be a Dimension, which has no grid
    grid = getattr(deriv.expr, 'grid', None)
    if grid is None or not grid.root.is_Curvilinear:
        return None

    grid = grid.root
    if any(d in grid.computational_dimensions for d in deriv.dims) or \
            not any(d.root in grid.dimensions for d in deriv.dims):
        return None
    return grid


def physical_derivative(deriv, expr, grid, **kwargs):
    """
    Compute a derivative on a CurvilinearGrid with the chain rule.

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
        raise NotImplementedError(f"The transpose of {deriv}, a rotated staggered "
                                  "derivative, is not supported on a CurvilinearGrid")

    # Terms at different points need different metric terms
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
    Split `deriv` into derivatives of order at most 2 in space.

    Second derivatives stay whole, so that they keep their compact stencils. For
    example, `u.dx2dy` becomes `d/dy (d^2/dx^2 u)`. Each factor is a tuple of
    `(Dimension, order)` pairs. An interpolation is one empty factor.
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
    Write `deriv` as nested Derivatives, one per factor (see `_factors`).

    A transpose nests them in reverse order. The `x0` of a Dimension goes to the
    outermost factor that differentiates along it.
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
                              fd_indices=deriv.fd_indices, subs=())
    return expr


def _chain_rule(deriv, expr, grid, factor, **kwargs):
    """
    Compute one factor of a derivative (see `_factors`) with the chain rule.

    The result is `sum_k c_k D_k expr`, where `D_k` are derivatives along the
    computational Dimensions and `c_k` are metric terms. A transpose gives
    `J^-1 sum_k D_k^T (J c_k expr)` instead.
    """
    dims = _space_dimensions(expr, grid)

    # For each Dimension: where `expr` is indexed, where its values are, and where
    # the derivative is evaluated
    indexed = {d: _index(expr, d) for d in dims}
    located = {d: _position(expr, d) for d in dims}
    evaluated = {d: deriv.x0.get(d, indexed[d]) for d in dims}

    transposed = deriv.transpose == transpose
    if transposed:
        # Devito mirrors a transposed stencil about the node: with `x0`, it writes
        # at `2*d - x0 + located - indexed`. This `x0` makes it write at `written`
        at = located
        written = {d: _written(deriv, d, evaluated[d], located[d], indexed[d])
                   for d in dims}
        x0 = {d: 2*d - written[d] + located[d] - indexed[d] for d in dims}
    else:
        at = {d: evaluated[d] + located[d] - indexed[d] for d in dims}
        x0 = evaluated
    x0 = {**{d: v for d, v in deriv.x0.items() if d not in dims}, **x0}

    # The Dimensions along which `expr` must be interpolated
    moving = tuple(d for d in dims if evaluated[d] != indexed[d])

    res = []
    for coefficient, term_dims, orders in _terms(grid, dims, factor, at):
        if coefficient == 0:
            continue
        # The term along the derivative's own Dimensions. On a flat map it is the
        # only nonzero term
        own = dict(zip(term_dims, orders, strict=True)) == dict(factor)
        if transposed:
            weight = grid.jacobian_at(_root(at))*coefficient
            stencil = _computational(deriv, weight*expr, grid, term_dims, orders, x0,
                                     moving, factor, own)
            res.append(stencil._evaluate(**kwargs))
        else:
            stencil = _computational(deriv, expr, grid, term_dims, orders, x0, moving,
                                     factor, own)
            res.append(coefficient*stencil._evaluate(**kwargs))

    if transposed:
        return grid.inverse_jacobian_at(_root(written))*Add(*res)
    return Add(*res)


def _terms(grid, dims, factor, at):
    """
    Return the terms of the chain rule for `factor`.

    Each term is a tuple `(coefficient, dims, orders)`. The coefficient is a
    metric term at the point `at`. A factor along time, or any non-space
    Dimension, gives a single term with coefficient 1.
    """
    space = [d for d, o in factor for _ in range(o) if d in dims]
    others = tuple((d, o) for d, o in factor if d not in dims)
    if not space:
        return [(1, tuple(d for d, _ in others), tuple(o for _, o in others))]

    roots = {d.root: d for d in dims}
    computational = dict(zip(grid.dimensions, grid.computational_dimensions,
                             strict=True))
    metrics = grid.metrics_at(_root(at))
    if len(space) == 1:
        physical, = space
        return [(metrics[(physical.root, computational[j])], (roots[j],), (1,))
                for j in roots]

    # A second derivative needs products of metric terms and second derivatives of
    # the map
    xi, xl = space[0].root, space[1].root
    curvatures = grid.curvatures_at(_root(at))
    terms = [(curvatures[(xi, xl, computational[j])], (roots[j],), (1,)) for j in roots]
    for j, k in product(roots, repeat=2):
        coefficient = metrics[(xi, computational[j])]*metrics[(xl, computational[k])]
        if j == k:
            terms.append((coefficient, (roots[j],), (2,)))
        else:
            terms.append((coefficient, (roots[j], roots[k]), (1, 1)))
    return terms


def _computational(deriv, expr, grid, dims, orders, x0, moving, factor, own):
    """
    Return one derivative along the computational Dimensions.

    The derivative has orders `orders` along `dims` and is evaluated at `x0`. If
    `x0` moves `expr` along a Dimension in `moving` that is not in `dims`, the
    derivative also interpolates `expr` along it.

    The `own` term uses the stencil options of `deriv`. The other terms use
    centered stencils.
    """
    xi = {d: _computational_dimension(grid, d) for d in (*dims, *x0)}
    fd_indices = {_computational_dimension(grid, d): v
                  for d, v in deriv.fd_indices.items()}
    settings = {'method': deriv.method, 'transpose': deriv.transpose,
                'halo': deriv.halo, 'subdomain': deriv.subdomain,
                'fd_indices': {xi[d]: fd_indices[xi[d]] for d in dims
                               if xi[d] in fd_indices}}
    differentiated = [(xi[d], o) for d, o in zip(dims, orders, strict=True)]

    # Forward, the own term is the derivative of a Grid, including its interpolation
    if own and dims and deriv.transpose != transpose:
        return type(deriv)(expr, *differentiated,
                           fd_order=tuple(deriv.fd_order[d] for d in dims),
                           x0={xi[d] if d in dims else d: v for d, v in x0.items()},
                           side=deriv.side, weights=deriv.weights, **settings)

    # A rotated stencil is evaluated at `x0` along every Dimension, so it needs no
    # interpolation
    moved = () if deriv.method == 'RSFD' else tuple(d for d in moving if d not in dims)

    # The interpolation order: the order given for a zero-order Dimension of
    # `deriv`, otherwise the default of a Grid for the own term and the stencil
    # order for the others
    fd_order = max((deriv.fd_order[d] for d, _ in factor), default=max(deriv.fd_order))
    asked = {d: o for d, n, o in zip(deriv.dims, deriv.deriv_order, deriv.fd_order,
                                     strict=True) if n == 0}
    default = getattr(expr, 'interp_order', 2) if own else fd_order
    interp = {d: asked.get(d, default) for d in moved}
    if not dims and not interp:
        return expr

    if own:
        if interp and (deriv.weights is not None or deriv.side not in (None, centered)):
            raise NotImplementedError(f"The transpose of {deriv}, with weights or a "
                                      "side, cannot interpolate along other Dimensions")
        fd_orders = tuple(deriv.fd_order[d] for d in dims)
        stencil = {'side': deriv.side, 'weights': deriv.weights}
    else:
        fd_orders = (fd_order,)*len(dims)
        stencil = {'side': centered}

    at = x0 if deriv.method == 'RSFD' else {d: x0[d] for d in (*dims, *interp) if d in x0}
    return type(deriv)(expr, *differentiated, *[(xi[d], 0) for d in interp],
                       fd_order=fd_orders + tuple(interp.values()),
                       x0={xi[d]: v for d, v in at.items()}, **stencil, **settings)


def _computational_dimension(grid, dim):
    """
    Return the computational Dimension that matches `dim`.

    For example, `xi` for `x` or for the SubDimension `ix`. A computational
    Dimension, or a non-space Dimension such as time, is returned unchanged.
    """
    if dim.root not in grid.dimensions or dim in grid.computational_dimensions:
        return dim
    return grid.computational_dimensions[grid.dimensions.index(dim.root)]


def _parts(expr, grid):
    """
    Split `expr` into sums of terms that are located at the same point.

    Return `[expr]` if all its terms are already at the same point.
    """
    dims = _space_dimensions(expr, grid)
    if len({_location(t, dims) for t in _terms_of(expr)}) <= 1:
        return [expr]

    groups = {}
    for term in _distributed(expr):
        groups.setdefault(_location(term, dims), []).append(term)
    return [Add(*g) for g in groups.values()]


def _terms_of(expr):
    """Return the terms of `expr` if it is a sum, else `(expr,)`."""
    return expr.args if expr.is_Add else (expr,)


def _distributed(expr):
    """Return the terms of `expr` with its products expanded."""
    if expr.is_Add:
        return [t for a in expr.args for t in _distributed(a)]
    if expr.is_Mul and any(a.is_Add for a in expr.args):
        factors = [_distributed(a) if a.is_Add else [a] for a in expr.args]
        return [Mul(*terms) for terms in product(*factors)]
    return [expr]


def _location(expr, dims):
    """Return the location of `expr` along each Dimension of `dims`."""
    return tuple(_position(expr, d) for d in dims)


def _space_dimensions(expr, grid):
    """
    Return the Dimensions that index `expr` along each space Dimension of
    `grid`.

    For a Function on a SubDomain, these are SubDimensions.
    """
    indexed = {d.root: d for d in getattr(expr, 'dimensions', ()) if d.is_Space}
    return tuple(indexed.get(d, d) for d in grid.dimensions)


def _root(location):
    """Replace each SubDimension in `location` with its parent Dimension."""
    return {d.root: sympy.sympify(v).subs(d, d.root) for d, v in location.items()}


def _index(expr, dim):
    """
    Return the index of `expr` along `dim`.

    Return `dim` itself if `expr` is not indexed along `dim`.
    """
    indices = getattr(expr, 'indices_ref', None)
    if indices is None or dim not in indices.getters:
        return dim
    return indices[dim]


def _written(deriv, dim, evaluated, located, indexed):
    """
    Return the location where the transpose `deriv` writes along `dim`.

    Devito mirrors a transposed stencil about the node. So along a Dimension
    that `deriv` differentiates or interpolates along, the output is at the
    mirror image of `evaluated`. Along the other Dimensions, the output stays at
    `located`.
    """
    orders = dict(zip(deriv.dims, deriv.deriv_order, strict=True))
    if orders.get(dim, 0) > 0 or evaluated != indexed:
        return 2*dim - evaluated + located - indexed
    return located


def _position(expr, dim):
    """
    Return the location of the values of `expr` along `dim`.

    - A Function: where it is indexed.
    - A derivative: at its `x0`. For example, `u.dx(x0=x + h_x/2)` is indexed at
      `x` but its values are at `x + h_x/2`.
    - A transposed derivative: where it writes its output.
    - Any other expression: at the location of the derivatives it contains, if
      they share one, else where it is indexed.
    """
    if getattr(expr, 'is_Derivative', False):
        index = _index(expr.expr, dim)
        evaluated = expr.x0.get(dim, index)
        if expr.transpose == transpose:
            return _written(expr, dim, evaluated, _position(expr.expr, dim), index)
        return evaluated + _position(expr.expr, dim) - index

    held = {_position(a, dim) for a in getattr(expr, 'args', ())
            if a.has(sympy.Derivative)}
    if len(held) == 1:
        return held.pop()
    return _index(expr, dim)
