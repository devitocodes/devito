"""
Derivatives w.r.t. the physical coordinates of a CurvilinearGrid.

The Dimensions of a CurvilinearGrid index its computational coordinates `xi_j`.
A Derivative along one of them is, unless `computational=True`, a derivative
w.r.t. the physical coordinate `x_i` of the same axis, evaluated through the
chain rule with the metric terms `d xi_j/d x_i` where it is evaluated,

    d/dx_i = sum_j (d xi_j/d x_i) d/d xi_j,

and a second derivative along one physical Dimension through

    d^2/dx_i^2 = sum_jk (d xi_j/d x_i)(d xi_k/d x_i) d^2/(d xi_j d xi_k)
                 + sum_j (d^2 xi_j/d x_i^2) d/d xi_j.

Higher and mixed derivatives are nested first and second derivatives. The
transpose `.T` of a physical derivative is its exact discrete transpose, so a
self-adjoint operator is written as one, e.g. `-(J*b*u.dx).dx.T/J`, symmetric
in the inner product weighted by the Jacobian `J`.
"""

import sympy

from .differentiable import Add
from .finite_difference import index_at
from .tools import centered, transpose

__all__ = ['curvilinear_grid', 'physical_derivative']


def curvilinear_grid(deriv):
    """
    The CurvilinearGrid whose metric terms `deriv` is evaluated with, None if the
    chain rule does not apply: for a derivative w.r.t. the computational
    coordinates, on any other grid, or along no Dimension of the grid.
    """
    grid = getattr(deriv.expr, 'grid', None)
    if deriv.computational or grid is None or not grid.root.is_Curvilinear:
        return None
    if any(o > 0 and d in grid.root.dimensions
           for d, o in zip(deriv.dims, deriv.deriv_order, strict=True)):
        return grid.root
    return None


def physical_derivative(deriv, expr, grid, **kwargs):
    """
    Evaluate `deriv`, of `expr`, w.r.t. the physical coordinates of `grid`.

    The derivative is split into factors, each of order at most two along a
    single physical Dimension, or along the other Dimensions (e.g. time), nested
    the first one outermost. With method='RSFD', every computational first
    derivative is a rotated staggered one, at the same point.
    """
    if deriv.method == 'RSFD' and (deriv.ndims > 1 or deriv.deriv_order[0] != 1):
        raise ValueError(f"{deriv} uses method='RSFD', which only provides first "
                         "derivatives along a single Dimension")
    for d, o in zip(deriv.dims, deriv.fd_order, strict=True):
        if d in grid.dimensions and o > grid.space_order:
            raise ValueError(f"{deriv} has fd_order={o} along {d}, larger than the "
                             f"space_order={grid.space_order} of its CurvilinearGrid "
                             "metrics; increase CurvilinearGrid(space_order=...)")

    spec = [(d, o) for d, o in zip(deriv.dims, deriv.deriv_order, strict=True) if o > 0]
    factors = [((d, o),) for d, o in spec if d in grid.dimensions
               for o in [2]*(o // 2) + [1]*(o % 2)]
    if len(factors) > 2:
        raise NotImplementedError(f"{deriv} nests {len(factors)} derivatives along the "
                                  "physical coordinates, beyond the two that the halo "
                                  "of the CurvilinearGrid metrics supports")
    others = tuple((d, o) for d, o in spec if d not in grid.dimensions)
    if others:
        factors.insert(0, others)

    if len(factors) > 1:
        res = _nested(deriv, expr, factors, **kwargs)
    else:
        (dim, order), = factors[0]
        res = _chain_rule(deriv, expr, grid, dim, order, **kwargs)

    for e in deriv._ppsubs:
        res = res.xreplace(e)
    return res


def _derivative(deriv, expr, dims, deriv_order, fd_order, **kwargs):
    """
    The derivative of `expr` along `dims`, inheriting the metadata of `deriv`
    except for its postponed substitutions, applied once to the whole expansion.
    """
    return deriv._rebuild(expr, *dims, deriv_order=deriv_order, fd_order=fd_order,
                          subs=(), _ppsubs=(), **kwargs)


def _nested(deriv, expr, factors, **kwargs):
    """
    Evaluate the product of derivative `factors`, the first one outermost. The
    transpose reverses the product, so that it is the exact transpose of the
    forward derivative.
    """
    if any(d not in deriv.dims for d in deriv.x0) or deriv.weights is not None:
        raise NotImplementedError(f"{deriv} combines several derivatives with custom "
                                  "weights or with x0 along Dimensions it does not "
                                  "differentiate, which is not supported on a "
                                  "CurvilinearGrid")
    if deriv.transpose == transpose:
        factors = factors[::-1]

    nested = expr
    for factor in reversed(factors):
        dims, orders = zip(*factor, strict=True)
        nested = _derivative(deriv, nested, dims, orders,
                             tuple(deriv.fd_order[d] for d in dims),
                             x0={d: v for d, v in deriv.x0.items() if d in dims})
    return nested._evaluate(**kwargs)


def _chain_rule(deriv, expr, grid, dim, order, **kwargs):
    """
    Evaluate the first or second derivative `deriv` of `expr` along the physical
    Dimension `dim`, a sum of computational derivatives weighted by the metric
    terms, `sum_k c_k D_k expr`, or its exact transpose `sum_k D_k^T (c_k expr)`.
    """
    held = _held_derivatives(expr, grid)
    if len({_held_location(h, grid) for h in held}) > 1:
        terms = _terms_by_location(expr, grid)
        if len(terms) > 1:
            # Differentiation is linear, and each term is differentiated where
            # its derivatives are evaluated
            return Add(*[_chain_rule(deriv, t, grid, dim, order, **kwargs)
                         for t in terms])

    if order == 1:
        terms = _first_terms(deriv, expr, grid, dim, held)
    else:
        terms = _second_terms(deriv, expr, grid, dim)

    res = []
    for coeff, *term in terms:
        if coeff == 0:
            continue
        if deriv.transpose == transpose:
            res.append(_computational(deriv, coeff*expr, *term)._evaluate(**kwargs))
        else:
            res.append(_evaluate(coeff, **kwargs) *
                       _computational(deriv, expr, *term)._evaluate(**kwargs))
    return Add(*res)


def _evaluate(expr, **kwargs):
    """`expr` evaluated, or itself if it is a number."""
    try:
        return expr._evaluate(**kwargs)
    except AttributeError:
        return expr


def _computational(deriv, expr, dims, deriv_order, fd_order, kwargs):
    """
    The derivative of `expr` w.r.t. the computational coordinates `dims`, with
    the metadata of `deriv` overridden by `kwargs`, and, if `kwargs` holds an
    `interp` Dimension, interpolated along it to the origin of `deriv` at the
    same order: in transpose mode, the transpose of the interpolation comes first.
    """
    kwargs = dict(kwargs)
    interp = kwargs.pop('interp', None)
    if interp is None:
        return _derivative(deriv, expr, dims, deriv_order, fd_order,
                           computational=True, **kwargs)

    # The interpolation, a derivative of order zero
    interp = ((interp, 0), {'fd_order': fd_order, 'x0': {interp: deriv.x0[interp]},
                            'computational': True, 'transpose': deriv.transpose})
    if deriv.transpose == transpose:
        return _derivative(deriv, expr.diff(interp[0], **interp[1]), dims, deriv_order,
                           fd_order, computational=True, **kwargs)
    return _derivative(deriv, expr, dims, deriv_order, fd_order, computational=True,
                       **kwargs).diff(interp[0], **interp[1])


def _first_terms(deriv, expr, grid, dim, held):
    """
    The terms `(c_j, dims, deriv_order, fd_order, kwargs)` of the first
    derivative `deriv` of `expr` along the physical Dimension `dim`: the metric
    term `c_j = d xi_j/d x` where the derivative is evaluated, and the
    derivative along the computational coordinate `xi_j` (see `_computational`).

    Along `dim` itself, or with method='RSFD', the latter is the stencil of
    `deriv`. Along another Dimension, it is centered, and interpolated along
    `dim` from where it indexes `expr` to the origin, at the order of `deriv`,
    by shifting the derivative as a whole as a staggered pair is formed, so
    that the two terms of a pair interpolate alike.
    """
    # The derivatives `held` by `expr` are evaluated first, which loses the
    # location of the expression they differentiate: their values are indexed
    # at the nodes
    indexed = {d: d if held else _location(expr, d) for d in grid.dimensions}
    if deriv.transpose == transpose:
        # Where the derivative it transposes ends, at `x0`, which the values of
        # `expr` are indexed as
        location = {d: deriv.x0.get(d, indexed[d]) for d in grid.dimensions}
    else:
        # Shifted as the values of `expr` are from where they are indexed
        values = _values_location(expr, held, grid)
        location = {d: deriv.x0.get(d, indexed[d]) + values[d] - indexed[d]
                    for d in grid.dimensions}
    metrics = grid.terms_at(location)[0]

    fd_order = (deriv.fd_order[dim],)
    across = {'side': deriv.side if deriv.side is not None else centered,
              'x0': {k: v for k, v in deriv.x0.items() if k is not dim}}
    if deriv.x0.get(dim, indexed[dim]) != indexed[dim]:
        across['interp'] = dim

    terms = []
    for d in grid.dimensions:
        # A rotated staggered derivative is at the origin along every Dimension
        kwargs = {} if d is dim or deriv.method == 'RSFD' else across
        terms.append((metrics[(dim, d)], (d,), (1,), fd_order, kwargs))
    return terms


def _second_terms(deriv, expr, grid, dim):
    """
    The terms `(c, dims, deriv_order, fd_order, kwargs)` of the second
    derivative `deriv` of `expr` along the physical Dimension `dim`, with the
    metric terms and their derivatives where the derivative is evaluated.

    Only the term along `dim` itself exists on a Cartesian Grid, and keeps the
    default stencil. The others, e.g. first derivatives from the metric
    derivatives, are centered so they keep the accuracy of `fd_order` (the
    default first derivative of order 2 is one-sided).
    """
    location = {d: deriv.x0.get(d, _location(expr, d)) for d in grid.dimensions}
    metrics = grid.terms_at(location)[0]
    metric_derivatives = grid.metric_derivatives_at(location)

    # Terms `(coefficient, computational Dimensions, derivative orders)`
    terms = []
    for i, di in enumerate(grid.dimensions):
        for dj in grid.dimensions[i:]:
            coeff = metrics[(dim, di)]*metrics[(dim, dj)]
            if di is dj:
                terms.append((coeff, (di,), (2,)))
            else:
                terms.append((2*coeff, (di, dj), (1, 1)))
    terms.extend((metric_derivatives[(dim, d)], (d,), (1,)) for d in grid.dimensions)

    side = deriv.side if deriv.side is not None else centered
    fd_order = deriv.fd_order[dim]
    return [(coeff, dims, orders, (fd_order,)*len(dims),
             {'side': deriv.side if (dims, orders) == ((dim,), (2,)) else side})
            for coeff, dims, orders in terms]


def _held_derivatives(expr, grid):
    """
    The outermost derivatives in `expr` w.r.t. the physical coordinates of
    `grid`, evaluated before a derivative of `expr`, which is evaluated where
    their values are (see `_values_location`).
    """
    physical = {d for d in expr.find(sympy.Derivative)
                if not d.computational and any(i in grid.dimensions for i in d.dims)}
    inner = {i for d in physical for i in physical & d.expr.find(sympy.Derivative)}
    return frozenset(physical - inner)


def _location(expr, dim):
    """Where `expr` sits along `dim`, the node if it does not say."""
    location = index_at(expr, dim)
    return dim if location is None else location


def _held_location(held, grid):
    """Where the derivative `held` is evaluated, along every Dimension of `grid`."""
    return tuple(held.x0.get(d, _location(held.expr, d)) for d in grid.dimensions)


def _values_location(expr, held, grid):
    """
    Where the values of `expr` are along every Dimension of `grid`: for the
    derivatives `held` by `expr`, where they are evaluated.
    """
    if not held:
        return {d: _location(expr, d) for d in grid.dimensions}
    locations = {_held_location(h, grid) for h in held}
    if len(locations) > 1:
        raise NotImplementedError(f"{expr} holds derivatives evaluated at different "
                                  f"points, {sorted(locations, key=str)}, which a "
                                  "derivative on a CurvilinearGrid cannot "
                                  "differentiate as a whole")
    return dict(zip(grid.dimensions, locations.pop(), strict=True))


def _terms_by_location(expr, grid):
    """
    `expr` as a sum of terms, one per set of points where the derivatives held
    by its terms are evaluated.
    """
    terms = [t for a in sympy.Add.make_args(expr)
             for t in sympy.Add.make_args(sympy.expand(a, deep=False, mul=True,
                                                       multinomial=False,
                                                       power_exp=False,
                                                       power_base=False, log=False))]
    groups = {}
    for term in terms:
        key = frozenset(_held_location(h, grid) for h in _held_derivatives(term, grid))
        groups.setdefault(key, []).append(term)
    return [Add(*terms) for terms in groups.values()]
