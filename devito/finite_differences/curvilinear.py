"""
Derivatives w.r.t. the physical coordinates of a CurvilinearGrid.

The Dimensions of a CurvilinearGrid index its computational coordinates `xi_j`.
A Derivative along one of them is, unless `computational=True`, a derivative
w.r.t. the physical coordinate `x_i` of the same axis, evaluated through the
chain rule with the metric terms `d xi_j/d x_i` and the Jacobian `J` of the
grid. A first derivative takes one of two discrete forms:

* the chain rule, `C_i f = sum_j (d xi_j/d x_i) d f/d xi_j`, with the metric
  terms where the derivative is evaluated;
* the conservation form, `K_i f = 1/J sum_j d/d xi_j (J (d xi_j/d x_i) f)`, with
  `J (d xi_j/d x_i)` where `f` lives and `1/J` where the derivative is evaluated.

Both are the physical derivative by the metric identities, and `K_i` is minus
the adjoint of `C_i` in the inner product weighted by `J`. A derivative takes
the conservation form when it is the second one of a pair:

* the outer derivative of a nested one, e.g. of `(b*u.dx).dx`, which is then
  `K(b C u)`;
* a derivative bringing a Function staggered along its Dimension back to the
  nodes, e.g. of `v_x` in `grad(v)`, its pair being the derivative from the
  nodes to the staggered points, e.g. of `tau_xx` in `div(tau)`.

Paired with the chain rule of the first derivative, the second one is minus its
adjoint, as the staggered pair is on a Cartesian Grid, where both forms are the
same stencil, so that operators such as `(b*u.dx).dx`, or `div` and `grad` of a
staggered velocity-stress system, are adjoint in the inner product weighted by
`J`.

In transpose mode, each form is replaced by its exact discrete transpose, `c*D`
by `D.T(c*.)`, and transposition swaps the roles of a nested pair: in
`(b*g.dx.T).dx.T`, the transpose of `(b*f.dx).dx`, the outer `.dx.T` is the
transpose of the inner chain rule and the inner `g.dx.T` the transpose of the
outer conservation form. The outer derivative therefore marks the physical
derivatives it holds to take the transpose of the conservation form, which
keeps the transpose of up to two nested derivatives exact.

A second derivative along one physical Dimension is evaluated through the chain
rule,

    d^2/dx_i^2 = sum_jk (d xi_j/d x_i)(d xi_k/d x_i) d^2/(d xi_j d xi_k)
                 + sum_j (d^2 xi_j/d x_i^2) d/d xi_j

and higher and mixed derivatives are nested first and second derivatives.
"""

from contextlib import suppress

import sympy

from .differentiable import Add, interp_for_fd
from .finite_difference import index_at
from .tools import centered, transpose

__all__ = ['curvilinear_grid', 'physical_derivative']


def curvilinear_grid(deriv):
    """
    The CurvilinearGrid whose metric terms `deriv` is evaluated with, None if the
    chain rule does not apply: on any other grid, for a derivative w.r.t. the
    computational coordinates, or for one along no Dimension of the grid.
    """
    if deriv.computational:
        return None
    grid = getattr(deriv.expr, 'grid', None)
    if grid is None or not grid.root.is_Curvilinear:
        return None
    grid = grid.root
    if not any(d in grid.dimensions and o > 0
               for d, o in zip(deriv.dims, deriv.deriv_order, strict=True)):
        return None
    return grid


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
        res = _along(deriv, expr, grid, *factors[0][0], **kwargs)

    for e in deriv._ppsubs:
        res = res.xreplace(e)
    return res


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
        dims = tuple(d for d, _ in factor)
        x0 = {d: v for d, v in deriv.x0.items() if d in dims}
        nested = deriv._rebuild(nested, *dims, deriv_order=tuple(o for _, o in factor),
                                fd_order=tuple(deriv.fd_order[d] for d in dims),
                                x0=x0, subs=(), _ppsubs=())
    return nested._evaluate(**kwargs)


def _along(deriv, expr, grid, dim, deriv_order, **kwargs):
    """
    Evaluate the first or second derivative `deriv` of `expr` along the physical
    Dimension `dim`, in the form of its place in a pair (see the module
    docstring).
    """
    held = _held_derivatives(expr, grid)
    staggered = (not held and deriv.x0.get(dim) == dim and
                 _location(expr, dim) != dim)
    if deriv.transpose == transpose:
        conservative = deriv._conservative or staggered
        expr = expr.xreplace({h: h._rebuild(_conservative=True) for h in held})
        held = _held_derivatives(expr, grid)
    else:
        conservative = bool(held) or staggered

    if deriv_order == 1 and conservative:
        return _conservation_form(deriv, expr, grid, dim, held, **kwargs)
    elif deriv_order == 1:
        return _first_chain_rule(deriv, expr, grid, dim, held, **kwargs)
    return _second_chain_rule(deriv, expr, grid, dim, **kwargs)


def _held_derivatives(expr, grid):
    """
    The outermost derivatives in `expr` w.r.t. the physical coordinates of
    `grid`, which make a derivative of `expr` the outer one of a nested pair.
    """
    physical = {d for d in expr.find(sympy.Derivative)
                if not d.computational and any(i in grid.dimensions for i in d.dims)}
    inner = {i for d in physical for i in physical & d.expr.find(sympy.Derivative)}
    return frozenset(physical - inner)


def _location(expr, dim):
    """Where `expr` sits along `dim`, the node if it does not say."""
    location = index_at(expr, dim)
    return dim if location is None else location


def _indexed_location(expr, held, dim):
    """
    Where a derivative of `expr` indexes it along `dim`. The derivatives `held`
    by `expr` are evaluated first, which loses the location of the expression
    they differentiate: their values are indexed at the nodes.
    """
    return dim if held else _location(expr, dim)


def _values_location(expr, held, dim):
    """
    Where the values of `expr` are along `dim`: for the derivatives `held` by
    `expr`, where they are evaluated, their `x0` or the location of the
    expression they differentiate.
    """
    if not held:
        return _location(expr, dim)
    locations = {h.x0.get(dim, _location(h.expr, dim)) for h in held}
    if len(locations) > 1:
        raise NotImplementedError(f"{expr} holds derivatives evaluated at different "
                                  f"points along {dim}, {sorted(locations, key=str)}, "
                                  "which a derivative on a CurvilinearGrid cannot "
                                  "differentiate as a whole")
    return locations.pop()


def _output_location(deriv, expr, held, grid):
    """
    Where `deriv` of `expr` is evaluated, along every Dimension of `grid`: at its
    `x0`, or where it indexes `expr`, shifted as the values of `expr` are from
    where they are indexed.
    """
    location = {}
    for d in grid.dimensions:
        indexed = _indexed_location(expr, held, d)
        location[d] = (deriv.x0.get(d, indexed) + _values_location(expr, held, d) -
                       indexed)
    return location


def _computational(deriv, expr, dims, deriv_order, fd_order, **kwargs):
    """
    The derivative of `expr` w.r.t. the computational coordinates `dims`,
    inheriting the metadata of `deriv` except for its postponed substitutions,
    applied once to the whole chain-rule expansion.
    """
    return deriv._rebuild(expr, *dims, deriv_order=deriv_order, fd_order=fd_order,
                          computational=True, subs=(), _ppsubs=(), **kwargs)


def _conservation_form(deriv, expr, grid, dim, held, **kwargs):
    """
    The first derivative `deriv` of `expr` along the physical Dimension `dim` in
    conservation form, or its exact transpose in transpose mode.
    """
    origin = _output_location(deriv, expr, held, grid)
    # The Jacobian where the derivative ends
    jacobian = grid.terms_at(origin)[1]
    # The metric terms and the Jacobian where the values of `expr` are, as the
    # chain rule of its pair takes them
    argument = {**origin, dim: _values_location(expr, held, dim)}
    metrics, weight_jacobian = grid.terms_at(argument)

    res = []
    for d in grid.dimensions:
        metric = metrics[(dim, d)]
        if metric == 0:
            continue
        weight = weight_jacobian*metric
        if deriv.transpose == transpose:
            res.append(weight*_computational_term(deriv, expr/jacobian, dim, d, held,
                                                  **kwargs))
        else:
            res.append(_computational_term(deriv, weight*expr, dim, d, held,
                                           **kwargs))

    res = Add(*res)
    return res if deriv.transpose == transpose else res/jacobian


def _first_chain_rule(deriv, expr, grid, dim, held, **kwargs):
    """
    The first derivative `deriv` of `expr` along the physical Dimension `dim`
    through the chain rule, with the metric terms where it is evaluated, or its
    exact transpose in transpose mode.
    """
    metrics = grid.terms_at(_output_location(deriv, expr, held, grid))[0]
    res = []
    for d in grid.dimensions:
        coeff = metrics[(dim, d)]
        if coeff == 0:
            continue
        if deriv.transpose == transpose:
            res.append(_computational_term(deriv, coeff*expr, dim, d, held,
                                           **kwargs))
        else:
            res.append(coeff*_computational_term(deriv, expr, dim, d, held,
                                                 **kwargs))
    return Add(*res)


def _computational_term(deriv, expr, dim, d, held, **kwargs):
    """
    The evaluated first derivative of `expr` w.r.t. the computational coordinate
    `d`, at the origin of `deriv` along the physical Dimension `dim`, or its exact
    transpose in transpose mode.

    Along `dim` itself, or with method='RSFD', it is the stencil of `deriv`.
    Along another Dimension, it is centered, and interpolated along `dim` from
    where it indexes `expr` to the origin, at the order of `deriv`, by shifting
    the derivative as a whole as a staggered pair is formed, so that the two
    terms of a pair interpolate alike.
    """
    fd_order = (deriv.fd_order[dim],)
    # A rotated staggered derivative is at the origin along every Dimension
    if d is dim or deriv.method == 'RSFD':
        return _computational(deriv, expr, (d,), (1,), fd_order)._evaluate(**kwargs)

    side = deriv.side if deriv.side is not None else centered
    origin = {k: v for k, v in deriv.x0.items() if k is not dim}
    location = _indexed_location(expr, held, dim)
    if deriv.x0.get(dim, location) == location:
        return _computational(deriv, expr, (d,), (1,), fd_order, side=side,
                              x0=origin)._evaluate(**kwargs)

    # The interpolation along `dim`, a derivative of order zero
    interp = {'fd_order': fd_order, 'x0': {dim: deriv.x0[dim]}, 'computational': True,
              'transpose': deriv.transpose}
    if deriv.transpose == transpose:
        return _computational(deriv, expr.diff((dim, 0), **interp), (d,), (1,),
                              fd_order, side=side, x0=origin)._evaluate(**kwargs)
    term = _computational(deriv, expr, (d,), (1,), fd_order, side=side, x0=origin)
    return term.diff((dim, 0), **interp)._evaluate(**kwargs)


def _second_chain_rule(deriv, expr, grid, dim, **kwargs):
    """
    The second derivative `deriv` of `expr` along the physical Dimension `dim`
    through the chain rule, or its exact transpose in transpose mode.
    """
    fd_order = deriv.fd_order[dim]
    metrics = grid.metrics

    # Terms `(coefficient, computational Dimensions, derivative orders)`
    terms = []
    for i, di in enumerate(grid.dimensions):
        for dj in grid.dimensions[i:]:
            coeff = metrics[(dim, di)]*metrics[(dim, dj)]
            if di is dj:
                terms.append((coeff, (di,), (2,)))
            else:
                terms.append((2*coeff, (di, dj), (1, 1)))
    for d in grid.dimensions:
        terms.append((grid.metric_derivatives[(dim, d)], (d,), (1,)))

    res = []
    for coeff, dims, orders in terms:
        if coeff == 0:
            continue
        # The metric terms are evaluated at the derivative's origin
        coeff = interp_for_fd(coeff, deriv.x0, **kwargs)
        fd_orders = (fd_order,)*len(dims)
        # Only the term along `dim` itself exists on a Cartesian Grid, and keeps
        # the default stencil. The others, e.g. first derivatives from the metric
        # derivatives, are centered so they keep the accuracy of `fd_order` (the
        # default first derivative of order 2 is one-sided)
        side = deriv.side
        if side is None and (dims, orders) != ((dim,), (2,)):
            side = centered
        if deriv.transpose == transpose:
            term = _computational(deriv, coeff*expr, dims, orders, fd_orders, side=side)
            res.append(term._evaluate(**kwargs))
        else:
            term = _computational(deriv, expr, dims, orders, fd_orders, side=side)
            with suppress(AttributeError):
                coeff = coeff._evaluate(**kwargs)
            res.append(coeff*term._evaluate(**kwargs))

    return Add(*res)
