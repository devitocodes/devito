from collections.abc import Mapping

import sympy

from devito.finite_differences.closure import Closure
from devito.finite_differences.tools import transpose

__all__ = ['index', 'interp_at', 'interp_mapper', 'position', 'written_at']


def interp_mapper(source, target, dims):
    """
    Build a `{dim: target_index}` mapper for dimensions in `dims` where
    `source[dim]` differs from `target[dim]`.

    `source` and `target` are dict-like `{dim: index_expr}` (e.g. a plain
    dict or a `DimensionTuple`). Dimensions missing from either side are
    skipped silently.
    """
    mapper = {}
    for d in dims:
        if d.is_Time:
            continue
        try:
            s = source[d]
            t = target[d]
        except (KeyError, IndexError):
            continue
        if s is not t:
            mapper[d] = t
    return mapper


def interp_at(expr, target, interp_order):
    """
    Build a symbolic 0-order FD interpolation operator on `expr` that maps its
    values from where they are (see `position`) to `target`, only on the
    dimensions where the two locations differ.
    """
    from devito.finite_differences.differentiable import Differentiable

    if not isinstance(expr, Differentiable):
        return expr

    source = {d: position(expr, d) for d in expr.dimensions}
    mapper = interp_mapper(source, target, expr.dimensions)
    if not mapper:
        return expr

    # The `x0` of the interpolation is an offset from where `expr` is indexed
    x0 = {d: v - source[d] + index(expr, d) for d, v in mapper.items()}
    # It is closed as the derivatives it interpolates are
    closures = {d: w for i in expr.find(sympy.Derivative)
                if isinstance(i.weights, Mapping) for d, w in i.weights.items()
                if isinstance(w, Closure) and d in mapper}
    return expr.diff(*mapper.keys(),
                     deriv_order=(0,) * len(mapper),
                     fd_order=(interp_order,) * len(mapper),
                     x0=x0, weights=closures or None)


def index(expr, dim):
    """
    Return the index of `expr` along `dim`, `dim` itself if `expr` is not indexed
    along it.
    """
    indices = getattr(expr, 'indices_ref', None)
    if indices is None or dim not in indices.getters:
        return dim
    return indices[dim]


def written_at(deriv, dim, evaluated, located, indexed):
    """
    Return where the transpose `deriv` of an argument indexed at `indexed` and
    located at `located` writes along `dim`, evaluated at `evaluated`.

    A transposed stencil is mirrored about the index. So along a Dimension that
    `deriv` differentiates or interpolates along, the output is at the mirror
    image of `evaluated`. Along the other Dimensions, it stays at `located`.
    """
    orders = dict(zip(deriv.dims, deriv.deriv_order, strict=True))
    if orders.get(dim, 0) > 0 or evaluated != indexed:
        return 2*dim - evaluated + located - indexed
    return located


def position(expr, dim):
    """
    Return where the values of `expr` are along `dim`.

    The `x0` of a derivative is an offset from where its argument is indexed, so a
    derivative may be indexed elsewhere than its values are:

    - A Function: where it is indexed.
    - A derivative: at its `x0`, offset by where its argument is. For example,
      `u.dx(x0=x + h_x/2)` is indexed at `x` but its values are at `x + h_x/2`.
    - A transposed derivative: where it writes its output.
    - Any other expression: at the location of the derivatives it contains, if
      they share one, else where it is indexed.
    """
    if getattr(expr, 'is_Derivative', False):
        indexed = index(expr.expr, dim)
        evaluated = expr.x0.get(dim, indexed)
        if expr.transpose == transpose:
            return written_at(expr, dim, evaluated, position(expr.expr, dim), indexed)
        return evaluated + position(expr.expr, dim) - indexed

    held = {position(a, dim) for a in getattr(expr, 'args', ())
            if a.has(sympy.Derivative)}
    if len(held) == 1:
        return held.pop()
    return index(expr, dim)
