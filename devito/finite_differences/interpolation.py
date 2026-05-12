from contextlib import suppress

__all__ = ['interp_at', 'interp_mapper', 'post_x0_indices']


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
        try:
            s = source[d]
            t = target[d]
        except (KeyError, IndexError):
            continue
        if s is not t:
            mapper[d] = t
    return mapper


def interp_at(expr, source, target, interp_order):
    """
    Build a symbolic 0-order FD interpolation operator on `expr` that maps
    values from `source` indices to `target` indices, only on the
    dimensions where the two locations differ.
    """
    from devito.finite_differences.differentiable import Differentiable

    if not isinstance(expr, Differentiable):
        return expr

    mapper = interp_mapper(source, target, expr.dimensions)
    if not mapper:
        return expr

    return expr.diff(*mapper.keys(),
                     deriv_order=(0,) * len(mapper),
                     fd_order=(interp_order,) * len(mapper),
                     x0=mapper)


def post_x0_indices(deriv, func):
    """
    Conceptual indices of `deriv` after setting `x0` on its own derivative
    dimensions to `func`'s indices. Derivative dims take `func`'s indices;
    other dims keep the underlying expression's natural location (so that
    `interp_for_fd` does not introduce a spurious second shift).
    """
    ref = {}
    for dim in deriv.dimensions:
        if dim in deriv.dims and dim in func.dimensions:
            ref[dim] = func.indices_ref[dim]
        else:
            with suppress(KeyError):
                ref[dim] = deriv.indices_ref[dim]
    return ref
