from contextlib import suppress

import sympy

from devito.types.lazy import Evaluable

__all__ = ['SubGridInterp', 'interp_at', 'interp_mapper', 'is_subgrid_read',
           'post_x0_indices']


class SubGridInterp(sympy.Expr, Evaluable):

    """
    A SubGrid Function read at the location of a parent-Grid Function,
    unevaluated.

    A SubGrid point `i` sits at parent coordinate `i*factor`, so reading it at
    the parent point `x` means evaluating at `x/factor` -- a fractional offset
    that varies with the iteration index, which the finite-difference
    machinery cannot turn into a compile-time stencil window.

    What makes it a stencil after all is that the offset is *periodic*. With
    `factor = p/q` the coordinate is `x*q/p`, so

        base = (x*q) // p        frac = ((x*q) % p) / p

    and `frac` depends on `x` only through `x % p`: there are `p` phases, and
    within a phase the tap weights are constants. Deriving that needs the
    iteration space, so it is done by a Cluster pass
    (`passes/clusters/subgrid.py`); this object just records *what* to relocate
    and survives FD lowering untouched, in the manner of `IndexDerivative`.

    Parameters
    ----------
    func : Function
        The SubGrid Function being read.
    target : Function
        The Function whose location to read it at.
    """

    __rargs__ = ('func', 'target')

    is_commutative = True

    def __new__(cls, func, target=None, **kwargs):
        # `target` rides along as a sympy arg so that generic reconstruction
        # (`expr.func(*expr.args)`, as `uxreplace` does) round-trips
        obj = sympy.Expr.__new__(cls, func, target)
        obj._func = func
        obj._target = target

        return obj

    @property
    def func_(self):
        """The SubGrid Function being read."""
        return self._func

    @property
    def target(self):
        """The Function whose location it is read at."""
        return self._target

    @property
    def subgrid(self):
        return self._func.function.grid

    @property
    def dtype(self):
        return self._func.function.dtype

    def __repr__(self):
        return f"SubGridInterp({self._func}, at={self._target})"

    __str__ = __repr__

    def _evaluate(self, **kwargs):
        # Deliberately inert: the relocation restructures the iteration space,
        # so it is lowered by a Cluster pass
        return self


def is_subgrid_read(func, target):
    """
    Whether reading `func` at `target`'s location is a SubGrid relocation.

    True when `func` lives on a SubGrid and `target` does not live on that same
    SubGrid. The exception matters: a Function read at a shift of its own
    Dimensions, as a recurrence does, is not a relocation.

    Only the Dimensions `target` actually iterates can be relocated, so a
    SubGrid sharing none of them is not a relocation either.
    """
    grid = getattr(func, 'grid', None)
    if grid is None or not getattr(grid, 'is_SubGrid', False):
        return False

    if getattr(target, 'grid', None) is grid:
        return False

    return any(d in target.dimensions for d in grid.parent.dimensions)


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
