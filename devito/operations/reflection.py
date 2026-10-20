"""
Reflections through the rows of a CurvilinearGrid.

On a Grid, an equation that writes a Function at one point and reads Functions
at the mirror point through a row of the last Dimension, e.g. the odd mirror of
a free surface at the row `s`,

    Eq(u.forward.subs(z, s - k), -u.forward.subs(z, s + k))

mirrors along the grid lines, which are normal to the rows. On a
CurvilinearGrid, the rows are curved surfaces and the grid lines are not normal
to them, so the read is the value at the reflection of the written point
through the surface of the row, normal to it, interpolated from the points at or
below the surface (see `CurvilinearGrid._reflection_weights`). An increment that
writes at the mirror point of the point it reads,

    Inc(v.subs(z, s + k), -v.subs(z, s - k))

is the transpose: it spreads the value read over the points that interpolate
its reflection, with the same weights. On a plane surface, either one is the
exact mirror of a field odd or even through the surface, and on a flat surface
it is the mirror along the grid lines.

The rows written may also vary over a Dimension of known range, e.g. over a
SubDomain of the rows above the surface,

    Eq(u.forward, -u.forward.subs(z, 2*s - z), subdomain=mirror)

which is lowered to one equation per row. Each row is written over the
SubDomain of that row, so that the equations hold under domain decomposition,
where a constant index of a distributed Dimension would be a local index.
"""

import numpy as np
import sympy

from devito.symbolics import retrieve_functions, retrieve_indexed
from devito.tools import memoized_func
from devito.types.equation import Eq, Inc
from devito.types.grid import SubDomain

__all__ = ['lower_reflections']


def lower_reflections(expressions):
    """
    Evaluate the reads at mirror points through the rows of a CurvilinearGrid
    in `expressions`, scalar equations whose derivatives are evaluated, as reads
    at the reflections normal to the rows. The other equations are unchanged.
    """
    processed, increments = [], []
    for expr in expressions:
        lowered = _lower_reflection(expr)
        if isinstance(expr, Inc) and lowered != [expr]:
            increments.extend(lowered)
            continue
        processed.extend(_merge_increments(increments) + lowered)
        increments = []
    return processed + _merge_increments(increments)


def _merge_increments(increments):
    """
    The increments of consecutive `Inc`s writing at mirror points, each over the
    SubDomain of its row (see `_on_row`), summed per point written and per kind
    of read, in its own column or at lateral offsets, unless one reads a point
    another writes.

    They are written `Eq(u, u + ...)`: an `Inc` over a SubDomain reading the
    Function it increments elsewhere becomes an array reduction, whose private
    copies of the Function read zero.
    """
    written = {(i.lhs.function, _row(i, i.lhs)) for i in increments}
    if any((a.function, _row(i, a)) in written
           for i in increments for a in _accesses(i.rhs)):
        merged = {(i.lhs, i.subdomain, i.implicit_dims, n): i.rhs
                  for n, i in enumerate(increments)}
    else:
        merged = {}
        for i in increments:
            key = (i.lhs, i.subdomain, i.implicit_dims, _reads_laterally(i))
            merged[key] = merged[key] + i.rhs if key in merged else i.rhs
    return [Eq(lhs, lhs + rhs, subdomain=subdomain, implicit_dims=implicit_dims)
            for (lhs, subdomain, implicit_dims, _), rhs in merged.items()]


def _row(eq, access):
    """The row of `access` in `eq`, written over the SubDomain of its row."""
    vertical = access.function.grid.dimensions[-1]
    indices = dict(zip(access.function.dimensions, _indexed(access).indices,
                       strict=True))
    return eq.subdomain.start + indices.get(vertical, vertical) - vertical


def _reads_laterally(eq):
    """True if `eq` reads a Function at lateral offsets from the point it writes."""
    grid = eq.lhs.function.grid
    written = _lateral_indices(eq.lhs, grid)
    return any(_lateral_indices(a, grid) != written for a in _accesses(eq.rhs)
               if _grid(a.function) is grid)


class Row(SubDomain):

    """The row `start` of the last Dimension of a grid, over its other Dimensions."""

    def __init__(self, start, grid):
        self.name = f'reflection_row{start}'
        self.start = start
        self._size = grid.shape[-1]
        super().__init__(grid=grid)

    def define(self, dimensions):
        *lateral, vertical = dimensions
        return {**{d: d for d in lateral},
                vertical: ('middle', self.start, self._size - self.start - 1)}


@memoized_func
def _row_subdomain(grid, start):
    """The Row `start` of `grid`, shared by the equations writing it."""
    return Row(start, grid)


def _grid(function):
    """The CurvilinearGrid of a Function, None for any other object."""
    grid = getattr(function, 'grid', None)
    if (grid is None or not grid.root.is_Curvilinear or
            not getattr(function, 'is_DiscreteFunction', False) or
            function.is_SparseFunction):
        return None
    grid = grid.root
    if not all(d in function.dimensions for d in grid.dimensions):
        return None
    return grid


def _indexed(access):
    """The Indexed of a Function access, or the Indexed itself."""
    return access if isinstance(access, sympy.Indexed) else access.indexify()


def _accesses(expr):
    """The Function accesses and the Indexeds of `expr`."""
    indexeds = list(retrieve_indexed(expr))
    # The Functions of the Indexeds are not accesses
    hidden = sympy.sympify(expr).xreplace({i: sympy.Dummy() for i in indexeds})
    return list(retrieve_functions(hidden)) + indexeds


def _positions(access, grid):
    """
    The position of a Function access or of an Indexed, in cells from the
    nodes, along each Dimension of `grid`.
    """
    function = access.function
    indices = dict(zip(function.dimensions, _indexed(access).indices, strict=True))
    return tuple(sympy.expand(indices[d] + o)
                 for d, o in zip(grid.dimensions, _offsets(function, grid),
                                 strict=True))


def _offsets(function, grid):
    """The staggering of `function`, in cells, along each Dimension of `grid`."""
    refs = dict(zip(function.dimensions, function.indices_ref, strict=True))
    return tuple(sympy.nsimplify((refs[d] - d).xreplace({d.spacing: 1}))
                 for d in grid.dimensions)


def _center(lhs_position, position):
    """
    The position of the row the two positions are the mirror of each other
    through, laterally aligned, None if they are not.
    """
    *lateral_lhs, vertical_lhs = lhs_position
    *lateral, vertical = position
    pairs = zip(lateral_lhs, lateral, strict=True)
    if any(sympy.expand(a - b) != 0 for a, b in pairs):
        return None
    center = sympy.expand(vertical_lhs + vertical)/2
    if not center.is_Number or sympy.expand(vertical_lhs - vertical) == 0:
        return None
    return center


def _check_halo(function, grid, smallest, window):
    """
    Raise an error unless the lateral offsets `smallest` to `smallest + window -
    1` along the lateral Dimensions of `grid` read `function` within its halo.
    """
    for d, s, n in zip(grid.dimensions[:-1], smallest, window, strict=True):
        halo = function._size_halo[d]
        if s < -halo.left or s + n - 1 > halo.right:
            raise ValueError(f"The reflection reads {function} at lateral offsets "
                             f"{s} to {s + n - 1}, beyond its halo; smooth the "
                             "surface or increase the space order")


def _shifted(access, grid, lateral=None, vertical=None):
    """
    The Indexed of `access` shifted by `lateral` cells along the lateral
    Dimensions of `grid`, and at the row `vertical` if given.
    """
    mapper = dict(zip(access.function.dimensions, _indexed(access).indices,
                      strict=True))
    for d, o in zip(grid.dimensions[:-1], lateral or (0,)*(grid.dim - 1), strict=True):
        mapper[d] = mapper[d] + o
    if vertical is not None:
        mapper[grid.dimensions[-1]] = vertical
    return access.function.indexed[tuple(mapper[d] for d in access.function.dimensions)]


def _lateral_indices(access, grid):
    """The indices of `access` along the lateral Dimensions of `grid`."""
    indices = dict(zip(access.function.dimensions, _indexed(access).indices,
                       strict=True))
    return tuple(indices[d] for d in grid.dimensions[:-1])


def _bounds(dim, eq, grid):
    """The first and last values of the Dimension `dim` in `eq`, None if unknown."""
    if dim is grid.dimensions[-1]:
        size = grid.shape[-1]
        definition = dim if eq.subdomain is None else \
            eq.subdomain.define(grid.dimensions).get(dim, dim)
        if definition is dim:
            return 0, size - 1
        side, *thickness = definition
        if side == 'left':
            return 0, thickness[0] - 1
        elif side == 'right':
            return size - thickness[0], size - 1
        return thickness[0], size - 1 - thickness[1]
    first, last = getattr(dim, 'symbolic_min', None), getattr(dim, 'symbolic_max', None)
    if isinstance(first, sympy.Number) and isinstance(last, sympy.Number):
        return int(first), int(last)
    return None


def _rows(eq, grid, dim):
    """
    The equations of `eq`, whose mirror points vary with the Dimension `dim`, one
    per value of `dim`, which write and read constant rows.

    A distributed Function written over a range of rows and read at lateral
    offsets in the rows of their mirror points cannot be told apart from a
    dependence along the lateral Dimensions, which would serialize them and
    drop their halo exchanges; with constant rows, it can.
    """
    bounds = _bounds(dim, eq, grid)
    if bounds is None:
        raise ValueError(f"{eq} reads mirror points through a row of a "
                         f"CurvilinearGrid over the values of {dim}, which are "
                         "unknown; write one equation per row")
    subdomain = eq.subdomain
    if subdomain is not None:
        definition = subdomain.define(grid.dimensions)
        if any(definition.get(d, d) is not d for d in grid.dimensions[:-1]):
            raise NotImplementedError(f"{eq} reads mirror points through a row of "
                                      "a CurvilinearGrid over a SubDomain that is "
                                      "not made of whole rows")
        subdomain = None
    return [eq.func(eq.lhs.subs(dim, value), eq.rhs.subs(dim, value),
                    subdomain=subdomain, implicit_dims=eq.implicit_dims)
            for value in range(bounds[0], bounds[1] + 1)]


def _ghost_row(center, position, offsets):
    """
    The ghost row of the reflection weights of a ghost at the vertical
    `position` above the row `center`, of a Function staggered by `offsets`.
    """
    fraction = (center - offsets[-1]) % 1
    return int(center - position - 1 + fraction)


def _lower_reflection(eq):
    """The equations evaluating the reads at mirror points of `eq`."""
    lhs = eq.lhs
    grid = _grid(getattr(lhs, 'function', None))
    if grid is None or not (lhs.is_Function or isinstance(lhs, sympy.Indexed)):
        return [eq]

    lhs_position = _positions(lhs, grid)
    reflected, centers = [], set()
    for access in _accesses(eq.rhs):
        if _grid(access.function) is not grid:
            continue
        center = _center(lhs_position, _positions(access, grid))
        if center is not None:
            reflected.append(access)
            centers.add(center)
    if not reflected:
        return [eq]
    if len(centers) > 1:
        raise ValueError(f"{eq} reads mirror points through several rows "
                         f"{sorted(centers)}")
    center = centers.pop()

    dims = [d for d in lhs_position[-1].free_symbols
            if getattr(d, 'is_Dimension', False)]
    if len(dims) > 1:
        raise NotImplementedError(f"{eq} reads mirror points through a row of a "
                                  f"CurvilinearGrid over several Dimensions {dims}")
    elif dims:
        return [e for i in _rows(eq, grid, dims[0]) for e in _lower_reflection(i)]

    if eq.subdomain is not None:
        raise NotImplementedError(f"{eq} writes a constant row over a SubDomain")
    if isinstance(eq, Inc):
        eqns = _scatter(eq, grid, center, reflected)
    elif eq.is_Reduction:
        raise NotImplementedError(f"{eq} reads mirror points through a row of a "
                                  "CurvilinearGrid, which only an Eq or an Inc can")
    else:
        eqns = _gather(eq, grid, center, reflected)
    return [_on_row(e, grid) for e in eqns]


def _on_row(eq, grid):
    """
    `eq`, which writes a constant row, over the SubDomain of the row, with the
    constant rows it reads relative to it: a constant index of a distributed
    Dimension is a local index on each rank.
    """
    vertical = grid.dimensions[-1]
    row = dict(zip(eq.lhs.function.dimensions, _indexed(eq.lhs).indices,
                   strict=True))[vertical]
    mapper = {a: _relative(a, vertical, row) for a in _accesses(eq.rhs)
              if vertical in a.function.dimensions}
    return eq.func(_relative(eq.lhs, vertical, row), eq.rhs.xreplace(mapper),
                   subdomain=_row_subdomain(grid, int(row)),
                   implicit_dims=eq.implicit_dims)


def _relative(access, vertical, row):
    """
    The Indexed of `access`, at a constant index along `vertical`, at that
    index relative to the row `row` of `vertical`.
    """
    indices = dict(zip(access.function.dimensions, _indexed(access).indices,
                       strict=True))
    if not sympy.sympify(indices[vertical]).is_Number:
        return _indexed(access)
    indices[vertical] = vertical + indices[vertical] - row
    return access.function.indexed[tuple(indices[d] for d in access.function.dimensions)]


def _linear_parts(rhs, accesses):
    """
    The part of `rhs` independent of `accesses` and the factor of each, None if
    `rhs` is not linear in them.
    """
    symbols = {a: sympy.Dummy() for a in accesses}
    expr = rhs.xreplace(symbols)
    zero = {s: 0 for s in symbols.values()}
    constant = expr.xreplace(zero)
    factors = {a: expr.xreplace({**zero, s: 1}) - constant for a, s in symbols.items()}
    pairs = zip(factors.values(), symbols.values(), strict=True)
    rest = constant + sum(f*s for f, s in pairs)
    if sympy.expand(expr - rest) != 0:
        return None
    return constant, factors


def _gather(eq, grid, center, reflected):
    """
    The equations of `eq`, which writes a ghost above the surface and reads
    Functions at its mirror point, reading them at its reflection.

    The reads of the written Function at lateral offsets are added by a second
    equation: an equation of a distributed Function reading it both in its own
    column and at lateral offsets misses the halo exchange of the lateral reads.
    """
    lhs = eq.lhs
    ghost = _offsets(lhs.function, grid)
    row = _ghost_row(center, _positions(lhs, grid)[-1], ghost)
    lateral_index = _lateral_indices(lhs, grid)

    column, lateral = {}, {}
    for access in reflected:
        function = access.function
        weights, table, first, start = grid._reflection_weights_function(
            center, ghost, _offsets(function, grid), function.space_order, (row,))
        _check_halo(function, grid, first, table.shape[grid.dim:-1])
        column[access], lateral[access] = 0, 0
        for stencil in np.ndindex(*table.shape[grid.dim:]):
            if not table[(Ellipsis,) + stencil].any():
                continue
            offsets = tuple(f + s for f, s in zip(first, stencil, strict=False))
            term = (weights.indexed[lateral_index + (0,) + stencil] *
                    _shifted(access, grid, offsets, start + stencil[-1]))
            if any(offsets):
                lateral[access] += term
            else:
                column[access] += term

    parts = _linear_parts(eq.rhs, reflected)
    if parts is None:
        mapper = {a: column[a] + lateral[a] for a in reflected}
        return [eq.func(lhs, eq.rhs.xreplace(mapper), subdomain=eq.subdomain,
                        implicit_dims=eq.implicit_dims)]

    constant, factors = parts
    own = [a for a in reflected if a.function is lhs.function]
    first_rhs = constant + sum(f*(column[a] + (0 if a in own else lateral[a]))
                               for a, f in factors.items())
    eqns = [eq.func(lhs, first_rhs, subdomain=eq.subdomain,
                    implicit_dims=eq.implicit_dims)]
    if any(lateral[a] != 0 for a in own):
        eqns.append(eq.func(lhs, lhs + sum(factors[a]*lateral[a] for a in own),
                            subdomain=eq.subdomain, implicit_dims=eq.implicit_dims))
    return eqns


def _scatter(eq, grid, center, reflected):
    """
    The increments of `eq`, which reads a ghost above the surface and writes at
    its mirror point, writing at the points that interpolate its reflection.

    The reads of the right-hand side are at the ghost, or at the written point.
    Each target point gathers the ghosts that write it, in its own column first,
    then at lateral offsets, so that no point is written beyond the domain. The
    increments of consecutive `Inc`s are then merged (see `_merge_increments`).
    """
    lhs = eq.lhs
    lhs_position = _positions(lhs, grid)
    ghost_position = _positions(reflected[0], grid)
    at_lhs = []
    for access in _accesses(eq.rhs):
        if _grid(access.function) is not grid or access in reflected:
            continue
        pairs = zip(_positions(access, grid), lhs_position, strict=True)
        if any(sympy.expand(a - b) != 0 for a, b in pairs):
            raise ValueError(f"{eq} writes at the mirror point of the point it "
                             f"reads, where {access} is neither read")
        at_lhs.append(access)
    if any(_positions(a, grid) != ghost_position for a in reflected):
        raise ValueError(f"{eq} reads several points above the surface")

    ghost = _offsets(reflected[0].function, grid)
    row = _ghost_row(center, ghost_position[-1], ghost)
    lateral_index = _lateral_indices(lhs, grid)
    weights, table, smallest, start = grid._reflection_weights_function(
        center, ghost, _offsets(lhs.function, grid), lhs.function.space_order,
        (row,), transposed=True)
    for access in reflected:
        _check_halo(access.function, grid, smallest, table.shape[grid.dim:-1])

    eqns = []
    for r in range(table.shape[grid.dim - 1]):
        target = {a: _shifted(a, grid, vertical=start + r) for a in at_lhs}
        column, lateral = 0, 0
        for offset in np.ndindex(*table.shape[grid.dim:-1]):
            if not table[(Ellipsis, r) + offset + (slice(None),)].any():
                continue
            shift = tuple(s + o for s, o in zip(smallest, offset, strict=True))
            mapper = {**target, **{a: _shifted(a, grid, shift) for a in reflected}}
            term = (weights.indexed[lateral_index + (r,) + offset + (0,)] *
                    eq.rhs.xreplace(mapper))
            if any(shift):
                lateral += term
            else:
                column += term
        written = _shifted(lhs, grid, vertical=start + r)
        eqns.extend(eq.func(written, part, subdomain=eq.subdomain,
                            implicit_dims=eq.implicit_dims)
                    for part in (column, lateral) if part != 0)
    return eqns
