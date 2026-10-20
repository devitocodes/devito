"""
Reflections through the rows of a CurvilinearGrid.

On a Grid, the odd mirror of a free surface at the row `s` sets each row `z`
above it to minus the row as far below it, written over a SubDomain of these
rows,

    Eq(u.forward, -u.forward.subs(z, 2*s - z), subdomain=mirror)

and reads along the grid lines, which are normal to the rows. On a
CurvilinearGrid, the rows are curved surfaces and the grid lines are not normal
to them, so the read is the value at the reflection of the written point
through the surface of the row, normal to it, interpolated from the points at or
below the surface. An increment that writes at the mirror point of the point it
reads,

    Inc(v, -v.subs(z, 2*s - z), subdomain=below)

is the transpose: it spreads the value read over the points that interpolate
its reflection, with the same weights. On a plane surface, either one is the
exact mirror of a field odd or even through the surface, and on a flat surface
it is the mirror along the grid lines.

The rows written are indexed by the last Dimension over a SubDomain of whole
rows. A read at the mirror point negates that Dimension, which a halo exchange
of a distributed Dimension cannot follow, so each row is lowered to its own
equation, over the SubDomain of that row, reading the rows relative to it. An
equation writing constant rows is refused: a constant index of a distributed
Dimension is a local index on each rank.
"""

import numpy as np
import sympy

from devito.mpi import MPI
from devito.symbolics import retrieve_functions, retrieve_indexed
from devito.types.dense import Function
from devito.types.dimension import DefaultDimension
from devito.types.equation import Inc
from devito.types.grid import SubDomain

__all__ = ['lower_reflections']


def lower_reflections(expressions):
    """
    Evaluate the reads at mirror points through the rows of a CurvilinearGrid
    in `expressions`, scalar equations whose derivatives are evaluated, as reads
    at the reflections normal to the rows. The other equations are unchanged.
    """
    lowering = _Lowering()
    processed, increments = [], []
    for expr in expressions:
        lowered = _lower_reflection(expr, lowering)
        if isinstance(expr, Inc) and lowered != [expr]:
            increments.extend(lowered)
            continue
        processed.extend(_sum_increments(increments) + lowered)
        increments = []
    return processed + _sum_increments(increments)


def _sum_increments(increments):
    """
    The increments of consecutive `Inc`s writing at mirror points, each over the
    SubDomain of the row it writes (see `_scatter`), summed into one increment per
    row written and kind of read, in its own column or at lateral offsets, unless
    one of them reads a point another one writes.
    """
    written = {(i.lhs.function, _row(i, i.lhs)) for i in increments}
    if any((a.function, _row(i, a)) in written
           for i in increments for a in _accesses(i.rhs)):
        return increments
    summed = {}
    for i in increments:
        key = (i.lhs, i.subdomain, i.implicit_dims, _reads_laterally(i))
        summed.setdefault(key, []).append(i.rhs)
    return [Inc(lhs, sympy.Add(*rhs), subdomain=subdomain, implicit_dims=implicit_dims)
            for (lhs, subdomain, implicit_dims, _), rhs in summed.items()]


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


class _Lowering:

    """
    The Row SubDomains and the reflection weights the lowering of the equations
    of one Operator creates, shared by the equations writing the same rows and
    reflecting through the same surface.
    """

    def __init__(self):
        self._rows = {}
        self._reflections = {}
        self._weights = {}

    def row(self, grid, start):
        """The Row `start` of `grid`."""
        key = (grid, start)
        if key not in self._rows:
            self._rows[key] = Row(start, grid)
        return self._rows[key]

    def reflection(self, grid, center, ghost, target, order, rows):
        """The `_Reflection` of these parameters."""
        key = (grid, center, ghost, target, order, rows)
        if key not in self._reflections:
            self._reflections[key] = _Reflection(*key)
        return self._reflections[key]

    def weights(self, grid, center, ghost, target, order, rows, transposed):
        """
        The weights of the `_Reflection` of these parameters, as a Function
        distributed like `grid` holding the local weights, the first lateral
        offset of their window, the first row of the target Function they read,
        and the stencil points they use on any rank (see `_Reflection.weights`).
        """
        key = (grid, center, ghost, target, order, rows, transposed)
        if key not in self._weights:
            reflection = self.reflection(grid, center, ghost, target, order, rows)
            table, first, used = reflection.weights(transposed)
            self._weights[key] = (_weights_function(grid, table, key), first,
                                  reflection.start, used)
        return self._weights[key]


def _lower_reflection(eq, lowering):
    """The equations evaluating the reads at mirror points of `eq`."""
    lhs = eq.lhs
    grid = _grid(getattr(lhs, 'function', None))
    if grid is None or not (lhs.is_Function or isinstance(lhs, sympy.Indexed)):
        return [eq]
    vertical = grid.dimensions[-1]

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

    dims = {d for d in lhs_position[-1].free_symbols if getattr(d, 'is_Dimension', False)}
    if dims != {vertical}:
        raise ValueError(f"{eq} reads mirror points through a row of a CurvilinearGrid "
                         f"and writes its rows along {sorted(dims, key=str)}; index the "
                         f"rows it writes by {vertical}, over a SubDomain of these rows "
                         "(a constant index of a distributed Dimension is local to each "
                         "rank)")
    if isinstance(eq, Inc):
        lower = _scatter
    elif eq.is_Reduction:
        raise NotImplementedError(f"{eq} reads mirror points through a row of a "
                                  "CurvilinearGrid, which only an Eq or an Inc can")
    else:
        lower = _gather
    first, last = _bounds(eq, grid)
    return [e for row in range(first, last + 1)
            for e in lower(eq, grid, center, reflected, row, lowering)]


def _gather(eq, grid, center, reflected, row, lowering):
    """
    The equations of `eq` over its row `row`, which writes ghosts above the
    surface and reads Functions at their mirror points, reading them at their
    reflection: over the SubDomain of the row, the rows read relative to it.

    The reads of the written Function at lateral offsets are added by a second
    equation. An equation reading the Function it writes both in its own column
    and at lateral offsets is sequential along all of its Dimensions, and the
    compiler exchanges no halo for such an equation (see
    tests/test_mpi.py::test_avoid_haloupdate_if_distr_but_sequential).
    """
    vertical = grid.dimensions[-1]
    lhs = _indexed(eq.lhs)
    ghost = _offsets(lhs.function, grid)
    position = _positions(lhs, grid)[-1].subs(vertical, row)
    ghost_row = _ghost_row(center, position, ghost)
    lateral_index = _lateral_indices(lhs, grid)
    subdomain = lowering.row(grid, row)

    column, lateral = {}, {}
    for access in reflected:
        function = access.function
        weights, first, start, used = lowering.weights(
            grid, center, ghost, _offsets(function, grid), function.space_order,
            (ghost_row,), False
        )
        _check_halo(function, grid, first, used.shape[1:-1])
        column[access], lateral[access] = [], []
        for stencil in np.ndindex(*used.shape[1:]):
            if not used[(0,) + stencil]:
                continue
            offsets = tuple(f + s for f, s in zip(first, stencil, strict=False))
            read = vertical + start + stencil[-1] - row
            term = (weights.indexed[lateral_index + (0,) + stencil] *
                    _shifted(access, grid, offsets, read))
            (lateral if any(offsets) else column)[access].append(term)
        column[access] = sympy.Add(*column[access])
        lateral[access] = sympy.Add(*lateral[access])

    own = [a for a in reflected if a.function is lhs.function]
    parts = _linear_parts(eq.rhs, reflected)
    if parts is None or not any(lateral[a] != 0 for a in own):
        mapper = {a: column[a] + lateral[a] for a in reflected}
        return [eq.func(lhs, eq.rhs.xreplace(mapper), subdomain=subdomain,
                        implicit_dims=eq.implicit_dims)]

    constant, factors = parts
    rhs = constant + sum(f*(column[a] + (0 if a in own else lateral[a]))
                         for a, f in factors.items())
    lateral_rhs = lhs + sum(factors[a]*lateral[a] for a in own)
    return [eq.func(lhs, rhs, subdomain=subdomain, implicit_dims=eq.implicit_dims),
            eq.func(lhs, lateral_rhs, subdomain=subdomain,
                    implicit_dims=eq.implicit_dims)]


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
    rest = constant + sum(f*s for f, s in zip(factors.values(), symbols.values(),
                                              strict=True))
    if sympy.expand(expr - rest) != 0:
        return None
    return constant, factors


def _scatter(eq, grid, center, reflected, row, lowering):
    """
    The increments of `eq` over its row `row`, which reads ghosts above the
    surface and writes at their mirror points, writing at the points that
    interpolate their reflection: each over the SubDomain of the row it writes,
    the rows read relative to it.

    The reads of the right-hand side are at the ghost, or at the written point.
    Each target point gathers the ghosts that write it, so that no point is
    written beyond the domain, those in its own column and those at lateral
    offsets by two increments, for the halo exchange of the lateral reads (see
    `_gather`).
    """
    lhs = eq.lhs
    vertical = grid.dimensions[-1]
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
    ghost_row = _ghost_row(center, ghost_position[-1].subs(vertical, row), ghost)
    lateral_index = _lateral_indices(lhs, grid)
    written_row = _vertical_index(lhs, grid, row)
    weights, smallest, start, used = lowering.weights(
        grid, center, ghost, _offsets(lhs.function, grid), lhs.function.space_order,
        (ghost_row,), True
    )
    for access in reflected:
        _check_halo(access.function, grid, smallest, used.shape[1:-1])

    eqns = []
    for r in range(used.shape[0]):
        target = start + r
        # Over the SubDomain of the target row, the reads at the written point move
        # with it, and the reads at the ghost stay at its row
        mapper = {a: _shifted(a, grid, vertical=vertical +
                              _vertical_index(a, grid, row) - written_row)
                  for a in at_lhs}
        column, lateral = [], []
        for offset in np.ndindex(*used.shape[1:-1]):
            if not used[(r,) + offset].any():
                continue
            shift = tuple(s + o for s, o in zip(smallest, offset, strict=True))
            mapper.update({a: _shifted(a, grid, shift, vertical +
                                       _vertical_index(a, grid, row) - target)
                           for a in reflected})
            term = (weights.indexed[lateral_index + (r,) + offset + (0,)] *
                    eq.rhs.xreplace(mapper))
            (lateral if any(shift) else column).append(term)
        written = _shifted(lhs, grid, vertical=vertical)
        eqns.extend(eq.func(written, sympy.Add(*terms),
                            subdomain=lowering.row(grid, target),
                            implicit_dims=eq.implicit_dims)
                    for terms in (column, lateral) if terms)
    return eqns


# Accesses on the rows of a CurvilinearGrid

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


def _offsets(function, grid):
    """The staggering of `function`, in cells, along each Dimension of `grid`."""
    refs = dict(zip(function.dimensions, function.indices_ref, strict=True))
    return tuple(sympy.nsimplify((refs[d] - d).xreplace({d.spacing: 1}))
                 for d in grid.dimensions)


def _positions(access, grid):
    """
    The position of a Function access or of an Indexed, in cells from the
    nodes, along each Dimension of `grid`, e.g. `7 - z + 1/2` for `w.subs(z, 7 - z)`
    with `w` staggered along `z`.
    """
    function = access.function
    indices = dict(zip(function.dimensions, _indexed(access).indices, strict=True))
    return tuple(sympy.expand(indices[d] + o)
                 for d, o in zip(grid.dimensions, _offsets(function, grid), strict=True))


def _lateral_indices(access, grid):
    """The indices of `access` along the lateral Dimensions of `grid`."""
    indices = dict(zip(access.function.dimensions, _indexed(access).indices,
                       strict=True))
    return tuple(indices[d] for d in grid.dimensions[:-1])


def _vertical_index(access, grid, row):
    """The index of `access` along the last Dimension of `grid` at its row `row`."""
    vertical = grid.dimensions[-1]
    position = _positions(access, grid)[-1] - _offsets(access.function, grid)[-1]
    return int(position.subs(vertical, row))


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


def _bounds(eq, grid):
    """
    The first and last rows of the SubDomain of `eq`, which must be made of whole
    rows of `grid`.
    """
    vertical = grid.dimensions[-1]
    if eq.subdomain is None:
        raise ValueError(f"{eq} reads mirror points through a row of a CurvilinearGrid "
                         "without a SubDomain; write it over a SubDomain of the rows "
                         "it writes")
    definition = eq.subdomain.define(grid.dimensions)
    if any(definition.get(d, d) is not d for d in grid.dimensions[:-1]):
        raise NotImplementedError(f"{eq} reads mirror points through a row of a "
                                  "CurvilinearGrid over a SubDomain that is not made "
                                  "of whole rows")
    size = grid.shape[-1]
    definition = definition.get(vertical, vertical)
    if definition is vertical:
        return 0, size - 1
    side, *thickness = definition
    if side == 'left':
        return 0, thickness[0] - 1
    elif side == 'right':
        return size - thickness[0], size - 1
    return thickness[0], size - 1 - thickness[1]


def _ghost_row(center, position, offsets):
    """
    The ghost row of the reflection weights of a ghost at the vertical
    `position` above the row `center`, of a Function staggered by `offsets`.
    """
    fraction = (center - offsets[-1]) % 1
    return int(center - position - 1 + fraction)


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


# Geometry of the reflections through the rows of a CurvilinearGrid

# The step, in cells, of the differences giving the tangents of a surface
_TANGENT_STEP = 1e-3

# The Gauss-Newton iterations finding the closest point of a surface, and their
# tolerance, in cells, above the roundoff of the physical coordinates
_FOOT_ITERATIONS = 40
_FOOT_TOLERANCE = 1e-10


def _surface_point(grid, lateral, level):
    """
    Physical position of the points at the lateral computational coordinates
    `lateral`, of shape `(npoint, ndim - 1)`, on the coordinate surface of
    vertical computational coordinate `level` of `grid`.
    """
    vertical = np.full((lateral.shape[0], 1), level)
    return grid.to_physical(np.concatenate([lateral, vertical], axis=1))


def _surface_foot(grid, physical, level):
    """
    The closest point of the coordinate surface of vertical computational
    coordinate `level` of `grid` to each of the `physical` points, of shape
    `(npoint, ndim)`, by Gauss-Newton iterations on the map, its tangents by
    centered differences of `_TANGENT_STEP` cells.
    """
    nlateral = grid.dim - 1
    lateral = grid.to_computational(physical)[:, :nlateral]
    steps = [_TANGENT_STEP*np.float64(h) for h in grid.spacing[:nlateral]]
    roundoff = 64*np.finfo(np.float64).eps*np.abs(physical).max(initial=0)
    tolerance = _FOOT_TOLERANCE*max(np.float64(h) for h in grid.spacing) + roundoff

    for _ in range(_FOOT_ITERATIONS):
        foot = _surface_point(grid, lateral, level)
        tangents = []
        for axis, step in enumerate(steps):
            shift = np.zeros(nlateral)
            shift[axis] = step
            tangents.append((_surface_point(grid, lateral + shift, level) -
                             _surface_point(grid, lateral - shift, level))/(2*step))
        tangents = np.stack(tangents, axis=2)
        residual = physical - foot
        normal = np.einsum('pda,pdb->pab', tangents, tangents)
        rhs = np.einsum('pda,pd->pa', tangents, residual)
        try:
            update = np.linalg.solve(normal, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError as e:
            raise ValueError(f"The surface at computational depth {level} is "
                             "degenerate: its tangents are not independent") from e
        lateral = lateral + update
        if np.abs(update).max(initial=0) < tolerance:
            return _surface_point(grid, lateral, level)
    raise ValueError(f"The closest points of the surface at computational depth "
                     f"{level} did not converge in {_FOOT_ITERATIONS} iterations; "
                     "smooth the surface")


def _lagrange_weights(position, npoint):
    """
    Lagrange interpolation weights at the fractional `position` (an array) of
    the `npoint` nodes `0, 1, ..., npoint - 1`, of shape `position.shape +
    (npoint,)`.
    """
    weights = np.ones(np.shape(position) + (npoint,))
    for p in range(npoint):
        for q in range(npoint):
            if q != p:
                weights[..., p] *= (position - q)/(p - q)
    return weights


class _Reflection:

    """
    The interpolation weights of the reflections of points through a coordinate
    surface of a grid, the row (or half row) `center` cells below its vertical
    origin.

    The points reflected, the ghosts, are the points of a Function staggered
    by `ghost`, the offsets in cells of its points from the nodes along each
    Dimension, above the surface: the ghost row `k` is `k + 1 - f` cells above
    it, with `f` the fractional part of `center - ghost[-1]`, for the rows
    `rows`. Each ghost, at physical position `p`, is reflected through the
    closest point `q` of the surface, normal to it, to its image `2*q - p`, which
    is interpolated with Lagrange polynomials of `order` points along each axis
    from the points of a Function staggered by `target` at or below the surface.
    Over a plane surface, the image of a field odd or even through the surface is
    exact.

    Each rank computes the weights of its own lateral nodes, in a window common
    to all ranks: `first`, the first lateral offset of the window from the ghost
    along each lateral Dimension, `window`, its size, `start`, the first row of
    the target Function it reads, and `depth`, the number of rows it reads.
    """

    def __init__(self, grid, center, ghost, target, order, rows):
        self.grid = grid
        self.center = float(center)
        self.ghost = np.array([float(o) for o in ghost])
        self.target = np.array([float(o) for o in target])
        self.order = order
        self.rows = rows
        self.nlateral = grid.dim - 1
        self.fraction = (self.center - self.ghost[-1]) % 1
        nrows = int(round(self.center - self.ghost[-1] + self.fraction))
        if not rows or min(rows) < 0 or max(rows) >= nrows:
            raise ValueError(f"The grid has {nrows} rows above the surface to "
                             f"reflect, not the rows {rows}")
        self.start = int(np.ceil(self.center - self.target[-1] - 1e-9))

        # The local lateral nodes, as a range of global indices along each
        # lateral Dimension
        self.local = tuple((n[0], n[-1] + 1) if len(n) else (0, 0)
                           for n in grid.distributor.glb_numb[:self.nlateral])
        _, _, offsets, base, _ = self._images(self.local)
        first = offsets.min(axis=0, initial=np.iinfo(np.int64).max)
        last = offsets.max(axis=0, initial=np.iinfo(np.int64).min)
        deepest = (base[:, -1] - self.start).max(initial=0)
        if grid.distributor.is_parallel:
            first = _allreduce(grid, first, MPI.MIN)
            last = _allreduce(grid, last, MPI.MAX)
            deepest = _allreduce(grid, np.array([deepest]), MPI.MAX)[0]
        self.first = tuple(int(f) for f in first)
        self.window = tuple(int(i) for i in last - first + order)
        self.depth = int(deepest) + order

    def _images(self, lateral):
        """
        The ghosts of the rows `rows` at the lateral nodes in the ranges of
        global indices `lateral`, as their global lateral indices, and the
        interpolation of their images: the row of each, the offsets of the first
        point of its stencil from it along each lateral Dimension, the first point
        of its stencil, and the Lagrange weights along each Dimension.
        """
        grid, nlateral, order = self.grid, self.nlateral, self.order
        if any(a == b for a, b in lateral):
            empty = np.zeros((0, nlateral), dtype=np.int64)
            return (empty, np.zeros(0, dtype=np.int64), empty,
                    np.zeros((0, grid.dim), dtype=np.int64),
                    [np.zeros((0, order)) for _ in range(grid.dim)])
        origin = np.array([np.float64(o) for o in grid.origin])
        spacing = np.array([np.float64(h) for h in grid.spacing])
        nodes = np.meshgrid(*[np.arange(*r) for r in lateral], indexing='ij')
        nodes = np.stack([n.ravel() for n in nodes], axis=1).astype(np.int64)
        ghosts = np.concatenate([nodes]*len(self.rows))
        rows = np.repeat(np.arange(len(self.rows)), len(nodes))
        ghost_rows = np.array(self.rows)[rows]

        # Ghosts at their positions in cells from the origin
        heights = self.center - (ghost_rows + 1 - self.fraction)
        cells = np.concatenate([ghosts + self.ghost[:-1], heights[:, None]], axis=1)
        physical = grid.to_physical(origin + spacing*cells)
        level = origin[-1] + spacing[-1]*self.center
        image = 2*_surface_foot(grid, physical, level) - physical
        position = (grid.to_computational(image) - origin)/spacing - self.target
        if (position[:, -1] + self.target[-1] < self.center - 1e-9).any():
            raise ValueError("The reflection of a point through the surface lies "
                             "above it; the surface is too curved for the grid "
                             "spacing")

        # Lagrange stencils, centered laterally, at or below the surface vertically
        base = np.floor(position).astype(np.int64) - order//2 + 1
        base[:, -1] = np.maximum(base[:, -1], self.start)
        weights = [_lagrange_weights(position[:, a] - base[:, a], order)
                   for a in range(grid.dim)]
        return ghosts, rows, base[:, :nlateral] - ghosts, base, weights

    def _table(self, lateral):
        """
        The weights of the ghosts at the lateral nodes in the ranges of global
        indices `lateral`, of shape `lateral shape + (ghost rows,) + window +
        (depth,)`.
        """
        ghosts, rows, offsets, base, weights = self._images(lateral)
        shape = tuple(b - a for a, b in lateral)
        table = np.zeros(shape + (len(self.rows),) + self.window + (self.depth,))
        index = tuple(ghosts[:, a] - lateral[a][0] for a in range(self.nlateral))
        for stencil in np.ndindex(*(self.order,)*self.grid.dim):
            weight = np.prod([w[:, s] for w, s in zip(weights, stencil, strict=True)],
                             axis=0)
            window = tuple(offsets[:, a] - self.first[a] + stencil[a]
                           for a in range(self.nlateral))
            vertical = base[:, -1] - self.start + stencil[-1]
            np.add.at(table, index + (rows,) + window + (vertical,), weight)
        return table

    def weights(self, transposed=False):
        """
        The local weights, and the stencil points they use on any rank.

        With `transposed`, the weights are gathered at the target points they
        read: entry `[x, r, o, k]` is the weight of the target point `(x, start +
        r)` in the image of the ghost of row `k` at `x + o`, the lateral offsets
        `o` counted from `smallest`, the smallest one, also returned. The ghosts
        a rank needs are those its target points gather, within a window of its
        own lateral nodes.
        """
        if not transposed:
            table, smallest = self._table(self.local), self.first
        else:
            shape = self.grid.shape[:self.nlateral]
            ghosts = tuple((max(0, a - f - w + 1), min(n, b - f))
                           for (a, b), f, w, n in zip(self.local, self.first,
                                                      self.window, shape, strict=True))
            table, smallest = _transpose_table(self._table(ghosts), self.first,
                                               self.nlateral)
            table = table[tuple(slice(a - g, b - g) for (a, b), (g, _)
                                in zip(self.local, ghosts, strict=True))]
        used = table.any(axis=tuple(range(self.nlateral)))
        if self.grid.distributor.is_parallel:
            used = _allreduce(self.grid, used.astype(np.uint8), MPI.MAX) > 0
        return table, smallest, used


def _allreduce(grid, values, op):
    """The elementwise reduction `op` of the integer array `values` over the ranks."""
    values = np.ascontiguousarray(values, dtype=np.int64)
    reduced = np.empty_like(values)
    grid.comm.Allreduce(values, reduced, op=op)
    return reduced


def _transpose_table(table, first, nlateral):
    """
    The reflection weights `table` of `_Reflection`, whose window starts
    at the lateral offsets `first` from each ghost, gathered at the target
    points they read: entry `[x, r, o, k]` is the weight of the target point
    `(x, r)` in the image of the ghost of row `k` at `x + o`, the lateral offsets
    `o` counted from the smallest one, which is also returned.
    """
    lateral_shape = table.shape[:nlateral]
    rows = table.shape[nlateral]
    window = table.shape[nlateral + 1:-1]
    transposed = np.zeros(lateral_shape + table.shape[-1:] + window + (rows,))
    for stencil in np.ndindex(*window):
        # The ghost at `xg` reads the target point at `xg + shift`
        shift = [f + i for f, i in zip(first, stencil, strict=True)]
        target = tuple(slice(max(0, sh), min(n, n + sh))
                       for n, sh in zip(lateral_shape, shift, strict=True))
        ghost = tuple(slice(i.start - sh, i.stop - sh)
                      for i, sh in zip(target, shift, strict=True))
        offset = tuple(n - 1 - i for n, i in zip(window, stencil, strict=True))
        for k in range(rows):
            transposed[target + (slice(None),) + offset + (k,)] = \
                table[ghost + (k,) + stencil + (slice(None),)]
    smallest = tuple(-(f + n - 1) for f, n in zip(first, window, strict=True))
    return transposed, smallest


def _weights_function(grid, table, key):
    """
    The reflection weights `table`, local to this rank, as a Function distributed
    like `grid`, named after the parameters `key` of the reflection, `(grid,
    center, ghost offsets, target offsets, order, rows, transposed)`.
    """
    _, center, ghost, target, order, rows, transposed = key
    halves = ''.join(str(int(2*o)) for o in ghost + target)
    name = (f"reflection{int(2*center)}_{halves}_o{order}_r{'_'.join(map(str, rows))}"
            f"{'_t' if transposed else ''}")
    extra = [DefaultDimension(name=f'{name}_{i}', default_value=n)
             for i, n in enumerate(table.shape[grid.dim - 1:])]
    shape = tuple(grid.shape[:grid.dim - 1]) + table.shape[grid.dim - 1:]
    weights = Function(name=name, grid=grid,
                       dimensions=grid.dimensions[:-1] + tuple(extra),
                       shape=shape, space_order=0, dtype=grid.dtype)
    weights.data_local[:] = table
    return weights
