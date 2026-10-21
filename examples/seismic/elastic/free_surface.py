"""
The isotropic elastic wave equation with a traction-free surface on a
CurvilinearGrid, discretized with summation-by-parts (SBP) operators along the
vertical axis as in

    O. O'Reilly, T.-Y. Yeh, K. B. Olsen, Z. Hu, A. Breuer, D. Roten and C. A.
    Goulet, A high-order finite-difference method on staggered curvilinear
    grids for seismic wave propagation applications with topography, Bulletin
    of the Seismological Society of America 112 (2022).
"""

import numpy as np
import sympy

from devito import Eq, SubDomain
from devito.finite_differences import generic_derivative
from devito.symbolics import indexify, retrieve_indexed
from examples.seismic.elastic.sbp import sbp_closure_rows, sbp_interior, sbp_matrix

__all__ = ['traction_free_update']


class Rows(SubDomain):

    """
    Rows `start` to `start + thickness - 1` of the last Dimension of a grid,
    over all of its other Dimensions.
    """

    def __init__(self, name, start, thickness, grid):
        self.name = name
        self._start = start
        self._thickness = thickness
        self._size = grid.shape[-1]
        super().__init__(grid=grid)

    def define(self, dimensions):
        *lateral, vertical = dimensions
        after = self._size - self._start - self._thickness
        return {**{d: d for d in lateral},
                vertical: ('middle', self._start, after)}


def _lateral(expr, kind, dim, from_half):
    """
    The interior SBP operator `kind` ('D' or 'P') along the lateral `dim` applied
    to `expr`, from the half points to the nodes if `from_half`, else from the
    nodes to the half points: the fourth-order staggered derivative or
    interpolation w.r.t. the computational coordinate.
    """
    weights, _ = sbp_interior(kind)
    x0 = dim if from_half else dim + dim.spacing/2
    return expr.diff((dim, int(kind == 'D')), x0={dim: x0}, fd_order=len(weights),
                     computational=True)


def _opposite(vgrid):
    """The other vertical SBP grid."""
    return '-' if vgrid == '+' else '+'


def _vertical_matrix(kinds, vgrid, traction_free=False):
    """
    The product of the vertical SBP operators `kinds`, applied right to left and
    alternating grids, onto `vgrid`, e.g. ('D', 'P') onto '+' is `D+ P-`. Each
    factor reaches its interior stencil past its closure, so the product does
    within the sum of their closures and stencil widths, and it has twice as
    many rows.
    """
    width = len(sbp_interior('D')[0])
    size = 2*len(kinds)*(sbp_closure_rows() + width)
    matrix, grid = None, vgrid
    for i, kind in enumerate(kinds):
        factor = sbp_matrix(kind, grid, size, traction_free and i == 0)
        matrix = factor if matrix is None else matrix[:, :size] @ factor
        size, grid = size + width, _opposite(grid)
    return matrix


def _closure_rows(matrix):
    """
    Number of boundary rows of `matrix` before its rows repeat its interior
    stencil, shifted by one column per row.
    """
    template = matrix.shape[0] // 2
    stencil = matrix[template]
    for row in range(template, -1, -1):
        shifted = np.roll(stencil, row - template)
        if row < template and not np.allclose(matrix[row], shifted, atol=1e-14):
            return row + 1
    return 0


def _closure(expr, dim, weights, offset):
    """
    The first derivative along `dim` of custom `weights`, in units of the
    spacing, applied to `expr` at `offset` cells from the point computed and on,
    as a finite difference of Devito over an odd number of points centred on its
    `x0`. A derivative nested in another is placed at the points of its argument,
    not at its own `x0`, so a product of vertical operators is one finite
    difference of the weights of the product.
    """
    nonzero = np.flatnonzero(weights)
    taps = list(weights[nonzero[0]:nonzero[-1] + 1])
    taps += [0.]*(1 - len(taps) % 2)
    centre = offset + int(nonzero[0]) + (len(taps) - 1)//2
    x0 = {dim: expr.indices_ref[dim] + centre*dim.spacing}
    return generic_derivative(expr, dim, len(taps) - 1, 1, x0=x0, weights=taps)


def _surface_row(grid):
    """The index of the surface row of `grid`, at computational depth 0."""
    index = -float(grid.origin[-1])/float(grid.spacing[-1])
    if index < 1 or not np.isclose(index, round(index)):
        raise ValueError(f"{grid} needs a row above the surface, at computational "
                         "depth 0, for the surface points of the vertically "
                         "staggered fields")
    return int(round(index))


class _ElasticSBP:

    """
    The equations of `traction_free_update`.

    Each field lives on the vertical `+` grid, the nodes from the surface down,
    or on the `-` grid, the surface and the half points below it, stored from
    the row above the surface. Each update is one equation per row of the
    boundary closures of its operators, then one for the rows below them.
    """

    def __init__(self, v, tau, buoyancy, lam, mu, damp):
        self.grid = grid = v[0].grid
        self.v = v
        self.tau = tau
        self.buoyancy = buoyancy
        self.lam = lam
        self.mu = mu
        self.damp = damp
        self.last = grid.dim - 1
        self.vertical = grid.dimensions[-1]
        surface = _surface_row(grid)
        self.bases = {'+': surface, '-': surface - 1}
        self.names = {'+': 'sbp_nodes', '-': 'sbp_half'}
        self.v_next = [vi.forward for vi in v]
        self.subdomains = {}
        # The metric terms of the `-` grid and those of its surface point, the
        # surface node, by Function
        self.surface_terms = {}

    def vgrid(self, i, j=None):
        """The vertical grid of `v[i]`, or of `tau[i, j]`."""
        staggered = {i} if j is None else ({i, j} if i != j else set())
        return '-' if self.last in staggered else '+'

    def lateral_half(self, i, j=None):
        """Whether `v[i]`, or `tau[i, j]`, is at half points along each lateral axis."""
        staggered = {i} if j is None else ({i, j} if i != j else set())
        return tuple(a in staggered for a in range(self.last))

    def rows_subdomain(self, name, start, thickness):
        """The SubDomain of `thickness` rows from row `start`."""
        key = (name, start, thickness)
        if key not in self.subdomains:
            self.subdomains[key] = Rows(f'{name}_{start}_{thickness}', start,
                                        thickness, self.grid)
        return self.subdomains[key]

    def terms(self, lateral_half, vgrid):
        """
        The Jacobian and the metric terms `d eta / d x_i` at the points staggered
        by half a cell along the lateral Dimensions `lateral_half` and on the
        vertical SBP grid `vgrid`, defined there. On the `-` grid, the surface
        point takes the terms of the surface node (see `surface`).
        """
        grid, vertical = self.grid, self.vertical
        lateral = tuple(d for d, half in zip(grid.dimensions, lateral_half,
                                             strict=False) if half)
        staggered = lateral + ((vertical,) if vgrid == '-' else ())
        metrics, jacobian = grid.staggered_terms(staggered)
        values = {f'eta_{d.name}': metrics[(d, vertical)] for d in grid.dimensions}
        values['J'] = jacobian
        if vgrid == '-':
            surface_metrics, surface_jacobian = grid.staggered_terms(lateral)
            surface = {f'eta_{d.name}': surface_metrics[(d, vertical)]
                       for d in grid.dimensions}
            surface['J'] = surface_jacobian
            self.surface_terms.update({v.function: surface[k] for k, v in values.items()
                                       if v.is_Function})
        return values

    def surface(self, expr, row):
        """
        `expr`, indexed and written at the row `row`, with the metric terms of
        the `-` grid read at its surface point, stored in the row above the
        surface, replaced by those of the surface node, the row below.
        """
        vertical = self.vertical
        mapper = {}
        for i in retrieve_indexed(expr):
            node = self.surface_terms.get(i.function)
            if node is None:
                continue
            index = dict(zip(i.function.dimensions, i.indices, strict=True))
            if index[vertical].subs(vertical, row) != self.bases['-']:
                continue
            index[vertical] += 1
            mapper[i] = node.indexed[tuple(index[d] for d in node.dimensions)]
        return expr.xreplace(mapper)

    def average(self, param, i, j=None, surface=False):
        """
        A parameter given on the nodes, interpolated to the points of `v[i]` or
        `tau[i, j]` by the average of its two neighbours along each staggered
        Dimension, or taken on the surface node at the surface point of the `-`
        grid.
        """
        if not isinstance(param, sympy.Basic) or param.is_Number:
            return param
        value = param
        for d, half in zip(self.grid.dimensions, self.lateral_half(i, j), strict=False):
            if half:
                value = value.diff((d, 0), x0={d: d + d.spacing/2}, fd_order=2,
                                   computational=True)
        if self.vgrid(i, j) == '-':
            vertical, h = self.vertical, self.vertical.spacing
            if surface:
                return value.evaluate.subs(vertical, vertical + h)
            value = value.diff((vertical, 0), x0={vertical: vertical + h/2}, fd_order=2,
                               computational=True)
        return value

    def vertical_operator(self, kinds, expr, vgrid, row, traction_free=False):
        """
        The product of the vertical SBP operators `kinds` (see `_vertical_matrix`)
        applied to `expr` at the closure row `row` of `vgrid`, or, if None, at
        the rows below the closures, whose stencil is the interior row of the
        product.
        """
        grid_in = vgrid if len(kinds) % 2 == 0 else _opposite(vgrid)
        matrix = _vertical_matrix(kinds, vgrid, traction_free)
        row = matrix.shape[0]//2 if row is None else row
        offset = self.bases[grid_in] - self.bases[vgrid] - row
        return _closure(expr, self.vertical, matrix[row], offset)

    def closures(self, vgrid, *kinds):
        """The number of closure rows of the vertical operators `kinds` onto `vgrid`."""
        return max(_closure_rows(_vertical_matrix(k, vgrid)) for k in kinds)

    def equation(self, lhs, rhs, vgrid, nrows, row):
        """
        The update `lhs = rhs` over the closure row `row` of `vgrid`, or over the
        interior if None, whose first row reads no surface point.
        """
        start = self.bases[vgrid] + (nrows if row is None else row)
        thickness = self.grid.shape[-1] - start if row is None else 1
        rhs = indexify(rhs.evaluate)
        surface = self.surface(rhs, start)
        if row is None and surface != rhs:
            raise ValueError("The interior stencils of the SBP operators read the "
                             "surface point; the closures need more rows")
        subdomain = self.rows_subdomain(self.names[vgrid], start, thickness)
        return Eq(lhs.indexify(), surface, subdomain=subdomain)

    def momentum(self, i):
        """
        The updates of `v[i]` from the divergence of the stress in conservation
        form, `(1/J) sum_k d/dxi_k (J dxi_k/dx_j tau[i, j])`, whose vertical
        derivative drops the flux through the surface, the traction.
        """
        grid, last, vertical = self.grid, self.last, self.vertical
        vgrid = self.vgrid(i)
        terms = self.terms(self.lateral_half(i), vgrid)
        # The lateral fluxes, and the vertical flux, from the lateral stresses
        # interpolated to the points of v[i] and from the vertical stress
        lateral, tangential = 0, 0
        for a, d in enumerate(grid.dimensions[:-1]):
            stress = self.tau[i, a]
            half = self.lateral_half(i, a)[a]
            terms_stress = self.terms(self.lateral_half(i, a), self.vgrid(i, a))
            lateral += _lateral(terms_stress['J']*stress, 'D', d, half)
            tangential += (terms['J']*terms[f'eta_{d.name}'] *
                           _lateral(stress, 'P', d, half))
        terms_normal = self.terms(self.lateral_half(i, last), self.vgrid(i, last))
        normal = (terms_normal['J']*terms_normal[f'eta_{vertical.name}'] *
                  self.tau[i, last])

        nrows = self.closures(vgrid, ('D', 'P'), ('D',))
        dt = grid.time_dim.spacing
        eqns = []
        for row in list(range(nrows)) + [None]:
            divergence = (lateral +
                          self.vertical_operator(('D', 'P'), tangential, vgrid, row,
                                                 traction_free=True) +
                          self.vertical_operator(('D',), normal, vgrid, row,
                                                 traction_free=True))
            buoyancy = self.average(self.buoyancy, i, surface=row == 0)
            update = self.damp*(self.v[i] + dt*buoyancy/terms['J']*divergence)
            eqns.append(self.equation(self.v_next[i], update, vgrid, nrows, row))
        return eqns

    def strain(self, i, j, row):
        """
        The derivative of the updated `v[i]` along the physical `x_j` at the
        points of `tau[i, j]` and at the closure row `row` of their vertical grid,
        or below the closures if None, by the chain rule.
        """
        grid, vertical = self.grid, self.vertical
        vgrid = self.vgrid(i, j)
        terms_stress = self.terms(self.lateral_half(i, j), vgrid)
        if j == self.last:
            return (terms_stress[f'eta_{vertical.name}'] *
                    self.vertical_operator(('D',), self.v_next[i], vgrid, row))
        d = grid.dimensions[j]
        half = self.lateral_half(i)[j]
        terms = self.terms(self.lateral_half(i), self.vgrid(i))
        gradient = self.vertical_operator(('P', 'D'), self.v_next[i], vgrid, row)
        return (_lateral(self.v_next[i], 'D', d, half) +
                _lateral(terms['J']*terms[f'eta_{d.name}']*gradient, 'P', d, half) /
                terms_stress['J'])

    def stress(self, i, j):
        """The updates of `tau[i, j]` from the strain rates of the updated velocity."""
        vgrid = self.vgrid(i, j)
        nrows = self.closures(vgrid, ('D',), ('P', 'D'))
        dt = self.grid.time_dim.spacing
        eqns = []
        for row in list(range(nrows)) + [None]:
            if i == j:
                dilatation = sum(self.strain(k, k, row) for k in range(self.grid.dim))
                rate = (self.average(self.lam, i, j)*dilatation +
                        2*self.average(self.mu, i, j)*self.strain(i, i, row))
            else:
                rate = (self.average(self.mu, i, j, surface=row == 0) *
                        (self.strain(i, j, row) + self.strain(j, i, row)))
            update = self.damp*(self.tau[i, j] + dt*rate)
            eqns.append(self.equation(self.tau[i, j].forward, update, vgrid, nrows,
                                      row))
        return eqns

    def equations(self):
        """The velocity updates, then the stress updates."""
        ndim = self.grid.dim
        return ([e for i in range(ndim) for e in self.momentum(i)] +
                [e for i in range(ndim) for j in range(i, ndim)
                 for e in self.stress(i, j)])


def traction_free_update(v, tau, buoyancy, lam, mu, damp=1):
    """
    Equations of one time step of the isotropic elastic wave equation in
    velocity-stress form on a CurvilinearGrid whose surface is traction free,

        rho dv/dt = div(tau),
        dtau/dt = lam tr(grad(v)) I + mu (grad(v) + grad(v)^T),
        tau . n = 0 on the surface,

    discretized as in O'Reilly et al. (2022). The momentum equation is taken in
    conservation form and the strain rates by the chain rule, with SBP
    operators along the vertical axis (see `examples.seismic.elastic.sbp`), whose
    interior stencils, fourth order, are also used laterally. The traction-free
    condition is imposed weakly, by dropping the flux through the surface from
    the vertical derivative of the momentum equation, `D - H^-1 B`. The
    semi-discrete system conserves the elastic energy, so the scheme is stable
    over any smooth topography, accurate to fourth order in the interior and
    second order at the surface, and the adjoint propagation is its exact
    transpose.

    The fields are on the usual staggered grid, `v[i]` half a cell along `x_i`
    and `tau[i, j]` along `x_i` and `x_j`, apart from the vertical grid of the
    fields staggered vertically, `v[-1]` and `tau[i, -1]`, which also holds a
    point on the surface: the row above it stores it, so the grid needs one
    row above the surface, and sparse functions reading or writing these
    fields must lie at least half a cell below the surface. The map must keep
    the lateral coordinates, as `TerrainFollowing` does.

    Parameters
    ----------
    v : VectorTimeFunction
        The particle velocity, of time order 1 and space order 4.
    tau : TensorTimeFunction
        The stress, of time order 1 and space order 4.
    buoyancy, lam, mu : Function or number
        The buoyancy `1/rho` and the Lame parameters on the nodes, averaged to
        the staggered points.
    damp : Function or number, optional
        A factor applied to each update, e.g. an absorbing layer.

    Returns
    -------
    list of Eq
        The updates of the velocity, then of the stress, each over the rows from
        the surface down. Sources follow them, e.g. injected into `tau.forward`.
    """
    grid = v[0].grid
    if grid is None or not grid.is_Curvilinear:
        raise ValueError(f"traction_free_update needs fields on a CurvilinearGrid, "
                         f"not on {grid}")
    for (i, j), value in grid.metrics.items():
        k, m = grid.dimensions.index(i), grid.dimensions.index(j)
        if m < grid.dim - 1 and value != int(k == m):
            raise ValueError(f"traction_free_update needs a map that keeps the "
                             f"lateral coordinates, x_i = xi_i, which "
                             f"{grid.mapping} does not")
    for f in list(v) + list(tau.values()):
        if f.grid is not grid:
            raise ValueError(f"{f} is not defined on {grid}")
        if f.space_order != 4 or f.time_order != 1:
            raise ValueError(f"traction_free_update needs fields of space order 4 "
                             f"and time order 1, not {f} of space order "
                             f"{f.space_order} and time order {f.time_order}")

    return _ElasticSBP(v, tau, buoyancy, lam, mu, damp).equations()
