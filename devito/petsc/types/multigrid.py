from functools import cached_property
from itertools import product as iterproduct

import numpy as np
import sympy
from sympy import Integer, Rational, finite_diff_weights

from devito.mpi import CoarseDistributor
from devito.symbolics import IntDiv
from devito.tools import Pickable, as_tuple, flatten
from devito.types.basic import Scalar
from devito.types.dimension import (ConditionalDimension, CustomDimension,
                                    SpaceDimension, Spacing, Thickness)
from devito.types.equation import Eq
from devito.types.grid import Grid
from devito.types.lazy import Evaluable


class CoarseParam:
    is_Input = True


class SubGridScalar(CoarseParam, Scalar):
    """
    """

    __rkwargs__ = Scalar.__rkwargs__ + ('value',)

    def __new__(cls, name, value=None, **kwargs):
        newobj = super().__new__(cls, name, **kwargs)
        newobj._value = value
        return newobj

    @property
    def default_value(self):
        return self._value


class GlobalStartScalar(CoarseParam, Scalar):
    """
    The global index of the first owned point on this MPI rank for a given
    dimension and grid level.

    Used in interpolation/restriction equations to convert between local and
    global indices without calling DMDAGetLocalInfo at runtime.
    _arg_values reads distributor.glb_slices[dim].start.
    """

    __rkwargs__ = ('name', 'dtype', 'is_const', 'dim', 'distributor', 'root')

    def __new__(cls, name, dim=None, distributor=None, root=None, **kwargs):
        kwargs.setdefault('dtype', np.int32)
        kwargs.setdefault('is_const', True)
        newobj = super().__new__(cls, name, **kwargs)
        newobj._dim = dim
        newobj._distributor = distributor
        newobj._root = root
        return newobj

    @property
    def root(self):
        return self._root

    def _arg_values(self, **kwargs):
        return {self.name: self._distributor.glb_slices[self._dim].start}


class FineGlobalStartScalar(GlobalStartScalar):
    """
    Like GlobalStartScalar but always holds the fine-grid global start —
    it is never coarsened by fix_mg_populate_calls. Every level's UserCtx
    gets the fine-grid value so callbacks can always access the fine start
    via ctx->field without special routing.
    """


class CoarseningFactorScalar(CoarseParam, Scalar):
    """
    The coarsening factor at a given multigrid level relative to the fine grid.
    """

    __rkwargs__ = ('name', 'dtype', 'is_const', 'depth')

    def __new__(cls, name, depth=0, **kwargs):
        kwargs.setdefault('dtype', np.int32)
        kwargs.setdefault('is_const', True)
        newobj = super().__new__(cls, name, **kwargs)
        newobj._depth = depth
        return newobj

    @property
    def depth(self):
        return self._depth

    def _arg_values(self, **kwargs):
        return {self.name: 2 ** self._depth}


class CoarseThickness(Thickness):
    """
    A Thickness token for a coarse grid level in a multigrid hierarchy.

    Stores a CoarseDistributor directly so _arg_values does not need the
    fine Grid. It uses self._distributor instead of grid.distributor.
    """

    __rkwargs__ = Thickness.__rkwargs__ + ('distributor',)

    def __new__(cls, *args, distributor=None, **kwargs):
        newobj = super().__new__(cls, *args, **kwargs)
        newobj._distributor = distributor
        return newobj

    def _arg_values(self, grid=None, **kwargs):
        rtkn = kwargs.get(self.name, self.value)
        if self._distributor is not None and rtkn is not None:
            if self.local:
                tkn = self._distributor.glb_to_loc(self.root, rtkn - 1, self.side)
                tkn = tkn + 1 if tkn is not None else 0
            else:
                tkn = self._distributor.glb_to_loc(self.root, rtkn, self.side) or 0
        else:
            tkn = rtkn or 0
        return {self.name: tkn}


class SubGrid(Grid):

    """
    A coarser level in a GridHierarchy of successively factor-2 coarsened
    Grids.

    Behaves like an independent Grid - its own per-level SpaceDimensions,
    spacing, and bounds (e.g. `x_d1_m`, `x_d1_M`) generated the same way a
    plain Grid generates `x_m`, `x_M` - but reuses the parent Grid's MPI comm
    and topology via a CoarseDistributor (which computes its own coarse
    decomposition), and shares the parent's TimeDimension.

    Not constructed directly by users - created by GridHierarchy.
    """

    def __init__(self, shape, parent, coarsening_depth):
        shape = as_tuple(shape)
        depth = coarsening_depth

        dimensions = tuple(
            SpaceDimension(
                name=f'{d.name}_d{depth}',
                spacing=Spacing(name=f'{d.spacing.name}_d{depth}',
                                dtype=parent.dtype, is_const=True)
            )
            for d in parent.dimensions
        )

        super().__init__(
            shape, extent=parent.extent, dimensions=dimensions,
            dtype=parent.dtype, time_dimension=parent.time_dim,
            comm=parent.distributor.comm, topology=parent.distributor.topology,
        )

        self._parent = parent
        self._coarsening_depth = depth
        # Grid.__init__ built a plain Distributor above; replace it with one
        # that reuses the parent's comm/topology but computes its own coarse
        # decomposition, keyed by this SubGrid's own Dimensions.
        self._distributor = CoarseDistributor(shape, dimensions, parent.distributor)

    def __repr__(self):
        return f'SubGrid[shape={self.shape}, dimensions={self.dimensions}]'

    @property
    def coarsening_depth(self):
        """
        Number of factor-2 coarsenings from the top-level Grid (1 = one
        halving, etc.).
        """
        return self._coarsening_depth

    @property
    def parent(self):
        return self._parent

    @property
    def root(self):
        return self.parent.root


class GridHierarchy:

    """
    A hierarchy of Grids for multi-resolution numerical methods (e.g.
    geometric multigrid).

    Applies successive factor-2 coarsenings to a fine Grid, producing a
    SubGrid per coarser level.

    Levels are numbered starting from 0 for the fine grid:
    levels[0] = fine Grid, levels[1] = first coarse SubGrid, etc.

    Parameters
    ----------
    fine_grid : Grid
        The finest level.
    nlevels : int
        Total number of levels including the fine grid (e.g. nlevels=3
        gives fine -> mid -> coarse).

    Examples
    --------
    >>> from devito import Grid
    >>> from devito.petsc.types.multigrid import GridHierarchy
    >>> grid = Grid(shape=(33,))
    >>> h = GridHierarchy(grid, nlevels=3)
    >>> h.levels
    (Grid[...shape=(33,)...], SubGrid[shape=(17,)...], SubGrid[shape=(9,)...])
    """

    def __init__(self, fine_grid, nlevels):
        self._fine = fine_grid
        self._nlevels = nlevels

        divisor = 2 ** (nlevels - 1)
        invalid = [
            (d, n) for d, n in zip(fine_grid.dimensions, fine_grid.shape)
            if (n - 1) % divisor != 0
        ]
        if invalid:
            msgs = ', '.join(
                f"{d}: size {n} ((n-1)={n-1} not divisible by {divisor})"
                for d, n in invalid
            )
            raise ValueError(
                f"Grid cannot be uniformly coarsened over {nlevels} levels: {msgs}. "
                f"Each (n-1) must be divisible by 2^(nlevels-1)={divisor}."
            )

        coarse_levels = []
        shape = fine_grid.shape
        for i in range(nlevels - 1):
            shape = tuple((s - 1) // 2 + 1 for s in shape)
            coarse_levels.append(SubGrid(shape, fine_grid, coarsening_depth=i + 1))
        self._coarse_levels = tuple(coarse_levels)

    def __repr__(self):
        shapes = ' -> '.join(str(l.shape) for l in self.levels)
        return f'GridHierarchy[{shapes}]'

    @property
    def fine(self):
        """
        The finest Grid.
        """
        return self._fine

    @property
    def coarse_levels(self):
        """
        Coarser SubGrids ordered finest-coarse to coarsest.
        """
        return self._coarse_levels

    @property
    def nlevels(self):
        return self._nlevels

    @property
    def levels(self):
        """
        All levels as a tuple: (fine_grid, subgrid_l1, ..., subgrid_lN).
        """
        return (self._fine,) + self._coarse_levels


RESTRICTION_TYPES = ('interpolation_transpose', 'full_weighting')


class MultigridMetadata:
    """
    PETSc-specific multigrid metadata: holds the GridHierarchy and the
    interpolation/restriction transfer equations for the target Function.
    """

    def __init__(self, hierarchy, target, restriction='interpolation_transpose'):
        # TODO: extend so that users can provide their own restriction equations
        if restriction not in RESTRICTION_TYPES:
            raise ValueError(
                f"restriction={restriction!r} not recognised; must be one of "
                f"{RESTRICTION_TYPES}."
            )
        self._hierarchy = hierarchy
        self._full_weighting = restriction == 'full_weighting'

        fine_grid = hierarchy.fine
        dims = fine_grid.dimensions
        distributor = fine_grid.distributor

        glb_starts_f = []
        for d in dims:
            root = Scalar(name=f'{d.name}_m_glb', dtype=np.int32, is_const=True)
            glb_starts_f.append(
                FineGlobalStartScalar(f'{d.name}_m_glb_d0', dim=d,
                                     distributor=distributor, root=root)
            )

        self._glb_start_syms_f = tuple(glb_starts_f)
        self._interpolation = GridTransferEquations(
            target, glb_starts_f=self._glb_start_syms_f,
            full_weighting=self._full_weighting
        )

    @property
    def hierarchy(self):
        return self._hierarchy

    @property
    def full_weighting(self):
        return self._full_weighting

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def glb_start_syms_f(self):
        """Per-dimension fine-level GlobalStartScalars (routed via fine_ctx)."""
        return self._glb_start_syms_f


def _field_shifts(field):
    """
    Per-space-Dimension staggering of `field`, as a tuple of booleans ordered
    like `field.space_dimensions` (True = staggered by half a cell). This is
    not a physical distance (cf. WeightedInterpolator._field_shifts in
    devito/operations/interpolators.py) -- the index math below works in
    units of a single cell, not physical spacing. `PETScArray.staggered`
    delegates to its target Function's own staggering, so this is
    `(False, ...)` for a non-staggered PETSc target, unchanged from today's
    behaviour.
    """
    staggered = field.staggered
    if not staggered or staggered.on_node:
        return (False,) * len(field.space_dimensions)
    return tuple(bool(s) for d, s in zip(field.dimensions, staggered, strict=True)
                 if d.is_Space)


class GridTransfer:
    """
    Builds the interpolation and restriction Eqs transferring values between
    a fine-level and a coarse-level object (anything exposing
    `.space_dimensions` and supporting explicit tuple indexing, e.g.
    `Function` or `PETScArray`) related by a single factor-2 coarsening.

    `fine` and `coarse` must have matching per-dimension staggering -- they
    are assumed to represent the same physical field at two resolutions, not
    two different staggered variables (e.g. interpolating a multigrid-solved
    pressure field between levels, never pressure <-> velocity).

    A fine-grid point either coincides with a coarse-grid point or lies at
    some fixed fractional position between two coarse-grid points, determined
    at compile time by the parity of its global index and the (matching)
    staggering. `ConditionalDimension` gates each of the 2**ndim parity
    combinations; within each, Lagrange interpolation weights (degree `so`)
    are evaluated at that fractional position. Restriction is the transpose
    of interpolation.

    `glb_starts_f`/`glb_starts_c` (per-dimension `GlobalStartScalar`s) convert
    between local and global indices so the parity/offset arithmetic is
    correct even when `fine` and `coarse` are decomposed independently under
    MPI. If not supplied, they are built from `fine.grid.distributor` /
    `coarse.grid.distributor` respectively.
    """

    def __init__(self, fine, coarse, so=None, glb_starts_f=None, glb_starts_c=None,
                 row_sum=None):
        self.fine = fine
        self.coarse = coarse
        self.fine_dims = fine.space_dimensions
        self.coarse_dims = coarse.space_dimensions
        self.so = so if so is not None else fine.space_order
        self.shifts = _field_shifts(fine)
        self.row_sum = row_sum

        if glb_starts_f is None:
            distributor = fine.grid.distributor
            glb_starts_f = []
            for d in self.fine_dims:
                root = Scalar(name=f'{d.name}_m_glb', dtype=np.int32, is_const=True)
                glb_starts_f.append(
                    GlobalStartScalar(f'{d.name}_m_glb_f', dim=d,
                                      distributor=distributor, root=root)
                )
        self.glb_starts_f = tuple(glb_starts_f)

        if glb_starts_c is None:
            distributor = coarse.grid.distributor
            glb_starts_c = []
            for i, d in enumerate(self.coarse_dims):
                glb_starts_c.append(
                    GlobalStartScalar(f'{d.name}_m_glb_c', dim=d,
                                      distributor=distributor,
                                      root=self.glb_starts_f[i].root)
                )
        self.glb_starts_c = tuple(glb_starts_c)

    def _offset_and_frac(self, shift, flag):
        """
        For a parity `flag` (0 or 1) and whether this dimension is staggered,
        return the compile-time integer coarse-index offset and the
        fractional position (in [0, 1)) at which to evaluate the Lagrange
        weights. Unstaggered: `frac` is 0 (coincident) or 1/2 (midpoint) --
        today's only supported case. Staggered (both fine and coarse, by
        construction): `frac` alternates between two asymmetric values (e.g.
        1/4, 3/4); there is no exact coincidence.
        """
        off = -Rational(1, 2) if shift else 0
        half = (flag + off) / 2
        extra = sympy.floor(half)
        return int(extra), half - extra

    def _weights(self, frac):
        """
        Lagrange weights (degree `self.so`) evaluated at fractional position
        `frac`, over a fixed window of `self.so` points.
        """
        start = -(self.so // 2 - 1)
        # so=2, pts=[0,1]
        pts = list(range(start, start + self.so))
        w = finite_diff_weights(0, pts, frac)[-1][-1]
        return pts, w

    @cached_property
    def interp_eqs(self):
        ndim = len(self.fine_dims)
        interp_eqs = []
        for flags in iterproduct([0, 1], repeat=ndim):
            conditions = [
                sympy.Eq(sympy.Mod(d + gsf, 2), f)
                for d, gsf, f in zip(self.fine_dims, self.glb_starts_f, flags)
            ]
            condition = (sympy.And(*conditions, evaluate=False)
                         if ndim > 1 else conditions[0])
            cd = ConditionalDimension(
                name='cd' + ''.join(str(f) for f in flags),
                parent=self.fine_dims[-1],
                condition=condition
            )

            lhs = self.fine[tuple(self.fine_dims)]

            dim_stencils = []
            for d, gsc, gsf, f, shift in zip(self.fine_dims, self.glb_starts_c,
                                             self.glb_starts_f, flags, self.shifts):
                extra, frac = self._offset_and_frac(shift, f)
                # C code 1D : (x+x_m_glb_f)/2) - x_m_glb_c + extra
                i_c = IntDiv(d + gsf - f, 2) - gsc + extra
                pts, w = self._weights(frac)
                dim_stencils.append([(i_c + j, wi) for j, wi in zip(pts, w)])

            rhs = Integer(0)
            for combo in iterproduct(*dim_stencils):
                weight = Integer(1)
                idx = []
                for i_c_expr, wi in combo:
                    weight *= wi
                    idx.append(i_c_expr)
                rhs += weight * self.coarse[tuple(idx)]

            interp_eqs.append(Eq(lhs, rhs, implicit_dims=(cd,)))

        return tuple(interp_eqs)

    @cached_property
    def restrict_eq(self):
        # R = P^T. Loop over coarse indices — the natural direction. For each
        # parity flag `f`, invert the interpolation index relation
        # `i_c = k + extra(f)` (so `k = i_c - extra(f)`, `gf = 2k + f`) to find
        # which fine points depend on a given coarse point and with what
        # weight.
        ndim = len(self.coarse_dims)
        rhs = Integer(0)
        for flags in iterproduct([0, 1], repeat=ndim):
            dim_stencils = []
            for d, gsc, gsf, f, shift in zip(self.coarse_dims, self.glb_starts_c,
                                             self.glb_starts_f, flags, self.shifts):
                extra, frac = self._offset_and_frac(shift, f)
                pts, w = self._weights(frac)
                dim_stencils.append(
                    [(2*(d + gsc - extra - j) + f - gsf, wi)
                     for j, wi in zip(pts, w)]
                )

            for combo in iterproduct(*dim_stencils):
                weight = Integer(1)
                fine_idx = []
                for idx_expr, wi in combo:
                    weight *= wi
                    fine_idx.append(idx_expr)
                rhs += weight * self.fine[tuple(fine_idx)]

        if self.row_sum is not None:
            rhs = rhs / self.row_sum[tuple(self.coarse_dims)]

        return Eq(self.coarse[tuple(self.coarse_dims)], rhs)


class GridTransferEquations:
    """
    """

    def __init__(self, target, glb_starts_f=None, full_weighting=False):
        # TODO: move imports
        from devito.petsc.types.array import PETScArray
        from devito.petsc.types.object import DMDALocalInfo

        self.target = target
        self.coarse_localinfo = DMDALocalInfo('cinfo')
        self.fine_localinfo = DMDALocalInfo('finfo')

        self.xc = PETScArray(
            name='x_' + target.name, target=target,
            liveness='eager', localinfo=self.coarse_localinfo
        )
        self.yf = PETScArray(
            name='y_' + target.name, target=target,
            liveness='eager', localinfo=self.fine_localinfo
        )
        self.row_sum = PETScArray(
            name='rs_' + target.name, target=target,
            liveness='eager', localinfo=self.coarse_localinfo
        ) if full_weighting else None
        self._build(glb_starts_f=glb_starts_f)

    def _build(self, fine=0, coarse=1, glb_starts_f=None):
        dims = self.target.space_dimensions
        so = self.target.space_order
        distributor = self.target.grid.distributor

        if glb_starts_f is None:
            glb_starts_f = []
            for d in dims:
                root = Scalar(name=f'{d.name}_m_glb', dtype=np.int32, is_const=True)
                glb_starts_f.append(
                    GlobalStartScalar(f'{d.name}_m_glb_d{fine}', dim=d,
                                      distributor=distributor, root=root)
                )

        glb_starts_c = []
        for i, d in enumerate(dims):
            glb_starts_c.append(
                GlobalStartScalar(f'{d.name}_m_glb_d{coarse}', dim=d,
                                  distributor=distributor, root=glb_starts_f[i].root)
            )

        transfer = GridTransfer(
            self.yf, self.xc, so=so,
            glb_starts_f=tuple(glb_starts_f), glb_starts_c=tuple(glb_starts_c),
            row_sum=self.row_sum
        )

        self._glb_start_syms_f = transfer.glb_starts_f
        self._glb_start_syms_c = transfer.glb_starts_c
        self._interp_eqs = transfer.interp_eqs
        self._restrict_eq = transfer.restrict_eq

    @property
    def interp_eqs(self):
        """Equations passed to `rcompile` for `InterpolationMult`."""
        return self._interp_eqs

    @property
    def restrict_eq(self):
        """Equations passed to `rcompile` for `RestrictionMult`."""
        return self._restrict_eq

    @property
    def glb_start_syms_f(self):
        """GlobalStartScalar instances for the fine-level global starts."""
        return self._glb_start_syms_f

    @property
    def glb_start_syms_c(self):
        """GlobalStartScalar instances for the coarse-level global starts."""
        return self._glb_start_syms_c


class UnevaluatedGridTransfer(sympy.Expr, Evaluable, Pickable):
    """
    Evaluates to a list of Eq objects representing a fine/coarse grid
    transfer. Mirrors UnevaluatedSparseOperation
    (devito/operations/interpolators.py).
    """

    __rargs__ = ('transfer',)

    def __new__(cls, transfer):
        obj = super().__new__(cls)
        obj.transfer = transfer
        return obj

    def _evaluate(self, **kwargs):
        return_value = self.operation(**kwargs)
        assert all(isinstance(i, Eq) for i in return_value)
        return return_value

    def __add__(self, other):
        return flatten([self, other])

    def __radd__(self, other):
        return flatten([other, self])


class GridInterpolation(UnevaluatedGridTransfer):
    """
    Represents interpolating a coarse-level object onto a fine-level one.
    Evaluates to a list of Eq objects.
    """

    def operation(self, **kwargs):
        return list(self.transfer.interp_eqs)

    def __repr__(self):
        return (f"GridInterpolation({repr(self.transfer.coarse)} onto "
                f"{repr(self.transfer.fine)})")

    __str__ = __repr__


class GridRestriction(UnevaluatedGridTransfer):
    """
    Represents restricting a fine-level object onto a coarse-level one.
    Evaluates to a list of Eq objects.
    """

    def operation(self, **kwargs):
        return [self.transfer.restrict_eq]

    def __repr__(self):
        return (f"GridRestriction({repr(self.transfer.fine)} onto "
                f"{repr(self.transfer.coarse)})")

    __str__ = __repr__


def _validate_transfer(source, target, *, want_finer_target):
    """
    Check that `source` and `target` are adjacent levels (exactly one
    factor-2 coarsening apart) of the same GridHierarchy, with `target` on
    the side (finer/coarser) that the calling function name promises, and
    that they have matching staggering.
    """
    source_depth = getattr(source.grid, 'coarsening_depth', 0)
    target_depth = getattr(target.grid, 'coarsening_depth', 0)
    expected_target_depth = source_depth - 1 if want_finer_target else source_depth + 1

    if source.grid.root is not target.grid.root or target_depth != expected_target_depth:
        direction = 'finer' if want_finer_target else 'coarser'
        raise ValueError(
            f"`target` must be exactly one factor-2 coarsening {direction} "
            f"than `source`, in the same GridHierarchy; got "
            f"source.grid={source.grid} (depth={source_depth}), "
            f"target.grid={target.grid} (depth={target_depth})"
        )
    if _field_shifts(source) != _field_shifts(target):
        raise ValueError(
            f"`source` and `target` must have matching staggering (they "
            f"should represent the same physical field at two resolutions); "
            f"got source.staggered={getattr(source, 'staggered', None)}, "
            f"target.staggered={getattr(target, 'staggered', None)}"
        )


def interpolate(source, target):
    """
    Interpolate `source` (coarse) onto `target` (fine) -- `target` must be
    exactly one factor-2 coarsening finer than `source`, in the same
    GridHierarchy.

    Parameters
    ----------
    source : Function
        The coarse-level Function to interpolate from (read).
    target : Function
        The fine-level Function to interpolate onto (written).

    Returns
    -------
    A lazily-evaluated object that expands to a list of Eq objects when
    passed to Operator (mirrors SparseFunction.interpolate/.inject, e.g.
    `Operator([Eq(f, f + 1)] + interpolate(source, target))`).
    """
    _validate_transfer(source, target, want_finer_target=True)
    return GridInterpolation(GridTransfer(target, source))


def restrict(source, target):
    """
    Restrict `source` (fine) onto `target` (coarse) -- `target` must be
    exactly one factor-2 coarsening coarser than `source`, in the same
    GridHierarchy.

    Parameters
    ----------
    source : Function
        The fine-level Function to restrict from (read).
    target : Function
        The coarse-level Function to restrict onto (written).

    Returns
    -------
    A lazily-evaluated object that expands to a list of Eq objects when
    passed to Operator (mirrors SparseFunction.interpolate/.inject, e.g.
    `Operator([Eq(f, f + 1)] + restrict(source, target))`).
    """
    _validate_transfer(source, target, want_finer_target=False)
    return GridRestriction(GridTransfer(source, target))