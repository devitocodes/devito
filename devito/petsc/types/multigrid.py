from functools import cached_property
from itertools import product as iterproduct

import numpy as np
import sympy
from sympy import Integer, Rational, finite_diff_weights

from devito.mpi import CoarseDistributor
from devito.symbolics import IntDiv
from devito.tools import as_tuple
from devito.types.basic import Scalar
from devito.types.dimension import ConditionalDimension, SpaceDimension, Spacing, Thickness
from devito.types.equation import Eq
from devito.types.grid import Grid


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


class MultigridMetadata:
    """
    PETSc-specific multigrid metadata: holds the GridHierarchy and the
    interpolation/restriction transfer equations for the target Function.

    Symbols are created once here and shared with GridTransferEquations so
    that interpolation/restriction and FormFunction/MatMult index transforms
    reference the same objects.
    """

    def __init__(self, hierarchy, target):
        self._hierarchy = hierarchy

        fine_grid = hierarchy.fine
        dims = fine_grid.dimensions
        distributor = fine_grid.distributor

        glb_starts_f = []
        gsc_c_syms = []
        for d in dims:
            root = Scalar(name=f'{d.name}_m_glb', dtype=np.int32, is_const=True)
            glb_starts_f.append(
                FineGlobalStartScalar(f'{d.name}_m_glb_d0', dim=d,
                                     distributor=distributor, root=root)
            )
            gsc_c_syms.append(
                GlobalStartScalar(f'{d.name}_m_glb', dim=d,
                                  distributor=distributor, root=root)
            )

        self._glb_start_syms_f = tuple(glb_starts_f)
        self._gsc_c = tuple(gsc_c_syms)
        self._factor = CoarseningFactorScalar('factor', depth=0)
        self._interpolation = GridTransferEquations(
            target, glb_starts_f=self._glb_start_syms_f
        )

    @property
    def hierarchy(self):
        return self._hierarchy

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def glb_start_syms_f(self):
        """Per-dimension fine-level GlobalStartScalars (routed via fine_ctx)."""
        return self._glb_start_syms_f

    @property
    def gsc_c(self):
        """Per-dimension canonical GlobalStartScalars for the current level."""
        return self._gsc_c

    @property
    def factor(self):
        """Canonical CoarseningFactorScalar (2^depth, populated per level)."""
        return self._factor


class GridTransferEquations:
    """
    """

    def __init__(self, target, glb_starts_f=None):
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
        self._build(glb_starts_f=glb_starts_f)

    def _build(self, fine=0, coarse=1, glb_starts_f=None):
        dims = self.target.space_dimensions
        so = self.target.space_order
        xc, yf = self.xc, self.yf
        ndim = len(dims)

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

        self._glb_start_syms_f = tuple(glb_starts_f)
        self._glb_start_syms_c = tuple(glb_starts_c)

        # Lagrange weights at the midpoint — shared by interpolation and restriction.
        start = -(so // 2 - 1)
        pts = list(range(start, start + so))
        w = finite_diff_weights(0, pts, Rational(1, 2))[-1][-1]

        # Interpolation: loop over fine indices. LHS is always yf[dims].
        # The parity of the global fine index (d + gsf) in each dimension determines
        # whether the fine point coincides with a coarse point (inject) or lies between
        # two coarse points (interpolate). A ConditionalDimension gates each of the
        # 2^ndim parity combinations. Within the conditional, (d + gsf - f) is always
        # even, so (d + gsf - f) // 2 is an exact integer coarse index.
        # Ghost coarse reads (local index -1) are handled naturally by the halo — no
        # loop-bound extension is needed.
        interp_eqs = []
        for flags in iterproduct([0, 1], repeat=ndim):
            conditions = [
                sympy.Eq(sympy.Mod(d + gsf, 2), f)
                for d, gsf, f in zip(dims, glb_starts_f, flags)
            ]
            condition = (sympy.And(*conditions, evaluate=False)
                         if ndim > 1 else conditions[0])
            cd = ConditionalDimension(
                name='cd' + ''.join(str(f) for f in flags),
                parent=dims[-1],
                condition=condition
            )

            lhs = yf[tuple(dims)]

            dim_stencils = []
            for d, gsc, gsf, f in zip(dims, glb_starts_c, glb_starts_f, flags):
                i_c = IntDiv(d + gsf - f, 2) - gsc
                if f:
                    dim_stencils.append([(i_c + j, wi) for j, wi in zip(pts, w)])
                else:
                    dim_stencils.append([(i_c, Integer(1))])

            rhs = Integer(0)
            for combo in iterproduct(*dim_stencils):
                weight = Integer(1)
                idx = []
                for i_c_expr, wi in combo:
                    weight *= wi
                    idx.append(i_c_expr)
                rhs += weight * xc[tuple(idx)]

            interp_eqs.append(Eq(lhs, rhs, implicit_dims=(cd,)))
        self._interp_eqs = tuple(interp_eqs)

        # Restriction: R = P^T. Loop over coarse indices — the natural direction.
        rhs = sympy.Integer(0)
        for flags in iterproduct([0, 1], repeat=ndim):
            offset_ranges = [pts if f else [0] for f in flags]
            for offsets in iterproduct(*offset_ranges):
                weight = sympy.Integer(1)
                fine_idx = []
                for d, gsc, gsf, f, j in zip(dims, glb_starts_c, glb_starts_f, flags, offsets):
                    if f:
                        weight *= w[pts.index(j)]
                        fine_idx.append(2*(d + gsc) - gsf - 2*j + 1)
                    else:
                        fine_idx.append(2*(d + gsc) - gsf)
                rhs += weight * yf[tuple(fine_idx)]

        self._restrict_eq = Eq(xc[tuple(dims)], rhs)

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