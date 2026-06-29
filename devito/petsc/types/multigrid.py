from functools import cached_property
from itertools import product as iterproduct

import numpy as np
import sympy
from sympy import Integer, Rational, finite_diff_weights

from devito.mpi import CoarseDistributor
from devito.symbolics import FieldFromComposite
from devito.tools import as_tuple
from devito.types.basic import Scalar
from devito.types.dimension import Spacing, Thickness
from devito.types.equation import Eq


class CoarseGridScalar(Scalar):
    """
    A rank-local dimension scalar for a coarse multigrid level.

    offset = -1 for *_M symbols, 0 for *_size symbols.
    is_Input = True so the operator's argument collection calls _arg_values.
    """

    is_Input = True

    __rkwargs__ = ('name', 'dtype', 'is_const', 'dim', 'distributor', 'offset')

    def __new__(cls, name, dim=None, distributor=None, offset=0, **kwargs):
        kwargs.setdefault('dtype', np.int32)
        kwargs.setdefault('is_const', True)
        newobj = super().__new__(cls, name, **kwargs)
        newobj._dim = dim
        newobj._distributor = distributor
        newobj._offset = offset
        return newobj

    def _arg_values(self, **kwargs):
        return {self.name: self._distributor.shape[self._dim] + self._offset}


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


class SubGrid:

    """
    A coarser-level grid in a multigrid hierarchy.

    Shares SpaceDimensions, physical extent, and dtype with the parent Grid
    but has a coarser shape and a CoarseDistributor for its MPI partition.
    Not constructed directly by users - created by GridHierarchy.
    """

    def __init__(self, shape, parent, coarsening_depth):
        self._shape = as_tuple(shape)
        self._parent = parent
        self._coarsening_depth = coarsening_depth
        self._distributor = CoarseDistributor(shape, parent.dimensions,
                                              parent.distributor)
        self._coarse_symbol_cache = {}

    def __repr__(self):
        return f'SubGrid[shape={self._shape}, dimensions={self.dimensions}]'

    @property
    def shape(self):
        return self._shape

    @property
    def dimensions(self):
        return self._parent.dimensions

    @property
    def dtype(self):
        return self._parent.dtype

    @property
    def extent(self):
        return self._parent.extent

    @cached_property
    def spacing(self):
        spacing = (np.array(self.extent) /
                   (np.array(self._shape) - 1)).astype(self.dtype)
        return as_tuple(spacing)

    @property
    def distributor(self):
        return self._distributor

    @property
    def coarsening_depth(self):
        """Number of factor-2 coarsenings from the fine Grid (1 = one halving, etc.)."""
        return self._coarsening_depth

    @property
    def parent(self):
        return self._parent

    def coarse_symbol_for(self, f):
        """Return the coarse-level equivalent of fine-grid symbol f."""
        try:
            return self._coarse_symbol_cache[f]
        except KeyError:
            coarse = self._build_coarse_symbol(f)
            self._coarse_symbol_cache[f] = coarse
            return coarse

    def _build_coarse_symbol(self, f):
        factor = 2 ** self._coarsening_depth
        if isinstance(f, Thickness):
            coarse_val = (f.value + factor - 1) // factor
            return CoarseThickness(
                name=f'{f.name}_d{self._coarsening_depth}',
                root=f.root, side=f.side, local=f.local,
                value=coarse_val, distributor=self._distributor
            )
        elif isinstance(f, Spacing):
            return f * factor
        elif isinstance(f, Scalar):
            for dim in self._parent.dimensions:
                if f.name in (f'{dim.name}_M', f'{dim.name}_size'):
                    offset = -1 if f.name.endswith('_M') else 0
                    return CoarseGridScalar(
                        name=f'{f.name}_d{self._coarsening_depth}',
                        dim=dim,
                        distributor=self._distributor,
                        offset=offset
                    )
        return f


class GridHierarchy:

    """
    A hierarchy of grids for geometric multigrid.

    Applies successive factor-2 coarsenings to a fine Grid, producing a
    SubGrid per coarser level. Each SubGrid shares the fine Grid's
    SpaceDimensions and physical extent.

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
        """The finest Grid."""
        return self._fine

    @property
    def coarse_levels(self):
        """Coarser SubGrids ordered finest-coarse to coarsest."""
        return self._coarse_levels

    @property
    def nlevels(self):
        return self._nlevels

    @property
    def levels(self):
        """All levels as a tuple: (fine_grid, subgrid_l1, ..., subgrid_lN)."""
        return (self._fine,) + self._coarse_levels


class MultigridMetadata:
    """
    PETSc-specific multigrid metadata: holds the GridHierarchy and the
    prolongation/restriction transfer equations for the target Function.
    """

    def __init__(self, hierarchy, field_data):
        self._hierarchy = hierarchy
        self._prolongation = GridTransferEquations(field_data.target)

    @property
    def hierarchy(self):
        return self._hierarchy

    @property
    def prolongation(self):
        return self._prolongation


class GridTransferEquations:
    """
    """

    def __init__(self, target):
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
        self._build()

    def _build(self):
        dims = self.target.space_dimensions
        so = self.target.space_order
        xc, yf = self.xc, self.yf

        petsc_letters = ['x', 'y', 'z']
        ndim = len(dims)
        xs_f = [
            FieldFromComposite(f'{petsc_letters[ndim - 1 - i]}s', self.fine_localinfo)
            for i in range(ndim)
        ]
        xs_c = [
            FieldFromComposite(f'{petsc_letters[ndim - 1 - i]}s', self.coarse_localinfo)
            for i in range(ndim)
        ]

        # Lagrange weights at the midpoint — shared by prolongation and restriction.
        start = -(so // 2 - 1)
        pts = list(range(start, start + so))
        w = finite_diff_weights(0, pts, Rational(1, 2))[-1][-1]

        prolong_eqs = []
        for flags in iterproduct([0, 1], repeat=ndim):
            lhs_idx = tuple(
                2*(d + xsc) - xsf + f
                for d, xsc, xsf, f in zip(dims, xs_c, xs_f, flags)
            )
            lhs = yf[lhs_idx]

            dim_stencils = []
            for d, f in zip(dims, flags):
                if f:
                    dim_stencils.append(list(zip([d + j for j in pts], w)))
                else:
                    dim_stencils.append([(d, Integer(1))])

            rhs = Integer(0)
            for combo in iterproduct(*dim_stencils):
                weight = Integer(1)
                idx = []
                for dim_offset, wi in combo:
                    weight *= wi
                    idx.append(dim_offset)
                rhs = rhs + weight * xc[tuple(idx)]

            prolong_eqs.append(Eq(lhs, rhs))
        self._prolong_eqs = tuple(prolong_eqs)

        # Restriction: R = P^T — same weights, reversed fine-array indices.
        rhs = sympy.Integer(0)
        for flags in iterproduct([0, 1], repeat=ndim):
            offset_ranges = [pts if f else [0] for f in flags]
            for offsets in iterproduct(*offset_ranges):
                weight = sympy.Integer(1)
                fine_idx = []
                for d, xsc, xsf, f, j in zip(dims, xs_c, xs_f, flags, offsets):
                    if f:
                        weight *= w[pts.index(j)]
                        fine_idx.append(2*(d + xsc) - xsf - 2*j + 1)
                    else:
                        fine_idx.append(2*(d + xsc) - xsf)
                rhs += weight * yf[tuple(fine_idx)]

        self._restrict_eq = Eq(xc[tuple(dims)], rhs)

    @property
    def prolong_eqs(self):
        """Equations passed to `rcompile` for `ProlongationMult`."""
        return self._prolong_eqs

    @property
    def restrict_eq(self):
        """Equations passed to `rcompile` for `RestrictionMult`."""
        return self._restrict_eq