from itertools import product as iterproduct

import sympy
from sympy import Integer, Rational, finite_diff_weights

from devito.symbolics import FieldFromComposite, IntDiv
from devito.types.basic import Scalar
from devito.types.dimension import Spacing
from devito.types.equation import Eq


def scale_param_for_level(param, level):
    """
    Scale a parameter to the correct value for a given MG level.

    Rules (factor = 2**level):
      Spacing   (h_x, h_y, ...)  -> param * factor   (grid spacing coarsens)
      Scalar _M (x_M, y_M, ...)  -> param // factor  (max index halves)
      Thickness / Scalar _m / other -> unchanged
    """
    factor = 2 ** level
    if isinstance(param, Spacing):
        return param * factor
    elif isinstance(param, Scalar) and param.name.endswith('_M'):
        return IntDiv(param, factor)
    return param


# TODO: Should really be generalised and not just used for `petscsolve`. It shouldn't really
# be a PETSc object..
class MultigridMetadata:
    """
    Class to hold metadata for geometric multigrid.
    """

    def __init__(self, field_data, solver_parameters):
        self.field_data = field_data
        self.solver_parameters = solver_parameters

        # TODO: what should I set the default to be? normally it is da_refine + 1 levels ...
        # maybe should just throw an error if not specifed by the user with pc_mg_levels?
        n_levels = int(solver_parameters.get('pc_mg_levels', 2))
        divisor = 2 ** (n_levels - 1)

        # Raise error for each dimension that cannot be uniformly coarsened to the specified number of levels
        # TODO: add test and check this for 1d,2d,3d cases
        grid = field_data.grid
        invalid = [
            (d, n) for d, n in zip(grid.dimensions, grid.shape)
            if (n - 1) % divisor != 0
        ]
        if invalid:
            msgs = ', '.join(
                f"dim {i}: size {n} (n-1={n-1} not divisible by {divisor})"
                for i, n in invalid
            )
            raise ValueError(
                f"Grid cannot be uniformly coarsened over {n_levels} levels: {msgs}. "
                f"Each (n-1) must be divisible by 2^(n_levels-1)={divisor}."
            )

        self._n_levels = n_levels
        self._fine_shape = grid.shape
        self._coarse_params = [
            {'shape': tuple((n - 1) // (2 ** i) + 1 for n in grid.shape), 'level': i}
            for i in range(1, n_levels)
        ]
        self._prolongation = GridTransferEquations(field_data.target)

    @property
    def n_levels(self):
        return self._n_levels

    @property
    def fine_shape(self):
        return self._fine_shape

    @property
    def coarse_params(self):
        return self._coarse_params

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

        self._prolong_eqs = [Eq(xc, yf)]

        self._restrict_eq = None

    @property
    def prolong_eqs(self):
        """Equations passed to `rcompile` for `ProlongationMult`."""
        return self._prolong_eqs

    @property
    def restrict_eq(self):
        """Equations passed to `rcompile` for `RestrictionMult`."""
        return self._restrict_eq
