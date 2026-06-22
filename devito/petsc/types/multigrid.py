from devito.symbolics import IntDiv
from devito.types.dimension import Spacing
from devito.types.basic import Scalar


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

    @property
    def n_levels(self):
        return self._n_levels

    @property
    def fine_shape(self):
        return self._fine_shape

    @property
    def coarse_params(self):
        return self._coarse_params
