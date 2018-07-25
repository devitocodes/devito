import devito.grid as grid

from devito.yask.function import Constant
from devito.yask.wrappers import contexts

__all__ = ['Grid']


class Grid(grid.Grid):

    def __init__(self, *args, **kwargs):
        super(Grid, self).__init__(*args, **kwargs)

        # Initialize a new YaskContext for this Grid
        contexts.putdefault(self)

    @property
    def _const(self):
        return Constant

    def _make_stepping_dim(self, time_dim, **kwargs):
        # In the `yask` backend, the stepping dimension is an alias of the
        # time dimension
        return time_dim

    def __setstate__(self, state):
        super(Grid, self).__setstate__(state)
        # A new context is created, as the unpickled Dimensions are new objects
        contexts.putdefault(self)
