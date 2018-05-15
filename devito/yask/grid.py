import devito.grid as grid

from devito.yask.function import Constant

__all__ = ['Grid']


class Grid(grid.Grid):

    @property
    def _const(self):
        return Constant
