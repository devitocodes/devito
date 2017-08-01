import devito.interfaces as interfaces
from devito.logger import yask as log

from devito.yask import exit
from devito.yask.wrappers import YaskGrid, yask_context

__all__ = ['DenseData', 'TimeData']


class DenseData(interfaces.DenseData):

    def _allocate_memory(self):
        """Allocate memory in terms of Yask grids."""

        log("Allocating YaskGrid for %s (%s)" % (self.name, str(self.shape)))

        context = yask_context(self.indices, self.shape, self.dtype, self.space_order)

        # Sanity check
        if self.name in context.grids:
            exit("A grid with name %s already exits" % self.name)

        # Only create a YaskGrid if the requested grid is dense
        dimensions = tuple(i.name for i in self.indices)
        # TODO : following check fails if not using BufferedDimension ('time' != 't')
        if dimensions in [context.dimensions, context.space_dimensions]:
            # Set up the grid in YASK-land
            grid = context.make_grid(self.name, dimensions, self.shape, self.space_order)
            self._data_object = YaskGrid(grid, dimensions, self.shape,
                                         context.halo, self.dtype)
        else:
            log("Failed. Reverting to plain allocation...")
            super(DenseData, self)._allocate_memory()

    def initialize(self):
        raise NotImplementedError


class TimeData(interfaces.TimeData, DenseData):
    pass
