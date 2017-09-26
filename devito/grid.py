from devito.tools import as_tuple
from devito.dimension import x, y, z

import numpy as np

__all__ = ['Grid']


class Grid(object):

    """
    A cartesian grid that encapsulates a physical domain over which
    to discretize :class:`Function`s.

    :param shape: Shape of the domain region in grid points.
    :param extent: Physical extent of the domain in m; defaults to a
                   unit box of extent 1m in all dimensions.
    :param origin: Physical coordinate of the origin of the domain;
                   defaults to 0. in all dimensions.
    :param dtype: Default data type to be inherited by all Functions
                  created from this :class:`Grid`.

    The :class:`Grid` encapsulates the topology and geometry
    information of the computational domain that :class:`Function`
    objects can be discretized on. As such it defines and provides the
    physical coordinate information of the logically cartesian grid
    underlying the discretized :class:`Function` objects. For example,
    the conventions for defining the coordinate space in 2D are:

    .. note::

       .. code-block:: python

          x ^
            |
            |           origin + extent
            |     x------------x
            |     |            |
            |     |            |
            |     |   DOMAIN   | extent[1]
            |     |            |
            |     |            |
            |     |  extent[0] |
            |     x------------x
            |  origin
            |
            |----------------------->
                       y
    """

    def __init__(self, shape, extent=None, origin=None, dimensions=None,
                 dtype=np.float32):
        self.shape = as_tuple(shape)
        self.extent = as_tuple(extent or tuple(1. for _ in shape))
        self.origin = as_tuple(origin or tuple(0. for _ in shape))
        self.dtype = dtype

        # TODO: Raise proper exceptions and logging
        assert(self.dim == len(self.origin) == len(self.extent) == len(self.spacing))

        # TODO: Create Dimensions locally instead of using global ones
        self.dimensions = (dimensions or (x, y, z))[:self.dim]

    @property
    def dim(self):
        """Problem dimension, or number of spatial dimensions."""
        return len(self.shape)

    @property
    def spacing(self):
        """Spacing between grid points in m."""
        return as_tuple(np.array(self.extent) / np.array(self.shape))

    @property
    def shape_domain(self):
        """Shape of the physical domain (without external boundary layer)"""
        return self.shape
