from devito.tools import as_tuple
from devito.dimension import SpaceDimension, TimeDimension, SteppingDimension
from devito.base import Constant

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
    :param dimensions: (Optional) list of :class:`SpaceDimension`
                       symbols that defines the spatial directions of
                       the physical domain encapsulated by this
                       :class:`Grid`.
    :param time_dimension: (Optional) :class:`TimeDimension` symbols
                           to to define the time dimension for all
                           :class:`TimeFunction` symbols created
                           from this :class:`Grid`.
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

    _default_dimensions = ('x', 'y', 'z')

    def __init__(self, shape, extent=None, origin=None, dimensions=None,
                 time_dimension=None, dtype=np.float32):
        self.shape = as_tuple(shape)
        self.extent = as_tuple(extent or tuple(1. for _ in shape))
        self.dtype = dtype
        origin = as_tuple(origin or tuple(0. for _ in shape))

        if dimensions is None:
            # Create the spatial dimensions and constant spacing symbols
            assert(self.dim <= 3)
            dim_names = self._default_dimensions[:self.dim]
            dim_spacing = tuple(Constant(name='h_%s' % name, value=val, dtype=self.dtype)
                                for name, val in zip(dim_names, self.spacing))
            self.dimensions = tuple(SpaceDimension(name=name, spacing=spc)
                                    for name, spc in zip(dim_names, dim_spacing))
        else:
            self.dimensions = dimensions

        self.origin = tuple(Constant(name='o_%s' % dim.name, value=val, dtype=self.dtype)
                            for dim, val in zip(self.dimensions, origin))
        # TODO: Raise proper exceptions and logging
        assert (self.dim == len(self.origin) == len(self.extent) == len(self.spacing))
        # Store or create default symbols for time and stepping dimensions
        if time_dimension is None:
            self.time_dim = TimeDimension(name='time',
                                          spacing=Constant(name='dt', dtype=self.dtype))
            self.stepping_dim = SteppingDimension(name='t', parent=self.time_dim)
        elif isinstance(time_dimension, TimeDimension):
            self.time_dim = time_dimension
            self.stepping_dim = SteppingDimension(name='%s_s' % time_dimension.name,
                                                  parent=self.time_dim)
        else:
            raise ValueError("`time_dimension` must be None or of type TimeDimension")

    def __repr__(self):
        return "Grid[extent=%s, shape=%s, dimensions=%s]" % (
            self.extent, self.shape, self.dimensions
        )

    @property
    def dim(self):
        """Problem dimension, or number of spatial dimensions."""
        return len(self.shape)

    @property
    def spacing(self):
        """Spacing between grid points in m."""
        spacing = (np.array(self.extent) / (np.array(self.shape) - 1)).astype(self.dtype)
        return as_tuple(spacing)

    @property
    def spacing_symbols(self):
        """Symbols representing the grid spacing in each :class:`SpaceDimension`"""
        return as_tuple(d.spacing for d in self.dimensions)

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each :class:`SpaceDimension`
        """
        return dict(zip(self.spacing_symbols, self.spacing))

    @property
    def shape_domain(self):
        """Shape of the physical domain (without external boundary layer)"""
        return self.shape
