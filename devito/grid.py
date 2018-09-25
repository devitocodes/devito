from collections import namedtuple

from devito.dimension import SpaceDimension, TimeDimension, SteppingDimension
from devito.function import Constant
from devito.mpi import Distributor
from devito.parameters import configuration
from devito.tools import ArgProvider, ReducerMap, as_tuple, Tag

from sympy import prod
import numpy as np

__all__ = ['SubDomain', 'INTERIOR', 'DOMAIN']


class Grid(ArgProvider):

    """
    A cartesian grid that encapsulates a computational domain over which
    to discretize :class:`Function`s.

    :param shape: Shape of the computational domain in grid points.
    :param extent: (Optional) physical extent of the domain in m; defaults
                   to a unit box of extent 1m in all dimensions.
    :param origin: (Optional) physical coordinate of the origin of the
                   domain; defaults to 0.0 in all dimensions.
    :param dimensions: (Optional) list of :class:`SpaceDimension`s
                       defining the spatial dimensions of the computational
                       domain encapsulated by this Grid.
    :param time_dimension: (Optional) a :class:`TimeDimension`, used to
                           define the time dimension for all
                           :class:`TimeFunction`s created from this Grid.
    :param dtype: (Optional) default data type to be inherited by all
                  :class:`Function`s created from this Grid. Defaults
                  to ``numpy.float32``.
    :param subdomains: (Optional) an iterable of :class:`SubDomain`s. If None
                       (as by default), then the Grid only has two subdomains,
                       namely ``INTERIOR`` and ``DOMAIN``.
    :param comm: (Optional) an MPI communicator defining the set of
                 processes among which the grid is distributed.

    A Grid encapsulates the topology and geometry information of the
    computational domain that :class:`Function`s can be discretized on.
    As such, it defines and provides the physical coordinate information of
    the logical cartesian grid underlying the discretized :class:`Function`s.
    For example, the conventions for defining the coordinate space in 2D are:

    .. note::

       .. code-block:: python

                      x
            |----------------------->
            |  origin
            |     o------------o
            |     |            |
            |     |            |
            |     |   DOMAIN   | extent[1]
        y   |     |            |
            |     |            |
            |     |  extent[0] |
            |     o------------o
            |             origin + extent
            |
            v
    """

    _default_dimensions = ('x', 'y', 'z')

    def __init__(self, shape, extent=None, origin=None, dimensions=None,
                 time_dimension=None, dtype=np.float32, subdomains=None,
                 comm=None):
        self._shape = as_tuple(shape)
        self._extent = as_tuple(extent or tuple(1. for _ in self.shape))
        self._subdomains = (DOMAIN, INTERIOR) + as_tuple(subdomains)
        self._dtype = dtype

        if dimensions is None:
            # Create the spatial dimensions and constant spacing symbols
            assert(self.dim <= 3)
            dim_names = self._default_dimensions[:self.dim]
            dim_spacing = tuple(self._const(name='h_%s' % n, value=v, dtype=self.dtype)
                                for n, v in zip(dim_names, self.spacing))
            self._dimensions = tuple(SpaceDimension(name=n, spacing=s)
                                     for n, s in zip(dim_names, dim_spacing))
        else:
            self._dimensions = dimensions

        origin = as_tuple(origin or tuple(0. for _ in self.shape))
        self._origin = tuple(self._const(name='o_%s' % d.name, value=v, dtype=self.dtype)
                             for d, v in zip(self.dimensions, origin))

        # Sanity check
        assert (self.dim == len(self.origin) == len(self.extent) == len(self.spacing))

        # Store or create default symbols for time and stepping dimensions
        if time_dimension is None:
            spacing = self._const(name='dt', dtype=self.dtype)
            self._time_dim = TimeDimension(name='time', spacing=spacing)
            self._stepping_dim = self._make_stepping_dim(self.time_dim, name='t')
        elif isinstance(time_dimension, TimeDimension):
            self._time_dim = time_dimension
            self._stepping_dim = self._make_stepping_dim(self.time_dim)
        else:
            raise ValueError("`time_dimension` must be None or of type TimeDimension")

        self._distributor = Distributor(self.shape, self.dimensions, comm)

    def __repr__(self):
        return "Grid[extent=%s, shape=%s, dimensions=%s]" % (
            self.extent, self.shape, self.dimensions
        )

    @property
    def extent(self):
        """Physical extent of the domain in m."""
        return self._extent

    @property
    def dtype(self):
        """Data type inherited by all :class:`Function`s defined on this Grid."""
        return self._dtype

    @property
    def origin(self):
        """Physical coordinates of the domain origin."""
        return self._origin

    @property
    def dimensions(self):
        """Spatial dimensions of the computational domain."""
        return self._dimensions

    @property
    def dim(self):
        """Problem dimension, or number of spatial dimensions."""
        return len(self.shape)

    @property
    def time_dim(self):
        """Time dimension associated with this Grid."""
        return self._time_dim

    @property
    def stepping_dim(self):
        """Stepping dimension associated with this Grid."""
        return self._stepping_dim

    @property
    def subdomains(self):
        """The :class:`SubDomain`s defined in this Grid."""
        return self._subdomains

    @property
    def volume_cell(self):
        """
        Volume of a single cell e.g  h_x*h_y*h_z in 3D
        """
        return prod(d.spacing for d in self.dimensions).subs(self.spacing_map)

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
    def origin_domain(self):
        """
        Origin of the local (per-process) physical domain.
        """
        grid_origin = [min(i) for i in self.distributor.glb_numb]
        assert len(grid_origin) == len(self.spacing)
        return tuple(i*h for i, h in zip(grid_origin, self.spacing))

    @property
    def shape(self):
        """Shape of the physical domain."""
        return self._shape

    @property
    def shape_domain(self):
        """Shape of the local (per-process) physical domain."""
        return self._distributor.shape

    @property
    def dimension_map(self):
        """
        Map between ``self``'s :class:`SpaceDimension` and their global and
        local size.
        """
        return {d: namedtuple('Size', 'glb loc')(g, l)
                for d, g, l in zip(self.dimensions, self.shape, self.shape_domain)}

    @property
    def distributor(self):
        """The :class:`Distributor` used for domain decomposition."""
        return self._distributor

    @property
    def _const(self):
        """Return the type to create constant symbols."""
        return Constant

    def is_distributed(self, dim):
        """Return True if ``dim`` is a distributed :class:`Dimension`,
        False otherwise."""
        return dim in self.distributor.dimensions

    def _make_stepping_dim(self, time_dim, name=None):
        """Create a stepping dimension for this Grid."""
        if name is None:
            name = '%s_s' % time_dim.name
        return SteppingDimension(name=name, parent=time_dim)

    def _arg_defaults(self):
        """
        Returns a map of default argument values defined by this Grid.
        """
        args = ReducerMap()

        if configuration['mpi']:
            distributor = self.distributor
            args[distributor._C_comm.name] = distributor._C_comm.value
            args[distributor._C_neighbours.obj.name] = distributor._C_neighbours.obj.value

        return args

    def __getstate__(self):
        state = self.__dict__.copy()
        # A Distributor wraps an MPI communicator, which can't and shouldn't be pickled
        state.pop('_distributor')
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._distributor = Distributor(self.shape, self.dimensions)


class SubDomain(Tag):

    """A :class:`Grid` subdomain."""

    pass


DOMAIN = SubDomain('DOMAIN')
"""
The entire computational domain (== boundary + interior).
"""

INTERIOR = SubDomain('INTERIOR')
"""
The interior of the computational domain (i.e., boundaries are excluded).
"""
