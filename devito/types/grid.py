from collections import namedtuple

import numpy as np
from sympy import prod
from math import floor

from cached_property import cached_property

from devito.mpi import Distributor
from devito.parameters import configuration
from devito.tools import ReducerMap, as_tuple
from devito.types.args import ArgProvider
from devito.types.constant import Constant
from devito.types.dense import Function
from devito.types.dimension import (Dimension, SpaceDimension, TimeDimension,
                                    SteppingDimension, SubDimension)

__all__ = ['Grid', 'SubDomain', 'SubDomains']


class Grid(ArgProvider):

    """
    A cartesian grid that encapsulates a computational domain over which
    to discretize a Function.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the computational domain in grid points.
    extent : tuple of floats, optional
        Physical extent of the domain in m; defaults to a unit box of extent 1m
        in all dimensions.
    origin : tuple of floats, optional
        Physical coordinate of the origin of the domain; defaults to 0.0 in all
        dimensions.
    dimensions : tuple of SpaceDimension, optional
        The dimensions of the computational domain encapsulated by this Grid.
    time_dimension : TimeDimension, optional
        The dimension used to define time in a `TimeFunction` created from
        this Grid.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type, used as default
        data type to be inherited by all Functions created from this Grid.
        Defaults to ``np.float32``.
    subdomains : tuple of SubDomain, optional
        If no subdomains are specified, the Grid only defines the two default
        subdomains ``interior`` and ``domain``.
    comm : MPI communicator, optional
        The set of processes over which the grid is distributed. Only relevant in
        case of MPI execution.

    Examples
    --------
    >>> from devito import Grid, Function
    >>> grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
    >>> f = Function(name='f', grid=grid)
    >>> f.shape
    (4, 4)
    >>> f.dimensions
    (x, y)
    >>> f.dtype
    <class 'numpy.float32'>

    In a Function, the domain defined by a Grid is often surrounded by a "halo
    region", which guarantees the correctness of stencil updates nearby the
    domain boundary. However, the size of the halo region does *not* depend on
    the Grid; for more information, refer to ``Function.__doc__``.

    >>> f.shape_with_halo
    (6, 6)

    Notes
    -----
    A Grid encapsulates the topology and geometry information of the
    computational domain that a Function can be discretized on.  As such, it
    defines and provides the physical coordinate information of the logical
    cartesian grid underlying the discretized Functions.  For example, the
    conventions for defining the coordinate space in 2D are:

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

        # Initialize SubDomains
        subdomains = tuple(i for i in (Domain(), Interior(), *as_tuple(subdomains)))
        for i in subdomains:
            i.__subdomain_finalize__(self.dimensions, self.shape)
        self._subdomains = subdomains

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
        """Data type inherited by all Functions defined on this Grid."""
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
        """The SubDomains defined in this Grid."""
        return {i.name: i for i in self._subdomains}

    @property
    def interior(self):
        """The interior SubDomain of the Grid."""
        return self.subdomains['interior']

    @property
    def volume_cell(self):
        """Volume of a single cell e.g  h_x*h_y*h_z in 3D."""
        return prod(d.spacing for d in self.dimensions).subs(self.spacing_map)

    @property
    def spacing(self):
        """Spacing between grid points in m."""
        spacing = (np.array(self.extent) / (np.array(self.shape) - 1)).astype(self.dtype)
        return as_tuple(spacing)

    @property
    def spacing_symbols(self):
        """Symbols representing the grid spacing in each SpaceDimension"""
        return as_tuple(d.spacing for d in self.dimensions)

    @property
    def spacing_map(self):
        """Map between spacing symbols and their values for each SpaceDimension."""
        return dict(zip(self.spacing_symbols, self.spacing))

    @property
    def origin_offset(self):
        """Offset of the local (per-process) origin from the domain origin."""
        grid_origin = [min(i) for i in self.distributor.glb_numb]
        assert len(grid_origin) == len(self.spacing)
        return tuple(i*h for i, h in zip(grid_origin, self.spacing))

    @property
    def shape(self):
        """Shape of the physical domain."""
        return self._shape

    @property
    def shape_local(self):
        """Shape of the local (per-process) physical domain."""
        return self._distributor.shape

    @property
    def dimension_map(self):
        """Map between SpaceDimensions and their global/local size."""
        return {d: namedtuple('Size', 'glb loc')(g, l)
                for d, g, l in zip(self.dimensions, self.shape, self.shape_local)}

    @property
    def distributor(self):
        """The Distributor used for domain decomposition."""
        return self._distributor

    def is_distributed(self, dim):
        """True if ``dim`` is a distributed Dimension, False otherwise."""
        return any(dim is d for d in self.distributor.dimensions)

    @property
    def _const(self):
        """The type to be used to create constant symbols."""
        return Constant

    def _make_stepping_dim(self, time_dim, name=None):
        """Create a stepping dimension for this Grid."""
        if name is None:
            name = '%s_s' % time_dim.name
        return SteppingDimension(name=name, parent=time_dim)

    def _arg_defaults(self):
        """A map of default argument values defined by this Grid."""
        args = ReducerMap()

        for k, v in self.dimension_map.items():
            args.update(k._arg_defaults(_min=0, size=v.loc))

        if configuration['mpi']:
            distributor = self.distributor
            args[distributor._obj_comm.name] = distributor._obj_comm.value
            args[distributor._obj_neighborhood.name] = distributor._obj_neighborhood.value

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


class SubDomain(object):

    """
    Base class to define Grid subdomains.

    To create a new SubDomain, all one needs to do is overriding :meth:`define`.
    This method takes as input a set of Dimensions and produce a mapper

        ``M : Dimensions -> {d, ('left', N), ('middle', N, M), ('right', N)}``

    so that:

        * If ``M(d) = d``, then the SubDomain spans the entire Dimension ``d``.
        * If ``M(d) = ('left', N)``, then the SubDomain spans a contiguous
          region of ``N`` points starting at ``d``\'s left extreme.
        * ``M(d) = ('right', N)`` is analogous to the case above.
        * If ``M(d) = ('middle', N, M)``, then the SubDomain spans a contiguous
          region of ``d_size - (N + M)`` points starting at ``N`` and finishing
          at ``d_sizeM - M``.

    Examples
    --------
    An "Inner" SubDomain, which spans the entire domain except for an exterior
    boundary region of ``thickness=3``, can be implemented as follows

    >>> from devito import SubDomain
    >>> class Inner(SubDomain):
    ...     name = 'inner'
    ...     def define(self, dimensions):
    ...         return {d: ('middle', 3, 3) for d in dimensions}

    Like before, but now spanning the entire ``y`` Dimension of a three-dimensional
    grid

    >>> class InnerY(SubDomain):
    ...     name = 'inner_y'
    ...     def define(self, dimensions):
    ...         x, y, z = dimensions
    ...         return {x: ('middle', 3, 3), y: y, z: ('middle', 3, 3)}

    See Also
    --------
    Domain : An example of preset SubDomain.
    Interior : An example of preset Subdomain.
    """

    name = None
    """A unique name for the SubDomain."""

    def __init__(self):
        if self.name is None:
            raise ValueError("SubDomain requires a `name`")
        self._dimensions = None

    def __subdomain_finalize__(self, dimensions, shape):
        # Create the SubDomain's SubDimensions
        sub_dimensions = []
        for k, v in self.define(dimensions).items():
            if isinstance(v, Dimension):
                sub_dimensions.append(v)
            else:
                try:
                    # Case ('middle', int, int)
                    side, thickness_left, thickness_right = v
                    if side != 'middle':
                        raise ValueError("Expected side 'middle', not `%s`" % side)
                    sub_dimensions.append(SubDimension.middle('%si' % k.name, k,
                                                              thickness_left,
                                                              thickness_right))
                except ValueError:
                    side, thickness = v
                    if side == 'left':
                        sub_dimensions.append(SubDimension.left('%sleft' % k.name, k,
                                                                thickness))
                    elif side == 'right':
                        sub_dimensions.append(SubDimension.right('%sright' % k.name, k,
                                                                 thickness))
                    else:
                        raise ValueError("Expected sides 'left|right', not `%s`" % side)
        self._dimensions = tuple(sub_dimensions)

        # Compute the SubDomain shape
        self._shape = tuple(s - (sum(d._thickness_map.values()) if d.is_Sub else 0)
                            for d, s in zip(self._dimensions, shape))

    def __eq__(self, other):
        if not isinstance(other, SubDomain):
            return False
        return self.name == other.name and self.dimensions == other.dimensions

    def __hash__(self):
        return hash((self.name, self.dimensions))

    def __str__(self):
        return "SubDomain %s[%s]" % (self.name, self.dimensions)

    __repr__ = __str__

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def dimension_map(self):
        return {d.root: d for d in self.dimensions}

    @property
    def shape(self):
        return self._shape

    def define(self, dimensions):
        """
        Parametrically describe the SubDomain w.r.t. a generic Grid.

        Notes
        -----
        This method should be overridden by each SubDomain subclass. For more
        information, refer to ``SubDomain.__doc__``.
        """
        raise NotImplementedError


class Domain(SubDomain):

    """
    The entire computational domain (== boundary + interior).
    """

    name = 'domain'

    def define(self, dimensions):
        return dict(zip(dimensions, dimensions))


class Interior(SubDomain):

    """
    The interior of the computational domain (i.e., boundaries are excluded).
    """

    name = 'interior'

    def define(self, dimensions):
        return {d: ('middle', 1, 1) for d in dimensions}


# FIXME: Name + example.
class SubDomains(SubDomain):
    """
    Class to define a set of N (a positive integer) subdomains.

    Parameters
    ----------
    **kwargs
        * N : int
            Number of subdomains.
        * bounds : tuple
            Tuple of numpy int32 arrays representing the bounds of
            each subdomain.

    Examples
    --------
    Include full example + example of shorthand notation here.
    """

    def __init__(self, **kwargs):
        super(SubDomains, self).__init__()
        self._n_domains = kwargs.get('N', 1)
        self._bounds = kwargs.get('bounds', None)

    def __subdomain_finalize__(self, dimensions, shape):
        super(SubDomains, self).__subdomain_finalize__(dimensions, shape)
        n = Dimension(name='n')
        self._implicit_dimension = n

    @property
    def n_domains(self):
        return self._n_domains

    @property
    def bounds(self):
        return self._bounds

    @cached_property
    def _implicit_eq_dat(self):
        if not len(self._bounds) == 2*len(self.dimensions):
            raise ValueError("Left and right bounds must be supplied for each dimension")
        n_domains = self.n_domains
        i_dim = self._implicit_dimension
        b_f = {}
        dat = []
        # Organise the data contained in 'bounds' into a form such that the
        # associated implicit equations can easily be created.
        for j in range(0, len(self._bounds)):
            index = floor(j/2)
            d = self.dimensions[index]
            if j % 2 == 0:
                fname = d.min_name
            else:
                fname = d.max_name
            b_f[fname] = Function(name=fname, shape=(n_domains, ),
                                  dimensions=(i_dim, ), dtype=np.int32)
            if isinstance(self._bounds[j], int):
                bounds = np.zeros((n_domains,), dtype=np.int32)
                b_f[fname].data[:] = bounds
            else:
                b_f[fname].data[:] = self._bounds[j]
            dat.append({'rhs': d.thickness[j % 2][0], 'lhs': b_f[fname][i_dim]})
        return as_tuple(dat)
