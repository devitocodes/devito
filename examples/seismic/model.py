import numpy as np
from sympy import sin, Abs


from devito import (Grid, SubDomain, Function, Constant,
                    SubDimension, Eq, Inc, Operator)
from devito.builtins import initialize_function, gaussian_smooth, mmax
from devito.tools import as_tuple

__all__ = ['SeismicModel', 'Model', 'ModelElastic',
           'ModelViscoelastic', 'ModelViscoacoustic']


def initialize_damp(damp, nbl, spacing, abc_type="damp"):
    """
    Initialise damping field with an absorbing boundary layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbl : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbl)

    eqs = [Eq(damp, 1.0)] if abc_type == "mask" else []
    for d in damp.dimensions:
        # left
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbl)
        pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if abc_type == "mask" else val
        eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbl)
        pos = Abs((nbl - (d.symbolic_max - dim_r) + 1) / float(nbl))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if abc_type == "mask" else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    Operator(eqs, name='initdamp')()


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbl):
        super(PhysicalDomain, self).__init__()
        self.nbl = nbl

    def define(self, dimensions):
        return {d: ('middle', self.nbl, self.nbl) for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbl=20,
                 dtype=np.float32, subdomains=(), bcs="damp",
                 grid=None):
        self.shape = shape
        self.space_order = space_order
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        if grid is None:
            origin_pml = tuple([dtype(o - s*nbl) for o, s in zip(origin, spacing)])
            phydomain = PhysicalDomain(self.nbl)
            subdomains = subdomains + (phydomain, )
            shape_pml = np.array(shape) + 2 * self.nbl
            # Physical extent is calculated per cell, so shape - 1
            extent = tuple(np.array(spacing) * (shape_pml - 1))
            self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml,
                             dtype=dtype, subdomains=subdomains)
        else:
            self.grid = grid

        if self.nbl != 0:
            # Create dampening field as symbol `damp`
            self.damp = Function(name="damp", grid=self.grid)
            if callable(bcs):
                bcs(self.damp, self.nbl)
            else:
                initialize_damp(self.damp, self.nbl, self.spacing, abc_type=bcs)
            self._physical_parameters = ['damp']
        else:
            self.damp = 1 if bcs == "mask" else 0
            self._physical_parameters = []

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self.physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    def _gen_phys_param(self, field, name, space_order, is_param=False,
                        default_value=0):
        if field is None:
            return default_value
        if isinstance(field, np.ndarray):
            function = Function(name=name, grid=self.grid, space_order=space_order,
                                parameter=is_param)
            initialize_function(function, field, self.nbl)
        else:
            function = Constant(name=name, value=field)
        self._physical_parameters.append(name)
        return function

    @property
    def physical_parameters(self):
        return as_tuple(self._physical_parameters)

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))


class SeismicModel(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : array_like or float
        Velocity in km/s.
    nbl : int, optional
        The number of absorbin layers for boundary damping.
    bcs: String or callable
        Absorbing boundary type ("damp" or "mask") or initializer
    dtype : np.float32 or np.float64
        Defaults to 32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.
    b : array_like or float
        Buoyancy
    rho : array_like or float
        Density
    vs : array_like or float
        S-wave velocity
    qp : array_like or float
        P-wave attenuation
    qs : array_like or float
        S-wave attenuation

    """
    def __init__(self, origin, spacing, shape, space_order, vp, nbl=20,
                 dtype=np.float32, subdomains=(), bcs="damp", grid=None, **kwargs):
        super(SeismicModel, self).__init__(origin, spacing, shape, space_order, nbl,
                                           dtype, subdomains, grid=grid, bcs=bcs)

        # All seismic models have at least a velocity
        self.vp = self._gen_phys_param(vp, 'vp', space_order)

        # Initialize physics
        self._initialize_physics(space_order, **kwargs)

    def _initialize_physics(self, space_order, **kwargs):
        """
        Initialize physical parameters and type of physics from inputs.
        The types of physics supportedare:
        - acoustic: vp and rho/b only
        - elastic: vp + vs + b
        - visco-acoustic: vp + b + qp
        - visco-elastic: vp + vs + b + qs
        - vti: epsilon + delta
        - tti: epsilon + delta + theta + phi
        """
        params = []
        self._scale = 1
        # Make sure only one of density and buoyancy is in input
        if 'rho' in kwargs.keys() and 'b' in kwargs.keys():
            assert 1 /  kwargs.get('rho') == kwargs.get('b')
            kwargs.pop('rho')
        # Initialize input physical parameters
        for k, v in kwargs.items():
            setattr(self, k, self._gen_phys_param(v, k, space_order))
            params.append(k)
        # Update scale for tti
        if 'epsilon' in params:
            self._scale += 2 * np.max(kwargs.get('epsilon')) 
        # Set CFL constant
        self._cfl_coeff = .85 / np.sqrt(self.dim)

    @property
    def _max_vp(self):
        return mmax(self.vp)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        dt = self._cfl_coeff * np.min(self.spacing) / (self._scale*self._max_vp)
        return self.dtype("%.3e" % dt)

    def update(self, name, value):
        """
        Update the physical parameter param
        """
        try:
            param = getattr(self, name)
        except AttributeError:
            # No physical parameter with tha name, create it
            setattr(self, name, self._gen_phys_param(name, value, self.space_order))
            return
        # Update the square slowness according to new value
        if isinstance(value, np.ndarray):
            if value.shape == param.shape:
                param.data[:] = value[:]
            elif value.shape == self.shape:
                initialize_function(param, value, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model of size" % value.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     param.shape))
        else:
            param.data = value

    @property
    def m(self):
        """
        Squared slowness
        """
        return 1 / (self.vp * self.vp)

    def smooth(self, physical_parameters, sigma=5.0):
        """
        Apply devito.gaussian_smooth to model physical parameters.

        Parameters
        ----------
        physical_parameters : string or tuple of string
            Names of the fields to be smoothed.
        sigma : float
            Standard deviation of the smoothing operator.
        """
        model_parameters = self.physical_params()
        for i in physical_parameters:
            gaussian_smooth(model_parameters[i], sigma=sigma)
        return


# For backward ompativility
Model = SeismicModel
ModelElastic = SeismicModel
ModelViscoelastic = SeismicModel
ModelViscoacoustic = SeismicModel
