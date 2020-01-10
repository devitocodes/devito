import numpy as np
from sympy import sin, Abs


from devito import (Grid, SubDomain, Function, Constant,
                    SubDimension, Eq, Inc, Operator)
from devito.builtins import initialize_function, gaussian_smooth
from devito.tools import as_tuple

__all__ = ['Model', 'ModelElastic', 'ModelViscoelastic']


def initialize_damp(damp, nbl, spacing, mask=False):
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

    eqs = [Eq(damp, 1.0)] if mask else []
    for d in damp.dimensions:
        # left
        dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                  thickness=nbl)
        pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbl)
        pos = Abs((nbl - (d.symbolic_max - dim_r) + 1) / float(nbl))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    # TODO: Figure out why yask doesn't like it with dse/dle
    Operator(eqs, name='initdamp', dse='noop', dle='noop')()


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
                 dtype=np.float32, subdomains=(), damp_mask=False):
        self.shape = shape
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbl) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbl)
        subdomains = subdomains + (phydomain, )
        shape_pml = np.array(shape) + 2 * self.nbl
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         subdomains=subdomains)

        if self.nbl != 0:
            # Create dampening field as symbol `damp`
            self.damp = Function(name="damp", grid=self.grid)
            initialize_damp(self.damp, self.nbl, self.spacing, mask=damp_mask)
            self._physical_parameters = ['damp']
        else:
            self.damp = 1 if damp_mask else 0
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


class Model(GenericModel):
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

    The `Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    m : array_like or float
        The square slowness of the wave.
    damp : Function
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, nbl=20,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 subdomains=(), **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbl, dtype,
                                    subdomains)

        # Create square slowness of the wave as symbol `m`
        self._vp = self._gen_phys_param(vp, 'vp', space_order)
        self._max_vp = np.max(vp)

        # Additional parameter fields for TTI operators
        self.epsilon = self._gen_phys_param(epsilon, 'epsilon', space_order)
        self.scale = 1 if epsilon is None else np.sqrt(1 + 2 * np.max(epsilon))

        self.delta = self._gen_phys_param(delta, 'delta', space_order)
        self.theta = self._gen_phys_param(theta, 'theta', space_order)
        if self.grid.dim > 2:
            self.phi = self._gen_phys_param(phi, 'phi', space_order)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*self._max_vp))
        return self.dtype("%.3e" % dt)

    @property
    def vp(self):
        """
        `numpy.ndarray` holding the model velocity in km/s.

        Notes
        -----
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type `Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model and update square slowness.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            if vp.shape == self.vp.shape:
                self.vp.data[:] = vp[:]
            elif vp.shape == self.shape:
                initialize_function(self._vp, vp, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model of size" % vp.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.vp.shape))
        else:
            self._vp.data = vp
        self._max_vp = np.max(vp)

    @property
    def m(self):
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


class ModelElastic(GenericModel):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats, optional
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : float or array
        P-wave velocity in km/s.
    vs : float or array
        S-wave velocity in km/s.
    nbl : int, optional
        The number of absorbing layers for boundary damping.
    rho : float or array, optional
        Density in kg/cm^3 (rho=1 for water).

    The `ModelElastic` provides a symbolic data objects for the
    creation of seismic wave propagation operators:

    damp : Function, optional
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, vs, rho, nbl=20,
                 dtype=np.float32):
        super(ModelElastic, self).__init__(origin, spacing, shape, space_order,
                                           nbl=nbl, dtype=dtype,
                                           damp_mask=True)

        self.maxvp = np.max(vp)
        self.lam = self._gen_phys_param((vp**2 - 2 * vs**2)*rho, 'lam', space_order,
                                        is_param=True)

        self.mu = self._gen_phys_param(vs**2 * rho, 'mu', space_order, is_param=True)

        self.irho = self._gen_phys_param(1/rho, 'irho', space_order, is_param=True)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt < h / (sqrt(ndim) * max(vp)))
        dt = .95*np.min(self.spacing) / (np.sqrt(3)*self.maxvp)
        return self.dtype("%.3e" % dt)


class ModelViscoelastic(ModelElastic):
    """
    The physical model used in seismic inversion processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats, optional
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    vp : float or array
        P-wave velocity in km/s.
    qp : float or array
        P-wave quality factor (dimensionless).
    vs : float or array
        S-wave velocity in km/s.
    qs : float or array
        S-wave qulaity factor (dimensionless).
    nbl : int, optional
        The number of absorbing layers for boundary damping.
    rho : float or array, optional
        Density in kg/cm^3 (rho=1 for water).

    The `ModelElastic` provides a symbolic data objects for the
    creation of seismic wave propagation operators:

    damp : Function, optional
        The damping field for absorbing boundary condition.
    """
    def __init__(self, origin, spacing, shape, space_order, vp, qp, vs, qs, rho,
                 nbl=20, dtype=np.float32):
        super(ModelViscoelastic, self).__init__(origin, spacing, shape,
                                                space_order, vp, vs, rho,
                                                nbl=nbl, dtype=dtype)

        self.qp = self._gen_phys_param(qp, 'qp', space_order, is_param=True)

        self.qs = self._gen_phys_param(qs, 'qs', space_order, is_param=True)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        # See Blanch, J. O., 1995, "A study of viscous effects in seismic modelling,
        # imaging, and inversion: methodology, computational aspects and sensitivity"
        # for further details:
        dt = .85*np.min(self.spacing) / (np.sqrt(self.grid.dim)*self.maxvp)
        return self.dtype("%.3e" % dt)
