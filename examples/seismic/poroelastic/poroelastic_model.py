import os

import numpy as np

from examples.seismic.utils import scipy_smooth
from devito import Grid, SubDomain, Function, Constant, warning

__all__ = ['Model', 'ModelPoroelastic', 'demo_model']


def demo_model(preset, **kwargs):
    """
    Utility function to create preset :class:`Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * `constant-poroelastic` : Constant parameter poroelastic isotropic model

    """
    space_order = kwargs.pop('space_order', 2)


    if preset.lower() in ['constant-poroelastic']:
        # A constant single-layer model in a 2D or 3D domain
        shape = kwargs.pop('shape', (250, 640)) # metres
        spacing = kwargs.pop('spacing', tuple([0.5 for _ in shape])) # metres
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)

        # Matrix Properties
        shm = np.ones(shape, dtype=dtype)*kwargs.pop('shm', 2.4e09)     # Shear Modulus, Pa = kg / (m * s**2)
        phi = np.ones(shape, dtype=dtype)*kwargs.pop('phi', 0.10)       # Porosity, %
        prm = np.ones(shape, dtype=dtype)*kwargs.pop('prm', 1.0e-14)    # Permeability, m**2
        kdr = np.ones(shape, dtype=dtype)*kwargs.pop('kdr', 6e09)       # Drained bulk modulus, Pa = kg / (m * s**2)
        T = np.ones(shape, dtype=dtype)*kwargs.pop('T', 2.42)           # Tortuosity, -

        # Solid Properties
        rhos = np.ones(shape, dtype=dtype)*kwargs.pop('rhos', 2250.)    # Solid grain density, kg/m**3
        ksg = np.ones(shape, dtype=dtype)*kwargs.pop('ksg', 36e10)      # Solid grain bulk modulus, Pa = kg / (m * s**2)

        # Fluid Properties
        rhof = np.ones(shape, dtype=dtype)*kwargs.pop('rhof', 937.)     # Fluid density, kg/m**3
        fvs = np.ones(shape, dtype=dtype)*kwargs.pop('fvs', 0.003)      # Fluid viscosity, Pa*s = (kg / (m * s**2)) * s
        kfl = np.ones(shape, dtype=dtype)*kwargs.pop('kfl', 2.25e9)     # Fluid bulk modulus, Pa = kg / (m * s**2)

        return ModelPoroelastic(space_order=space_order, rhos=rhos,
        rhof=rhof, phi=phi, prm=prm, fvs=fvs, kdr=kdr, ksg=ksg, kfl=kfl, shm=shm,
        T=T, origin=origin, shape=shape, dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    else:
        raise ValueError("Unknown model preset name")


def initialize_damp(damp, nbpml, spacing, mask=False):
    """Initialise damping field with an absorbing PML layer.

    :param damp: The :class:`Function` for the damping field.
    :param nbpml: Number of points in the damping layer.
    :param spacing: Grid spacing coefficient.
    :param mask: whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """

    phy_shape = damp.grid.subdomains['phydomain'].shape
    data = np.ones(phy_shape) if mask else np.zeros(phy_shape)

    pad_widths = [(nbpml, nbpml) for i in range(damp.ndim)]
    data = np.pad(data, pad_widths, 'edge')

    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)

    assert all(damp._offset_domain[0] == i for i in damp._offset_domain)

    for i in range(damp.ndim):
        for j in range(nbpml):
            # Dampening coefficient
            pos = np.abs((nbpml - j + 1) / float(nbpml))
            val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
            if mask:
                val = -val
            # : slices
            all_ind = [slice(0, d) for d in data.shape]
            # Left slice for dampening for dimension i
            all_ind[i] = slice(j, j+1)
            data[tuple(all_ind)] += val/spacing[i]
            # right slice for dampening for dimension i
            all_ind[i] = slice(data.shape[i]-j, data.shape[i]-j+1)
            data[tuple(all_ind)] += val/spacing[i]

    initialize_function(damp, data, 0)


def initialize_function(function, data, nbpml, pad_mode='edge'):
    """Initialize a :class:`Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this function.

    :param function: The :class:`Function` to be initialised with some data.
    :param data: The data array used for initialisation.
    :param nbpml: Number of PML layers for boundary damping.
    :param pad_mode: A string or a suitable padding function as explained in
                     :func:`numpy.pad`.
    """
    pad_widths = [(nbpml + i.left, nbpml + i.right) for i in function._size_halo]
    data = np.pad(data, pad_widths, pad_mode)
    function.data_with_halo[:] = data


class PhysicalDomain(SubDomain):

    name = 'phydomain'

    def __init__(self, nbpml):
        super(PhysicalDomain, self).__init__()
        self.nbpml = nbpml

    def define(self, dimensions):
        return {d: ('middle', self.nbpml, self.nbpml) for d in dimensions}


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbpml=20,
                 dtype=np.float32):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])

        # Origin of the computational domain with PML to inject/interpolate
        # at the correct index
        origin_pml = tuple([dtype(o - s*nbpml) for o, s in zip(origin, spacing)])
        phydomain = PhysicalDomain(self.nbpml)
        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=origin_pml, dtype=dtype,
                         subdomains=phydomain)

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self._physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

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
        Map between spacing symbols and their values for each :class:`SpaceDimension`
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
    """The physical model used in seismic inversion processes.

    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param space_order: Order of the spatial stencil discretisation
    :param vp: Velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param theta: Tilt angle in radian
    :param phi: Asymuth angle in radian

    The :class:`Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    :param m: The square slowness of the wave
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, shape, space_order, vp, nbpml=20,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbpml, dtype)

        # Are we provided with an existing grid?
        grid = kwargs.get('grid')
        if grid is not None:
            assert self.grid.extent == grid.extent
            assert self.grid.shape == grid.shape
            self.grid = grid

        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = Function(name="m", grid=self.grid, space_order=space_order)
        else:
            self.m = Constant(name="m", value=1/vp**2)
        self._physical_parameters = ('m',)
        # Set model velocity, which will also set `m`
        self.vp = vp

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                self._physical_parameters += ('epsilon',)
                self.epsilon = Function(name="epsilon", grid=self.grid)
                initialize_function(self.epsilon, 1 + 2 * epsilon, self.nbpml)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if np.max(self.epsilon.data_with_halo[:]) > 0:
                    self.scale = np.sqrt(np.max(self.epsilon.data_with_halo[:]))
            else:
                self.epsilon = 1 + 2 * epsilon
                self.scale = epsilon
        else:
            self.epsilon = 1

        if delta is not None:
            if isinstance(delta, np.ndarray):
                self._physical_parameters += ('delta',)
                self.delta = Function(name="delta", grid=self.grid)
                initialize_function(self.delta, np.sqrt(1 + 2 * delta), self.nbpml)
            else:
                self.delta = delta
        else:
            self.delta = 1

        if theta is not None:
            if isinstance(theta, np.ndarray):
                self._physical_parameters += ('theta',)
                self.theta = Function(name="theta", grid=self.grid,
                                      space_order=space_order)
                initialize_function(self.theta, theta, self.nbpml)
            else:
                self.theta = theta
        else:
            self.theta = 0

        if phi is not None:
            if self.grid.dim < 3:
                warning("2D TTI does not use an azimuth angle Phi, ignoring input")
                self.phi = 0
            elif isinstance(phi, np.ndarray):
                self._physical_parameters += ('phi',)
                self.phi = Function(name="phi", grid=self.grid, space_order=space_order)
                initialize_function(self.phi, phi, self.nbpml)
            else:
                self.phi = phi
        else:
            self.phi = 0

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*np.max(self.vp)))
        return .001 * int(1000 * dt)

    @property
    def vp(self):
        """:class:`numpy.ndarray` holding the model velocity in km/s.
        .. note::
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type :class:`Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """Set a new velocity model and update square slowness

        :param vp : new velocity in km/s
        """
        self._vp = vp

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            initialize_function(self.m, 1 / (self.vp * self.vp), self.nbpml)
        else:
            self.m.data = 1 / vp**2


class ModelPoroelastic(GenericModel):
    """The physical model used in seismic inversion processes.
    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param space_order: Order of the spatial stencil discretisation
    :param rhos: Solid grain density, kg / m**3
    :param rhof: Fluid density, kg / m**3
    :param kdr: Drained bulk modulus, Pa [ kg / (m * s**2) ]
    :param ksg: Solid grain bulk modulus, Pa [ kg / (m * s**2) ]
    :param kfl: Fluid bulk modulus, [ kg / (m * s**2) ]
    :param fvs: Fluid viscosity, Pa*s [ kg / (m * s)]
    :param phi: Porosity, %
    :param prm: Permeability, m**2
    :param shm: Shear modulus
    :param T: Tortuosity, -
    
    
    
    # Original Elastic Units Below
    :param vp: P-wave velocity in km/s
    :param vs: S-wave velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param rho: Density in kg/cm^3 (rho=1 for water)
    The :class:`ModelElastic` provides a symbolic data objects for the
    creation of seismic wave propagation operators:
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, shape, space_order, rhos, rhof, kdr, ksg,
                 kfl, fvs, phi, prm, shm, T, nbpml=20, dtype=np.float32):
        super(ModelPoroelastic, self).__init__(origin, spacing, shape, space_order,
                                           nbpml=nbpml, dtype=dtype)

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        initialize_damp(self.damp, self.nbpml, self.spacing, mask=True)

        # Create solid grain density
        if isinstance(rhos, np.ndarray):
            self.rhos = Function(name="rhos", grid=self.grid, space_order=space_order)
            initialize_function(self.rhos, rhos, self.nbpml)
        else:
            self.rhos = Constant(name="rhos", value=rhos)
        self._physical_parameters = ('rhos',)

        # Create fluid density
        if isinstance(rhof, np.ndarray):
            self.rhof = Function(name="rhof", grid=self.grid, space_order=space_order)
            initialize_function(self.rhof, rhof, self.nbpml)
        else:
            self.rhof = Constant(name="rhof", value=rhof)
        self._physical_parameters = ('rhof',)

        # Create drained bulk modulus
        if isinstance(kdr, np.ndarray):
            self.kdr = Function(name="kdr", grid=self.grid, space_order=space_order)
            initialize_function(self.kdr, kdr, self.nbpml)
        else:
            self.kdr = Constant(name="kdr", value=kdr)
        self._physical_parameters = ('kdr',)

        # Create solid grain bulk modulus
        if isinstance(ksg, np.ndarray):
            self.ksg = Function(name="ksg", grid=self.grid, space_order=space_order)
            initialize_function(self.ksg, ksg, self.nbpml)
        else:
            self.ksg = Constant(name="ksg", value=ksg)
        self._physical_parameters = ('ksg',)

        # Create fluid bulk modulus
        if isinstance(kfl, np.ndarray):
            self.kfl = Function(name="kfl", grid=self.grid, space_order=space_order)
            initialize_function(self.kfl, kfl, self.nbpml)
        else:
            self.kfl = Constant(name="kfl", value=kfl)
        self._physical_parameters = ('kfl',)

        # Create fluid viscosity
        if isinstance(fvs, np.ndarray):
            self.fvs = Function(name="fvs", grid=self.grid, space_order=space_order)
            initialize_function(self.fvs, fvs, self.nbpml)
        else:
            self.fvs = Constant(name="fvs", value=fvs)
        self._physical_parameters = ('fvs',)

        # Create porosity
        if isinstance(phi, np.ndarray):
            self.phi = Function(name="phi", grid=self.grid, space_order=space_order)
            initialize_function(self.phi, phi, self.nbpml)
        else:
            self.phi = Constant(name="phi", value=phi)
        self._physical_parameters = ('phi',)

        # Create permeability
        if isinstance(prm, np.ndarray):
            self.prm = Function(name="prm", grid=self.grid, space_order=space_order)
            initialize_function(self.prm, prm, self.nbpml)
        else:
            self.prm = Constant(name="prm", value=prm)
        self._physical_parameters = ('prm',)

        # Create shear modulus
        if isinstance(shm, np.ndarray):
            self.shm = Function(name="shm", grid=self.grid, space_order=space_order)
            initialize_function(self.shm, shm, self.nbpml)
        else:
            self.shm = Constant(name="shm", value=shm)
        self._physical_parameters = ('shm',)

        # Create formation tortuosity
        if isinstance(T, np.ndarray):
            self.T = Function(name="T", grid=self.grid, space_order=space_order)
            initialize_function(self.T, T, self.nbpml)
        else:
            self.T = Constant(name="T", value=T)
        self._physical_parameters = ('T',)
        
        # Generate derived values
        
        # Biot Coefficient, -
        self.alpha = Function(name="alpha", grid=self.grid, space_order=space_order)
        initialize_function(self.alpha, 1.0 - self.kdr.data/self.ksg.data, self.nbpml)
        self._physical_parameters += ('alpha',)

        # Biot Modulus
        self.M = Function(name="M", grid=self.grid, space_order=space_order)
        initialize_function(self.M, 1.0/ ((self.alpha.data - self.phi.data)/self.ksg.data + self.phi.data/self.kfl.data), self.nbpml)
        self._physical_parameters += ('M',)
        
        # Undrained Bulk Modulus
        self.K_u = Function(name="K_u", grid=self.grid, space_order=space_order)
        initialize_function(self.K_u, self.kdr.data + self.alpha.data**2 * self.M.data, self.nbpml)
        self._physical_parameters += ('K_u',)

        # Bulk Density
        self.rhob = Function(name="rhob", grid=self.grid, space_order=space_order)
        initialize_function(self.rhob, self.rhos.data  * (1.0-self.phi.data) + self.rhof.data * self.phi.data, self.nbpml)
        self._physical_parameters += ('rhob',)

        # P Wave Velocity
        self.Vp = Function(name="Vp", grid=self.grid, space_order=space_order)
        initialize_function(self.Vp, ( (self.K_u.data + 4.0*self.shm.data/3.0) / self.rhob.data )**0.5, self.nbpml)
        self._physical_parameters += ('Vp',)      

        # S Wave Velocity
        self.Vs = Function(name="Vs", grid=self.grid, space_order=space_order)
        initialize_function(self.Vs, ( self.shm.data / self.rhob.data)**0.5, self.nbpml)
        self._physical_parameters += ('Vs',)                

        # Lame Parameter of Saturated Medium,     # Pa        
        self.l_u = Function(name="l_u", grid=self.grid, space_order=space_order)
        initialize_function(self.l_u, self.K_u.data - 2.0/3.0 * self.shm.data, self.nbpml)
        self._physical_parameters += ('l_u',)
        

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        # 
        # Assuming fourth order differencing weights
        # The CFL condtion is then given by
        # dt < h / (sqrt(2) * max(vp)))
        return self.dtype(.5*np.min(self.spacing) / (np.sqrt(2)*(9./8. + 1.0/24.)*np.max(self.Vp.data)))
        
        
        
