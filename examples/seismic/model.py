import numpy as np
import os

from devito import Grid, Function, Constant
from devito.logger import error, warning


__all__ = ['Model', 'demo_model']


def demo_model(preset, **kwargs):
    """
    Utility function to create preset :class:`Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * `constant-isotropic` : Constant velocity (1.5km/sec) isotropic model
    * `constant-tti` : Constant anisotropic model. Velocity is 1.5 km/sec and
                      Thomsen parameters are epsilon=.3, delta=.2, theta = .7rad
                      and phi=.35rad for 3D. 2d/3d is defined from the input shape
    * 'layers-isotropic': Simple two-layer model with velocities 1.5 km/s
                 and 2.5 km/s in the top and bottom layer respectively.
                 2d/3d is defined from the input shape
    * 'layers-tti': Simple two-layer TTI model with velocities 1.5 km/s
                    and 2.5 km/s in the top and bottom layer respectively.
                    Thomsen parameters in the top layer are 0 and in the lower layer
                    are epsilon=.3, delta=.2, theta = .5rad and phi=.1 rad for 3D.
                    2d/3d is defined from the input shape
    * 'circle-isotropic': Simple camembert model with velocities 1.5 km/s
                 and 2.5 km/s in a circle at the center. 2D only.
    * 'marmousi2d-isotropic': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    * 'marmousi2d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    * 'marmousi3d-tti': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    """
    space_order = kwargs.pop('space_order', 2)

    if preset.lower() in ['constant-isotropic']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        vp = kwargs.pop('vp', 1.5)

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['constant-tti']:
        # A constant single-layer model in a 2D or 3D domain
        # with velocity 1.5km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        dtype = kwargs.pop('dtype', np.float32)
        v = np.empty(shape, dtype=dtype)
        v[:] = 1.5
        epsilon = .3*np.ones(shape, dtype=dtype)
        delta = .2*np.ones(shape, dtype=dtype)
        theta = .7*np.ones(shape, dtype=dtype)
        phi = None
        if len(shape) > 2:
            phi = .35*np.ones(shape, dtype=dtype)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    elif preset.lower() in ['layers-isotropic', 'twolayer-isotropic',
                            '2layer-isotropic']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, **kwargs)

    elif preset.lower() in ['layers-tti', 'twolayer-tti', '2layer-tti']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        epsilon = .3*(v - 1.5)
        delta = .2*(v - 1.5)
        theta = .5*(v - 1.5)
        phi = None
        if len(shape) > 2:
            phi = .1*(v - 1.5)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    elif preset.lower() in ['layers-tti-noazimuth', 'twolayer-tti-noazimuth',
                            '2layer-tti-noazimuth']:
        # A two-layer model in a 2D or 3D domain with two different
        # velocities split across the height dimension:
        # By default, the top part of the domain has 1.5 km/s,
        # and the bottom part of the domain has 2.5 km/s.\
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        dtype = kwargs.pop('dtype', np.float32)
        nbpml = kwargs.pop('nbpml', 10)
        ratio = kwargs.pop('ratio', 2)
        vp_top = kwargs.pop('vp_top', 1.5)
        vp_bottom = kwargs.pop('vp_bottom', 2.5)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=dtype)
        v[:] = vp_top  # Top velocity (background)
        v[..., int(shape[-1] / ratio):] = vp_bottom  # Bottom velocity

        epsilon = .3*(v - 1.5)
        delta = .2*(v - 1.5)
        theta = .5*(v - 1.5)

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, **kwargs)

    elif preset.lower() in ['circle-isotropic']:
        # A simple circle in a 2D domain with a background velocity.
        # By default, the circle velocity is 2.5 km/s,
        # and the background veloity is 3.0 km/s.
        dtype = kwargs.pop('dtype', np.float32)
        shape = kwargs.pop('shape', (101, 101))
        spacing = kwargs.pop('spacing', tuple([10. for _ in shape]))
        origin = kwargs.pop('origin', tuple([0. for _ in shape]))
        nbpml = kwargs.pop('nbpml', 10)
        vp = kwargs.pop('vp', 3.0)
        vp_background = kwargs.pop('vp_background', 2.5)
        r = kwargs.pop('r', 15)

        # Only a 2D preset is available currently
        assert(len(shape) == 2)

        v = np.empty(shape, dtype=dtype)
        v[:] = vp_background

        a, b = shape[0] / 2, shape[1] / 2
        y, x = np.ogrid[-a:shape[0]-a, -b:shape[1]-b]
        v[x*x + y*y <= r*r] = vp

        return Model(space_order=space_order, vp=v, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml)

    elif preset.lower() in ['marmousi-isotropic', 'marmousi2d-isotropic']:
        shape = (1601, 401)
        spacing = (7.5, 7.5)
        origin = (0., 0.)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.get('data_path', None)
        if data_path is None:
            error("Path to opesci/data not found! Please specify with "
                  "'data_path=<path/to/opesci/data>'")
            raise ValueError("Path to model data unspecified")
        path = os.path.join(data_path, 'Simple2D/vp_marmousi_bi')
        v = np.fromfile(path, dtype='float32', sep="")
        v = v.reshape(shape)

        # Cut the model to make it slightly cheaper
        v = v[301:-300, :]

        return Model(space_order=space_order, vp=v, origin=origin, shape=v.shape,
                     dtype=np.float32, spacing=spacing, nbpml=20)

    elif preset.lower() in ['marmousi-tti2d', 'marmousi2d-tti']:

        shape_full = (201, 201, 70)
        shape = (201, 70)
        spacing = (10., 10.)
        origin = (0., 0.)
        nbpml = kwargs.pop('nbpml', 20)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            error("Path to opesci/data not found! Please specify with "
                  "'data_path=<path/to/opesci/data>'")
            raise ValueError("Path to model data unspecified")
        path = os.path.join(data_path, 'marmousi3D/vp_marmousi_bi')

        # velocity
        vp = 1e-3 * np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiVP.raw'),
                                dtype='float32', sep="")
        vp = vp.reshape(shape_full)
        vp = vp[101, :, :]
        # Epsilon, in % in file, resale between 0 and 1
        epsilon = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiEps.raw'),
                              dtype='float32', sep="") * 1e-2
        epsilon = epsilon.reshape(shape_full)
        epsilon = epsilon[101, :, :]
        # Delta, in % in file, resale between 0 and 1
        delta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiDelta.raw'),
                            dtype='float32', sep="") * 1e-2
        delta = delta.reshape(shape_full)
        delta = delta[101, :, :]
        # Theta, in degrees in file, resale in radian
        theta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiTilt.raw'),
                            dtype='float32', sep="")
        theta = np.float32(np.pi / 180 * theta.reshape(shape_full))
        theta = theta[101, :, :]

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, **kwargs)

    elif preset.lower() in ['marmousi-tti3d', 'marmousi3d-tti']:
        shape = (201, 201, 70)
        spacing = (10., 10., 10.)
        origin = (0., 0., 0.)
        nbpml = kwargs.pop('nbpml', 20)

        # Read 2D Marmousi model from opesc/data repo
        data_path = kwargs.pop('data_path', None)
        if data_path is None:
            error("Path to opesci/data not found! Please specify with "
                  "'data_path=<path/to/opesci/data>'")
            raise ValueError("Path to model data unspecified")
        path = os.path.join(data_path, 'marmousi3D/vp_marmousi_bi')

        # Velcoity
        vp = 1e-3 * np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiVP.raw'),
                                dtype='float32', sep="")
        vp = vp.reshape(shape)
        # Epsilon, in % in file, resale between 0 and 1
        epsilon = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiEps.raw'),
                              dtype='float32', sep="") * 1e-2
        epsilon = epsilon.reshape(shape)
        # Delta, in % in file, resale between 0 and 1
        delta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiDelta.raw'),
                            dtype='float32', sep="") * 1e-2
        delta = delta.reshape(shape)
        # Theta, in degrees in file, resale in radian
        theta = np.fromfile(os.path.join(data_path, 'marmousi3D/MarmousiTilt.raw'),
                            dtype='float32', sep="")
        theta = np.float32(np.pi / 180 * theta.reshape(shape))
        # Phi, in degrees in file, resale in radian
        phi = np.fromfile(os.path.join(data_path, 'marmousi3D/Azimuth.raw'),
                          dtype='float32', sep="")
        phi = np.float32(np.pi / 180 * phi.reshape(shape))

        return Model(space_order=space_order, vp=vp, origin=origin, shape=shape,
                     dtype=np.float32, spacing=spacing, nbpml=nbpml, epsilon=epsilon,
                     delta=delta, theta=theta, phi=phi, **kwargs)

    else:
        error('Unknown model preset name %s' % preset)


def damp_boundary(damp, nbpml, spacing):
    """Initialise damping field with an absorbing PML layer.

    :param damp: The :class:`Function` for the damping field.
    :param nbpml: Number of points in the damping layer.
    :param spacing: Grid spacing coefficent.
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)
    assert all(damp._offset_domain[0] == i for i in damp._offset_domain)
    for i in range(damp.ndim):
        for j in range(nbpml):
            # Dampening coefficient
            pos = np.abs((nbpml - j + 1) / float(nbpml))
            val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
            # : slices
            all_ind = [slice(0, d) for d in damp.data.shape]
            # Left slice for dampening for dimension i
            all_ind[i] = slice(j, j+1)
            damp.data[all_ind] += val/spacing[i]
            # right slice for dampening for dimension i
            all_ind[i] = slice(damp.data.shape[i]-j, damp.data.shape[i]-j+1)
            damp.data[all_ind] += val/spacing[i]


def initialize_function(function, data, nbpml):
    """Initialize a :class:`Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this method.

    :param function: The :class:`Function` to be initialised with some data.
    :param data: The data array used for initialisation.
    :param nbpml: Number of PML layers for boundary damping.
    """
    pad_list = [(nbpml + i.left, nbpml + i.right) for i in function._offset_domain]
    function.data_with_halo[:] = np.pad(data, pad_list, 'edge')


class Model(object):
    """The physical model used in seismic inversion processes.

    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param space_order: Order of the spatial stencil discretisation
    :param vp: Velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param rho: Density in kg/cm^3 (rho=1 for water)
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
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = tuple([dtype(o) for o in origin])

        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml,
                         origin=origin, dtype=dtype)

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
        damp_boundary(self.damp, self.nbpml, spacing=self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                self._physical_parameters += ('epsilon',)
                self.epsilon = Function(name="epsilon", grid=self.grid)
                initialize_function(self.epsilon, 1 + 2 * epsilon, self.nbpml)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if np.max(self.epsilon.data_with_halo) > 0:
                    self.scale = np.sqrt(np.max(self.epsilon.data_with_halo))
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
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

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
