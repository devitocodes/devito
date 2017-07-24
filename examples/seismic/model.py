import numpy as np

from devito import DenseData, ConstantData
from devito.logger import error


__all__ = ['Model', 'demo_model']


def demo_model(preset, **kwargs):
    """
    Utility function to create preset :class:`Model` objects for
    demonstration and testing purposes. The particular presets are ::

    * 'layer2D': Simple two-layer model with velocities 1.5 km/s
                 and 2.5 km/s in the top and bottom layer respectively.
    * 'marmousi2D': Loads the 2D Marmousi data set from the given
                    filepath. Requires the ``opesci/data`` repository
                    to be available on your machine.
    """
    if preset.lower() == 'layer2d':
        # 2D two-layer model with domain shape (101, 101),
        # grid spacing of 10m and the origin at (1., 0.).
        # The physical extent with defaults will be 1km x 1km.

        shape = kwargs.get('shape', (101, 101))
        spacing = kwargs.get('spacing', (10., 10.))
        origin = kwargs.get('origin', (0., 0.))
        nbpml = kwargs.get('nbpml', 20)

        # Define a velocity profile in km/s
        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        return Model(vp=v, origin=origin, shape=shape,
                     spacing=spacing, nbpml=nbpml)
    else:
        error('Unknown model preset name %s' % preset)


def damp_boundary(damp, nbpml, spacing):
    """Initialise damping field with an absorbing PML layer.

    :param damp: Array data defining the damping field
    :param nbpml: Number of points in the damping layer
    :param spacing: Grid spacing coefficent
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)
    ndim = len(damp.shape)
    for i in range(nbpml):
        pos = np.abs((nbpml - i + 1) / float(nbpml))
        val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
        if ndim == 2:
            damp[i, :] += val/spacing[0]
            damp[-(i + 1), :] += val/spacing[0]
            damp[:, i] += val/spacing[1]
            damp[:, -(i + 1)] += val/spacing[1]
        else:
            damp[i, :, :] += val/spacing[0]
            damp[-(i + 1), :, :] += val/spacing[0]
            damp[:, i, :] += val/spacing[1]
            damp[:, -(i + 1), :] += val/spacing[1]
            damp[:, :, i] += val/spacing[2]
            damp[:, :, -(i + 1)] += val/spacing[2]


class Model(object):
    """The physical model used in seismic inversion processes.

    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
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
    def __init__(self, origin, spacing, shape, vp, nbpml=0, dtype=np.float32,
                 epsilon=None, delta=None, theta=None, phi=None):
        self._vp = vp
        self.origin = origin
        self.spacing = spacing
        self.nbpml = nbpml
        self.dtype = dtype
        self.shape = shape
        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = DenseData(name="m", shape=self.shape_domain, dtype=self.dtype)
            self.m.data[:] = self.pad(1 / (self.vp * self.vp))
        else:
            self.m = ConstantData(name="m", value=1/vp**2, dtype=self.dtype)

        # Create dampening field as symbol `damp`
        self.damp = DenseData(name="damp", shape=self.shape_domain,
                              dtype=self.dtype)
        damp_boundary(self.damp.data, nbpml, spacing=self.get_spacing())

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                self.epsilon = DenseData(name="epsilon", shape=self.shape_domain,
                                         dtype=self.dtype)
                self.epsilon.data[:] = self.pad(1 + 2 * epsilon)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if np.max(self.epsilon.data) > 0:
                    self.scale = np.sqrt(np.max(self.epsilon.data))
            else:
                self.epsilon = 1 + 2 * epsilon
                self.scale = epsilon
        else:
            self.epsilon = 1

        if delta is not None:
            if isinstance(delta, np.ndarray):
                self.delta = DenseData(name="delta", shape=self.shape_domain,
                                       dtype=self.dtype)
                self.delta.data[:] = self.pad(np.sqrt(1 + 2 * delta))
            else:
                self.delta = delta
        else:
            self.delta = 1

        if theta is not None:
            if isinstance(theta, np.ndarray):
                self.theta = DenseData(name="theta", shape=self.shape_domain,
                                       dtype=self.dtype)
                self.theta.data[:] = self.pad(theta)
            else:
                self.theta = theta
        else:
            self.theta = 0

        if phi is not None:
            if isinstance(phi, np.ndarray):
                self.phi = DenseData(name="phi", shape=self.shape_domain,
                                     dtype=self.dtype)
                self.phi.data[:] = self.pad(phi)
            else:
                self.phi = phi
        else:
            self.phi = 0

    @property
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple(d * s for d, s in zip(self.shape, self.spacing))

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        return coeff * np.min(self.spacing) / (self.scale*np.max(self.vp))

    @property
    def vp(self):
        """:class:`numpy.ndarray` holding the model velocity in km/s.

        .. note::

        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type :class:`DenseData`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """Set a new velocity model and update square slowness

        :param vp : new velocity in km/s
        """
        self._vp = vp
        self.m.data[:] = self.pad(1 / (self.vp * self.vp))

    def get_spacing(self):
        """Return the grid size"""
        return self.spacing

    def pad(self, data):
        """Padding function PNL layers in every direction for for the
        absorbing boundary conditions.

        :param data : Data array to be padded"""
        pad_list = [(self.nbpml, self.nbpml) for _ in self.shape]
        return np.pad(data, pad_list, 'edge')
