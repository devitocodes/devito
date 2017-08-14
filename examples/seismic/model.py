import numpy as np

from devito import DenseData, ConstantData


__all__ = ['Model']


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
        self.vp = vp
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
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        return coeff * np.min(self.spacing) / (self.scale*np.max(self.vp))

    def set_vp(self, vp):
        """Set a new velocity model and update square slowness

        :param vp : new velocity in km/s
        """
        self.vp = vp
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
