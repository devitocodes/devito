import numpy as np

from devito import DenseData


__all__ = ['Model']


def damp_boundary(damp, nbpml, spacing):
    """Initialise damping field with an absorbing PML layer.

    :param damp: Array data defining the damping field
    :param nbpml: Number of points in the damping layer
    :param spacing: Grid spacing coefficent
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * spacing)
    ndim = len(damp.shape)
    for i in range(nbpml):
        pos = np.abs((nbpml - i + 1) / float(nbpml))
        val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
        if ndim == 2:
            damp[i, :] += val
            damp[-(i + 1), :] += val
            damp[:, i] += val
            damp[:, -(i + 1)] += val
        else:
            damp[i, :, :] += val
            damp[-(i + 1), :, :] += val
            damp[:, i, :] += val
            damp[:, -(i + 1), :] += val
            damp[:, :, i] += val
            damp[:, :, -(i + 1)] += val


class Model(object):
    """The physical model used in seismic inversion processes.

    :param origin: Origin of the model in m as a tuple
    :param spacing: Grid size in m as a Tuple
    :param vp: Velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param rho: Density in kg/cm^3 (rho=1 for water)
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param theta: Tilt angle in radian
    :param phi : Asymuth angle in radian

    The :class:`Model` provides two symbolic data objects for the
    creation of seismic wve propagation operators:
    :param m: The square slowness of the wave
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, vp, nbpml=0, dtype=np.float32,
                 rho=None, epsilon=None, delta=None, theta=None, phi=None):
        self.vp = vp
        self.origin = origin
        self.spacing = spacing
        self.nbpml = nbpml
        self.dtype = dtype

        # Create square slowness of the wave as symbol `m`
        self.m = DenseData(name="m", shape=self.shape_pml, dtype=self.dtype)
        self.m.data[:] = self.pad(1 / (self.vp * self.vp))

        # Create dampening field as symbol `damp`
        self.damp = DenseData(name="damp", shape=self.shape_pml,
                              dtype=self.dtype)
        damp_boundary(self.damp.data, nbpml, spacing=self.get_spacing())

        # Additional parameter fields for TTI operators
        self.rho = rho
        if epsilon is not None:
            self.epsilon = 1 + 2 * epsilon
            # Maximum velocity is scale*max(vp) is epsilon>0
            self.scale = np.sqrt(np.max(self.epsilon)) if np.max(self.epsilon) > 0 else 1
        else:
            self.scale = 1
            self.epsilon = None

        if delta is not None:
            self.delta = np.sqrt(1 + 2 * delta)
        else:
            self.delta = None

        self.theta = theta
        self.phi = phi

    @property
    def shape(self):
        """Original shape of the model without PML layers"""
        return self.vp.shape

    @property
    def shape_pml(self):
        """Computational shape of the model with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        return coeff * self.spacing[0] / (self.scale*np.max(self.vp))

    def set_vp(self, vp):
        """Set a new velocity model and update square slowness

        :param vp : new velocity in km/s
        """
        self.vp = vp
        self.m.data[:] = self.pad(1 / (self.vp * self.vp))

    def get_spacing(self):
        """Return the grid size"""
        return self.spacing[0]

    def pad(self, data):
        """Padding function PNL layers in every direction for for the
        absorbing boundary conditions.

        :param data : Data array to be padded"""
        pad_list = [(self.nbpml, self.nbpml) for _ in self.vp.shape]
        return np.pad(data, pad_list, 'edge')
