import numpy as np


class Model(object):
    """The physical model as used in seismic inversion processes.

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
    def __init__(self, origin, spacing, vp, nbpml=0, rho=None,
                 epsilon=None, delta=None, theta=None, phi=None):
        self.vp = vp
        self.rho = rho
        self.origin = origin
        self.spacing = spacing
        self.nbpml = nbpml

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

    def get_spacing(self):
        """Return the grid size"""
        return self.spacing[0]

    def padm(self):
        """Padding function extending self.vp by `self.nbpml` in every direction
        for the absorbing boundary conditions"""
        return self.pad(1 / (self.vp * self.vp))

    def pad(self, m):
        """Padding function extending m by `self.nbpml` in every direction
        for the absorbing boundary conditions
        :param m : physical parameter to be extended"""
        pad_list = []
        for dim_index in range(len(self.vp.shape)):
            pad_list.append((self.nbpml, self.nbpml))
        return np.pad(m, pad_list, 'edge')
