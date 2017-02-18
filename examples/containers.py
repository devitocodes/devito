# coding: utf-8
import numpy as np


class IGrid:
    """
    Class to setup a physical model

    :param origin: Origin of the model in m as a Tuple
    :param spacing:grid size in m as a Tuple
    :param vp: Velocity in km/s
    :param rho: Density in kg/cm^3 (rho=1 for water)
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param: theta: Tilt angle in radian
    :param phi : Asymuth angle in radian
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

    def get_critical_dt(self):
        """ Return the computational time step value from the CFL condition"""
        # limit for infinite stencil of √(a1/a2) where a1 is the
        #  sum of absolute values of the time discretisation
        # and a2 is the sum of the absolute values of the space discretisation
        #
        # example, 2nd order in time and space in 2D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 2*(1+2+1)  = 8
        # coeff = √(1/2) = 0.7
        # example, 2nd order in time and space in 3D
        # a1 = 1 + 2 + 1 = 4
        # a2 = 3*(1+2+1)  = 12
        # coeff = √(1/3) = 0.57

        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        if len(self.vp.shape) == 3:
            coeff = 0.38
        else:
            coeff = 0.42
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


class ISource:
    """Source class, currently not implemented"""

    def __init__(self):
        raise NotImplementedError

    def get_source(self):
        """ List of size nt
        """
        raise NotImplementedError

    def get_corner(self):
        """ Tuple of (x, y) or (x, y, z)
        """
        return self._corner

    def get_weights(self):
        """ List of [w1, w2, w3, w4] or [w1, w2, w3, w4, w5, w6, w7, w8]
        """
        return self._weights


class IShot:
    """Class seting up the acquisition geometry"""
    def set_source(self, time_serie, dt, location):
        """Set the source signature"""
        self.source_sign = time_serie
        self.source_coords = location
        self.sample_interval = dt

    def set_receiver_pos(self, pos):
        """Set the receivers position"""
        self.receiver_coords = pos

    def set_shape(self, nt, nrec):
        """Set the data array shape"""
        self.shape = (nt, nrec)

    def set_traces(self, traces):
        """ Add traces data  """
        self.traces = traces

    def get_source(self, ti=None):
        """Return the source signature"""
        if ti is None:
            return self.source_sign

        return self.source_sign[ti]

    def get_nrec(self):
        """Return the snumber of receivers"""
        ntraces, nsamples = self.traces.shape

        return ntraces

    def reinterpolate(self, dt):
        raise NotImplementedError

    def __str__(self):
        return "Source: "+str(self.source_coords)+", Receiver:"+str(self.receiver_coords)
