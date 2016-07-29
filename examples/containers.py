# coding: utf-8
import numpy as np


class IGrid:
    def get_shape(self):
        """Tuple of (x, y) or (x, y, z)
        """
        return self.vp.shape

    def get_critical_dt(self):
        # limit for infinite stencil of √(a1/a2) where a1 is the sum of absolute values of the time discretisation
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
        return coeff * self.spacing[0] / (2*np.max(self.vp))

    def get_spacing(self):
        return self.spacing[0]

    def create_model(self, origin, spacing, vp, epsilon=None, delta=None, theta=None, phi=None):
        self.vp = vp
        self.epsilon = epsilon
        self.delta = delta
        self.theta = theta
        self.phi = phi
        self.spacing = spacing
        self.origin = origin

    def set_origin(self, shift):
        norig = len(self.origin)
        aux = []
        for i in range(0, norig):
            aux.append(self.origin[i] - shift * self.spacing[i])
        self.origin = aux

    def get_origin(self):
        return self.origin


class ISource:
    def get_source(self):
        """ List of size nt
        """
        return self._source

    def get_corner(self):
        """ Tuple of (x, y) or (x, y, z)
        """
        return self._corner

    def get_weights(self):
        """ List of [w1, w2, w3, w4] or [w1, w2, w3, w4, w5, w6, w7, w8]
        """
        return self._weights


class IShot:
    def get_data(self):
        """ List of ISource objects, of size ntraces
        """
        return self._shots

    def set_source(self, time_serie, dt, location):
        self.source_sign = time_serie
        self.source_coords = location
        self.sample_interval = dt

    def set_receiver_pos(self, pos):
        self.receiver_coords = pos

    def set_shape(self, nt, nrec):
        self.traces = np.zeros((nrec, nt))

    def get_source(self, ti=None):
        if ti is None:
            return self.source_sign
        return self.source_sign[ti]

    def get_nrec(self):
        ntraces, nsamples = self.traces.shape
        return ntraces

    def reinterpolate(self, dt):
        pass

    def __str__(self):
        return "Source: "+str(self.source_coords)+", Receiver:"+str(self.receiver_coords)
