import numpy as np


class IGrid:
    def get_shape(self):
        """Tuple of (x, y) or (x, y, z)
        """
        return self.vp.shape

    def get_critical_dt(self):
        return 0.5 * self.spacing[0] / (np.max(self.vp))
    
    def get_spacing(self):
        return self.spacing[0]

    def create_model(self, origin, spacing, vp):
        self.vp = vp
        self.spacing = spacing
        self.origin = origin
    
    def set_origin(self, origin):
        assert(1==2)
        self.origin = origin
    
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
        self.nt = nt
        self.nrec = nrec
        self.traces = np.zeros((nrec, nt))
    
    def get_source(self, ti = None):
        if ti is None:
            return self.source_sign
        return self.source_sign[ti]
    
    def get_nrec(self):
        return self.nrec