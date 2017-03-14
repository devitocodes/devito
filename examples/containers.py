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
