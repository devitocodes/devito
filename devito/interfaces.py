import numpy as np
from sympy import IndexedBase


class MatrixData(IndexedBase):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.pointer = None
        self.initializer = None

    def set_initializer(self, lambda_initializer):
        assert(callable(lambda_initializer))
        self.initializer = lambda_initializer

    def initialize(self):
        assert(self.initializer is not None)
        self.initializer(self.data)

    def _allocate_memory(self):
        self.pointer = np.zeros(self.shape, self.dtype, order='C')

    @property
    def data(self):
        if self.pointer is None:
            self._allocate_memory()
        return self.pointer


class TimeData(MatrixData):
    def __init__(self, name, spc_shape, time_dim, time_order, save, dtype):
        if save:
            time_dim = time_dim + time_order
        else:
            time_dim = time_order + 1
        shape = tuple((time_dim,) + spc_shape)
        super(TimeData, self).__init__(name, shape, dtype)
        self.save = save
        self.time_order = time_order

    def _allocate_memory(self):
        MatrixData._allocate_memory(self)
        if self.pad_time:
            self.pointer = self.pointer[self.time_order]


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
