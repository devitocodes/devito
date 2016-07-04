import numpy as np
from sympy import IndexedBase
from tools import aligned


__all__ = ['DenseData', 'TimeData', 'PointData']


class DenseData(IndexedBase):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.dtype = dtype
        self.pointer = None
        self.initializer = None
        super(DenseData, self).__init__(name)

    def __new__(cls, *args, **kwargs):
        return IndexedBase.__new__(cls, args[0], shape=args[1])

    def set_initializer(self, lambda_initializer):
        assert(callable(lambda_initializer))
        self.initializer = lambda_initializer

    def _allocate_memory(self):
        self.pointer = aligned(np.zeros(self.shape, self.dtype, order='C'), alignment=64)

    @property
    def data(self):
        if self.pointer is None:
            self._allocate_memory()
        return self.pointer

    def initialize(self):
        if self.initializer is not None:
            self.initializer(self.data)
        # Ignore if no initializer exists - assume no initialisation necessary


class TimeData(DenseData):
    # The code here is getting increasingly messy because python wants two types
    # of constructors for everything. Since the parent class is Immutable, some
    # constructor work needs to be moved into the __new__ method while some is in
    # __init__. This makes it important to override both __new__ and __init__ in
    # every child class.
    def __init__(self, name, spc_shape, time_dim, time_order, save, dtype, pad_time=False):
        if save:
            time_dim = time_dim + time_order
        else:
            time_dim = time_order + 1
        shape = tuple((time_dim,) + spc_shape)
        super(TimeData, self).__init__(name, shape, dtype)
        self.save = save
        self.time_order = time_order
        self.pad_time = pad_time

    def _allocate_memory(self):
        super(TimeData, self)._allocate_memory()
        if self.pad_time is True:
            self.pointer = self.pointer[self.time_order:, :, :]

    def __new__(cls, name, spc_shape, time_dim, time_order, save, dtype, pad_time=False):
        if save:
            time_dim = time_dim + time_order
        else:
            time_dim = time_order + 1
        shape = tuple((time_dim,) + spc_shape)
        return IndexedBase.__new__(cls, name, shape=shape)


class PointData(DenseData):
    """This class is expected to eventually evolve into a full-fledged
    sparse data container. For now, the naming follows the use in the
    current problem.
    """
    def __init__(self, name, npoints, nt, dtype):
        self.npoints = npoints
        self.nt = nt
        super(PointData, self).__init__(name, (nt, npoints), dtype)

    def __new__(cls, name, npoints, nt, *args):
        return IndexedBase.__new__(cls, name, shape=(nt, npoints))
