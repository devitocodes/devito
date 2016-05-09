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
