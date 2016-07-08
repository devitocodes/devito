import numpy as np
from sympy import IndexedBase
from tools import aligned
from random import randint
from hashlib import sha1
import os

## managing memmap files
# list containing string path to memmap file created
memmap_file_list = []
# functons for managing memmap files
def remove_memmap_files():
    for f in memmap_file_list:
        try:
            os.remove(f)
            print("removed file: " + f)
        except OSError:
            print("error removing " + f + " file may no longer exists, skipping")
            pass

def get_memmap_file():
    return memmap_file_list
## managing memmap files

class DenseData(IndexedBase):

    # hash function used for memmap file name generation
    _hashing_function = sha1

    def __init__(self, name, shape, dtype, disk_path=None):
        self.name = name
        self.dtype = dtype
        self.pointer = None
        self.initializer = None
        # saving disk path
        self.disk_path = disk_path
        super(DenseData, self).__init__(name)

    def __new__(cls, *args, **kwargs):
        return IndexedBase.__new__(cls, args[0], shape=args[1])

    def set_initializer(self, lambda_initializer):
        assert(callable(lambda_initializer))
        self.initializer = lambda_initializer
    
    # function to generate a hased file name used for memmap file
    def _generate_memmap_filenmae(self):
        hash_string = self.name + str(randint(0, 100000000))
        return self._hashing_function(hash_string).hexdigest()
    
    # if self.disk_path is defined, memmap files will be used, otherwise ndarray is used
    def _allocate_memory(self):
        if self.disk_path == None:
             self.pointer = aligned(np.zeros(self.shape, self.dtype, order='C'), alignment=64)
        else:
            f = self.disk_path + "/data_" + self.name + "_" + self._generate_memmap_filenmae()
            self.pointer = aligned(np.memmap(filename=f, dtype=self.dtype, mode='w+', shape=tuple(self.shape), order='C'), alignment=64)
            memmap_file_list.append(f)
            print("memmap file written to: " + f)

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
    def __init__(self, name, spc_shape, time_dim, time_order, save, dtype, pad_time=False, disk_path=None):
        if save:
            time_dim = time_dim + time_order
        else:
            time_dim = time_order + 1
        shape = tuple((time_dim,) + spc_shape)
        # saving disk path
        self.disk_path = disk_path
        # propagate disk path to parent class
        super(TimeData, self).__init__(name, shape, dtype, self.disk_path)
        self.save = save
        self.time_order = time_order
        self.pad_time = pad_time

    def _allocate_memory(self):
        super(TimeData, self)._allocate_memory()
        if self.pad_time is True:
            self.pointer = self.pointer[self.time_order:, :, :]

    def __new__(cls, name, spc_shape, time_dim, time_order, save, dtype, pad_time=False, disk_path=None):
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
    def __init__(self, name, npoints, nt, dtype, disk_path=None):
        self.npoints = npoints
        self.nt = nt
        # saving disk path
        self.disk_path = disk_path
        # propagate disk path to parent class
        super(PointData, self).__init__(name, (nt, npoints), dtype, self.disk_path)

    def __new__(cls, name, npoints, nt, *args):
        return IndexedBase.__new__(cls, name, shape=(nt, npoints))
