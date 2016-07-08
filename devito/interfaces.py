import numpy as np
from sympy import IndexedBase
from tools import aligned
from hashlib import sha1
from random import randint
import os


class DenseData(IndexedBase):

    # hash function used for generating part of memmap file name
    _hashing_function = sha1
    # this is the directory where memmap files will be created
    _disk_path = None
    # holds the string name of all memmap file created
    _memmap_file_list = []

    ## functions for managing memmap files
    # call this method to specify where you want memmap file to be created
    @staticmethod
    def set_disk_path(disk_path):
        if not os.path.exists(disk_path):
            print("the directory" + disk_path + " does not exit")
            os.makedirs(disk_path)
            print(disk_path + " is created")
        else:
            print("directory " + disk_path + " found")
        DenseData._disk_path = disk_path
    
    @staticmethod
    def get_memmap_file_list():
        return DenseData._memmap_file_list

    @staticmethod
    def remove_memmap_file():
        for f in DenseData._memmap_file_list:
            try:
                os.remove(f)
                print("removed file: " + f)
            except OSError:
                print("error removing " + f + " file may no longer exists, skipping")
                pass
    ##end of functions for managing memmap files

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

    # function to generate a hased file name used for memmap file
    def _generate_memmap_filenmae(self):
        hash_string = self.name + str(randint(0, 100000000))
        return self._hashing_function(hash_string).hexdigest()

    # function to allocate memory for this data, if _disk_path is not None, a numpy memmap is used
    # if not a numpy ndarray is used, by default _disk_path is None
    def _allocate_memory(self):
        self._disk_path = DenseData._disk_path
        if self._disk_path == None:
            self.pointer = aligned(np.zeros(self.shape, self.dtype, order='C'), alignment=64)
        else:
            f = self._disk_path + "/data_" + self.name + "_" + self._generate_memmap_filenmae()
            self.pointer = aligned(np.memmap(filename=f, dtype=self.dtype, mode='w+', shape=tuple(self.shape), order='C'), alignment=64)
            DenseData._memmap_file_list.append(f)
            print("memmap file written to: " + f)
            print("if the script call DenseData.remove_memmap_file() at the end, this file will be deleted automatically. if you have interupted this script, please go to " + self._disk_path + " to delete this file.")

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
