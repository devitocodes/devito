import numpy as np
from sympy import IndexedBase
from tools import aligned
from hashlib import sha1
from random import randint
import os


class DenseData(IndexedBase):

    def __init__(self, name, shape, dtype, disk_path=None):
        self.name = name
        self.dtype = dtype
        self.pointer = None
        self.initializer = None
        self.disk_path = disk_path
        super(DenseData, self).__init__(name)

    def __new__(cls, *args, **kwargs):
        return IndexedBase.__new__(cls, args[0], shape=args[1])

    # hash function used for generating part of memmap file name
    _hashing_function = sha1
    # this is the directory where memmap files will be created
    _default_disk_path = None
    # holds the string name of all memmap file created
    _memmap_file_list = []

    ## functions for managing memmap files
    # call this method to specify where you want memmap file to be created
    @staticmethod
    def set_default_disk_path(disk_path):
        if not os.path.exists(disk_path):
            os.makedirs(disk_path)
        DenseData._default_disk_path = disk_path
        print("default disk path set to: " + DenseData._default_disk_path)

    @staticmethod
    def get_memmap_file_list():
        return DenseData._memmap_file_list

    @staticmethod
    def remove_memmap_file():
        for f in DenseData._memmap_file_list:
            try:
                os.remove(f)
            except OSError:
                print("error removing " + f + " skipping")
                pass
    ##end of functions for managing memmap files

    # function to allocate memory for this data, if _disk_path is not None, a numpy memmap is used
    # if not a numpy ndarray is used, by default _disk_path is None
    def _allocate_memory(self):
        """allocate memmory for this data. if either _defualt_disk_path or self.disk_path is not None,
           a numpy memmap is used, if not a numpy ndarray is used."""
        if DenseData._default_disk_path == None and self.disk_path == None:
            # not disk_path use ndarray
            self.pointer = aligned(np.zeros(self.shape, self.dtype, order='C'), alignment=64)
            return
        elif self.disk_path == None:
            # using defualt disk_path
            self.disk_path = DenseData._default_disk_path
        elif not os.path.exists(self.disk_path):
            # create disk path
            os.makedirs(self.disk_path)
        # allocate memory    
        f = self.disk_path + "/data_" + self.name
        self.pointer = aligned(np.memmap(filename=f, dtype=self.dtype, mode='w+', shape=tuple(self.shape), order='C'), alignment=64)
        DenseData._memmap_file_list.append(f)
        print("memmap file written to: " + f)

    def set_initializer(self, lambda_initializer):
        assert(callable(lambda_initializer))
        self.initializer = lambda_initializer

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
        super(TimeData, self).__init__(name, shape, dtype)
        self.save = save
        self.time_order = time_order
        self.pad_time = pad_time
        self.disk_path = disk_path

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
        super(PointData, self).__init__(name, (nt, npoints), dtype)
        self.disk_path = disk_path

    def __new__(cls, name, npoints, nt, *args, **kwargs):
        return IndexedBase.__new__(cls, name, shape=(nt, npoints))
