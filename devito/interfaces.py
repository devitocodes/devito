import atexit
import os
import sys
from signal import SIGABRT, SIGINT, SIGSEGV, SIGTERM, signal
from tempfile import gettempdir

import numpy as np
from sympy import Function, IndexedBase, as_finite_diff
from sympy.abc import h, s, t, x, y, z

from devito.finite_difference import cross_derivative
from tools import aligned

__all__ = ['DenseData', 'TimeData', 'PointData']


# This cache stores a reference to each created data object
# so that we may re-create equivalent symbols during symbolic
# manipulation with the correct shapes, pointers, etc.
_SymbolCache = {}


class SymbolicData(Function):
    """Base class for data classes that provides symbolic behaviour.

    :param name: Symbolic name to give to the resulting function. Must
                 be given as keyword argument.
    :param shape: Shape of the underlying spatial data. Must be given
                  as keyword argument.

    This class implements the symbolic behaviour of Devito's data
    objects by inheriting from and mimicking the behaviour of :class
    sympy.Function:. In order to maintain meta information across the
    numerous re-instantiation SymPy performs during symbolic
    manipulation, we inject the symbol name as the class name and
    cache all created objects on that name. This entails that data
    object should implement `__init__` in the following format:

    def __init__(self, \*args, \*\*kwargs):
        if self._cached():
            SymbolicData.__init__(self)
            return
        else:
            ... # Initialise object properties from kwargs

            self._cache_put(self)

    Note: The parameters :param name: and :param shape: must always be
    present and given as keyword arguments, since SymPy uses `*args`
    to (re-)create the dimension arguments of the symbolic function.
    """

    def __new__(cls, *args, **kwargs):
        if cls not in _SymbolCache:
            name = kwargs.get('name')
            shape = kwargs.get('shape')

            if len(args) < 1:
                args = cls.indices(shape)

            # Create a new type instance from cls and inject name
            newcls = type(name, (cls, ), dict(cls.__dict__))

            # Create the new Function object and invoke __init__
            newobj = Function.__new__(newcls, *args)
            newobj.__init__(*args, **kwargs)

            return newobj

        return Function.__new__(cls, *args)

    def __init__(self):
        """Initialise from a cached instance by shallow copying __dict__."""
        original = _SymbolCache[self.__class__]
        self.__dict__ = original.__dict__.copy()

    @classmethod
    def _cached(cls):
        """Test if current class is already in the symbol cache."""
        return cls in _SymbolCache

    @classmethod
    def _cache_put(cls, obj):
        """Store given object instance in symbol cache.

        :param obj: Object to be cached.
        """
        _SymbolCache[cls] = obj

    @classmethod
    def indices(cls, shape):
        """Abstract class method to determine the default dimension indices.

        :param shape: Given shape of the data.
        :raises NotImplementedError: 'Abstract class `SymbolicData` does not have default indices'.
        """
        raise NotImplementedError('Abstract class `SymbolicData` does not have default indices')


class DenseData(SymbolicData):
    """Data object for spatially varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param shape: Shape of the spatial data grid
    :param dtype: Data type of the buffered data
    :param space_order: Discretisation order for space derivatives

    Note: :class:`DenseData` objects are assumed to be constant in time and
    therefore do not support time derivatives. Use :class:`TimeData` for
    time-varying griad data.
    """
    def __init__(self, *args, **kwargs):
        if self._cached():
            # Initialise instance from symbol cache
            SymbolicData.__init__(self)
            return
        else:
            self.name = kwargs.get('name')
            self.shape = kwargs.get('shape')
            self.dtype = kwargs.get('dtype', np.float32)
            self.space_order = kwargs.get('space_order', 1)
            self._data = kwargs.get('_data', None)
            self.initializer = None
            MemmapManager.setup(self, *args, **kwargs)
            # Store new instance in symbol cache
            self._cache_put(self)

    @classmethod
    def indices(cls, shape):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: Indices used for axis.
        """
        _indices = [x, y, z]

        return _indices[:len(shape)]

    @property
    def dim(self):
        """Returns the spatial dimension of the data object"""
        return len(self.shape)

    @property
    def indexed(self):
        """:return: Base symbol as sympy.IndexedBase"""
        return IndexedBase(self.name, shape=self.shape)

    def indexify(self):
        """Convert base symbol and dimensions to indexed data accesses

        :return: Index corrosponding to the indices
        """
        indices = [a.subs({h: 1, s: 1}) for a in self.args]

        return self.indexed[indices]

    def set_initializer(self, lambda_initializer):
        """Set data intialising function to given lambda function.

        :param lambda_initializer: Given lambda function.
        """
        assert(callable(lambda_initializer))

        self.initializer = lambda_initializer

    def _allocate_memory(self):
        """Function to allocate memmory in terms of numpy ndarrays.

        Note: memmap is a subclass of ndarray.
        """
        if self.memmap:
            self._data = np.memmap(filename=self.f, dtype=self.dtype, mode='w+', shape=self.shape, order='C')
        else:
            self._data = aligned(np.zeros(self.shape, self.dtype, order='C'), alignment=64)

    @property
    def data(self):
        """Reference to the :class:`numpy.ndarray` containing the data

        :returns: The ndarray containing the data
        """
        if self._data is None:
            self._allocate_memory()

        return self._data

    def initialize(self):
        """Apply the data initilisation function, if it is not None."""
        if self.initializer is not None:
            self.initializer(self.data)
        # Ignore if no initializer exists - assume no initialisation necessary

    @property
    def dx2(self):
        """Symbol for the second derivative wrt the x dimension"""
        width_h = int(self.space_order/2)
        indx = [(x + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(x, x), indx)

    @property
    def dy2(self):
        """Symbol for the second derivative wrt the y dimension"""
        width_h = int(self.space_order/2)
        indy = [(y + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(y, y), indy)

    @property
    def dz2(self):
        """Symbol for the second derivative wrt the z dimension"""
        width_h = int(self.space_order/2)
        indz = [(z + i * h) for i in range(-width_h, width_h + 1)]

        return as_finite_diff(self.diff(z, z), indz)

    @property
    def laplace(self):
        """Symbol for the second derivative wrt all spatial dimenions"""
        derivs = ['dx2', 'dy2', 'dz2']

        return sum([getattr(self, d) for d in derivs[:self.dim]])

    @property
    def dxy(self):
        """Symbol for the cross derivative wrt the x and y dimension"""
        return cross_derivative(self, order=int(self.space_order/2), dims=(x, y))

    @property
    def dxz(self):
        """Symbol for the cross derivative wrt the x and z dimension"""
        return cross_derivative(self, order=int(self.space_order/2), dims=(x, z))

    @property
    def dyz(self):
        """Symbol for the cross derivative wrt the y and z dimension"""
        return cross_derivative(self, order=int(self.space_order/2), dims=(y, z))


class TimeData(DenseData):
    """Data object for time-varying data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param shape: Shape of the spatial data grid
    :param dtype: Data type of the buffered data
    :param save: Save the intermediate results to the data buffer. Defaults
                 to `False`, indicating the use of alternating buffers.
    :param time_dim: Size of the time dimension that dictates the leading
                     dimension of the data buffer if :param save: is True.
    :param time_order: Order of the time discretization which affects the
                       final size of the leading time dimension of the
                       data buffer.

    Note: The parameter :shape: should only define the spatial shape of the
    grid. The temporal dimension will be inserted automatically as the
    leading dimension, according to the :param time_dim:, :param time_order:
    and whether we want to write intermediate timesteps in the buffer.
    """

    def __init__(self, *args, **kwargs):
        if self._cached():
            # Initialise instance from symbol cache
            SymbolicData.__init__(self)
            return
        else:
            super(TimeData, self).__init__(*args, **kwargs)
            self._full_data = self._data.view() if self._data else None
            time_dim = kwargs.get('time_dim')
            self.time_order = kwargs.get('time_order', 1)
            self.save = kwargs.get('save', False)
            self.pad_time = kwargs.get('pad_time', False)

            if self.save:
                time_dim += self.time_order
            else:
                time_dim = self.time_order + 1

            self.shape = (time_dim,) + self.shape

            # Store final instance in symbol cache
            self._cache_put(self)

    @classmethod
    def indices(cls, shape):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: Indices used for axis.
        """
        _indices = [t, x, y, z]

        return _indices[:len(shape) + 1]

    def _allocate_memory(self):
        """function to allocate memmory in terms of numpy ndarrays."""
        super(TimeData, self)._allocate_memory()

        self._full_data = self._data.view()

        if self.pad_time:
            self._data = self._data[self.time_order:, :, :]

    def init_data(self, timestep, data):
        """Function to initialize the initial time steps

        :param timestep: Time step to initialize. Must be negative since calculated timesteps start from 0.
        :param data: :class:`numpy.ndarray` containing the initial spatial data
        """
        if self._full_data is None:
            self._allocate_memory()

        assert timestep < 0, "Timestep must be negative"
        assert data.shape == self._full_data[0].shape, "Data must have the same shape as the spatial data"

        # Adds the time_order to the index to access padded indexes
        timestep += self.time_order
        self._full_data[timestep] = data

    def get_data(self, timestep=0):
        """Returns the calculated data at the specified timestep

        :param timestep: The timestep from which we want to retrieve the data.
                         Specify only in the case :obj:`self.save` is True
        """
        timestep += self.time_order

        return self._full_data[timestep, :]

    @property
    def dim(self):
        """Returns the spatial dimension of the data object"""
        return len(self.shape[1:])

    @property
    def forward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1

        return self.subs(t, t + i * s)

    @property
    def backward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1

        return self.subs(t, t - i * s)

    @property
    def dt(self):
        """Symbol for the first derivative wrt the time dimension"""
        if self.time_order == 1:
            # This hack is needed for the first-order diffusion test
            indices = [t, t + s]
        else:
            width = int(self.time_order / 2)
            indices = [(t + i * s) for i in range(-width, width + 1)]

        return as_finite_diff(self.diff(t), indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        width_t = int(self.time_order/2)
        indt = [(t + i * s) for i in range(-width_t, width_t + 1)]

        return as_finite_diff(self.diff(t, t), indt)


class PointData(DenseData):
    """Data object for sparse point data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param point: Number of points to sample
    :param nt: Size of the time dimension for point data
    :param dtype: Data type of the buffered data

    Note: This class is expected to eventually evolve into a
    full-fledged sparse data container. For now, the naming and
    symbolic behaviour follows the use in the current problem.
    """

    def __init__(self, *args, **kwargs):
        if self._cached():
            # Initialise instance from symbol cache
            SymbolicData.__init__(self)
            return
        else:
            self.nt = kwargs.get('nt')
            self.npoint = kwargs.get('npoint')
            kwargs['shape'] = (self.nt, self.npoint)
            DenseData.__init__(self, *args, **kwargs)
            # Store final instance in symbol cache
            self._cache_put(self)

    def __new__(cls, *args, **kwargs):
        nt = kwargs.get('nt')
        npoint = kwargs.get('npoint')
        kwargs['shape'] = (nt, npoint)

        return DenseData.__new__(cls, *args, **kwargs)

    @classmethod
    def indices(cls, shape):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: indices used for axis.
        """
        _indices = [t, x, y, z]

        return _indices[:len(shape) + 1]


class MemmapManager():
    """Class for managing all memmap related settings"""
    # used to enable memmap as default
    _use_memmap = False
    # flag for registering exit func
    _registered = False
    _default_disk_path = os.path.join(gettempdir(), "devito_disk")
    # contains str name of all memmap file created
    _created_data = set()
    _default_exit_code = 0

    @staticmethod
    def set_memmap(memmap):
        """Call this method to set default value of memmap"""
        MemmapManager._use_memmap = memmap

    @staticmethod
    def set_default_disk_path(default_disk_path):
        """Call this method to change the default disk path for memmap"""
        MemmapManager._default_disk_path = default_disk_path

    @staticmethod
    def setup(data_self, *args, **kwargs):
        """This method is used to setup memmap parameters for data classes.

        :param name: Name of data, must be unique
        :param memmap: Boolean indicates whether memmap is used. Optional
        :param disk_path: String indicates path to create memmap file. Optional

        Note: If memmap or disk_path are not provided, the default values
        are used.
        """
        data_self.memmap = kwargs.get('memmap', MemmapManager._use_memmap)

        if data_self.memmap:
            disk_path = kwargs.get('disk_path', MemmapManager._default_disk_path)

            if not os.path.exists(disk_path):
                os.makedirs(disk_path)

            data_self.f = disk_path + "/data_" + kwargs.get('name')
            MemmapManager._created_data.add(data_self.f)

            if not MemmapManager._registered:
                MemmapManager._register_remove_memmap_file_signal()
                MemmapManager._registered = True

    @staticmethod
    def _reomve_memmap_file():
        """This method is used to clean up memmap file"""
        for f in MemmapManager._created_data:
            try:
                os.remove(f)
            except OSError:
                print("error removing " + f + " it may be already removed, skipping")
                pass

    @staticmethod
    def _remove_memmap_file_on_signal(*args):
        """This method is used to clean memmap file on signal, internal method"""
        sys.exit(MemmapManager._default_exit_code)

    @staticmethod
    def _register_remove_memmap_file_signal():
        """This method is used to register clean up method for chosen signals"""
        atexit.register(MemmapManager._reomve_memmap_file)

        for sig in (SIGABRT, SIGINT, SIGSEGV, SIGTERM):
            signal(sig, MemmapManager._remove_memmap_file_on_signal)
