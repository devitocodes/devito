from collections import namedtuple
from ctypes import POINTER, Structure, c_int, c_ulong, c_void_p, cast, byref
from functools import wraps, reduce
from operator import mul

import numpy as np
import sympy
from functools import cached_property

from devito.builtins import assign
from devito.data import (DOMAIN, OWNED, HALO, NOPAD, FULL, LEFT, CENTER, RIGHT,
                         Data, default_allocator)
from devito.data.allocators import DataReference
from devito.deprecations import deprecations
from devito.exceptions import InvalidArgument
from devito.logger import debug, warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.symbolics import FieldFromPointer, normalize_args, IndexedPointer
from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.finite_differences.tools import fd_weights_registry
from devito.tools import (ReducerMap, as_tuple, c_restrict_void_p, flatten,
                          is_integer, memoized_meth, dtype_to_ctype, humanbytes,
                          mpi4py_mapper)
from devito.types.dimension import Dimension
from devito.types.args import ArgProvider
from devito.types.caching import CacheManager
from devito.types.basic import AbstractFunction, Size
from devito.types.utils import Buffer, DimensionTuple, NODE, CELL, host_layer, Staggering

__all__ = ['Function', 'TimeFunction', 'SubFunction', 'TempFunction']


RegionMeta = namedtuple('RegionMeta', 'offset size')
Offset = namedtuple('Offset', 'left right')


class DiscreteFunction(AbstractFunction, ArgProvider, Differentiable):

    """
    Tensor symbol representing a discrete function in symbolic equations.
    Unlike an Array, a DiscreteFunction carries data.

    Notes
    -----
    Users should not instantiate this class directly. Use Function or
    SparseFunction (or their subclasses) instead.
    """

    # Default method for the finite difference approximation weights computation.
    _default_fd = 'taylor'

    # Required by SymPy, otherwise the presence of __getitem__ will make SymPy
    # think that a DiscreteFunction is actually iterable, thus breaking many of
    # its key routines (e.g., solve)
    _iterable = False

    is_Input = True
    is_DiscreteFunction = True

    _DataType = Data
    """
    The type of the underlying data object.
    """

    __rkwargs__ = AbstractFunction.__rkwargs__ + ('staggered', 'coefficients')

    def __init_finalize__(self, *args, function=None, **kwargs):
        # Now that *all* __X_setup__ hooks have been called, we can let the
        # superclass constructor do its job
        super().__init_finalize__(*args, **kwargs)

        # Symbolic (finite difference) coefficients
        self._coefficients = self.__coefficients_setup__(**kwargs)

        # Data-related properties
        self._data = None
        self._first_touch = kwargs.get('first_touch', configuration['first-touch'])
        self._allocator = kwargs.get('allocator') or default_allocator()

        # Data initialization
        initializer = kwargs.get('initializer')

        if self.alias:
            self._initializer = None
        elif function is not None:
            # An object derived from user-level AbstractFunction (e.g.,
            # `f(x+1)`), so we just copy the reference to the original data
            self._initializer = None
            self._data = function._data
        elif isinstance(self._allocator, DataReference):
            # Don't want to reinitialise array if DataReference used as allocator;
            # create a no-op intialiser to avoid overwriting the original array.
            self._initializer = lambda x: None
        elif initializer is None or callable(initializer) or self.alias:
            # Initialization postponed until the first access to .data
            self._initializer = initializer
        elif isinstance(initializer, (np.ndarray, list, tuple)):
            # Allocate memory and initialize it. Note that we do *not* hold
            # a reference to the user-provided buffer
            self._initializer = None
            if len(initializer) > 0:
                self.data_with_halo[:] = initializer[:]
            else:
                # This is a corner case -- we might get here, for example, when
                # running with MPI and some processes get 0-size arrays after
                # domain decomposition. We touch the data anyway to avoid the
                # case `self._data is None`
                self.data
        else:
            raise ValueError("`initializer` must be callable or buffer, not %s"
                             % type(initializer))

    _subs = Differentiable._subs

    def _allocate_memory(func):
        """Allocate memory as a Data."""
        @wraps(func)
        def wrapper(self):
            if self._data is None:
                if self._alias:
                    # Aliasing Functions must not allocate data
                    return

                debug(f"Allocating host memory for {self.name}{self.shape_allocated} "
                      f"[{humanbytes(self.nbytes)}]")

                # Clear up both SymPy and Devito caches to drop unreachable data
                CacheManager.clear(force=False)

                # Allocate the actual data object
                self._data = self._DataType(self.shape_allocated, self.dtype,
                                            modulo=self._mask_modulo,
                                            allocator=self._allocator,
                                            distributor=self._distributor,
                                            padding=self._size_ghost)

                # Initialize data
                if self._first_touch:
                    assign(self, 0)
                if callable(self._initializer):
                    if self._first_touch:
                        warning("`first touch` together with `initializer` causing "
                                "redundant data initialization")
                    try:
                        self._initializer(self.data_with_halo)
                    except ValueError:
                        # Perhaps user only wants to initialise the physical domain
                        self._initializer(self.data)
                else:
                    self.data_with_halo.fill(0)

            return func(self)
        return wrapper

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dtype = kwargs.get('dtype')
        if dtype is not None:
            return dtype
        elif grid is not None:
            return grid.dtype
        else:
            return np.float32

    def __coefficients_setup__(self, **kwargs):
        """
        Setup finite-differences coefficients mode
        """
        coeffs = kwargs.get('coefficients', self._default_fd)
        if coeffs not in fd_weights_registry:
            if coeffs == 'symbolic':
                deprecations.symbolic_warn
            else:
                raise ValueError(f"coefficients must be one of {str(fd_weights_registry)}"
                                 f" not {coeffs}")
        return coeffs

    @cached_property
    def _functions(self):
        return {self.function}

    @property
    def _data_alignment(self):
        return self._allocator.guaranteed_alignment

    @property
    def _mem_external(self):
        return True

    @property
    def _mem_heap(self):
        return True

    @property
    def coefficients(self):
        """Form of the coefficients of the function."""
        return self._coefficients

    @property
    def _shape_with_outhalo(self):
        return self.shape_with_halo

    @cached_property
    def shape(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.
        """
        return self._shape

    @cached_property
    def shape_domain(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.
        Alias to `self.shape`.
        """
        return self.shape

    @cached_property
    def shape_with_halo(self):
        """
        Shape of the domain+outhalo region. The outhalo is the region
        surrounding the domain that may be read by an Operator.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.
        Further, note that the outhalo of inner ranks is typically empty, while
        the outhalo of boundary ranks contains a number of elements depending
        on the rank position in the decomposed grid (corner, side, ...).
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape, self._size_outhalo))

    @cached_property
    def _shape_with_inhalo(self):
        """
        Shape of the domain+inhalo region. The inhalo region comprises the
        outhalo as well as any additional "ghost" layers for MPI halo
        exchanges. Data in the inhalo region are exchanged when running
        Operators to maintain consistent values as in sequential runs.

        Notes
        -----
        Typically, this property won't be used in user code, but it may come
        in handy for testing or debugging
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape, self._halo))

    @cached_property
    def shape_allocated(self):
        """
        Shape of the allocated data. It includes the domain and inhalo regions,
        as well as any additional padding surrounding the halo.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.
        """
        return DimensionTuple(*[j + i + k for i, (j, k) in zip(self._shape_with_inhalo,
                                                               self._padding)],
                              getters=self.dimensions)

    @cached_property
    def shape_global(self):
        """
        Global shape of the domain region. The domain constitutes the area of
        the data written to by an Operator.

        Notes
        -----
        In an MPI context, this is the *global* domain region shape, which is
        therefore identical on all MPI ranks.

        Issues
        ------
        * https://github.com/devitocodes/devito/issues/1498
        """
        if self.grid is None:
            return self.shape
        retval = []
        for d, s in zip(self.dimensions, self.shape):
            size = self.grid.size_map.get(d)
            retval.append(size.glb if size is not None else s)
        return tuple(retval)

    @property
    def symbolic_shape(self):
        return tuple(self._C_get_field(FULL, d).size for d in self.dimensions)

    @property
    def size_global(self):
        """
        The global number of elements this object is expected to store in memory.
        Note that this would need to be combined with self.dtype to give the actual
        size in bytes.
        """
        return reduce(mul, self.shape_global)

    @property
    def _offset_inhalo(self):
        return super()._offset_halo

    @property
    def _size_inhalo(self):
        return super()._size_halo

    @cached_property
    def _size_outhalo(self):
        """Number of points in the outer halo region."""

        if self._distributor is None:
            # Computational domain is not distributed and hence the outhalo
            # and inhalo correspond
            return self._size_inhalo

        left = [abs(min(i.loc_abs_min-i.glb_min-j, 0)) if i and not i.loc_empty else 0
                for i, j in zip(self._decomposition, self._size_inhalo.left)]
        right = [max(i.loc_abs_max+j-i.glb_max, 0) if i and not i.loc_empty else 0
                 for i, j in zip(self._decomposition, self._size_inhalo.right)]

        sizes = tuple(Size(i, j) for i, j in zip(left, right))

        if self._distributor.is_parallel and (any(left) or any(right)):
            try:
                warning_msg = """A space order of {0} and a halo size of {1} has been
                                 set but the current rank ({2}) has a domain size of
                                 only {3}""".format(self._space_order,
                                                    max(self._size_inhalo),
                                                    self._distributor.myrank,
                                                    min(self.grid.shape_local))
                if not self._distributor.is_boundary_rank:
                    warning(warning_msg)
                else:
                    left_dist = [i for i, d in zip(left, self.dimensions) if d
                                 in self._distributor.dimensions]
                    right_dist = [i for i, d in zip(right, self.dimensions) if d
                                  in self._distributor.dimensions]
                    for i, j, k, l in zip(left_dist, right_dist,
                                          self._distributor.mycoords,
                                          self._distributor.topology):
                        if l > 1 and ((j > 0 and k == 0) or (i > 0 and k == l-1)):
                            warning(warning_msg)
                            break
            except AttributeError:
                pass

        return DimensionTuple(*sizes, getters=self.dimensions, left=left, right=right)

    @property
    def size_allocated(self):
        """
        The number of elements this object is expected to store in memory.
        Note that this would need to be combined with self.dtype to give the actual
        size in bytes.
        """
        return reduce(mul, self.shape_allocated)

    @cached_property
    def _mask_modulo(self):
        """Boolean mask telling which Dimensions support modulo-indexing."""
        return tuple(True if i.is_Stepping else False for i in self.dimensions)

    @cached_property
    def _mask_domain(self):
        """Slice-based mask to access the domain region of the allocated data."""
        return tuple(slice(i, j) for i, j in
                     zip(self._offset_domain, self._offset_halo.right))

    @cached_property
    def _mask_inhalo(self):
        """Slice-based mask to access the domain+inhalo region of the allocated data."""
        return tuple(slice(i.left, i.right + j.right) for i, j in
                     zip(self._offset_inhalo, self._size_inhalo))

    @cached_property
    def _mask_outhalo(self):
        """Slice-based mask to access the domain+outhalo region of the allocated data."""
        return tuple(slice(i.start - j.left, i.stop and i.stop + j.right or None)
                     for i, j in zip(self._mask_domain, self._size_outhalo))

    @cached_property
    def _decomposition(self):
        """
        Tuple of Decomposition objects, representing the domain decomposition.
        None is used as a placeholder for non-decomposed Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        mapper = {d: self._distributor.decomposition[d] for d in self._dist_dimensions}
        return tuple(mapper.get(d) for d in self.dimensions)

    @cached_property
    def _decomposition_outhalo(self):
        """
        Tuple of Decomposition objects, representing the domain+outhalo
        decomposition. None is used as a placeholder for non-decomposed Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        return tuple(v.reshape(*self._size_inhalo[d]) if v is not None else v
                     for d, v in zip(self.dimensions, self._decomposition))

    @property
    def data(self):
        """
        The domain data values, as a numpy.ndarray.

        Elements are stored in row-major format.

        Notes
        -----
        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use :meth:`data_ro`
        instead.
        """
        return self.data_domain

    def data_gather(self, start=None, stop=None, step=1, rank=0):
        """
        Gather distributed `Data` attached to a `Function` onto a single rank.

        Parameters
        ----------
        rank : int
            The rank onto which the data will be gathered.
        step : int or tuple of ints
            The `slice` step in each dimension.
        start : int or tuple of ints
            The `slice` start in each dimension.
        stop : int or tuple of ints
            The final point of the `slice` to include.

        Notes
        -----
        Alias to `self.data._gather`.

        Note that gathering data from large simulations onto a single rank may
        result in memory blow-up and hence should use this method judiciously.
        """
        return self.data._gather(start=start, stop=stop, step=step, rank=rank)

    @property
    @_allocate_memory
    def data_domain(self):
        """
        The domain data values.

        Elements are stored in row-major format.

        Notes
        -----
        Alias to `self.data`.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_domain` instead.
        """
        self._is_halo_dirty = True
        return self._data._global(self._mask_domain, self._decomposition)

    @property
    @_allocate_memory
    def data_with_halo(self):
        """
        The domain+outhalo data values.

        Elements are stored in row-major format.

        Notes
        -----
        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_with_halo` instead.
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return self._data._global(self._mask_outhalo, self._decomposition_outhalo)

    _data_with_outhalo = data_with_halo

    @property
    @_allocate_memory
    def _data_with_inhalo(self):
        """
        The domain+inhalo data values.

        Elements are stored in row-major format.

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_with_inhalo` instead.

        Typically, this accessor won't be used in user code to set or read data
        values. Instead, it may come in handy for testing or debugging
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return np.asarray(self._data[self._mask_inhalo])

    @property
    @_allocate_memory
    def _data_allocated(self):
        """
        The allocated data values, that is domain+inhalo+padding.

        Elements are stored in row-major format.

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use
        :meth:`data_ro_allocated` instead.

        Typically, this accessor won't be used in user code to set or read data
        values. Instead, it may come in handy for testing or debugging
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return np.asarray(self._data)

    def _data_buffer(self, **kwargs):
        """
        Reference to the data. Unlike :attr:`data` and :attr:`data_with_halo`,
        this *never* returns a view of the data. This method is for internal
        use only.
        """
        return self._data_allocated

    def _data_in_region(self, region, dim, side):
        """
        The data values in a given region.

        Parameters
        ----------
        region : DataRegion
            The data region of interest (e.g., OWNED, HALO) for which a view
            is produced.
        dim : Dimension
            The dimension of interest.
        side : DataSide
            The side of interest (LEFT, RIGHT).

        Notes
        -----
        This accessor does *not* support global indexing.

        With this accessor you are claiming that you will modify the values you
        get back.

        Typically, this accessor won't be used in user code to set or read
        data values.
        """
        self._is_halo_dirty = True
        offset = getattr(getattr(self, f'_offset_{region.name}')[dim], side.name)
        size = getattr(getattr(self, f'_size_{region.name}')[dim], side.name)
        index_array = [
            slice(offset, offset+size) if d is dim else slice(pl, s - pr)
            for d, s, (pl, pr)
            in zip(self.dimensions, self.shape_allocated, self._padding)
        ]
        return np.asarray(self._data[index_array])

    @property
    @_allocate_memory
    def data_ro_domain(self):
        """Read-only view of the domain data values."""
        view = self._data._global(self._mask_domain, self._decomposition)
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_with_halo(self):
        """Read-only view of the domain+outhalo data values."""
        view = self._data._global(self._mask_outhalo, self._decomposition_outhalo)
        view.setflags(write=False)
        return view

    _data_ro_with_outhalo = data_ro_with_halo

    @property
    @_allocate_memory
    def _data_ro_with_inhalo(self):
        """
        Read-only view of the domain+inhalo data values.

        Notes
        -----
        This accessor does *not* support global indexing.
        """
        view = self._data[self._mask_inhalo]
        view.setflags(write=False)
        return np.asarray(view)

    @property
    @_allocate_memory
    def _data_ro_allocated(self):
        """
        Read-only view of the domain+inhalo+padding data values.

        Notes
        -----
        This accessor does *not* support global indexing.
        """
        view = self._data
        view.setflags(write=False)
        return np.asarray(view)

    @cached_property
    def local_indices(self):
        """
        Tuple of slices representing the global indices that logically
        belong to the calling MPI rank.

        Notes
        -----
        Given a Function `f(x, y)` with shape `(nx, ny)`, when *not* using
        MPI this property will return `(slice(0, nx-1), slice(0, ny-1))`. On
        the other hand, when MPI is used, the local ranges depend on the domain
        decomposition, which is carried by `self.grid`.
        """
        if self._distributor is None:
            return tuple(slice(0, s) for s in self.shape)
        else:
            return tuple(self._distributor.glb_slices.get(d, slice(0, s))
                         for s, d in zip(self.shape, self.dimensions))

    @property
    def initializer(self):
        if isinstance(self._data, np.ndarray):
            return self.data_with_halo.view(np.ndarray)
        else:
            return self._initializer

    _C_structname = 'dataobj'
    _C_field_data = 'data'
    _C_field_size = 'size'
    _C_field_nbytes = 'nbytes'
    _C_field_nopad_size = 'npsize'
    _C_field_domain_size = 'dsize'
    _C_field_halo_size = 'hsize'
    _C_field_halo_ofs = 'hofs'
    _C_field_owned_ofs = 'oofs'
    _C_field_dmap = 'dmap'

    _C_ctype = POINTER(type(_C_structname, (Structure,),
                            {'_fields_': [(_C_field_data, c_restrict_void_p),
                                          (_C_field_size, POINTER(c_int)),
                                          (_C_field_nbytes, c_ulong),
                                          (_C_field_nopad_size, POINTER(c_ulong)),
                                          (_C_field_domain_size, POINTER(c_ulong)),
                                          (_C_field_halo_size, POINTER(c_int)),
                                          (_C_field_halo_ofs, POINTER(c_int)),
                                          (_C_field_owned_ofs, POINTER(c_int)),
                                          (_C_field_dmap, c_void_p)]}))

    def _C_make_dataobj(self, alias=None, **args):
        """
        A ctypes object representing the DiscreteFunction that can be passed to
        an Operator.
        """
        key = alias or self
        data = args[key.name]

        dataobj = byref(self._C_ctype._type_())
        dataobj._obj.data = data.ctypes.data_as(c_restrict_void_p)
        dataobj._obj.size = (c_int*self.ndim)(*data.shape)
        dataobj._obj.nbytes = data.nbytes

        # MPI-related fields
        dataobj._obj.npsize = (c_ulong*self.ndim)(*[i - sum(j) for i, j in
                                                    zip(data.shape, self._size_padding)])
        dataobj._obj.dsize = (c_ulong*self.ndim)(*self._size_domain)
        dataobj._obj.hsize = (c_int*(self.ndim*2))(*flatten(self._size_halo))
        dataobj._obj.hofs = (c_int*(self.ndim*2))(*flatten(self._offset_halo))
        dataobj._obj.oofs = (c_int*(self.ndim*2))(*flatten(self._offset_owned))

        # Fields used only within C-land
        dataobj._obj.dmap = c_void_p(0)

        # stash a reference to the array on _obj, so we don't let it get freed
        # while we hold onto _obj
        dataobj._obj.underlying_array = data

        return dataobj

    def _C_as_ndarray(self, dataobj):
        """Cast the data carried by a DiscreteFunction dataobj to an ndarray."""
        shape = tuple(dataobj._obj.size[i] for i in range(self.ndim))
        ctype_1d = dtype_to_ctype(self.dtype) * int(reduce(mul, shape))
        buf = cast(dataobj._obj.data, POINTER(ctype_1d)).contents
        return np.frombuffer(buf, dtype=self.dtype).reshape(shape)

    @memoized_meth
    def _C_make_index(self, dim, side=None):
        # Depends on how fields are populated in `_C_make_dataobj`
        idx = self.dimensions.index(dim)
        if side is not None:
            idx = idx*2 + (0 if side is LEFT else 1)
        return idx

    @memoized_meth
    def _C_get_field(self, region, dim, side=None):
        """Symbolic representation of a given data region."""
        ffp = lambda f, i: IndexedPointer(FieldFromPointer(f"{f}", self._C_symbol), i)
        if region is DOMAIN:
            offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
            size = ffp(self._C_field_domain_size, self._C_make_index(dim))
        elif region is OWNED:
            if side is LEFT:
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, RIGHT))
            elif side is CENTER:
                # Note: identical to region=HALO, side=CENTER
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_domain_size, self._C_make_index(dim))
            else:
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, RIGHT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, LEFT))
        elif region is HALO:
            if side is LEFT:
                offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, LEFT))
            elif side is CENTER:
                # Note: identical to region=OWNED, side=CENTER
                offset = ffp(self._C_field_owned_ofs, self._C_make_index(dim, LEFT))
                size = ffp(self._C_field_domain_size, self._C_make_index(dim))
            else:
                offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, RIGHT))
                size = ffp(self._C_field_halo_size, self._C_make_index(dim, RIGHT))
        elif region is NOPAD:
            offset = ffp(self._C_field_halo_ofs, self._C_make_index(dim, LEFT))
            size = ffp(self._C_field_nopad_size, self._C_make_index(dim))
        elif region is FULL:
            offset = 0
            size = ffp(self._C_field_size, self._C_make_index(dim))
        else:
            raise ValueError(f"Unknown region `{str(region)}`")

        return RegionMeta(offset, size)

    def _halo_exchange(self):
        """Perform the halo exchange with the neighboring processes."""
        if not MPI.Is_initialized() or \
                MPI.COMM_WORLD.size == 1 or \
                not configuration['mpi'] or \
                self.grid is None:
            # Nothing to do
            return
        if MPI.COMM_WORLD.size > 1 and self._distributor is None:
            raise RuntimeError(f"`{self.name}` cannot perform a halo exchange as it has "
                               "no Grid attached")

        neighborhood = self._distributor.neighborhood
        comm = self._distributor.comm
        comm_dtype = mpi4py_mapper.get(self.dtype, self.dtype)

        for d in self._dist_dimensions:
            for i in [LEFT, RIGHT]:
                # Get involved peers
                dest = neighborhood[d][i]
                source = neighborhood[d][i.flip()]

                # Gather send data
                data = self._data_in_region(OWNED, d, i)
                sendbuf = np.ascontiguousarray(data.view(comm_dtype))

                # Setup recv buffer
                shape = self._data_in_region(HALO, d, i.flip()).shape
                recvbuf = np.ndarray(shape=shape, dtype=comm_dtype)

                # Communication
                comm.Sendrecv(sendbuf, dest=dest, recvbuf=recvbuf, source=source)

                # Scatter received data
                if recvbuf is not None and source != MPI.PROC_NULL:
                    self._data_in_region(HALO, d, i.flip())[:] = recvbuf.view(self.dtype)

        self._is_halo_dirty = False

    @property
    def _arg_names(self):
        """Tuple of argument names introduced by this function."""
        return (self.name,)

    def _arg_defaults(self, alias=None, metadata=None, estimate_memory=False):
        """
        A map of default argument values defined by this symbol.

        Parameters
        ----------
        alias : DiscreteFunction, optional
            To bind the argument values to different names.
        """
        key = alias or self
        # Avoid touching the data if just estimating memory usage
        if estimate_memory:
            args = ReducerMap({key.name: self})
        else:
            args = ReducerMap({key.name: self._data_buffer(metadata=metadata)})

        # Collect default dimension arguments from all indices
        for a, i, s in zip(key.dimensions, self.dimensions, self.shape):
            args.update(i._arg_defaults(_min=0, size=s, alias=a))

        return args

    def _arg_values(self, metadata=None, estimate_memory=False, **kwargs):
        """
        A map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        Parameters
        ----------
        **kwargs
            Dictionary of user-provided argument overrides.
        """
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, DiscreteFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self, metadata=metadata,
                                           estimate_memory=estimate_memory)
                values = values.reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {self.name: new}
                # Add value overrides for all associated dimensions
                for i, s in zip(self.dimensions, new.shape):
                    size = s - sum(self._size_nodomain[i])
                    values.update(i._arg_defaults(size=size))
        else:
            values = self._arg_defaults(alias=self, metadata=metadata,
                                        estimate_memory=estimate_memory)
            values = values.reduce_all()

        return values

    def _arg_check(self, args, intervals, **kwargs):
        """
        Check that `args` contains legal runtime values bound to `self`.

        Raises
        ------
        InvalidArgument
            If an incompatibility is detected.
        """
        if self.name not in args:
            raise InvalidArgument(f"No runtime value for `{self.name}`")

        data = args[self.name]

        if len(data.shape) != self.ndim:
            raise InvalidArgument(f"Shape {data.shape} of runtime value `{self.name}` "
                                  f"does not match dimensions {self.dimensions}")
        if data.dtype != self.dtype:
            warning(f"Data type {data.dtype} of runtime value `{self.name}` "
                    f"does not match the Function data type {self.dtype}")

        # Check each Dimension for potential OOB accesses
        for i, s in zip(self.dimensions, data.shape):
            i._arg_check(args, s, intervals[i])

        if args.options['index-mode'] == 'int32' and \
           args.options['linearize'] and \
           self.is_regular and \
           data.size - 1 >= np.iinfo(np.int32).max:
            raise InvalidArgument(f"`{self.name}`, with its {data.size} elements, is too "
                                  "big for int32 pointer arithmetic. Consider using the "
                                  "'index-mode=int64' option, the save=Buffer(..) "
                                  "API (TimeFunction only), or domain "
                                  "decomposition via MPI")

    def _arg_finalize(self, args, alias=None):
        key = alias or self
        return {key.name: self._C_make_dataobj(alias=key, **args)}

    # Pickling support

    @property
    def _pickle_rkwargs(self):
        # Picklying carries data over, if available
        return tuple(self.__rkwargs__) + ('initializer',)


class Function(DiscreteFunction):

    """
    Tensor symbol representing a discrete function in symbolic equations.

    A Function carries multi-dimensional data and provides operations to create
    finite-differences approximations.

    A Function encapsulates space-varying data; for data that also varies in time,
    use TimeFunction instead.

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the Function. When grid is not
        provided, shape and dimensions must be given. For MPI execution, a
        Grid is compulsory.
    space_order : int or 3-tuple of ints, optional, default=1
        Discretisation order for space derivatives.
        `space_order` also impacts the number of points available around a
        generic point of interest.  By default, `space_order` points are
        available on both sides of a generic point of interest, including those
        nearby the grid boundary. Sometimes, fewer points suffice; in other
        scenarios, more points are necessary. In such cases, instead of an
        integer, one can pass:
          * a 3-tuple `(o, lp, rp)` indicating the discretization order
            (`o`) as well as the number of points on the left (`lp`) and
            right (`rp`) sides of a generic point of interest;
          * a 2-tuple `(o, ((lp0, rp0), (lp1, rp1), ...))` indicating the
            discretization order (`o`) as well as the number of points on
            the left/right sides of a generic point of interest for each
            SpaceDimension.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if `grid`
        isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if `grid` isn't
        given.
    dtype : data-type, optional, default=np.float32
        Any object that can be interpreted as a numpy data type.
    staggered : Dimension or tuple of Dimension or Stagger, optional, default=None
        Define how the Function is staggered.
    initializer : callable or any object exposing the buffer interface, default=None
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.
    padding : int or tuple of ints, optional
        Allocate extra grid points to maximize data access alignment. When a tuple
        of ints, one int per Dimension should be provided.

    Examples
    --------
    Creation

    >>> from devito import Grid, Function
    >>> grid = Grid(shape=(4, 4))
    >>> f = Function(name='f', grid=grid)
    >>> f
    f(x, y)
    >>> g = Function(name='g', grid=grid, space_order=2)
    >>> g
    g(x, y)

    First-order derivatives through centered finite-difference approximations

    >>> f.dx
    Derivative(f(x, y), x)
    >>> f.dy
    Derivative(f(x, y), y)
    >>> g.dx
    Derivative(g(x, y), x)
    >>> (f + g).dx
    Derivative(f(x, y) + g(x, y), x)

    First-order derivatives through left/right finite-difference approximations

    >>> f.dxl
    Derivative(f(x, y), x)

    Note that the fact that it's a left-derivative isn't captured in the representation.
    However, upon derivative expansion, this becomes clear

    >>> f.dxl.evaluate
    f(x, y)/h_x - f(x - h_x, y)/h_x
    >>> f.dxr
    Derivative(f(x, y), x)

    Second-order derivative through centered finite-difference approximation

    >>> g.dx2
    Derivative(g(x, y), (x, 2))

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy
    uses `*args` to (re-)create the dimension arguments of the symbolic object.
    """

    is_Function = True

    is_autopaddable = True

    __rkwargs__ = (DiscreteFunction.__rkwargs__ +
                   ('space_order', 'shape_global', 'dimensions'))

    def _cache_meta(self):
        # Attach additional metadata to self's cache entry
        return {'nbytes': self.size}

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        # Staggering
        self._staggered = self.__staggered_setup__(self.dimensions,
                                                   staggered=kwargs.get('staggered'))

        # Space order
        space_order = kwargs.get('space_order', 1)
        if is_integer(space_order):
            self._space_order = space_order
        elif isinstance(space_order, tuple) and len(space_order) >= 2:
            self._space_order = space_order[0]
        else:
            raise TypeError("Invalid `space_order`")

        # Acquire derivative shortcuts
        if self is self.function:
            self._fd = self.__fd_setup__()
        else:
            # E.g., `self is f(x + i0, y)` and `self.function is f(x, y)`
            # Dynamically genereating derivative shortcuts is expensive; we
            # can clearly avoid that here though!
            self._fd = self.function._fd

        # Flag whether it is a parameter or a variable.
        # Used at operator evaluation to evaluate the Function at the
        # variable location (i.e. if the variable is staggered in x the
        # parameter has to be computed at x + hx/2)
        self._is_parameter = kwargs.get('parameter', False)

    def __fd_setup__(self):
        """
        Dynamically add derivative short-cuts.
        """
        return generate_fd_shortcuts(self.dimensions, self.space_order)

    @cached_property
    def _fd_priority(self):
        return 1 if self.staggered.on_node else 2

    @property
    def is_parameter(self):
        return self._is_parameter

    def _eval_at(self, func):
        if not self.is_parameter or self.staggered == func.staggered:
            return self
        mapper = {self.indices_ref[d]: func.indices_ref[d]
                  for d in self.dimensions
                  if self.indices_ref[d] is not func.indices_ref[d]}
        if mapper:
            return self.subs(mapper)
        return self

    @classmethod
    def __staggered_setup__(cls, dimensions, staggered=None, **kwargs):
        """
        Setup staggering-related metadata. This method assigns:

            * 0 to non-staggered dimensions;
            * 1 to staggered dimensions.
        """
        if not staggered:
            processed = ()
        elif staggered is CELL:
            processed = (sympy.S.One,)*len(dimensions)
        elif staggered is NODE:
            processed = (sympy.S.Zero,)*len(dimensions)
        elif all(is_integer(s) for s in as_tuple(staggered)):
            # Staggering is already a tuple likely from rebuild
            assert len(staggered) == len(dimensions)
            processed = staggered
        else:
            processed = []
            for d in dimensions:
                if d in as_tuple(staggered):
                    processed.append(sympy.S.One)
                elif -d in as_tuple(staggered):
                    processed.append(sympy.S.NegativeOne)
                else:
                    processed.append(sympy.S.Zero)
        return Staggering(*processed, getters=dimensions)

    @classmethod
    def __indices_setup__(cls, *args, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        staggered = kwargs.get('staggered')

        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions

        staggered = cls.__staggered_setup__(dimensions, staggered=staggered)
        if args:
            assert len(args) == len(dimensions)
            staggered_indices = tuple(args)
        else:
            if not staggered:
                staggered_indices = dimensions
            else:
                staggered_indices = (d + i * d.spacing / 2
                                     for d, i in zip(dimensions, staggered))
        return tuple(dimensions), tuple(staggered_indices)

    @property
    def staggered(self):
        """The staggered indices of the object."""
        return self._staggered

    @property
    def is_Staggered(self):
        return bool(self.staggered)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        shape = kwargs.get('shape', kwargs.get('shape_global'))
        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
        elif shape is None:
            if dimensions is not None and dimensions != grid.dimensions:
                raise TypeError("Need `shape` as not all `dimensions` are in `grid`")
            shape = grid.shape_local
        elif dimensions is None:
            raise TypeError("`dimensions` required if both `grid` and "
                            "`shape` are provided")
        else:
            # Got `grid`, `dimensions`, and `shape`. We sanity-check that the
            # Dimensions in `dimensions` also appearing in `grid` have same size
            # (given by `shape`) as that provided in `grid`
            if len(shape) != len(dimensions):
                raise ValueError("`shape` and `dimensions` must have the "
                                 "same number of entries")
            loc_shape = []
            for d, s in zip(dimensions, shape):
                if d in grid.dimensions:
                    size = grid.size_map[d]
                    if size.glb != s and s is not None:
                        raise ValueError(f"Dimension `{d}` is given size `{s}`, "
                                         f"while `grid` says `{d}` has size `{size.glb}`")
                    else:
                        loc_shape.append(size.loc)
                else:
                    loc_shape.append(s)
            shape = tuple(loc_shape)
        return shape

    def __halo_setup__(self, **kwargs):
        if self._distributor and self._distributor.loc_empty:
            # No need to assign a halo on a completely empty rank
            return DimensionTuple(*[(0, 0) for i in self.dimensions],
                                  getters=self.dimensions)
        halo = kwargs.get('halo')
        if halo is not None:
            if isinstance(halo, DimensionTuple):
                halo = tuple(halo[d] for d in self.dimensions)
        else:
            space_order = kwargs.get('space_order', 1)
            if is_integer(space_order):
                v = (space_order, space_order)
                halo = [v if i.is_Space else (0, 0) for i in self.dimensions]

            elif isinstance(space_order, tuple) and len(space_order) == 3:
                _, l, r = space_order
                halo = [(l, r) if i.is_Space else (0, 0) for i in self.dimensions]

            elif isinstance(space_order, tuple) and len(space_order) == 2:
                _, space_halo = space_order
                if not isinstance(space_halo, tuple) or \
                   not all(isinstance(i, tuple) for i in space_halo) or \
                   len(space_halo) != len(self.space_dimensions):
                    raise TypeError("Invalid `space_order`")
                v = list(space_halo)
                halo = [v.pop(0) if i.is_Space else (0, 0)
                        for i in self.dimensions]

            else:
                raise TypeError("Invalid `space_order`")
        return DimensionTuple(*halo, getters=self.dimensions)

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding')
        if padding is None:
            if self.is_autopaddable:
                padding = self.__padding_setup_smart__(**kwargs)
            else:
                padding = super().__padding_setup__(**kwargs)

        elif isinstance(padding, DimensionTuple):
            padding = tuple(padding[d] for d in self.dimensions)

        elif is_integer(padding):
            padding = tuple((0, padding) if d.is_Space else (0, 0)
                            for d in self.dimensions)

        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            padding = tuple((0, i) if is_integer(i) else i for i in padding)

        else:
            raise TypeError(f"`padding` must be int or {self.ndim}-tuple of ints")
        return DimensionTuple(*padding, getters=self.dimensions)

    @property
    def space_order(self):
        """The space order."""
        return self._space_order

    def sum(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the sum of `p` points
        along the spatial dimensions `dims`.

        Parameters
        ----------
        p : int, optional
            The number of summands. Defaults to the halo size.
        dims : tuple of Dimension, optional
            The Dimensions along which the sum is computed. Defaults to
            `self`'s spatial dimensions.
        """
        points = []
        for d in (as_tuple(dims) or self.space_dimensions):
            if p is None:
                lp = self._size_inhalo[d].left
                rp = self._size_inhalo[d].right
            else:
                lp = p // 2 + p % 2
                rp = p // 2
            indices = [d - i for i in range(lp, 0, -1)]
            indices.extend([d + i for i in range(rp)])
            points.extend([self.subs({d: i}) for i in indices])
        return sum(points)

    def avg(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the average of `p` points
        along the spatial dimensions `dims`.

        Parameters
        ----------
        p : int, optional
            The number of summands. Defaults to the halo size.
        dims : tuple of Dimension, optional
            The Dimensions along which the average is computed. Defaults to
            `self`'s spatial dimensions.
        """
        tot = self.sum(p, dims)
        return tot / len(tot.args)


class TimeFunction(Function):

    """
    Tensor symbol representing a discrete function in symbolic equations.

    A TimeFunction carries multi-dimensional data and provides operations to create
    finite-differences approximations, in both space and time.

    A TimeFunction encapsulates space- and time-varying data.

    Parameters
    ----------
    name : str
        Name of the symbol.
    grid : Grid, optional
        Carries shape, dimensions, and dtype of the Function. When grid is not
        provided, shape and dimensions must be given. For MPI execution, a
        Grid is compulsory.
    space_order : int or 3-tuple of ints, optional, default=1
        Discretisation order for space derivatives.
        `space_order` also impacts the number of points available around a
        generic point of interest.  By default, `space_order` points are
        available on both sides of a generic point of interest, including those
        nearby the grid boundary. Sometimes, fewer points suffice; in other
        scenarios, more points are necessary. In such cases, instead of an
        integer, one can pass:
          * a 3-tuple `(o, lp, rp)` indicating the discretization order
            (`o`) as well as the number of points on the left (`lp`) and
            right (`rp`) sides of a generic point of interest;
          * a 2-tuple `(o, ((lp0, rp0), (lp1, rp1), ...))` indicating the
            discretization order (`o`) as well as the number of points on
            the left/right sides of a generic point of interest for each
            SpaceDimension.
    time_order : int, optional
        Discretization order for time derivatives. Defaults to 1.
    shape : tuple of ints, optional
        Shape of the domain region in grid points. Only necessary if `grid`
        isn't given.
    dimensions : tuple of Dimension, optional
        Dimensions associated with the object. Only necessary if `grid` isn't
        given.
    dtype : data-type, optional, default=np.float32
        Any object that can be interpreted as a numpy data type.
    save : int or Buffer, optional, default=None
        By default, `save=None`, which indicates the use of alternating
        buffers. This enables cyclic writes to the TimeFunction. For example,
        if the TimeFunction `u(t, x)` has shape (3, 100), then, in an Operator,
        `t` will assume the values `1, 2, 0, 1, 2, 0, 1, ...` (note that the
        very first value depends on the stencil equation in which `u` is
        written.). The default size of the time buffer when `save=None` is
        `time_order + 1`.  To specify a different size for the time buffer, one
        should use the syntax `save=Buffer(mysize)`.  Alternatively, if all of
        the intermediate results are required (or, simply, to avoid using an
        alternating buffer), an explicit value for `save` ( an integer) must be
        provided.
    time_dim : Dimension, optional, default=grid.time_dim
        TimeDimension to be used in the TimeFunction.
    staggered : Dimension or tuple of Dimension or Stagger, optional, default=None
        Define how the Function is staggered.
    initializer : callable or any object exposing the buffer interface, default=None
        Data initializer. If a callable is provided, data is allocated lazily.
    allocator : MemoryAllocator, optional
        Controller for memory allocation. To be used, for example, when one wants
        to take advantage of the memory hierarchy in a NUMA architecture. Refer to
        `default_allocator.__doc__` for more information.
    padding : int or tuple of ints, optional
        Allocate extra grid points to maximize data access alignment. When a tuple
        of ints, one int per Dimension should be provided.

    Examples
    --------

    Creation

    >>> from devito import Grid, TimeFunction
    >>> grid = Grid(shape=(4, 4))
    >>> f = TimeFunction(name='f', grid=grid)
    >>> f
    f(t, x, y)
    >>> g = TimeFunction(name='g', grid=grid, time_order=2)
    >>> g
    g(t, x, y)

    First-order derivatives through centered finite-difference approximations

    >>> f.dx
    Derivative(f(t, x, y), x)
    >>> f.dt
    Derivative(f(t, x, y), t)
    >>> g.dt
    Derivative(g(t, x, y), t)

    When using the alternating buffer protocol, the size of the time dimension
    is given by `time_order + 1`

    >>> f.shape
    (2, 4, 4)
    >>> g.shape
    (3, 4, 4)

    One can drop the alternating buffer protocol specifying a value for `save`

    >>> h = TimeFunction(name='h', grid=grid, save=20)
    >>> h
    h(time, x, y)
    >>> h.shape
    (20, 4, 4)

    Notes
    -----
    The parameters must always be given as keyword arguments, since SymPy uses
    `*args` to (re-)create the dimension arguments of the symbolic object.
    If the parameter `grid` is provided, the values for `shape`,
    `dimensions` and `dtype` will be derived from it. When present, the
    parameter `shape` should only define the spatial shape of the grid. The
    temporal dimension will be inserted automatically as the leading dimension.
    """

    is_TimeFunction = True
    is_TimeDependent = True

    _time_position = 0
    """Position of time index among the function indices."""

    __rkwargs__ = Function.__rkwargs__ + ('time_order', 'save', 'time_dim')

    def __init_finalize__(self, *args, **kwargs):
        self.time_dim = kwargs.get('time_dim', self.dimensions[self._time_position])
        self._time_order = kwargs.get('time_order', 1)
        super().__init_finalize__(*args, **kwargs)

        if not is_integer(self.time_order):
            raise TypeError("`time_order` must be int")

        self.save = kwargs.get('save')

    def __fd_setup__(self):
        """
        Dynamically add derivative short-cuts.
        """
        return generate_fd_shortcuts(self.dimensions, self.space_order,
                                     to=self.time_order)

    @classmethod
    def __indices_setup__(cls, *args, **kwargs):
        dimensions = kwargs.get('dimensions')

        if dimensions is None:
            save = kwargs.get('save')
            grid = kwargs.get('grid')
            time_dim = kwargs.get('time_dim')

            if time_dim is None:
                time_dim = grid.time_dim if is_integer(save) else grid.stepping_dim
            elif not (isinstance(time_dim, Dimension) and time_dim.is_Time):
                raise TypeError("`time_dim` must be a time dimension")
            dimensions = list(Function.__indices_setup__(**kwargs)[0])
            dimensions.insert(cls._time_position, time_dim)

        return Function.__indices_setup__(
            *args, dimensions=dimensions, staggered=kwargs.get('staggered')
        )

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        save = kwargs.get('save') or None  # Force to None if 0/False/None/...
        dimensions = kwargs.get('dimensions')
        shape = kwargs.get('shape', kwargs.get('shape_global'))
        time_order = kwargs.get('time_order', 1)

        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
            if save is not None:
                raise TypeError("Ambiguity detected: provide either `grid` and `save` "
                                "or just `shape` ")
        elif shape is None:
            shape = list(grid.shape_local)
            if save is None:
                shape.insert(cls._time_position, time_order + 1)
            elif isinstance(save, Buffer):
                shape.insert(cls._time_position, save.val)
            elif is_integer(save):
                shape.insert(cls._time_position, save)
            else:
                raise TypeError(f"`save` can be None, int or Buffer, not {type(save)}")
        elif dimensions is None:
            raise TypeError("`dimensions` required if both `grid` and "
                            "`shape` are provided")
        else:
            shape = super().__shape_setup__(
                grid=grid, shape=shape, dimensions=dimensions
            )

        return tuple(shape)

    @cached_property
    def _fd_priority(self):
        return 2.1 if self.staggered.on_node else 2.2

    @property
    def time_order(self):
        """The time order."""
        return self._time_order

    @property
    def forward(self):
        """Symbol for the time-forward state of the TimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.dimensions[self._time_position]

        return self._subs(_t, _t + i * _t.spacing)

    @property
    def backward(self):
        """Symbol for the time-backward state of the TimeFunction."""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.dimensions[self._time_position]

        return self._subs(_t, _t - i * _t.spacing)

    @property
    def layer(self):
        """
        The memory hierarchy layer in which the TimeFunction is stored.
        """
        return host_layer

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    @property
    def time_size(self):
        return self._time_size

    @property
    def _time_buffering(self):
        return not is_integer(self.save)

    @property
    def _time_buffering_default(self):
        return self._time_buffering and not isinstance(self.save, Buffer)

    def _arg_check(self, args, intervals, **kwargs):
        super()._arg_check(args, intervals, **kwargs)

        key_time_size = args[self.name].shape[self._time_position]
        if self._time_buffering and self._time_size != key_time_size:
            raise InvalidArgument(f"Expected `time_size={self._time_size}` for runtime "
                                  f"value `{self.name}`, found `{key_time_size}` instead")


class SubFunction(Function):

    """
    A Function bound to a "parent" DiscreteFunction.

    A SubFunction hands control of argument binding and halo exchange to the
    DiscreteFunction it's bound to.
    """

    __rkwargs__ = DiscreteFunction.__rkwargs__ + ('dimensions', 'shape')

    def __init_finalize__(self, *args, **kwargs):
        self._parent = kwargs.pop('parent', None)
        super().__init_finalize__(*args, **kwargs)

    def __padding_setup__(self, **kwargs):
        # SubFunctions aren't expected to be used in time-consuming loops
        return tuple((0, 0) for i in range(self.ndim))

    def _halo_exchange(self):
        return

    def _arg_values(self, **kwargs):
        if self._parent is not None and self.parent.name not in kwargs:
            return self._parent._arg_defaults(alias=self._parent).reduce_all()
        elif self.name in kwargs:
            raise RuntimeError(f"`{self.name}` is a SubFunction, so it can't be assigned "
                               "a value dynamically")
        else:
            return self._arg_defaults(alias=self)

    def _arg_apply(self, *args, **kwargs):
        if self._parent is not None:
            return self._parent._arg_apply(*args, **kwargs)
        return super()._arg_apply(*args, **kwargs)

    @property
    def origin(self):
        # SubFunction have zero origin
        return DimensionTuple(*(0 for _ in range(self.ndim)), getters=self.dimensions)

    @property
    def parent(self):
        return self._parent


class TempFunction(DiscreteFunction):

    """
    Tensor symbol used to store an intermediate sub-expression extracted from
    one or more symbolic equations.

    Users should not instantiate this class directly. TempFunctions may be created
    by Devito to store intermediate sub-expressions ("temporary values") when the
    user supplies the `cire-ftemps` option to an Operator.

    Unlike other DiscreteFunction types, TempFunctions do not carry data directly.
    However, they can generate Functions to override the TempFunction at Operator
    application time (see the Examples section below).

    TempFunctions are useful if the user wants to retain control over the allocation
    and deletion of temporary storage (by default, instead, Devito uses Arrays, which
    are allocated and deallocated upon entering and exiting C-land, respectively).

    Examples
    --------
    The `make` method makes the TempFunction create a new Function. For more info,
    refer to TempFunction.make.__doc__.

      .. code-block:: python

        op = Operator(...)
        cfuncs = [i for i in op.input if i.is_TempFunction]
        kwargs = {i.name: i.make(grid.shape) for i in cfuncs}
        op.apply(..., **kwargs)
    """

    is_TempFunction = True

    __rkwargs__ = DiscreteFunction.__rkwargs__ + ('dimensions', 'pointer_dim')

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self._pointer_dim = kwargs.get('pointer_dim')

    @classmethod
    def __indices_setup__(cls, **kwargs):
        pointer_dim = kwargs.get('pointer_dim')
        dimensions = as_tuple(kwargs['dimensions'])
        if pointer_dim not in dimensions:
            # This is a bit hacky but it does work around duplicate dimensions when
            # it gets to pickling
            dimensions = as_tuple(pointer_dim) + dimensions

        # Sanity check
        assert not any(d.is_NonlinearDerived for d in dimensions)

        return dimensions, dimensions

    def __halo_setup__(self, **kwargs):
        pointer_dim = kwargs.get('pointer_dim')
        dimensions = as_tuple(kwargs['dimensions'])
        halo = as_tuple(kwargs.get('halo'))
        if halo is None:
            halo = tuple((0, 0) for _ in dimensions)
        if pointer_dim is not None and pointer_dim not in dimensions:
            halo = ((0, 0),) + as_tuple(halo)
        return halo

    @property
    def data(self):
        # Any attempt at allocating data by the user should fail miserably
        raise TypeError("TempFunction cannot allocate data")

    data_domain = data
    data_with_halo = data
    data_ro_domain = data
    data_ro_with_halo = data

    @property
    def pointer_dim(self):
        return self._pointer_dim

    @property
    def dim(self):
        return self.pointer_dim

    @property
    def shape(self):
        domain = [i.symbolic_size for i in self.dimensions]
        return DimensionTuple(*domain, getters=self.dimensions)

    @property
    def shape_with_halo(self):
        domain = self.shape
        halo = [sympy.Add(*i, evaluate=False) for i in self._size_halo]
        ret = tuple(sum(i) for i in zip(domain, halo))
        return DimensionTuple(*ret, getters=self.dimensions)

    shape_allocated = AbstractFunction.symbolic_shape
    symbolic_shape = AbstractFunction.symbolic_shape

    def make(self, shape=None, initializer=None, allocator=None, **kwargs):
        """
        Create a Function which can be used to override this TempFunction
        in a call to `op.apply(...)`.

        Parameters
        ----------
        shape : tuple of ints, optional
            Shape of the domain region in grid points.
        initializer : callable or any object exposing the buffer interface, optional
            Data initializer. If a callable is provided, data is allocated lazily.
        allocator : MemoryAllocator, optional
            Controller for memory allocation. To be used, for example, when one wants
            to take advantage of the memory hierarchy in a NUMA architecture. Refer to
            `default_allocator.__doc__` for more information.
        **kwargs
            Mapper of Operator overrides. Used to automatically derive the shape
            if not explicitly provided.
        """
        if shape is None:
            if len(kwargs) == 0:
                raise ValueError("Either `shape` or `kwargs` (Operator overrides) "
                                 "must be provided.")
            shape = []
            args = normalize_args(kwargs)
            for n, i in enumerate(self.shape):
                v = i.subs(args)
                if not v.is_Integer:
                    raise ValueError(f"Couldn't resolve `shape[{n}]={i}` with the given "
                                     f"kwargs (obtained: `{v}`)")
                shape.append(int(v))
            shape = tuple(shape)
        elif len(shape) != self.ndim:
            raise ValueError(f"`shape` must contain {self.ndim} integers, "
                             f"not {len(shape)}")
        elif not all(is_integer(i) for i in shape):
            raise ValueError(f"`shape` must contain integers (got `{str(shape)}`)")

        return Function(name=self.name, dtype=self.dtype, dimensions=self.dimensions,
                        shape=shape, halo=self.halo, initializer=initializer,
                        allocator=allocator)

    def _make_pointer(self, dim):
        return TempFunction(name=f'p{self.name}', dtype=self.dtype, pointer_dim=dim,
                            dimensions=self.dimensions, halo=self.halo)

    def _arg_defaults(self, alias=None):
        raise RuntimeError("TempFunction does not have default arguments ")

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, DiscreteFunction):
                # Set new values and re-derive defaults
                return new._arg_defaults().reduce_all()
            else:
                raise InvalidArgument(f"Illegal runtime value for `{self.name}`")
        else:
            raise InvalidArgument(f"TempFunction `{self.name}` lacks override")
