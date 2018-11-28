from collections import OrderedDict, namedtuple
from itertools import product

import sympy
import numpy as np
from psutil import virtual_memory
from cached_property import cached_property

from devito.builtins import assign
from devito.cgen_utils import INT, cast_mapper
from devito.data import Data, default_allocator
from devito.dimension import Dimension, ConditionalDimension, DefaultDimension
from devito.equation import Eq, Inc
from devito.exceptions import InvalidArgument
from devito.logger import debug, warning
from devito.mpi import MPI, SparseDistributor
from devito.parameters import configuration
from devito.symbolics import Add, indexify, retrieve_function_carriers
from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.types import (AbstractCachedFunction, AbstractCachedSymbol, Symbol, Scalar,
                          OWNED, HALO, LEFT, RIGHT)
from devito.tools import (EnrichedTuple, Tag, ReducerMap, ArgProvider, as_tuple,
                          flatten, is_integer, prod, powerset, filter_ordered,
                          memoized_meth)

__all__ = ['Constant', 'Function', 'TimeFunction', 'SparseFunction',
           'SparseTimeFunction', 'PrecomputedSparseFunction',
           'PrecomputedSparseTimeFunction', 'Buffer', 'NODE', 'CELL']


class Constant(AbstractCachedSymbol, ArgProvider):

    """
    Symbol representing constant values in symbolic equations.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_Input = True
    is_Constant = True
    is_Scalar = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self._value = kwargs.get('value', 0)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def data(self):
        """The value of the data object, as a scalar (int, float, ...)."""
        return self.dtype(self._value)

    @data.setter
    def data(self, val):
        self._value = val

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this symbol."""
        return (self.name,)

    @memoized_meth
    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.
        """
        key = alias or self
        return {key.name: self.data}

    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, Constant):
                return new._arg_defaults(alias=self)
            else:
                return {self.name: new}
        else:
            return self._arg_defaults()

    def _arg_check(self, args, intervals):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for %s" % self.name)
        key = args[self.name]
        try:
            # Might be a plain number, w/o a dtype field
            if key.dtype != self.dtype:
                warning("Data type %s of runtime value `%s` does not match the "
                        "Constant data type %s" % (key.dtype, self.name, self.dtype))
        except AttributeError:
            pass

    _pickle_kwargs = AbstractCachedSymbol._pickle_kwargs + ['_value']


class TensorFunction(AbstractCachedFunction, ArgProvider):

    """
    Utility class to encapsulate all symbolic types that represent
    tensor (array) data.

    .. note::

        Users should not instantiate this class. Use :class:`Function` or
        :class:`SparseFunction` (or their subclasses) instead.
    """

    # Required by SymPy, otherwise the presence of __getitem__ will make SymPy
    # think that a TensorFunction is actually iterable, thus breaking many of
    # its key routines (e.g., solve)
    _iterable = False

    is_Input = True
    is_TensorFunction = True
    is_Tensor = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(TensorFunction, self).__init__(*args, **kwargs)

            # There may or may not be a `Grid` attached to the TensorFunction
            self._grid = kwargs.get('grid')

            # A `Distributor` to handle domain decomposition (only relevant for MPI)
            self._distributor = self.__distributor_setup__(**kwargs)

            # Staggering metadata
            self._staggered = self.__staggered_setup__(**kwargs)

            # Data-related properties and data initialization
            self._data = None
            self._first_touch = kwargs.get('first_touch', configuration['first-touch'])
            self._allocator = kwargs.get('allocator', default_allocator())
            initializer = kwargs.get('initializer')
            if initializer is None or callable(initializer):
                # Initialization postponed until the first access to .data
                self._initializer = initializer
            elif isinstance(initializer, (np.ndarray, list, tuple)):
                # Allocate memory and initialize it. Note that we do *not* hold
                # a reference to the user-provided buffer
                self._initializer = None
                if len(initializer) > 0:
                    self.data_with_halo[:] = initializer
                else:
                    # This is a corner case -- we might get here, for example, when
                    # running with MPI and some processes get 0-size arrays after
                    # domain decomposition. We touch the data anyway to avoid the
                    # case ``self._data is None``
                    self.data
            else:
                raise ValueError("`initializer` must be callable or buffer, not %s"
                                 % type(initializer))

    def _allocate_memory(func):
        """Allocate memory as a :class:`Data`."""
        def wrapper(self):
            if self._data is None:
                debug("Allocating memory for %s%s" % (self.name, self.shape_allocated))
                self._data = Data(self.shape_allocated, self.dtype,
                                  modulo=self._mask_modulo, allocator=self._allocator)
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

    def __staggered_setup__(self, **kwargs):
        """
        Setup staggering-related metadata. This method assigns: ::

            * 0 to non-staggered dimensions;
            * 1 to staggered dimensions.
        """
        staggered = kwargs.get('staggered')
        if staggered is None:
            self.is_Staggered = False
            return tuple(0 for _ in self.indices)
        else:
            self.is_Staggered = True
            if staggered is NODE:
                staggered = ()
            elif staggered is CELL:
                staggered = self.indices
            else:
                staggered = as_tuple(staggered)
            mask = []
            for d in self.indices:
                if d in staggered:
                    mask.append(1)
                elif -d in staggered:
                    mask.append(-1)
                else:
                    mask.append(0)
            return tuple(mask)

    def __distributor_setup__(self, **kwargs):
        grid = kwargs.get('grid')
        # There may or may not be a `Distributor`. In the latter case, the
        # TensorFunction is to be considered "local" to each MPI rank
        return kwargs.get('distributor') if grid is None else grid.distributor

    @property
    def _data_buffer(self):
        """Reference to the data. Unlike :attr:`data` and :attr:`data_with_halo`,
        this *never* returns a view of the data. This method is for internal use only."""
        return self._data_allocated

    @property
    def _mem_external(self):
        return True

    @property
    def grid(self):
        return self._grid

    @property
    def staggered(self):
        return self._staggered

    @cached_property
    def shape(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an :class:`Operator`.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.
        """
        return self.shape_domain

    @cached_property
    def shape_domain(self):
        """
        Shape of the domain region. The domain constitutes the area of the
        data written to by an :class:`Operator`.

        Notes
        -----
        In an MPI context, this is the *local* domain region shape.

        Alias to ``self.shape``.
        """
        return tuple(i - j for i, j in zip(self._shape, self.staggered))

    @cached_property
    def shape_with_halo(self):
        """
        Shape of the domain+outhalo region. The outhalo is the region
        surrounding the domain that may be read by an :class:`Operator`.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.

        Further, note that the outhalo of inner ranks is typically empty, while
        the outhalo of boundary ranks contains a number of elements depending
        on the rank position in the decomposed grid (corner, side, ...).
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_domain,
                                                    self._extent_outhalo))

    _shape_with_outhalo = shape_with_halo

    @cached_property
    def _shape_with_inhalo(self):
        """
        Shape of the domain+inhalo region. The inhalo region comprises the
        outhalo as well as any additional "ghost" layers for MPI halo
        exchanges. Data in the inhalo region are exchanged when running
        :class:`Operator`s to maintain consistent values as in sequential runs.

        Notes
        -----
        Typically, this property won't be used in user code, but it may come
        in handy for testing or debugging
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_domain, self._halo))

    @cached_property
    def shape_allocated(self):
        """
        Shape of the allocated data. It includes the domain and inhalo regions,
        as well as any additional padding surrounding the halo.

        Notes
        -----
        In an MPI context, this is the *local* with_halo region shape.
        """
        return tuple(j + i + k for i, (j, k) in zip(self._shape_with_inhalo,
                                                    self._padding))

    @cached_property
    def shape_global(self):
        """
        Global shape of the domain region. The domain constitutes the area of
        the data written to by an :class:`Operator`.

        Notes
        -----
        In an MPI context, this is the *global* domain region shape, which is
        therefore identical on all MPI ranks.
        """
        if self.grid is None:
            return self.shape
        retval = []
        for d, s in zip(self.dimensions, self.shape):
            size = self.grid.dimension_map.get(d)
            retval.append(size.glb if size is not None else s)
        return tuple(retval)

    _offset_inhalo = AbstractCachedFunction._offset_halo
    _extent_inhalo = AbstractCachedFunction._extent_halo

    @cached_property
    def _extent_outhalo(self):
        """
        The number of points in the outer halo region.
        """
        if self._distributor is None:
            return self._extent_inhalo

        left = [self._distributor.glb_to_loc(d, i, LEFT, strict=False)
                for d, i in zip(self.dimensions, self._extent_inhalo.left)]
        right = [self._distributor.glb_to_loc(d, i, RIGHT, strict=False)
                 for d, i in zip(self.dimensions, self._extent_inhalo.right)]

        Extent = namedtuple('Extent', 'left right')
        extents = tuple(Extent(i, j) for i, j in zip(left, right))

        return EnrichedTuple(*extents, getters=self.dimensions, left=left, right=right)

    @cached_property
    def _mask_modulo(self):
        """
        A boolean mask telling which :class:`Dimension`s support modulo-indexing.
        """
        return tuple(True if i.is_Stepping else False for i in self.dimensions)

    @cached_property
    def _mask_domain(self):
        """
        A mask to access the domain region of the allocated data.
        """
        return tuple(slice(i, -j) if j != 0 else slice(i, None)
                     for i, j in self._offset_domain)

    @cached_property
    def _mask_inhalo(self):
        """
        A mask to access the domain+inhalo region of the allocated data.
        """
        return tuple(slice(i, -j) if j != 0 else slice(i, None)
                     for i, j in self._offset_inhalo)

    @cached_property
    def _mask_outhalo(self):
        """
        A mask to access the domain+outhalo region of the allocated data.
        """
        return tuple(slice(i.start - j.left, i.stop and i.stop + j.right or None)
                     for i, j in zip(self._mask_domain, self._extent_outhalo))

    @cached_property
    def _decomposition(self):
        """
        A tuple of :class:`Decomposition`s, representing the domain
        decomposition.  None is used as a placeholder for non-decomposed
        Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        mapper = {d: self._distributor.decomposition[d] for d in self._dist_dimensions}
        return tuple(mapper.get(d) for d in self.dimensions)

    @cached_property
    def _decomposition_outhalo(self):
        """
        A tuple of :class:`Decomposition`s, representing the domain+outhalo
        decomposition.  None is used as a placeholder for non-decomposed
        Dimensions.
        """
        if self._distributor is None:
            return (None,)*self.ndim
        return tuple(v.reshape(*self._extent_inhalo[d]) if v is not None else v
                     for d, v in zip(self.dimensions, self._decomposition))

    @property
    def data(self):
        """
        The domain data values, as a :class:`numpy.ndarray`.

        Elements are stored in row-major format.

        Notes
        -----
        With this accessor you are claiming that you will modify the values you
        get back. If you only need to look at the values, use :meth:`data_ro`
        instead.
        """
        return self.data_domain

    @property
    @_allocate_memory
    def data_domain(self):
        """
        The domain data values.

        Elements are stored in row-major format.

        Notes
        -----
        Alias to ``self.data``.

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

    @property
    @_allocate_memory
    def data_ro_domain(self):
        """
        A read-only view of the domain data values.
        """
        view = self._data._global(self._mask_domain, self._decomposition)
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_with_halo(self):
        """
        A read-only view of the domain+outhalo data values.
        """
        view = self._data._global(self._mask_outhalo, self._decomposition_outhalo)
        view.setflags(write=False)
        return view

    _data_ro_with_outhalo = data_ro_with_halo

    @property
    @_allocate_memory
    def _data_ro_with_inhalo(self):
        """
        A read-only view of the domain+inhalo data values.

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
        A read-only view of the domain+inhalo+padding data values.

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
        A tuple of slices representing the global indices that logically
        belong to the calling MPI rank.

        Notes
        -----
        Given a Function ``f(x, y)`` with shape ``(nx, ny)``, when *not* using
        MPI this property will return ``(slice(0, nx-1), slice(0, ny-1))``.  On
        the other hand, when MPI is used, the local ranges depend on the domain
        decomposition, which is carried by ``self.grid``.
        """
        if self._distributor is None:
            return tuple(slice(0, s) for s in self.shape)
        else:
            return tuple(self._distributor.glb_slices.get(d, slice(0, s))
                         for s, d in zip(self.shape, self.dimensions))

    @cached_property
    def space_dimensions(self):
        """Tuple of :class:`Dimension`s that define physical space."""
        return tuple(d for d in self.indices if d.is_Space)

    @cached_property
    def _dist_dimensions(self):
        """Tuple of MPI-distributed :class:`Dimension`s."""
        if self._distributor is None:
            return ()
        return tuple(d for d in self.indices if d in self._distributor.dimensions)

    @property
    def initializer(self):
        if self._data is not None:
            return self.data_with_halo.view(np.ndarray)
        else:
            return self._initializer

    @cached_property
    def symbolic_shape(self):
        """
        The symbolic shape of the object. This includes: ::

            * the domain, halo, and padding regions. While halo and padding are
              known quantities (integers), the domain size is represented by a symbol.
            * the shifting induced by the ``staggered`` mask.
        """
        symbolic_shape = super(TensorFunction, self).symbolic_shape
        ret = tuple(Add(i, -j) for i, j in zip(symbolic_shape, self.staggered))
        return EnrichedTuple(*ret, getters=self.dimensions)

    def _halo_exchange(self):
        """Perform the halo exchange with the neighboring processes."""
        if not MPI.Is_initialized() or MPI.COMM_WORLD.size == 1:
            # Nothing to do
            return
        if MPI.COMM_WORLD.size > 1 and self._distributor is None:
            raise RuntimeError("`%s` cannot perform a halo exchange as it has "
                               "no Grid attached" % self.name)
        if self._in_flight:
            raise RuntimeError("`%s` cannot initiate a halo exchange as previous "
                               "exchanges are still in flight" % self.name)
        for i in self.space_dimensions:
            self.__halo_begin_exchange(i)
            self.__halo_end_exchange(i)
        self._is_halo_dirty = False
        assert not self._in_flight

    def __halo_begin_exchange(self, dim):
        """Begin a halo exchange along a given :class:`Dimension`."""
        neighbours = self._distributor.neighbours
        comm = self._distributor.comm
        for i in [LEFT, RIGHT]:
            neighbour = neighbours[dim][i]
            owned_region = self._get_view(OWNED, dim, i)
            halo_region = self._get_view(HALO, dim, i)
            sendbuf = np.ascontiguousarray(owned_region)
            recvbuf = np.ndarray(shape=halo_region.shape, dtype=self.dtype)
            self._in_flight.append((dim, i, recvbuf, comm.Irecv(recvbuf, neighbour)))
            self._in_flight.append((dim, i, None, comm.Isend(sendbuf, neighbour)))

    def __halo_end_exchange(self, dim):
        """End a halo exchange along a given :class:`Dimension`."""
        for d, i, payload, req in list(self._in_flight):
            if d == dim:
                status = MPI.Status()
                req.Wait(status=status)
                if payload is not None and status.source != MPI.PROC_NULL:
                    # The MPI.Request `req` originated from a `comm.Irecv`
                    # Now need to scatter the data to the right place
                    self._get_view(HALO, d, i)[:] = payload
            self._in_flight.remove((d, i, payload, req))

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this function."""
        return (self.name,)

    @memoized_meth
    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param alias: (Optional) name under which to store values.
        """
        key = alias or self
        args = ReducerMap({key.name: self._data_buffer})

        # Collect default dimension arguments from all indices
        for i, s, o in zip(key.indices, self.shape, self.staggered):
            args.update(i._arg_defaults(start=0, size=s+o))

        # Add MPI-related data structures
        if self.grid is not None:
            args.update(self.grid._arg_defaults())

        return args

    def _arg_values(self, **kwargs):
        """
        Returns a map of argument values after evaluating user input. If no
        user input is provided, return a default value.

        :param kwargs: Dictionary of user-provided argument overrides.
        """
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, TensorFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self).reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {self.name: new}
                # Add value overrides for all associated dimensions
                for i, s, o in zip(self.indices, new.shape, self.staggered):
                    values.update(i._arg_defaults(size=s+o-sum(self._offset_domain[i])))
                # Add MPI-related data structures
                if self.grid is not None:
                    values.update(self.grid._arg_defaults())
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_check(self, args, intervals):
        """
        Check that ``args`` contains legal runtime values bound to ``self``.

        :raises InvalidArgument: If, given the runtime arguments ``args``, an
                                 out-of-bounds access will be performed.
        """
        if self.name not in args:
            raise InvalidArgument("No runtime value for `%s`" % self.name)
        key = args[self.name]
        if len(key.shape) != self.ndim:
            raise InvalidArgument("Shape %s of runtime value `%s` does not match "
                                  "dimensions %s" % (key.shape, self.name, self.indices))
        if key.dtype != self.dtype:
            warning("Data type %s of runtime value `%s` does not match the "
                    "Function data type %s" % (key.dtype, self.name, self.dtype))
        for i, s in zip(self.indices, key.shape):
            i._arg_check(args, s, intervals[i])

    # Pickling support
    _pickle_kwargs = AbstractCachedFunction._pickle_kwargs +\
        ['grid', 'staggered', 'initializer']


class Function(TensorFunction, Differentiable):
    """A :class:`TensorFunction` providing operations to express
    finite-difference approximation. A ``Function`` encapsulates
    space-varying data; for time-varying data, use :class:`TimeFunction`.

    :param name: Name of the symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param space_order: Discretisation order for space derivatives. By default,
                        ``space_order`` points are available on both sides of
                        a generic point of interest, including those on the grid
                        border. Sometimes, fewer points may be necessary; in
                        other cases, depending on the PDE being approximated,
                        more points may be necessary. In such cases, one
                        can pass a 3-tuple ``(o, lp, rp)`` instead of a single
                        integer representing the discretization order. Here,
                        ``o`` is the discretization order, while ``lp`` and ``rp``
                        indicate how many points are expected on left (``lp``)
                        and right (``rp``) of a point of interest.
    :param shape: (Optional) shape of the domain region in grid points.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param staggered: (Optional) a :class:`Dimension`, or a tuple of :class:`Dimension`s,
                      or a :class:`Stagger`, defining how the function is staggered.
                      For example:
                      * ``staggered=x`` entails discretization on x edges,
                      * ``staggered=y`` entails discretization on y edges,
                      * ``staggered=(x, y)`` entails discretization on xy facets,
                      * ``staggered=NODE`` entails discretization on node,
                      * ``staggerd=CELL`` entails discretization on cell.
    :param padding: (Optional) allocate extra grid points at a space dimension
                    boundary. These may be used for data alignment. Defaults to 0.
                    In alternative to an integer, a tuple, indicating the padding
                    in each dimension, may be passed; in this case, an error is
                    raised if such tuple has fewer entries then the number of space
                    dimensions.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

    .. note::

       :class:`Function` objects are assumed to be constant in time
       and therefore do not support time derivatives. Use
       :class:`TimeFunction` for time-varying grid data.
    """

    is_Function = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(Function, self).__init__(*args, **kwargs)

            # Space order
            space_order = kwargs.get('space_order', 1)
            if isinstance(space_order, int):
                self._space_order = space_order
            elif isinstance(space_order, tuple) and len(space_order) == 3:
                self._space_order, _, _ = space_order
            else:
                raise TypeError("`space_order` must be int or 3-tuple of ints")

            # Dynamically add derivative short-cuts
            self._fd = generate_fd_shortcuts(self)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        dimensions = kwargs.get('dimensions')
        if grid is None:
            if dimensions is None:
                raise TypeError("Need either `grid` or `dimensions`")
        elif dimensions is None:
            dimensions = grid.dimensions
        return dimensions

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
                    size = grid.dimension_map[d]
                    if size.glb != s and s is not None:
                        raise ValueError("Dimension `%s` is given size `%d`, "
                                         "while `grid` says `%s` has size `%d` "
                                         % (d, s, d, size.glb))
                    else:
                        loc_shape.append(size.loc)
                else:
                    loc_shape.append(s)
            shape = tuple(loc_shape)
        return shape

    def __halo_setup__(self, **kwargs):
        halo = kwargs.get('halo')
        if halo is not None:
            return halo
        else:
            space_order = kwargs.get('space_order', 1)
            if isinstance(space_order, int):
                halo = (space_order, space_order)
            elif isinstance(space_order, tuple) and len(space_order) == 3:
                _, left_points, right_points = space_order
                halo = (left_points, right_points)
            else:
                raise TypeError("`space_order` must be int or 3-tuple of ints")
            return tuple(halo if i.is_Space else (0, 0) for i in self.indices)

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding', 0)
        if isinstance(padding, int):
            return tuple((padding,)*2 for i in range(self.ndim))
        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            return tuple((i,)*2 if isinstance(i, int) else i for i in padding)
        else:
            raise TypeError("`padding` must be int or %d-tuple of ints" % self.ndim)

    @property
    def space_order(self):
        return self._space_order

    def sum(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the sum of ``p`` points
        along the spatial dimensions ``dims``.

        :param p: (Optional) the number of summands. Defaults to the
                  halo extent.
        :param dims: (Optional) the :class:`Dimension`s along which the
                     sum is computed. Defaults to ``self``'s spatial
                     dimensions.
        """
        points = []
        for d in (as_tuple(dims) or self.space_dimensions):
            if p is None:
                lp = self._extent_inhalo[d].left
                rp = self._extent_inhalo[d].right
            else:
                lp = p // 2 + p % 2
                rp = p // 2
            indices = [d - i for i in range(lp, 0, -1)]
            indices.extend([d + i for i in range(rp)])
            points.extend([self.subs(d, i) for i in indices])
        return sum(points)

    def avg(self, p=None, dims=None):
        """
        Generate a symbolic expression computing the average of ``p`` points
        along the spatial dimensions ``dims``.

        :param p: (Optional) the number of summands. Defaults to the
                  halo extent.
        :param dims: (Optional) the :class:`Dimension`s along which the
                     sum is computed. Defaults to ``self``'s spatial
                     dimensions.
        """
        tot = self.sum(p, dims)
        return tot / len(tot.args)

    # Pickling support
    _pickle_kwargs = TensorFunction._pickle_kwargs +\
        ['space_order', 'shape_global', 'dimensions']


class TimeFunction(Function):
    """
    A special :class:`Function` encapsulating time-varying data.

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param grid: :class:`Grid` object from which to infer the data shape
                 and :class:`Dimension` indices.
    :param space_order: Discretisation order for space derivatives. By default,
                        ``space_order`` points are available on both sides of
                        a generic point of interest, including those on the grid
                        border. Sometimes, fewer points may be necessary; in
                        other cases, depending on the PDE being approximated,
                        more points may be necessary. In such cases, one
                        can pass a 3-tuple ``(o, lp, rp)`` instead of a single
                        integer representing the discretization order. Here,
                        ``o`` is the discretization order, while ``lp`` and ``rp``
                        indicate how many points are expected on left (``lp``)
                        and right (``rp``) of a point of interest.
    :param time_order: Discretization order for time derivatives.
    :param shape: (Optional) shape of the domain region in grid points.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param save: (Optional) defaults to `None`, which indicates the use of
                 alternating buffers. This enables cyclic writes to the
                 TimeFunction. For example, if the TimeFunction ``u(t, x)`` has
                 shape (3, 100), then, in an :class:`Operator`, ``t`` will
                 assume the values ``1, 2, 0, 1, 2, 0, 1, ...`` (note that the
                 very first value depends on the stencil equation in which
                 ``u`` is written.). The default size of the time buffer when
                 ``save=None`` is ``time_order + 1``.  To specify a different
                 size for the time buffer, one should use the syntax
                 ``save=Buffer(mysize)``. Alternatively, if all of the
                 intermediate results are required (or, simply, to forbid the
                 usage of an alternating buffer), an explicit value for ``save``
                 (i.e., an integer) must be provided.
    :param time_dim: (Optional) the :class:`Dimension` object to use to represent
                     time in this symbol. Defaults to the time dimension provided
                     by the :class:`Grid`.
    :param staggered: (Optional) a :class:`Dimension`, or a tuple of :class:`Dimension`s,
                      or a :class:`Stagger`, defining how the function is staggered.
                      For example:
                      * ``staggered=x`` entails discretization on x edges,
                      * ``staggered=y`` entails discretization on y edges,
                      * ``staggered=(x, y)`` entails discretization on xy facets,
                      * ``staggered=NODE`` entails discretization on node,
                      * ``staggerd=CELL`` entails discretization on cell.
    :param padding: (Optional) allocate extra grid points at a space dimension
                    boundary. These may be used for data alignment. Defaults to 0.
                    In alternative to an integer, a tuple, indicating the padding
                    in each dimension, may be passed; in this case, an error is
                    raised if such tuple has fewer entries then the number of
                    space dimensions.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses ``*args`` to (re-)create the dimension arguments of the
        symbolic function.

    .. note::

       If the parameter ``grid`` is provided, the values for ``shape``,
       ``dimensions`` and ``dtype`` will be derived from it.

       The parameter ``shape`` should only define the spatial shape of
       the grid. The temporal dimension will be inserted automatically
       as the leading dimension, according to the ``time_dim``,
       ``time_order`` and whether we want to write intermediate
       timesteps in the buffer. The same is true for explicitly
       provided dimensions, which will be added to the automatically
       derived time dimensions symbol. For example:

       .. code-block:: python

          In []: TimeFunction(name="a", dimensions=(x, y, z))
          Out[]: a(t, x, y, z)

          In []: TimeFunction(name="a", shape=(20, 30))
          Out[]: a(t, x, y)
    """

    is_TimeFunction = True

    _time_position = 0
    """Position of time index among the function indices."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.time_dim = kwargs.get('time_dim', self.indices[self._time_position])
            self._time_order = kwargs.get('time_order', 1)
            super(TimeFunction, self).__init__(*args, **kwargs)

            # Check we won't allocate too much memory for the system
            available_mem = virtual_memory().available
            if np.dtype(self.dtype).itemsize * self.size > available_mem:
                warning("Trying to allocate more memory for symbol %s " % self.name +
                        "than available on physical device, this will start swapping")
            if not isinstance(self.time_order, int):
                raise TypeError("`time_order` must be int")

            self.save = kwargs.get('save')

    @classmethod
    def __indices_setup__(cls, **kwargs):
        dimensions = kwargs.get('dimensions')
        if dimensions is None:
            save = kwargs.get('save')
            grid = kwargs.get('grid')
            time_dim = kwargs.get('time_dim')

            if time_dim is None:
                time_dim = grid.time_dim if isinstance(save, int) else grid.stepping_dim
            elif not (isinstance(time_dim, Dimension) and time_dim.is_Time):
                raise TypeError("`time_dim` must be a time dimension")

            dimensions = list(Function.__indices_setup__(**kwargs))
            dimensions.insert(cls._time_position, time_dim)

        return tuple(dimensions)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        save = kwargs.get('save') or None  # Force to None if 0/False/None/...
        shape = kwargs.get('shape')
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
            elif isinstance(save, int):
                shape.insert(cls._time_position, save)
            else:
                raise TypeError("`save` can be None, int or Buffer, not %s" % type(save))
        return tuple(shape)

    @property
    def time_order(self):
        return self._time_order

    @property
    def forward(self):
        """Symbol for the time-forward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[self._time_position]

        return self.subs(_t, _t + i * _t.spacing)

    @property
    def backward(self):
        """Symbol for the time-backward state of the function"""
        i = int(self.time_order / 2) if self.time_order >= 2 else 1
        _t = self.indices[self._time_position]

        return self.subs(_t, _t - i * _t.spacing)

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    @property
    def _time_buffering(self):
        return not is_integer(self.save)

    @property
    def _time_buffering_default(self):
        return self._time_buffering and not isinstance(self.save, Buffer)

    def _arg_check(self, args, intervals):
        super(TimeFunction, self)._arg_check(args, intervals)
        key_time_size = args[self.name].shape[self._time_position]
        if self._time_buffering and self._time_size != key_time_size:
            raise InvalidArgument("Expected `time_size=%d` for runtime "
                                  "value `%s`, found `%d` instead"
                                  % (self._time_size, self.name, key_time_size))

    # Pickling support
    _pickle_kwargs = Function._pickle_kwargs + ['time_order', 'save']


class SubFunction(Function):
    """
    A :class:`Function` that is bound to another "parent" TensorFunction.

    A SubFunction hands control of argument binding and halo exchange to its
    parent TensorFunction.
    """

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SubFunction, self).__init__(*args, **kwargs)
            self._parent = kwargs['parent']

    def _halo_exchange(self):
        return

    def _arg_values(self, **kwargs):
        if self.name in kwargs:
            raise RuntimeError("`%s` is a SubFunction, so it can't be assigned "
                               "a value dynamically" % self.name)
        else:
            return self._parent._arg_defaults(alias=self._parent).reduce_all()

    @property
    def parent(self):
        return self._parent

    _pickle_kwargs = Function._pickle_kwargs + ['parent']


class AbstractSparseFunction(TensorFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """

    _sparse_position = -1
    """Position of sparse index among the function indices."""

    _radius = 0
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ()
    """:class:`SubFunction`s encapsulated within this AbstractSparseFunction."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseFunction, self).__init__(*args, **kwargs)
            self._npoint = kwargs['npoint']
            self._space_order = kwargs.get('space_order', 0)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """
        Return the default dimension indices for a given data shape.
        """
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions
        else:
            return (Dimension(name='p_%s' % kwargs["name"]),)

    @classmethod
    def __shape_setup__(cls, **kwargs):
        grid = kwargs.get('grid')
        # A Grid must have been provided
        if grid is None:
            raise TypeError('Need `grid` argument')
        shape = kwargs.get('shape')
        npoint = kwargs['npoint']
        if shape is None:
            glb_npoint = SparseDistributor.decompose(npoint, grid.distributor)
            shape = (glb_npoint[grid.distributor.myrank],)
        return shape

    @property
    def npoint(self):
        return self.shape[self._sparse_position]

    @property
    def space_order(self):
        return self._space_order

    @property
    def _sparse_dim(self):
        return self.dimensions[self._sparse_position]

    @property
    def gridpoints(self):
        """
        The *reference* grid point corresponding to each sparse point.

        Notes
        -----
        When using MPI, this property refers to the *physically* owned
        sparse points.
        """
        raise NotImplementedError

    @property
    def _support(self):
        """
        The grid points surrounding each sparse point within the radius of self's
        injection/interpolation operators.
        """
        ret = []
        for i in self.gridpoints:
            support = [range(max(0, j - self._radius + 1), min(M, j + self._radius + 1))
                       for j, M in zip(i, self.grid.shape)]
            ret.append(tuple(product(*support)))
        return ret

    @property
    def _dist_datamap(self):
        """
        Mapper ``M : MPI rank -> required sparse data``.
        """
        ret = {}
        for i, s in enumerate(self._support):
            # Sparse point `i` is "required" by the following ranks
            for r in self.grid.distributor.glb_to_rank(s):
                ret.setdefault(r, []).append(i)
        return {k: filter_ordered(v) for k, v in ret.items()}

    @property
    def _dist_scatter_mask(self):
        """
        A mask to index into ``self.data``, which creates a new data array that
        logically contains N consecutive groups of sparse data values, where N
        is the number of MPI ranks. The i-th group contains the sparse data
        values accessible by the i-th MPI rank.  Thus, sparse data values along
        the boundary of two or more MPI ranks are duplicated.
        """
        dmap = self._dist_datamap
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        ret = [slice(None) for i in range(self.ndim)]
        ret[self._sparse_position] = mask
        return ret

    @property
    def _dist_subfunc_scatter_mask(self):
        """
        This method is analogous to :meth:`_dist_scatter_mask`, although
        the mask is now suitable to index into self's SubFunctions, rather
        than into ``self.data``.
        """
        return self._dist_scatter_mask[self._sparse_position]

    @property
    def _dist_gather_mask(self):
        """
        A mask to index into the ``data`` received upon returning from
        ``self._dist_alltoall``. This mask creates a new data array in which
        duplicate sparse data values have been discarded. The resulting data
        array can thus be used to populate ``self.data``.
        """
        ret = list(self._dist_scatter_mask)
        mask = ret[self._sparse_position]
        ret[self._sparse_position] = [mask.tolist().index(i)
                                      for i in filter_ordered(mask)]
        return ret

    @property
    def _dist_count(self):
        """
        A 2-tuple of comm-sized iterables, which tells how many sparse points
        is this MPI rank expected to send/receive to/from each other MPI rank.
        """
        dmap = self._dist_datamap
        comm = self.grid.distributor.comm

        ssparse = np.array([len(dmap.get(i, [])) for i in range(comm.size)], dtype=int)
        rsparse = np.empty(comm.size, dtype=int)
        comm.Alltoall(ssparse, rsparse)

        return ssparse, rsparse

    @cached_property
    def _dist_reorder_mask(self):
        """
        An ordering mask that puts ``self._sparse_position`` at the front.
        """
        ret = (self._sparse_position,)
        ret += tuple(i for i, d in enumerate(self.indices) if d is not self._sparse_dim)
        return ret

    @property
    def _dist_alltoall(self):
        """
        The metadata necessary to perform an ``MPI_Alltoallv`` distributing the
        sparse data values across the MPI ranks needing them.
        """
        ssparse, rsparse = self._dist_count

        # Per-rank shape of send/recv data
        sshape = []
        rshape = []
        for s, r in zip(ssparse, rsparse):
            handle = list(self.shape)
            handle[self._sparse_position] = s
            sshape.append(tuple(handle))

            handle = list(self.shape)
            handle[self._sparse_position] = r
            rshape.append(tuple(handle))

        # Per-rank count of send/recv data
        scount = [prod(i) for i in sshape]
        rcount = [prod(i) for i in rshape]

        # Per-rank displacement of send/recv data (it's actually all contiguous,
        # but the Alltoallv needs this information anyway)
        sdisp = np.concatenate([[0], np.cumsum(scount)[:-1]])
        rdisp = np.concatenate([[0], tuple(np.cumsum(rcount))[:-1]])

        # Total shape of send/recv data
        sshape = list(self.shape)
        sshape[self._sparse_position] = sum(ssparse)
        rshape = list(self.shape)
        rshape[self._sparse_position] = sum(rsparse)

        # May have to swap axes, as `MPI_Alltoallv` expects contiguous data, and
        # the sparse dimension may not be the outermost
        sshape = [sshape[i] for i in self._dist_reorder_mask]
        rshape = [rshape[i] for i in self._dist_reorder_mask]

        return sshape, scount, sdisp, rshape, rcount, rdisp

    @property
    def _dist_subfunc_alltoall(self):
        """
        The metadata necessary to perform an ``MPI_Alltoallv`` distributing
        self's SubFunction values across the MPI ranks needing them.
        """
        raise NotImplementedError

    def _dist_scatter(self):
        """
        Return a ``numpy.ndarray`` containing up-to-date data values belonging
        to the calling MPI rank. A data value belongs to a given MPI rank R
        if its coordinates fall within R's local domain.
        """
        raise NotImplementedError

    def _dist_gather(self, data):
        """
        Return a ``numpy.ndarray`` containing up-to-date data and coordinate values
        suitable for insertion into ``self.data``.
        """
        raise NotImplementedError

    @memoized_meth
    def _arg_defaults(self, alias=None):
        key = alias or self
        mapper = {self: key}
        mapper.update({getattr(self, i): getattr(key, i) for i in self._sub_functions})
        args = ReducerMap()

        # Add in the sparse data (as well as any SubFunction data) belonging to
        # self's local domain only
        for k, v in self._dist_scatter().items():
            args[mapper[k].name] = v
            for i, s, o in zip(mapper[k].indices, v.shape, k.staggered):
                args.update(i._arg_defaults(start=0, size=s+o))

        # Add MPI-related data structures
        args.update(self.grid._arg_defaults())

        return args

    def _arg_values(self, **kwargs):
        # Add value override for own data if it is provided, otherwise
        # use defaults
        if self.name in kwargs:
            new = kwargs.pop(self.name)
            if isinstance(new, AbstractSparseFunction):
                # Set new values and re-derive defaults
                values = new._arg_defaults(alias=self).reduce_all()
            else:
                # We've been provided a pure-data replacement (array)
                values = {}
                for k, v in self._dist_scatter(new).items():
                    values[k.name] = v
                    for i, s, o in zip(k.indices, v.shape, k.staggered):
                        values.update(i._arg_defaults(size=s+o-sum(k._offset_domain[i])))
                # Add MPI-related data structures
                values.update(self.grid._arg_defaults())
        else:
            values = self._arg_defaults(alias=self).reduce_all()

        return values

    def _arg_apply(self, data, alias=None):
        key = alias if alias is not None else self
        if isinstance(key, AbstractSparseFunction):
            key._dist_gather(data)
        elif self.grid.distributor.nprocs > 1:
            raise NotImplementedError("Don't know how to gather data from an "
                                      "object of type `%s`" % type(key))

    # Pickling support
    _pickle_kwargs = TensorFunction._pickle_kwargs + ['npoint', 'space_order']


class AbstractSparseTimeFunction(AbstractSparseFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    time functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """

    _time_position = 0
    """Position of time index among the function indices."""

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseTimeFunction, self).__init__(*args, **kwargs)
            self.time_dim = self.indices[self._time_position]
            self._time_order = kwargs.get('time_order', 1)
            if not isinstance(self.time_order, int):
                raise ValueError("`time_order` must be int")

    @classmethod
    def __indices_setup__(cls, **kwargs):
        """
        Return the default dimension indices for a given data shape.
        """
        dimensions = kwargs.get('dimensions')
        if dimensions is not None:
            return dimensions
        else:
            return (kwargs['grid'].time_dim, Dimension(name='p_%s' % kwargs["name"]))

    @classmethod
    def __shape_setup__(cls, **kwargs):
        shape = kwargs.get('shape')
        if shape is None:
            nt = kwargs.get('nt')
            if not isinstance(nt, int):
                raise TypeError('Need `nt` int argument')
            if nt <= 0:
                raise ValueError('`nt` must be > 0')

            shape = list(AbstractSparseFunction.__shape_setup__(**kwargs))
            shape.insert(cls._time_position, nt)

        return tuple(shape)

    @property
    def nt(self):
        return self.shape[self._time_position]

    @property
    def time_order(self):
        return self._time_order

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['nt', 'time_order']


class SparseFunction(AbstractSparseFunction, Differentiable):
    """
    A special :class:`TensorFunction` representing a set of sparse point
    objects that are not aligned with the computational grid.

    A :class:`SparseFunction` provides symbolic interpolation routines
    to convert between grid-aligned :class:`Function` objects and sparse
    data points. These are based upon standard [bi,tri]linear interpolation.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: (Optional) discretisation order for space derivatives.
    :param shape: (Optional) shape of the function. Defaults to ``(npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.

    .. note::

        About SparseFunction and MPI. There is a clear difference between: ::

            * Where the sparse points *physically* live, i.e., on which MPI rank. This
              depends on the user code, particularly on how the data is set up.
            * and which MPI rank *logically* owns a given sparse point. The logical
              ownership depends on where the sparse point is located within `self.grid`.

        Right before running an Operator (i.e., upon a call to `op.apply()`), a
        SparseFunction `scatter`s its physically owned sparse points so that each
        MPI rank gets temporary access to all of its logically owned sparse points.
        A `gather` operation, executed before returning control to user-land,
        updates the physically owned sparse points in `self.data` by collecting
        the values computed during `op.apply()` from different MPI ranks.
    """

    is_SparseFunction = True

    _radius = 1
    """The radius of the stencil operators provided by the SparseFunction."""

    _sub_functions = ('coordinates',)

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SparseFunction, self).__init__(*args, **kwargs)

            # Set up sparse point coordinates
            coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
            if isinstance(coordinates, Function):
                self._coordinates = coordinates
            else:
                dimensions = (self.indices[-1], Dimension(name='d'))
                # Only retain the local data region
                if coordinates is not None:
                    coordinates = np.array(coordinates)
                self._coordinates = SubFunction(name='%s_coords' % self.name, parent=self,
                                                dtype=self.dtype, dimensions=dimensions,
                                                shape=(self.npoint, self.grid.dim),
                                                space_order=0, initializer=coordinates,
                                                distributor=self._distributor)
                if self.npoint == 0:
                    # This is a corner case -- we might get here, for example, when
                    # running with MPI and some processes get 0-size arrays after
                    # domain decomposition. We "touch" the data anyway to avoid the
                    # case ``self._data is None``
                    self.coordinates.data

    def __distributor_setup__(self, **kwargs):
        """A `SparseDistributor` handles the SparseFunction decomposition based on
        physical ownership, and allows to convert between global and local indices."""
        return SparseDistributor(kwargs['npoint'], self._sparse_dim,
                                 kwargs['grid'].distributor)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def coordinates_data(self):
        return self.coordinates.data.view(np.ndarray)

    @property
    def _coefficients(self):
        """Symbolic expression for the coefficients for sparse point
        interpolation according to:
        https://en.wikipedia.org/wiki/Bilinear_interpolation.

        :returns: List of coefficients, eg. [b_11, b_12, b_21, b_22]
        """
        # Grid indices corresponding to the corners of the cell ie x1, y1, z1
        indices1 = tuple(sympy.symbols('%s1' % d) for d in self.grid.dimensions)
        indices2 = tuple(sympy.symbols('%s2' % d) for d in self.grid.dimensions)
        # 1, x1, y1, z1, x1*y1, ...
        indices = list(powerset(indices1))
        indices[0] = (1,)
        point_sym = list(powerset(self._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self._point_increments:
            # substitute x1 by x2 if increment in that dimension
            subs = dict((indices1[d], indices2[d] if i[d] == 1 else indices1[d])
                        for d in range(len(i)))
            A += [[1] + [a.subs(subs) for a in ref_A[1:]]]

        A = sympy.Matrix(A)
        # Coordinate values of the sparse point
        p = sympy.Matrix([[np.prod(ind)] for ind in point_sym])

        # reference cell x1:0, x2:h_x
        left = dict((a, 0) for a in indices1)
        right = dict((b, dim.spacing) for b, dim in zip(indices2, self.grid.dimensions))
        reference_cell = {**left, **right}
        # Substitute in interpolation matrix
        A = A.subs(reference_cell)
        return A.inv().T * p

    @cached_property
    def _point_symbols(self):
        """Symbol for coordinate value in each dimension of the point."""
        return tuple(Scalar(name='p%s' % d, dtype=self.dtype)
                     for d in self.grid.dimensions)

    @cached_property
    def _point_increments(self):
        """Index increments in each dimension for each point symbol."""
        return tuple(product(range(2), repeat=self.grid.dim))

    @cached_property
    def _coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension."""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @cached_property
    def _coordinate_indices(self):
        """Symbol for each grid index according to the coordinates."""
        indices = self.grid.dimensions
        return tuple([INT(sympy.Function('floor')((c - o) / i.spacing))
                      for c, o, i in zip(self._coordinate_symbols, self.grid.origin,
                                         indices[:self.grid.dim])])

    @cached_property
    def _coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point."""
        indices = self.grid.dimensions
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i in zip(self._coordinate_symbols,
                                              self.grid.origin,
                                              self._coordinate_indices,
                                              indices[:self.grid.dim])])

    @memoized_meth
    def _index_matrix(self, offset):
        # Note about the use of *memoization*
        # Since this method is called by `_interpolation_indices`, using
        # memoization avoids a proliferation of symbolically identical
        # ConditionalDimensions for a given set of indirection indices

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self._coordinate_indices))
                        for inc in self._point_increments]

        # A unique symbol for each indirection index
        indices = filter_ordered(flatten(index_matrix))
        points = OrderedDict([(p, Symbol(name='ii_%s_%d' % (self.name, i)))
                              for i, p in enumerate(indices)])

        return index_matrix, points

    def _interpolation_indices(self, variables, offset=0):
        """Generate interpolation indices for the :class:`TensorFunction`s
        in ``variables``."""
        index_matrix, points = self._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                lb = sympy.And(p >= d.symbolic_start - self._radius, evaluate=False)
                ub = sympy.And(p <= d.symbolic_end + self._radius, evaluate=False)
                condition = sympy.And(lb, ub, evaluate=False)
                mapper[d] = ConditionalDimension(p.name, self._sparse_dim,
                                                 condition=condition, indirect=True)

            # Track Indexed substitutions
            idx_subs.append(OrderedDict([(v, v.subs(mapper)) for v in variables
                                         if v.function is not self]))

        # Equations for the indirection dimensions
        eqns = [Eq(v, k) for k, v in points.items()]
        # Equations (temporaries) for the coefficients
        eqns.extend([Eq(p, c) for p, c in
                     zip(self._point_symbols, self._coordinate_bases)])

        return idx_subs, eqns

    @property
    def gridpoints(self):
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        ret = []
        for coords in self.coordinates.data._local:
            ret.append(tuple(int(sympy.floor((c - o.data)/i.spacing.data)) for c, o, i in
                             zip(coords, self.grid.origin, self.grid.dimensions)))
        return ret

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : sympy.Expr
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        variables = list(retrieve_function_carriers(expr))

        # List of indirection indices for all adjacent grid points
        idx_subs, eqns = self._interpolation_indices(variables, offset)

        # Substitute coordinate base symbols into the coefficients
        args = [expr.subs(v_sub) * b.subs(v_sub)
                for b, v_sub in zip(self._coefficients, idx_subs)]

        # Accumulate point-wise contributions into a temporary
        rhs = Scalar(name='sum', dtype=self.dtype)
        summands = [Eq(rhs, 0.)] + [Inc(rhs, i) for i in args]

        # Write/Incr `self`
        lhs = self.subs(self_subs)
        last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

        return eqns + summands + last

    def inject(self, field, expr, offset=0):
        """Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : sympy.Expr
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """

        variables = list(retrieve_function_carriers(expr)) + [field]

        # List of indirection indices for all adjacent grid points
        idx_subs, eqns = self._interpolation_indices(variables, offset)

        # Substitute coordinate base symbols into the coefficients
        eqns.extend([Inc(field.subs(vsub), expr.subs(vsub) * b)
                     for b, vsub in zip(self._coefficients, idx_subs)])

        return eqns

    def guard(self, expr=None, offset=0):
        """
        Generate a guarded expression, that is expressions that are
        evaluated by an Operator only if certain conditions are met.
        The introduced condition, here, is that all grid points in the
        support of a sparse value must fall within the grid domain (i.e.,
        *not* on the halo).

        Parameters
        ----------
        expr : sympy.Expr, optional
            Input expression, from which the guarded expression is derived.
            If not specified, defaults to ``self``.
        offset : int, optional
            Relax the guard condition by introducing a tolerance offset.
        """
        _, points = self._index_matrix(offset)

        # Guard through ConditionalDimension
        conditions = {}
        for d, idx in zip(self.grid.dimensions, self._coordinate_indices):
            p = points[idx]
            lb = sympy.And(p >= d.symbolic_start - offset, evaluate=False)
            ub = sympy.And(p <= d.symbolic_end + offset, evaluate=False)
            conditions[p] = sympy.And(lb, ub, evaluate=False)
        condition = sympy.And(*conditions.values(), evaluate=False)
        cd = ConditionalDimension("%s_g" % self._sparse_dim, self._sparse_dim,
                                  condition=condition)

        if expr is None:
            out = self.indexify().xreplace({self._sparse_dim: cd})
        else:
            functions = {f for f in retrieve_function_carriers(expr)
                         if f.is_SparseFunction}
            out = indexify(expr).xreplace({f._sparse_dim: cd for f in functions})

        # Equations for the indirection dimensions
        eqns = [Eq(v, k) for k, v in points.items() if v in conditions]

        return out, eqns

    @cached_property
    def _decomposition(self):
        mapper = {self._sparse_dim: self._distributor.decomposition[self._sparse_dim]}
        return tuple(mapper.get(d) for d in self.dimensions)

    @property
    def _dist_subfunc_alltoall(self):
        ssparse, rsparse = self._dist_count

        # Per-rank shape of send/recv `coordinates`
        sshape = [(i, self.grid.dim) for i in ssparse]
        rshape = [(i, self.grid.dim) for i in rsparse]

        # Per-rank count of send/recv `coordinates`
        scount = [prod(i) for i in sshape]
        rcount = [prod(i) for i in rshape]

        # Per-rank displacement of send/recv `coordinates` (it's actually all
        # contiguous, but the Alltoallv needs this information anyway)
        sdisp = np.concatenate([[0], np.cumsum(scount)[:-1]])
        rdisp = np.concatenate([[0], tuple(np.cumsum(rcount))[:-1]])

        # Total shape of send/recv `coordinates`
        sshape = list(self.coordinates.shape)
        sshape[0] = sum(ssparse)
        rshape = list(self.coordinates.shape)
        rshape[0] = sum(rsparse)

        return sshape, scount, sdisp, rshape, rcount, rdisp

    def _dist_scatter(self, data=None):
        data = data if data is not None else self.data._local
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.coordinates: self.coordinates.data}

        comm = distributor.comm
        mpitype = MPI._typedict[np.dtype(self.dtype).char]

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = data[self._dist_scatter_mask]
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        # Send out the sparse point values
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_alltoall
        scattered = np.empty(shape=rshape, dtype=self.dtype)
        comm.Alltoallv([data, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        data = scattered
        # Unpack data values so that they follow the expected storage layout
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))

        # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
        coords = self.coordinates.data._local[self._dist_subfunc_scatter_mask]
        # Send out the sparse point coordinates
        _, scount, sdisp, rshape, rcount, rdisp = self._dist_subfunc_alltoall
        scattered = np.empty(shape=rshape, dtype=self.coordinates.dtype)
        comm.Alltoallv([coords, scount, sdisp, mpitype],
                       [scattered, rcount, rdisp, mpitype])
        coords = scattered

        # Translate global coordinates into local coordinates
        coords = coords - np.array(self.grid.origin_offset, dtype=self.dtype)

        return {self: data, self.coordinates: coords}

    def _dist_gather(self, data):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        comm = distributor.comm

        # Pack sparse data values so that they can be sent out via an Alltoallv
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        # Send back the sparse point values
        sshape, scount, sdisp, _, rcount, rdisp = self._dist_alltoall
        gathered = np.empty(shape=sshape, dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, rcount, rdisp, mpitype],
                       [gathered, scount, sdisp, mpitype])
        data = gathered
        # Unpack data values so that they follow the expected storage layout
        data = np.ascontiguousarray(np.transpose(data, self._dist_reorder_mask))
        self._data[:] = data[self._dist_gather_mask]

        # Note: this method "mirrors" `_dist_scatter`: a sparse point that is sent
        # in `_dist_scatter` is here received; a sparse point that is received in
        # `_dist_scatter` is here sent. However, the `coordinates` SubFunction
        # values are not distributed, as this is a read-only field.

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['coordinates_data']


class SparseTimeFunction(AbstractSparseTimeFunction, SparseFunction):
    """
    A time-dependent :class:`SparseFunction`.

    :param name: Name of the function.
    :param nt: Size of the time dimension for point data.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: (Optional) coordinate data for the sparse points.
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) Data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.
    :param allocator: (Optional) an object of type :class:`MemoryAllocator` to
                      specify where to allocate the function data when running
                      on a NUMA architecture. Refer to ``default_allocator()``'s
                      __doc__ for more information about possible allocators.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_SparseTimeFunction = True

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param increment: (Optional) if True, perform an increment rather
                          than an assignment. Defaults to False.
        """
        # Apply optional time symbol substitutions to expr
        subs = {}
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs({time: u_t, t: u_t})

        if p_t is not None:
            subs = {self.time_dim: p_t}

        return super(SparseTimeFunction, self).interpolate(expr, offset=offset,
                                                           increment=increment,
                                                           self_subs=subs)

    def inject(self, field, expr, offset=0, u_t=None, p_t=None):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.time_dim, u_t)
        if p_t is not None:
            expr = expr.subs(self.time_dim, p_t)

        return super(SparseTimeFunction, self).inject(field, expr, offset=offset)

    # Pickling support
    _pickle_kwargs = AbstractSparseTimeFunction._pickle_kwargs +\
        SparseFunction._pickle_kwargs


class PrecomputedSparseFunction(AbstractSparseFunction):
    """
    A specialised type of SparseFunction where the interpolation is externally defined.
    Currently, this means that the grid points and associated coefficients for each
    sparse point is precomputed at the time this object is being created.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param r: The number of gridpoints in each dimension to interpolate a single sparse
              point to. e.g. 2 for linear interpolation.
    :param gridpoints: The *reference* grid point corresponding to each sparse point.
                       Of all the gridpoints that one sparse point would be interpolated
                       to, this is the grid point closest to the origin, i.e. the one
                       with the lowest value of each coordinate dimension. Must be a
                       two-dimensional array of shape [npoint][grid.ndim].
    :param coefficients: An array containing the coefficient for each of the r^2 (2D) or
                         r^3 (3D) gridpoints that each sparsefunction will be interpolated
                         to. The coefficient is split across the n dimensions such that
                         the contribution of the point (i, j, k) will be multiplied by
                         coefficients[..., i]*coefficients[..., j]*coefficients[...,k]. So
                         for r=6, we will store 18 coefficients per sparse point (instead
                         of potentially 216). Shape must be [npoint][grid.ndim][r].
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_PrecomputedSparseFunction = True

    _sub_functions = ('gridpoints', 'coefficients')

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PrecomputedSparseFunction, self).__init__(*args, **kwargs)

            # Grid points per sparse point (2 in the case of bilinear and trilinear)
            r = kwargs.get('r')
            if not isinstance(r, int):
                raise TypeError('Need `r` int argument')
            if r <= 0:
                raise ValueError('`r` must be > 0')
            self.r = r

            gridpoints = SubFunction(name="%s_gridpoints" % self.name, dtype=np.int32,
                                     dimensions=(self.indices[-1], Dimension(name='d')),
                                     shape=(self.npoint, self.grid.dim), space_order=0,
                                     parent=self)

            gridpoints_data = kwargs.get('gridpoints', None)
            assert(gridpoints_data is not None)
            gridpoints.data[:] = gridpoints_data[:]
            self._gridpoints = gridpoints

            coefficients = SubFunction(name="%s_coefficients" % self.name,
                                       dimensions=(self.indices[-1], Dimension(name='d'),
                                                   Dimension(name='i')),
                                       shape=(self.npoint, self.grid.dim, self.r),
                                       dtype=self.dtype, space_order=0, parent=self)
            coefficients_data = kwargs.get('coefficients', None)
            assert(coefficients_data is not None)
            coefficients.data[:] = coefficients_data[:]
            self._coefficients = coefficients
            warning("Ensure that the provided coefficient and grid point values are " +
                    "computed on the final grid that will be used for other " +
                    "computations.")

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, increment=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param increment: (Optional) if True, perform an increment rather
                          than an assignment. Defaults to False.
        """
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            time = self.grid.time_dim
            t = self.grid.stepping_dim
            expr = expr.subs(t, u_t).subs(time, u_t)

        p, _, _ = self.coefficients.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.coefficients[p, i, rd])
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(self.indices[0], p_t)
        rhs = prod(coeffs) * expr.subs(dim_subs)

        return [Eq(lhs, lhs + rhs)]

    def inject(self, field, expr, offset=0, u_t=None, p_t=None):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        expr = indexify(expr)
        field = indexify(field)

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(self.indices[0], p_t)

        p, _ = self.gridpoints.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.coefficients[p, i, rd])
        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        return [Eq(field, field + rhs.subs(dim_subs))]

    @property
    def gridpoints(self):
        return self._gridpoints

    @property
    def coefficients(self):
        return self._coefficients

    def _dist_scatter(self, data=None):
        data = data if data is not None else self.data
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return {self: data, self.gridpoints: self.gridpoints.data,
                    self.coefficients: self.coefficients.data}

        raise NotImplementedError

    def _dist_gather(self, data):
        distributor = self.grid.distributor

        # If not using MPI, don't waste time
        if distributor.nprocs == 1:
            return

        raise NotImplementedError


class PrecomputedSparseTimeFunction(AbstractSparseTimeFunction,
                                    PrecomputedSparseFunction):
    """
    A specialised type of SparseFunction where the interpolation is externally defined.
    Currently, this means that the grid points and associated coefficients for each
    sparse point is precomputed at the time this object is being created.

    :param name: Name of the function.
    :param npoint: Number of points to sample.
    :param grid: :class:`Grid` object defining the computational domain.
    :param r: The number of gridpoints in each dimension to interpolate a single sparse
              point to. e.g. 2 for linear interpolation.
    :param gridpoints: The *reference* grid point corresponding to each sparse point.
                       Of all the gridpoints that one sparse point would be interpolated
                       to, this is the grid point closest to the origin, i.e. the one
                       with the lowest value of each coordinate dimension. Must be a
                       two-dimensional array of shape [npoint][grid.ndim].
    :param coefficients: An array containing the coefficient for each of the r^2 (2D) or
                         r^3 (3D) gridpoints that each sparsefunction will be interpolated
                         to. The coefficient is split across the n dimensions such that
                         the contribution of the point (i, j, k) will be multiplied by
                         coefficients[..., i]*coefficients[..., j]*coefficients[...,k]. So
                         for r=6, we will store 18 coefficients per sparse point (instead
                         of potentially 216). Shape must be [npoint][grid.ndim][r].
    :param space_order: (Optional) discretisation order for space derivatives.
                        Default to 0.
    :param time_order: (Optional) discretisation order for time derivatives.
                       Default to 1.
    :param shape: (Optional) shape of the function. Defaults to ``(nt, npoint,)``.
    :param dimensions: (Optional) symbolic dimensions that define the
                       data layout and function indices of this symbol.
    :param dtype: (Optional) data type of the buffered data.
    :param initializer: (Optional) a callable or an object exposing buffer interface
                        used to initialize the data. If a callable is provided,
                        initialization is deferred until the first access to
                        ``data``.

    .. note::

        The parameters must always be given as keyword arguments, since
        SymPy uses `*args` to (re-)create the dimension arguments of the
        symbolic function.
    """

    is_PrecomputedSparseTimeFunction = True


# Additional Function-related APIs

class Buffer(Tag):

    def __init__(self, value):
        super(Buffer, self).__init__('Buffer', value)


class Stagger(Tag):
    """Stagger region."""
    pass
NODE = Stagger('node')  # noqa
CELL = Stagger('cell')
