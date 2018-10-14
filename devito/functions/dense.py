import sympy
import numpy as np
from psutil import virtual_memory

from devito.data import Data, default_allocator, first_touch
from devito.exceptions import InvalidArgument
from devito.logger import debug, warning
from devito.mpi import MPI
from devito.parameters import configuration
from devito.finite_differences import Differentiable, generate_fd_shortcuts
from devito.functions.dimension import Dimension
from devito.functions.utils import Buffer, NODE, CELL
from devito.functions.basic import AbstractCachedFunction, OWNED, HALO, LEFT, RIGHT
from devito.tools import (EnrichedTuple, ReducerMap, ArgProvider, as_tuple,
                          is_integer)

__all__ = ['Function', 'TimeFunction']


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

            # Staggering metadata
            self._staggered = self.__staggered_setup__(**kwargs)

            # Data-related properties and data initialization
            self._data = None
            self._first_touch = kwargs.get('first_touch', configuration['first_touch'])
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
                    self.data_allocated[:] = initializer
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
                self._data = Data(self.shape_allocated, self.indices, self.dtype,
                                  allocator=self._allocator)
                if self._first_touch:
                    first_touch(self)
                if callable(self._initializer):
                    if self._first_touch:
                        warning("`first touch` together with `initializer` causing "
                                "redundant data initialization")
                    try:
                        self._initializer(self._data)
                    except ValueError:
                        # Perhaps user only wants to initialise the physical domain
                        self._initializer(self._data[self._mask_domain])
                else:
                    self._data.fill(0)
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

    @property
    def _data_buffer(self):
        """Reference to the data. Unlike ``data, data_with_halo, data_allocated``,
        this *never* returns a view of the data. This method is for internal use only."""
        return self.data_allocated

    @property
    def _mem_external(self):
        return True

    @property
    def grid(self):
        return self._grid

    @property
    def staggered(self):
        return self._staggered

    @property
    def shape(self):
        """
        Shape of the domain associated with this :class:`TensorFunction`.
        The domain constitutes the area of the data written to in a
        stencil update.
        """
        return self.shape_domain

    @property
    def shape_domain(self):
        """
        Shape of the domain associated with this :class:`TensorFunction`.
        The domain constitutes the area of the data written to in a
        stencil update.

        .. note::

            Alias to ``self.shape``.
        """
        return tuple(i - j for i, j in zip(self._shape, self.staggered))

    @property
    def shape_with_halo(self):
        """
        Shape of the domain plus the read-only stencil boundary associated
        with this :class:`Function`.
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_domain, self._halo))

    @property
    def shape_allocated(self):
        """
        Shape of the allocated data associated with this :class:`Function`.
        It includes the domain and halo regions, as well as any additional
        padding outside of the halo.
        """
        return tuple(j + i + k for i, (j, k) in zip(self.shape_with_halo, self._padding))

    @property
    def data(self):
        """
        The domain data values, as a :class:`numpy.ndarray`.

        Elements are stored in row-major format.

        .. note::

            With this accessor you are claiming that you will modify
            the values you get back. If you only need to look at the
            values, use :meth:`data_ro` instead.
        """
        return self.data_domain

    @property
    @_allocate_memory
    def data_domain(self):
        """
        The domain data values.

        Elements are stored in row-major format.

        .. note::

            With this accessor you are claiming that you will modify
            the values you get back. If you only need to look at the
            values, use :meth:`data_ro_domain` instead.

        .. note::

            Alias to ``self.data``.
        """
        self._is_halo_dirty = True
        return self._data[self._mask_domain]

    @property
    @_allocate_memory
    def data_interior(self):
        """
        The interior data values.

        Elements are stored in row-major format.

        .. note::

            With this accessor you are claiming that you will modify
            the values you get back. If you only need to look at the
            values, use :meth:`data_ro_interior` instead.
        """
        self._is_halo_dirty = True
        return self._data[self._mask_interior]

    @property
    @_allocate_memory
    def data_with_halo(self):
        """
        The domain+halo data values.

        Elements are stored in row-major format.

        .. note::

            With this accessor you are claiming that you will modify
            the values you get back. If you only need to look at the
            values, use :meth:`data_ro_with_halo` instead.
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return self._data[self._mask_with_halo]

    @property
    @_allocate_memory
    def data_allocated(self):
        """
        The allocated data values, that is domain+halo+padding.

        Elements are stored in row-major format.

        .. note::

            With this accessor you are claiming that you will modify
            the values you get back. If you only need to look at the
            values, use :meth:`data_ro_allocated` instead.
        """
        self._is_halo_dirty = True
        self._halo_exchange()
        return self._data

    @property
    @_allocate_memory
    def data_ro_domain(self):
        """
        A read-only view of the domain data values.
        """
        view = self._data[self._mask_domain]
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_interior(self):
        """
        A read-only view of the interior data values.
        """
        view = self._data[self._mask_interior]
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_with_halo(self):
        """A read-only view of the domain+halo data values."""
        view = self._data[self._mask_with_halo]
        view.setflags(write=False)
        return view

    @property
    @_allocate_memory
    def data_ro_allocated(self):
        """A read-only view of the domain+halo+padding data values."""
        view = self._data.view()
        view.setflags(write=False)
        return view

    @property
    def space_dimensions(self):
        """Tuple of :class:`Dimension`s that define physical space."""
        return tuple(d for d in self.indices if d.is_Space)

    @property
    def initializer(self):
        if self._data is not None:
            return self._data.view(np.ndarray)
        else:
            return self._initializer

    @property
    def symbolic_shape(self):
        """
        Return the symbolic shape of the object. This includes: ::

            * the padding, halo, and domain regions. While halo and padding are
              known quantities (integers), the domain size is represented by a symbol.
            * the shifting induced by the ``staggered`` mask
        """
        symbolic_shape = super(TensorFunction, self).symbolic_shape
        ret = tuple(sympy.Add(i, -j, evaluate=False)
                    for i, j in zip(symbolic_shape, self.staggered))
        return EnrichedTuple(*ret, getters=self.dimensions)

    @property
    def _mask_interior(self):
        """A mask to access the interior region of the allocated data."""
        if self.grid is None:
            glb_pos_map = {d: [LEFT, RIGHT] for d in self.dimensions}
        else:
            glb_pos_map = self.grid.distributor.glb_pos_map
        ret = []
        for d, i in zip(self.dimensions, self._mask_domain):
            if d.is_Space:
                lshift = int(LEFT in glb_pos_map.get(d, []))
                rshift = int(RIGHT in glb_pos_map.get(d, []))
                if i.stop is None:
                    ret.append(slice(i.start + lshift, -rshift or None))
                else:
                    ret.append(slice(i.start + lshift, i.stop - rshift))
            else:
                ret.append(i)
        return tuple(ret)

    def _halo_exchange(self):
        """Perform the halo exchange with the neighboring processes."""
        if not MPI.Is_initialized() or MPI.COMM_WORLD.size == 1:
            # Nothing to do
            return
        if MPI.COMM_WORLD.size > 1 and self.grid is None:
            raise RuntimeError("`%s` cannot perfom a halo exchange as it has "
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
        distributor = self.grid.distributor
        neighbours = distributor.neighbours
        comm = distributor.comm
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
        shape = kwargs.get('shape')
        if grid is None:
            if shape is None:
                raise TypeError("Need either `grid` or `shape`")
        elif shape is None:
            shape = grid.shape_domain
        elif dimensions is None:
            raise TypeError("`dimensions` required if both `grid` and "
                            "`shape` are provided")
        else:
            # Got `grid`, `dimensions`, and `shape`. We sanity-check that the
            # Dimensions in `dimensions` which also appear in `grid` have
            # size (given by `shape`) matching the one in `grid`
            if len(shape) != len(dimensions):
                raise TypeError("`shape` and `dimensions` must have the "
                                "same number of entries")
            loc_shape = []
            for d, s in zip(dimensions, shape):
                if d in grid.dimensions:
                    size = grid.dimension_map[d]
                    if size.glb != s:
                        raise TypeError("Dimension `%s` is given size `%d`, "
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
                lp = self._extent_halo[d].left
                rp = self._extent_halo[d].right
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
        ['space_order', 'shape', 'dimensions', 'staggered']


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
            shape = list(grid.shape_domain)
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
