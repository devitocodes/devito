from collections import OrderedDict
from functools import partial
from itertools import product

import sympy
import numpy as np
from cached_property import cached_property
from psutil import virtual_memory
from mpi4py import MPI

from devito.cgen_utils import INT, cast_mapper
from devito.data import Data, default_allocator, first_touch
from devito.dimension import Dimension, DefaultDimension
from devito.equation import Eq, Inc
from devito.exceptions import InvalidArgument
from devito.finite_difference import (centered, cross_derivative,
                                      first_derivative, left, right,
                                      second_derivative, generic_derivative,
                                      second_cross_derivative)
from devito.logger import debug, warning
from devito.parameters import configuration
from devito.symbolics import indexify, retrieve_indexed
from devito.types import (AbstractCachedFunction, AbstractCachedSymbol,
                          OWNED, HALO, LEFT, RIGHT)
from devito.tools import (Tag, ReducerMap, as_mapper, as_tuple, flatten, is_integer,
                          prod, powerset)

__all__ = ['Constant', 'Function', 'TimeFunction', 'SparseFunction',
           'SparseTimeFunction', 'PrecomputedSparseFunction',
           'PrecomputedSparseTimeFunction', 'Buffer']


class Constant(AbstractCachedSymbol):

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
            self.dtype = kwargs.get('dtype', np.float32)
            self._value = kwargs.get('value')

    @property
    def data(self):
        """The value of the data object, as a scalar (int, float, ...)."""
        return self._value

    @data.setter
    def data(self, val):
        self._value = val

    @property
    def base(self):
        return self

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this symbol."""
        return (self.name,)

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

    _pickle_kwargs = AbstractCachedSymbol._pickle_kwargs + ['dtype', '_value']


class TensorFunction(AbstractCachedFunction):

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

            # Staggered mask
            self._staggered = kwargs.get('staggered', tuple(0 for _ in self.indices))
            if len(self.staggered) != len(self.indices):
                raise ValueError("`staggered` needs %s entries for indices %s"
                                 % (len(self.indices), self.indices))

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
                self.data_allocated[:] = initializer
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
    def staggered(self):
        return self._staggered

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
        return tuple(sympy.Add(i, -j, evaluate=False)
                     for i, j in zip(symbolic_shape, self.staggered))

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
        if MPI.COMM_WORLD.size == 1:
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
        for i, s, o, k in zip(self.indices, self.shape, self.staggered, key.indices):
            args.update(i._arg_defaults(start=0, size=s+o, alias=k))

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
        ['dtype', 'grid', 'staggered', 'initializer']


class Function(TensorFunction):
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
    :param staggered: (Optional) tuple containing staggering offsets.
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

    .. note::

       The tuple :param staggered: contains a ``1`` in each dimension
       entry that should be staggered, and ``0`` otherwise. For example,
       ``staggered=(1, 0, 0)`` entails discretization on horizontal edges,
       ``staggered=(0, 0, 1)`` entails discretization on vertical edges,
       ``staggered=(0, 1, 1)`` entails discretization side facets and
       ``staggered=(1, 1, 1)`` entails discretization on cells.
    """

    is_Function = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(Function, self).__init__(*args, **kwargs)

            # Space order
            space_order = kwargs.get('space_order', 1)
            if isinstance(space_order, int):
                self.space_order = space_order
            elif isinstance(space_order, tuple) and len(space_order) == 3:
                self.space_order, _, _ = space_order
            else:
                raise TypeError("`space_order` must be int or 3-tuple of ints")

            # Dynamically add derivative short-cuts
            self._initialize_derivatives()

    def _initialize_derivatives(self):
        """
        Dynamically create notational shortcuts for space derivatives.
        """
        for dim in self.space_dimensions:
            name = dim.parent.name if dim.is_Derived else dim.name
            # First derivative, centred
            dx = partial(first_derivative, order=self.space_order,
                         dim=dim, side=centered)
            setattr(self.__class__, 'd%s' % name,
                    property(dx, 'Return the symbolic expression for '
                             'the centered first derivative wrt. '
                             'the %s dimension' % name))

            # First derivative, left
            dxl = partial(first_derivative, order=self.space_order,
                          dim=dim, side=left)
            setattr(self.__class__, 'd%sl' % name,
                    property(dxl, 'Return the symbolic expression for '
                             'the left-sided first derivative wrt. '
                             'the %s dimension' % name))

            # First derivative, right
            dxr = partial(first_derivative, order=self.space_order,
                          dim=dim, side=right)
            setattr(self.__class__, 'd%sr' % name,
                    property(dxr, 'Return the symbolic expression for '
                             'the right-sided first derivative wrt. '
                             'the %s dimension' % name))

            # Second derivative
            dx2 = partial(generic_derivative, deriv_order=2, dim=dim,
                          fd_order=int(self.space_order / 2))
            setattr(self.__class__, 'd%s2' % name,
                    property(dx2, 'Return the symbolic expression for '
                             'the second derivative wrt. the '
                             '%s dimension' % name))

            # Fourth derivative
            dx4 = partial(generic_derivative, deriv_order=4, dim=dim,
                          fd_order=max(int(self.space_order / 2), 2))
            setattr(self.__class__, 'd%s4' % name,
                    property(dx4, 'Return the symbolic expression for '
                             'the fourth derivative wrt. the '
                             '%s dimension' % name))

            for dim2 in self.space_dimensions:
                name2 = dim2.parent.name if dim2.is_Derived else dim2.name
                # First cross derivative
                dxy = partial(cross_derivative, order=self.space_order,
                              dims=(dim, dim2))
                setattr(self.__class__, 'd%s%s' % (name, name2),
                        property(dxy, 'Return the symbolic expression for '
                                 'the first cross derivative wrt. the '
                                 '%s and %s dimensions' %
                                 (name, name2)))

                # Second cross derivative
                dx2y2 = partial(second_cross_derivative, dims=(dim, dim2),
                                order=self.space_order)
                setattr(self.__class__, 'd%s2%s2' % (dim.name, name2),
                        property(dx2y2, 'Return the symbolic expression for '
                                 'the second cross derivative wrt. the '
                                 '%s and %s dimensions' %
                                 (name, name2)))

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
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative wrt. all spatial dimensions.
        """
        derivs = tuple('d%s2' % d.name for d in self.space_dimensions)

        return sum([getattr(self, d) for d in derivs[:self.ndim]])

    def laplace2(self, weight=1):
        """
        Generates a symbolic expression for the double Laplacian
        wrt. all spatial dimensions.
        """
        order = self.space_order/2
        first = sum([second_derivative(self, dim=d, order=order)
                     for d in self.space_dimensions])
        return sum([second_derivative(first * weight, dim=d, order=order)
                    for d in self.space_dimensions])

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
        ['space_order', 'shape', 'dimensions']


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
    :param staggered: (Optional) tuple containing staggering offsets.
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
            super(TimeFunction, self).__init__(*args, **kwargs)

            # Check we won't allocate too much memory for the system
            available_mem = virtual_memory().available
            if np.dtype(self.dtype).itemsize * self.size > available_mem:
                warning("Trying to allocate more memory for symbol %s " % self.name +
                        "than available on physical device, this will start swapping")

            self.time_dim = kwargs.get('time_dim', self.indices[self._time_position])
            self.time_order = kwargs.get('time_order', 1)
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
    def dt(self):
        """Symbol for the first derivative wrt the time dimension"""
        _t = self.indices[self._time_position]
        if self.time_order == 1:
            # This hack is needed for the first-order diffusion test
            indices = [_t, _t + _t.spacing]
        else:
            width = int(self.time_order / 2)
            indices = [(_t + i * _t.spacing) for i in range(-width, width + 1)]

        return self.diff(_t).as_finite_difference(indices)

    @property
    def dt2(self):
        """Symbol for the second derivative wrt the t dimension"""
        _t = self.indices[0]
        width_t = int(self.time_order / 2)
        indt = [(_t + i * _t.spacing) for i in range(-width_t, width_t + 1)]

        return self.diff(_t, _t).as_finite_difference(indt)

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


class AbstractSparseFunction(TensorFunction):
    """
    An abstract class to define behaviours common to any kind of sparse
    functions, whether using precomputed coefficients or computing them
    on the fly. This is an internal class only and should never be
    instantiated.
    """
    # Symbols that are encapsulated within this symbol (e.g. coordinates)
    _child_functions = []

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(AbstractSparseFunction, self).__init__(*args, **kwargs)

            npoint = kwargs.get('npoint')
            if not isinstance(npoint, int):
                raise TypeError('SparseFunction needs `npoint` int argument')
            if npoint < 0:
                raise ValueError('`npoint` must be >= 0')
            self.npoint = npoint

            # A Grid must have been provided
            if self.grid is None:
                raise TypeError('SparseFunction needs `grid` argument')

            self.space_order = kwargs.get('space_order', 0)

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
        return kwargs.get('shape', (kwargs.get('npoint'),))

    def _arg_defaults(self, alias=None):
        """
        Returns a map of default argument values defined by this symbol.

        :param alias: (Optional) name under which to store values.
        """
        key = alias or self
        args = super(AbstractSparseFunction, self)._arg_defaults(alias=alias)
        for child_name in self._child_functions:
            child = getattr(self, child_name)
            args.update(child._arg_defaults(alias=getattr(key, child_name)))
        return args

    @property
    def _arg_names(self):
        """Return a tuple of argument names introduced by this function."""
        return tuple([self.name] + [x for x in self._child_functions])

    def _is_owned(self, point):
        """Return True if ``point`` is in self's local domain, False otherwise."""
        point = as_tuple(point)
        if len(point) != self.grid.dim:
            raise ValueError("`%s` is an %dD point (expected %dD)"
                             % (point, len(point), self.grid.dim))
        distributor = self.grid.distributor
        return all(distributor.glb_to_loc(d, p) is not None
                   for d, p in zip(self.grid.dimensions, point))

    @cached_property
    def _dist_datamap(self):
        """
        Return a mapper ``M : MPI rank -> logically owned sparse data``
        """
        targets = self.grid.distributor.glb_to_rank(self.gridpoints)
        return as_mapper(range(self.npoint), lambda i: targets[i])

    @cached_property
    def _dist_counts(self):
        """
        Return a 2-tuple of iterables, describing how much data is each MPI rank
        expected to send/receive to/from each other MPI rank.
        """
        dmap = self._dist_datamap
        comm = self.grid.distributor.comm

        sendcount = np.array([len(dmap.get(i, [])) for i in range(comm.size)], dtype=int)
        recvcount = np.empty(comm.size, dtype=int)
        comm.Alltoall(sendcount, recvcount)

        return sendcount, recvcount

    def _dist_scatter(self):
        """
        Return a ``numpy.ndarray`` containing up-to-date data and coordinate values
        belonging to the calling MPI rank. A data value belongs to a given MPI rank R
        if its coordinates fall in R's local domain region.
        """
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        dmap = self._dist_datamap
        comm = self.grid.distributor.comm

        # First, determine what (i) needs to be sent to and (ii) what needs to
        # be received from each MPI rank
        sendcount, recvcount = self._dist_counts

        # Pack (reordered) data values so that they can be sent out via an Alltoallv
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        data = self.data[mask]

        # Send out the sparse point values
        senddisp = np.concatenate([[0], np.cumsum(sendcount)[:-1]])
        recvdisp = np.concatenate([[0], tuple(np.cumsum(recvcount))[:-1]])
        scattered = np.empty(sum(recvcount), dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, sendcount, senddisp, mpitype],
                       [scattered, recvcount, recvdisp, mpitype])
        data = scattered

        # Pack (reordered) coordinates so that they can be sent out via an Alltoallv
        coords = self.coordinates.data[mask]

        # Send out the sparse point coordinates
        ndim = self.grid.dim
        scattered = np.empty(shape=(sum(recvcount), ndim), dtype=self.coordinates.dtype)
        comm.Alltoallv([coords, sendcount*ndim, senddisp*ndim, mpitype],
                       [scattered, recvcount*ndim, recvdisp*ndim, mpitype])
        coords = scattered

        return data, coords

    def _dist_gather(self, data):
        """
        Return a ``numpy.ndarray`` containing up-to-date data and coordinate values
        suitable for insertion into ``self.data``.
        """
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        dmap = self._dist_datamap
        comm = self.grid.distributor.comm

        # First, determine what (i) needs to be sent to and (ii) what needs to
        # be received from each MPI rank
        sendcount, recvcount = self._dist_counts

        # Send back the sparse point values
        senddisp = np.concatenate([[0], np.cumsum(sendcount)[:-1]])
        recvdisp = np.concatenate([[0], tuple(np.cumsum(recvcount))[:-1]])
        gathered = np.empty(sum(sendcount), dtype=self.dtype)
        mpitype = MPI._typedict[np.dtype(self.dtype).char]
        comm.Alltoallv([data, recvcount, recvdisp, mpitype],
                       [gathered, sendcount, senddisp, mpitype])
        data = gathered

        # Insert back into `self.data` based on the original (expected) data layout
        mask = np.array(flatten(dmap[i] for i in sorted(dmap)), dtype=int)
        self.data[:] = data[mask]

        # Note: this method is almost the dual of `_dist_scatter`, except for the
        # fact that `coordinates` are not sent back. This is because `coordinates`
        # should never be written to

    @property
    def gridpoints(self):
        """The *reference* grid point corresponding to each sparse point."""
        raise NotImplementedError

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

            nt = kwargs.get('nt')
            if not isinstance(nt, int):
                raise TypeError('Sparse TimeFunction needs `nt` int argument')
            if nt <= 0:
                raise ValueError('`nt` must be > 0')
            self.nt = nt

            self.time_dim = self.indices[self._time_position]
            self.time_order = kwargs.get('time_order', 1)
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
        return kwargs.get('shape', (kwargs.get('nt'), kwargs.get('npoint'),))

    @property
    def _time_size(self):
        return self.shape_allocated[self._time_position]

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['nt', 'time_order']


class SparseFunction(AbstractSparseFunction):
    """
    A special :class:`TensorFunction` representing a set of sparse point
    objects that are not aligned with the computational grid.

    A :class:`SparseFunction` provides symbolic interpolation routines
    to convert between grid-aligned :class:`Function` objects and sparse
    data points.

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
    """

    is_SparseFunction = True
    _child_functions = ['coordinates']

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(SparseFunction, self).__init__(*args, **kwargs)

            # Set up coordinates of sparse points
            coordinates = kwargs.get('coordinates')
            if isinstance(coordinates, Function):
                self._coordinates = coordinates
            elif coordinates is not None:
                dimensions = (self.indices[-1], Dimension(name='d'))
                self._coordinates = Function(name='%s_coords' % self.name,
                                             dtype=self.dtype, dimensions=dimensions,
                                             shape=(self.npoint, self.grid.dim),
                                             space_order=0, initializer=coordinates)

    @property
    def coordinates(self):
        return self._coordinates

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

    @property
    def _point_symbols(self):
        """Symbol for coordinate value in each dimension of the point"""
        return tuple(sympy.symbols('p%s' % d) for d in self.grid.dimensions)

    @property
    def _point_increments(self):
        """Index increments in each dimension for each point symbol"""
        return tuple(product(range(2), repeat=self.grid.dim))

    @property
    def _coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension"""
        p_dim = self.indices[-1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.grid.dim)])

    @property
    def _coordinate_indices(self):
        """Symbol for each grid index according to the coordinates"""
        indices = self.grid.dimensions
        return tuple([INT(sympy.Function('floor')((c - o) / i.spacing))
                      for c, o, i in zip(self._coordinate_symbols, self.grid.origin,
                                         indices[:self.grid.dim])])

    @property
    def _coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        indices = self.grid.dimensions
        return tuple([cast_mapper[self.dtype](c - o - idx * i.spacing)
                      for c, o, idx, i in zip(self._coordinate_symbols,
                                              self.grid.origin,
                                              self._coordinate_indices,
                                              indices[:self.grid.dim])])

    def _interpolation_indices(self, variables, offset=0):
        """
        Get interpolation indices for the variables
        """
        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self._coordinate_indices))
                        for inc in self._point_increments]

        # Generate index substitutions for all grid variables except
        # the `SparseFunction` types
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            ind_subs = dict([(dim, ind) for dim, ind in zip(self.grid.dimensions, idx)])
            v_subs = [(v, v.subs(ind_subs))
                      for v in variables if not v.base.function.is_SparseFunction]
            idx_subs += [OrderedDict(v_subs)]

        # Substitute coordinate base symbols into the coefficients
        return OrderedDict(zip(self._point_symbols, self._coordinate_bases)), idx_subs

    @property
    def gridpoints(self):
        if self.coordinates._data is None:
            raise ValueError("No coordinates attached to this SparseFunction")
        ret = []
        for coords in self.coordinates.data:
            ret.append(tuple(int(sympy.floor((c - o.data)/i.spacing.data)) for c, o, i in
                             zip(coords, self.grid.origin, self.grid.dimensions)))
        return ret

    def interpolate(self, expr, offset=0, cummulative=False, self_subs={}):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param cummulative: (Optional) If True, perform an increment rather
                            than an assignment. Defaults to False.
        """
        expr = indexify(expr)

        variables = list(retrieve_indexed(expr))
        # List of indirection indices for all adjacent grid points
        subs, idx_subs = self._interpolation_indices(variables, offset)
        # Substitute coordinate base symbols into the coefficients
        rhs = sum([expr.subs(vsub) * b.subs(subs)
                   for b, vsub in zip(self._coefficients, idx_subs)])

        lhs = self.subs(self_subs)
        rhs = rhs + lhs if cummulative else rhs

        return [Inc(lhs, rhs)] if cummulative else [Eq(lhs, rhs)]

    def inject(self, field, expr, offset=0):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        """
        expr = indexify(expr)
        field = indexify(field)
        variables = list(retrieve_indexed(expr)) + [field]

        # List of indirection indices for all adjacent grid points
        subs, idx_subs = self._interpolation_indices(variables, offset)

        # Substitute coordinate base symbols into the coefficients
        return [Inc(field.subs(vsub),
                    field.subs(vsub) + expr.subs(subs).subs(vsub) * b.subs(subs))
                for b, vsub in zip(self._coefficients, idx_subs)]

    # Pickling support
    _pickle_kwargs = AbstractSparseFunction._pickle_kwargs + ['coordinates']


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

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, cummulative=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param cummulative: (Optional) If True, perform an increment rather
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
                                                           cummulative=cummulative,
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
    _child_functions = ['gridpoints', 'coefficients']

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PrecomputedSparseFunction, self).__init__(*args, **kwargs)

            # Grid points per sparse point (2 in the case of bilinear and trilinear)
            r = kwargs.get('r')
            if not isinstance(r, int):
                raise TypeError('Interpolation needs `r` int argument')
            if r <= 0:
                raise ValueError('`r` must be > 0')
            self.r = r

            gridpoints = Function(name="%s_gridpoints" % self.name, dtype=np.int32,
                                  dimensions=(self.indices[-1], Dimension(name='d')),
                                  shape=(self.npoint, self.grid.dim), space_order=0)

            gridpoints_data = kwargs.get('gridpoints', None)
            assert(gridpoints_data is not None)
            gridpoints.data[:] = gridpoints_data[:]
            self._gridpoints = gridpoints

            coefficients = Function(name="%s_coefficients" % self.name, dtype=self.dtype,
                                    dimensions=(self.indices[-1], Dimension(name='d'),
                                                Dimension(name='i')),
                                    shape=(self.npoint, self.grid.dim, self.r),
                                    space_order=0)
            coefficients_data = kwargs.get('coefficients', None)
            assert(coefficients_data is not None)
            coefficients.data[:] = coefficients_data[:]
            self._coefficients = coefficients
            warning("Ensure that the provided coefficient and grid point values are " +
                    "computed on the final grid that will be used for other " +
                    "computations.")

    def interpolate(self, expr, offset=0, u_t=None, p_t=None, cummulative=False):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        :param cummulative: (Optional) If True, perform an increment rather
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
