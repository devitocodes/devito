from ctypes import POINTER, Structure, c_void_p, c_ulong
from math import ceil

import numpy as np
from cached_property import cached_property
from sympy import Expr, Number

from devito.parameters import configuration
from devito.tools import (Reconstructable, as_tuple, c_restrict_void_p,
                          dtype_to_ctype, dtypes_vector_mapper)
from devito.types.basic import AbstractFunction
from devito.types.utils import CtypesFactory, DimensionTuple

__all__ = ['Array', 'ArrayMapped', 'ArrayObject', 'PointerArray', 'Bundle']


class ArrayBasic(AbstractFunction):

    is_ArrayBasic = True

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])

    @property
    def _C_name(self):
        if self._mem_stack or self._mem_constant:
            # No reason to distinguish between two different names, that is
            # the _C_name and the name -- just `self.name` is enough
            return self.name
        else:
            return super()._C_name

    @property
    def shape(self):
        return self.symbolic_shape

    shape_allocated = shape


class Array(ArrayBasic):

    """
    Tensor symbol representing an array in symbolic equations.

    An Array behaves similarly to a Function, but unlike a Function it carries
    no user data.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dimensions : tuple of Dimension
        Dimensions associated with the object.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type. Defaults
        to ``np.float32``.
    halo : iterable of 2-tuples, optional
        The halo region of the object.
    padding : iterable of 2-tuples, optional
        The padding region of the object.
    liveness : str, optional
        The liveness of the object. Allowed values: 'eager', 'lazy'. Defaults
        to 'lazy'. Used to override `_mem_internal_eager` and `_mem_internal_lazy`.
    space : str, optional
        The memory space. Allowed values: 'local', 'mapped', 'host'. Defaults
        to 'local'. Used to override `_mem_local` and `_mem_mapped`.
    scope : str, optional
        The scope in the given memory space. Allowed values: 'heap', 'stack',
        'static', 'constant', 'shared'. 'static' refers to a static array in a
        C/C++ sense. 'constant' and 'shared' mean that the Array represents an
        object allocated in so called constant and shared memory, respectively,
        which are typical of device architectures. If 'shared' is specified but
        the underlying architecture doesn't have something akin to shared memory,
        the behaviour is unspecified. If 'constant' is specified but the underlying
        architecture doesn't have something akin to constant memory, the Array
        falls back to a global, const, static array in a C/C++ sense.
        Note that not all scopes make sense for a given space.
    initvalue : array-like, optional
        The initial content of the Array. Must be None if `scope='heap'`.

    Warnings
    --------
    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_Array = True

    __rkwargs__ = (AbstractFunction.__rkwargs__ +
                   ('dimensions', 'liveness', 'space', 'scope', 'initvalue'))

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        space = kwargs.get('space', 'local')

        if cls is Array and space == 'mapped':
            return AbstractFunction.__new__(ArrayMapped, *args, **kwargs)
        else:
            return AbstractFunction.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(Array, self).__init_finalize__(*args, **kwargs)

        self._liveness = kwargs.get('liveness', 'lazy')
        assert self._liveness in ['eager', 'lazy']

        self._space = kwargs.get('space', 'local')
        assert self._space in ['local', 'mapped', 'host']

        self._scope = kwargs.get('scope', 'heap')
        assert self._scope in ['heap', 'stack', 'static', 'constant', 'shared']

        self._initvalue = kwargs.get('initvalue')
        assert self._initvalue is None or self._scope != 'heap'

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding')
        if padding is None:
            padding = [(0, 0) for _ in range(self.ndim)]
            if kwargs.get('autopadding', configuration['autopadding']):
                # Heuristic 1; Arrays are typically introduced for temporaries
                # introduced during compilation, and are almost always used together
                # with loop blocking.  Since the typical block size is a multiple of
                # the SIMD vector length, `vl`, padding is made such that the
                # NODOMAIN size is a multiple of `vl` too

                # Heuristic 2: the right-NODOMAIN size is not only a multiple of
                # `vl`, but also guaranteed to be *at least* greater or equal than
                # `vl`, so that the compiler can tweak loop trip counts to maximize
                # the effectiveness of SIMD vectorization

                # Let UB be a function that rounds up a value `x` to the nearest
                # multiple of the SIMD vector length
                vl = configuration['platform'].simd_items_per_reg(self.dtype)
                ub = lambda x: int(ceil(x / vl)) * vl

                fvd_halo_size = sum(self.halo[-1])
                fvd_pad_size = (ub(fvd_halo_size) - fvd_halo_size) + vl

                padding[-1] = (0, fvd_pad_size)
            return tuple(padding)
        elif isinstance(padding, int):
            return tuple((0, padding) for _ in range(self.ndim))
        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            return tuple((0, i) if isinstance(i, int) else i for i in padding)
        else:
            raise TypeError("`padding` must be int or %d-tuple of ints" % self.ndim)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    @property
    def liveness(self):
        return self._liveness

    @property
    def space(self):
        return self._space

    @property
    def scope(self):
        return self._scope

    @property
    def _C_ctype(self):
        return POINTER(dtype_to_ctype(self.dtype))

    @property
    def _mem_internal_eager(self):
        return self._liveness == 'eager'

    @property
    def _mem_internal_lazy(self):
        return self._liveness == 'lazy'

    @property
    def _mem_local(self):
        return self._space == 'local'

    @property
    def _mem_mapped(self):
        return self._space == 'mapped'

    @property
    def _mem_host(self):
        return self._space == 'host'

    @property
    def _mem_stack(self):
        return self._scope in ('stack', 'shared')

    @property
    def _mem_heap(self):
        return self._scope == 'heap'

    @property
    def _mem_shared(self):
        return self._scope == 'shared'

    @property
    def _mem_constant(self):
        return self._scope == 'constant'

    @property
    def initvalue(self):
        return self._initvalue

    @cached_property
    def free_symbols(self):
        return super().free_symbols - {d for d in self.dimensions if d.is_Default}

    def _make_pointer(self, dim):
        return PointerArray(name='p%s' % self.name, dimensions=dim, array=self)


class ArrayMapped(Array):

    _C_structname = 'array'
    _C_field_data = 'data'
    _C_field_nbytes = 'nbytes'
    _C_field_dmap = 'dmap'

    _C_ctype = POINTER(type(_C_structname, (Structure,),
                            {'_fields_': [(_C_field_data, c_restrict_void_p),
                                          (_C_field_nbytes, c_ulong),
                                          (_C_field_dmap, c_void_p)]}))


class ArrayObject(ArrayBasic):

    # TODO: Cannot inherit from LocalObject too due to Python complaining via
    # `TypeError: Cannot create a consistent method resolution`
    # Perhaps the Object sub-hierarchy should become a set of mixin classes

    """
    Tensor symbol representing an array of objects.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dimensions : tuple of Dimension
        Dimensions associated with the object.
    fields : tuple of 2-tuple <name, ctype>, optional
        The fields of the underlying C Struct this Array represents, if any.

    Warnings
    --------
    ObjectArrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_ObjectArray = True

    __rkwargs__ = list(ArrayBasic.__rkwargs__) + ['dimensions', 'fields', 'pname']
    __rkwargs__.remove('dtype')

    def __init_finalize__(self, *args, **kwargs):
        name = kwargs['name']
        fields = tuple(kwargs.pop('fields', ()))

        self._fields = fields
        self._pname = kwargs.pop('pname', 't%s' % name)

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        pname = kwargs.get('pname', 't%s' % kwargs['name'])
        pfields = cls.__pfields_setup__(**kwargs)
        return CtypesFactory.generate(pname, pfields)

    @classmethod
    def __pfields_setup__(cls, **kwargs):
        return [(i._C_name, i._C_ctype) for i in kwargs.get('fields', [])]

    @property
    def _C_ctype(self):
        return self.dtype

    @property
    def fields(self):
        return self._fields

    @cached_property
    def pfields(self):
        # TODO: Currently lifted from CompositeObject
        return tuple(self.dtype._type_._fields_)

    @property
    def pname(self):
        return self._pname

    @property
    def _mem_local(self):
        return True

    @property
    def _mem_stack(self):
        return True


class PointerArray(ArrayBasic):

    """
    Symbol representing a pointer to an Array.

    Parameters
    ----------
    name : str
        Name of the symbol.
    dimensions : Dimension
        The pointer Dimension.
    array : Array
        The pointed Array.

    Warnings
    --------
    PointerArrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_PointerArray = True

    __rkwargs__ = ('name', 'dimensions', 'array')

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractFunction.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(PointerArray, self).__init_finalize__(*args, **kwargs)

        self._array = kwargs['array']
        assert self._array.is_Array

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs['array'].dtype

    @property
    def _C_ctype(self):
        return POINTER(POINTER(dtype_to_ctype(self.dtype)))

    @property
    def dim(self):
        """Shortcut for self.dimensions[0]."""
        return self.dimensions[0]

    @property
    def array(self):
        return self._array


class Bundle(ArrayBasic):

    """
    Tensor symbol representing an unrolled vector of AbstractFunctions.

    Parameters
    ----------
    name : str
        Name of the symbol.
    components : tuple of AbstractFunctions
        The AbstractFunctions of the Bundle. They must have same type.

    Warnings
    --------
    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_Bundle = True

    __rkwargs__ = AbstractFunction.__rkwargs__ + ('components',)

    def __init__(self, *args, components=(), **kwargs):
        self._components = tuple(components)

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        components = kwargs.get('components', ())
        if len(components) <= 1:
            raise ValueError("Expected at least two components")

        klss = {type(i).__base__ for i in components}
        if len(klss) != 1:
            raise ValueError("Components must be of same type")
        if not issubclass(klss.pop(), AbstractFunction):
            raise ValueError("Component type must be subclass of AbstractFunction")

        return args, kwargs

    @classmethod
    def __dtype_setup__(cls, components=(), **kwargs):
        dtypes = {i.dtype for i in components}
        if len(dtypes) > 1:
            raise ValueError("Components must have the same dtype")
        dtype = dtypes.pop()
        try:
            return dtypes_vector_mapper[(dtype, len(components))]
        except KeyError:
            raise NotImplementedError("Unsupported vector type `%s`" % dtype)

    @classmethod
    def __indices_setup__(cls, components=(), **kwargs):
        dimensionss = {i.dimensions for i in components}
        if len(dimensionss) > 1:
            raise ValueError("Components must have the same dimensions")
        dimensions = dimensionss.pop()
        return as_tuple(dimensions), as_tuple(dimensions)

    def __halo_setup__(self, components=(), **kwargs):
        halos = {i.halo for i in components}
        if len(halos) > 1:
            raise ValueError("Components must have the same halo")
        return halos.pop()

    # Class attributes overrides

    @property
    def is_DiscreteFunction(self):
        return self.c0.is_DiscreteFunction

    @property
    def is_TimeFunction(self):
        return self.c0.is_TimeFunction

    # Other properties and methods

    @property
    def components(self):
        return self._components

    @property
    def c0(self):
        # Shortcut for self.components[0]
        return self.components[0]

    @property
    def grid(self):
        if self.is_DiscreteFunction:
            return self.c0.grid
        else:
            return None

    @property
    def symbolic_shape(self):
        # A Bundle may be defined over a SteppingDimension, which is of unknown
        # size, hence we gotta use the actual numeric size instead
        ret = []
        for d, s, v in zip(self.dimensions, super().symbolic_shape, self.c0.shape):
            if d.is_Stepping:
                ret.append(Number(v))
            else:
                ret.append(s)
        return DimensionTuple(*ret, getters=self.dimensions)

    @property
    def initvalue(self):
        return None

    # CodeSymbol overrides defaulting to self.c0's behaviour

    for i in ['_mem_internal_eager', '_mem_internal_lazy', '_mem_local',
              '_mem_mapped', '_mem_host', '_mem_stack', '_mem_constant',
              '_mem_shared', '_size_domain', '_size_halo', '_size_owned',
              '_size_padding', '_size_nopad', '_size_nodomain', '_offset_domain',
              '_offset_halo', '_offset_owned', '_dist_dimensions', '_C_get_field']:
        locals()[i] = property(lambda self, v=i: getattr(self.c0, v))

    @property
    def _mem_heap(self):
        return not self._mem_stack

    @property
    def _dist_dimensions(self):
        return self.c0._dist_dimensions

    def _C_get_field(self, region, dim, side=None):
        return self.c0._C_get_field(region, dim, side=side)

    def __getitem__(self, index):
        index = as_tuple(index)
        if len(index) == self.ndim:
            return super().__getitem__(index)
        elif len(index) == self.ndim + 1:
            component_index, indices = index[0], index[1:]
            return ComponentAccess(self.indexed[indices], component_index)
        else:
            raise ValueError("Expected %d or %d indices, got %d instead"
                             % (self.ndim, self.ndim + 1, len(index)))

    _C_structname = ArrayMapped._C_structname
    _C_field_data = ArrayMapped._C_field_data
    _C_field_nbytes = ArrayMapped._C_field_nbytes
    _C_field_dmap = ArrayMapped._C_field_dmap

    @property
    def _C_ctype(self):
        if self._mem_mapped:
            return ArrayMapped._C_ctype
        else:
            return POINTER(dtype_to_ctype(self.dtype))


class ComponentAccess(Expr, Reconstructable):

    _component_names = ('x', 'y', 'z', 'w')

    __rkwargs__ = ('index',)

    def __new__(cls, arg, index=0, **kwargs):
        if not arg.is_Indexed:
            raise ValueError("Expected Indexed, got `%s` instead" % type(arg))
        if not isinstance(index, int) or index > 3:
            raise ValueError("Expected 0 <= index < 4")

        obj = Expr.__new__(cls, arg)
        obj._index = index

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self._index,)

    def __str__(self):
        return "%s.%s" % (self.base, self.sindex)

    __repr__ = __str__

    func = Reconstructable._rebuild

    def _sympystr(self, printer):
        return str(self)

    @property
    def base(self):
        return self.args[0]

    @property
    def index(self):
        return self._index

    @property
    def sindex(self):
        return self._component_names[self.index]

    @property
    def function(self):
        return self.base.function

    @property
    def indices(self):
        return self.base.indices

    @property
    def dtype(self):
        return self.function.dtype
