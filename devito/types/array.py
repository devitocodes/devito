from ctypes import POINTER, Structure, c_void_p, c_int, c_uint64
from functools import cached_property

import numpy as np
from sympy import Expr, cacheit

from devito.tools import (Pickable, as_tuple, c_restrict_void_p,
                          dtype_to_ctype, dtypes_vector_mapper, is_integer)
from devito.types.basic import AbstractFunction, LocalType
from devito.types.utils import CtypesFactory, DimensionTuple

__all__ = ['Array', 'ArrayMapped', 'ArrayObject', 'PointerArray', 'Bundle',
           'ComponentAccess', 'Bag', 'BundleView']


class ArrayBasic(AbstractFunction, LocalType):

    is_ArrayBasic = True

    __rkwargs__ = AbstractFunction.__rkwargs__ + ('is_const', 'liveness')

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self._liveness = kwargs.get('liveness', 'lazy')
        assert self._liveness in ['eager', 'lazy']

        self._is_const = kwargs.get('is_const', False)

    @classmethod
    def __indices_setup__(cls, *args, **kwargs):
        dimensions = kwargs['dimensions']

        if args:
            indices = args
        else:
            indices = dimensions

        return as_tuple(dimensions), as_tuple(indices)

    @property
    def _C_name(self):
        if self._mem_stack or self._mem_constant or self._mem_rvalue:
            # No reason to distinguish between two different names, that is
            # the _C_name and the name -- just `self.name` is enough
            return self.name
        else:
            return super()._C_name

    @cached_property
    def shape(self):
        ret = [i.symbolic_size for i in self.dimensions]
        return DimensionTuple(*ret, getters=self.dimensions)

    @property
    def shape_allocated(self):
        return self.symbolic_shape

    @property
    def is_const(self):
        return self._is_const

    @property
    def c0(self):
        # ArrayBasic can be used as a base class for tensorial objects (that is,
        # arrays whose components are AbstractFunctions). This property enables
        # treating the two cases uniformly in some lowering passes
        return self


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
        'static', 'constant', 'shared', 'shared-remote', 'registers', 'rvalue'.
        'static' refers to a static array in a C/C++ sense. 'constant' and
        'shared' mean that the Array represents an object allocated in so
        called constant and shared memory, respectively, which are typical of
        device architectures. If 'shared' is specified but the underlying
        architecture doesn't have something akin to shared memory, the
        behaviour is unspecified. If 'constant' is specified but the underlying
        architecture doesn't have something akin to constant memory, the Array
        falls back to a global, const, static array in a C/C++ sense.
        'registers' is used to indicate that the Array has a small static size
        and, as such, it could be allocated in registers. If `rvalue`, the
        Array is treated as a temporary or "transient" object, just like
        C++'s rvalue references or C's compound literals. Defaults to 'heap'.
        Note that not all scopes make sense for a given space.
    grid : Grid, optional
        Only necessary for distributed-memory parallelism; a Grid contains
        information about the distributed Dimensions, hence it is necessary
        if (and only if) an Operator requires to perform a halo exchange on
        an Array.
    initvalue : array-like, optional
        The initial content of the Array. Must be None if `scope='heap'`.

    Warnings
    --------
    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_Array = True

    __rkwargs__ = (ArrayBasic.__rkwargs__ +
                   ('dimensions', 'scope', 'initvalue'))

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        space = kwargs.setdefault('space', 'local')

        if cls is Array and space == 'mapped':
            return AbstractFunction.__new__(ArrayMapped, *args, **kwargs)
        else:
            return AbstractFunction.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self._scope = kwargs.get('scope', 'heap')
        assert self._scope in ['heap', 'stack', 'static', 'constant', 'shared',
                               'shared-remote', 'registers', 'rvalue']

        self._initvalue = kwargs.get('initvalue')
        assert self._initvalue is None or self._scope != 'heap'

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return kwargs.get('dtype', np.float32)

    def __padding_setup__(self, **kwargs):
        padding = kwargs.get('padding')
        if padding is None:
            padding = ((0, 0),)*self.ndim
        elif isinstance(padding, DimensionTuple):
            padding = tuple(padding[d] for d in self.dimensions)
        elif is_integer(padding):
            padding = tuple((0, padding) for _ in range(self.ndim))
        elif isinstance(padding, tuple) and len(padding) == self.ndim:
            padding = tuple((0, i) if is_integer(i) else i for i in padding)
        else:
            raise TypeError("`padding` must be int or %d-tuple of ints" % self.ndim)
        return DimensionTuple(*padding, getters=self.dimensions)

    @property
    def scope(self):
        return self._scope

    @property
    def _C_ctype(self):
        return POINTER(dtype_to_ctype(self.dtype))

    @property
    def _mem_stack(self):
        return self._scope in ('stack', 'shared', 'registers')

    @property
    def _mem_heap(self):
        return self._scope == 'heap'

    @property
    def _mem_shared(self):
        return self._scope == 'shared'

    @property
    def _mem_shared_remote(self):
        return self._scope == 'shared-remote'

    @property
    def _mem_registers(self):
        return self._scope == 'registers'

    @property
    def _mem_constant(self):
        return self._scope == 'constant'

    @property
    def _mem_rvalue(self):
        return self._scope == 'rvalue'

    @property
    def initvalue(self):
        return self._initvalue

    @cached_property
    def free_symbols(self):
        return super().free_symbols - {d for d in self.dimensions if d.is_Default}

    def _make_pointer(self, dim):
        return PointerArray(name='p%s' % self.name, dimensions=dim, array=self)


class MappedArrayMixin:

    _C_structname = 'array'
    _C_field_data = 'data'
    _C_field_dmap = 'dmap'
    _C_field_shape = 'shape'
    _C_field_size = 'size'
    _C_field_nbytes = 'nbytes'
    _C_field_arity = 'arity'

    _C_ctype = POINTER(type(_C_structname, (Structure,),
                            {'_fields_': [(_C_field_data, c_restrict_void_p),
                                          (_C_field_dmap, c_void_p),
                                          (_C_field_shape, POINTER(c_int)),
                                          (_C_field_size, c_uint64),
                                          (_C_field_nbytes, c_uint64),
                                          (_C_field_arity, c_uint64)]}))


class ArrayMapped(MappedArrayMixin, Array):
    pass


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
        super().__init_finalize__(*args, **kwargs)

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


class Bundle(MappedArrayMixin, ArrayBasic):

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

    def __new__(cls, *args, components=(), **kwargs):
        return super().__new__(cls, *args, components=as_tuple(components), **kwargs)

    def __init_finalize__(self, *args, components=(), **kwargs):
        super().__init_finalize__(*args, components=components, **kwargs)

        self._components = tuple(components)

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        components = kwargs.get('components', ())
        klss = {type(i).__base__ for i in components}
        if len(klss) != 1:
            raise ValueError("Components must be of same type")
        if not issubclass(klss.pop(), AbstractFunction):
            raise ValueError("Component type must be subclass of AbstractFunction")
        if len({i.__padding_dtype__ for i in components}) != 1:
            raise ValueError("Components must have the same padding dtype")
        if len({i.properties for i in components}) != 1:
            raise ValueError("Components must have the same properties")

        return args, kwargs

    @classmethod
    def __dtype_setup__(cls, components=(), **kwargs):
        dtypes = {i.dtype for i in components}
        if len(dtypes) > 1:
            raise ValueError("Components must have the same dtype")
        dtype = dtypes.pop()
        count = len(components)
        try:
            return dtypes_vector_mapper[(dtype, count)]
        except KeyError:
            dtypes_vector_mapper.add_dtype('⊥', count)
            return dtypes_vector_mapper[(dtype, count)]

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

    @property
    def c0(self):
        return self.components[0]

    # Class attributes overrides

    @property
    def is_DiscreteFunction(self):
        return self.c0.is_DiscreteFunction

    @property
    def is_TimeFunction(self):
        return self.c0.is_TimeFunction

    @property
    def is_Input(self):
        return all(i.is_Input for i in self.components)

    @property
    def is_autopaddable(self):
        return all(i.is_autopaddable for i in self.components)

    # Other properties and methods

    @property
    def handles(self):
        return (self,)

    @property
    def components(self):
        return self._components

    @property
    def ncomp(self):
        return len(self.components)

    @property
    def initvalue(self):
        return None

    # Defaulting to self.c0's behaviour
    for i in ('_mem_internal_eager', '_mem_internal_lazy', '_mem_local',
              '_mem_mapped', '_mem_host', '_mem_stack', '_mem_constant',
              '_mem_shared', '_mem_shared_remote', '_mem_registers',
              '_mem_rvalue', '__padding_dtype__', '_size_domain', '_size_halo',
              '_size_owned', '_size_padding', '_size_nopad', '_size_nodomain',
              '_offset_domain', '_offset_halo', '_offset_owned',
              '_dist_dimensions', '_C_get_field', 'grid',
              *AbstractFunction.__properties__):
        locals()[i] = property(lambda self, v=i: getattr(self.c0, v))

    # Other overrides

    @cached_property
    def symbolic_shape(self):
        if self._mem_mapped:
            # E.g., `(uv_vec->shape[0], uv_vec->shape[1], uv_vec->shape[2])`
            from devito.symbolics import FieldFromPointer, IndexedPointer  # noqa
            ffp = FieldFromPointer(self._C_field_shape, self._C_symbol)
            ret = [s if is_integer(s) else IndexedPointer(ffp, i)
                   for i, s in enumerate(super().symbolic_shape)]
            return DimensionTuple(*ret, getters=self.dimensions)
        else:
            # There's no accompanying C struct, so we simply return `c0`'s symbolic
            # shape, i.e. something along the lines of  `(x_size, y_size, z_size)`
            return self.c0.symbolic_shape

    @property
    def _mem_heap(self):
        return not any([self._mem_stack, self._mem_shared, self._mem_shared_remote])

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

    @property
    def _C_ctype(self):
        if self._mem_mapped:
            return super()._C_ctype
        else:
            return POINTER(dtype_to_ctype(self.dtype))


class Bag(Bundle):

    """
    A Bag is like a Bundle but it doesn't represent a concrete object
    in the generated code. It's used by the compiler because, in certain
    passes, treating groups of Function homogeneously is more practical
    than keeping them separated.
    """

    @property
    def handles(self):
        return self.components


class BundleView(Bundle):

    """
    A BundleView is like a Bundle but it doesn't represent a concrete object
    in the generated code. It's used by the compiler to represent a subset
    of the components of a Bundle.
    """

    __rkwargs__ = Bundle.__rkwargs__ + ('parent',)

    def __new__(cls, *args, parent=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj._parent = parent

        return obj

    @property
    def parent(self):
        return self._parent

    @property
    def handles(self):
        return (self.parent,)

    @property
    def component_indices(self):
        return tuple(self.parent.components.index(i) for i in self.components)


class ComponentAccess(Expr, Pickable):

    _component_names = ('x', 'y', 'z', 'w')

    __rargs__ = ('arg',)
    __rkwargs__ = ('index',)

    def __new__(cls, arg, index=0, **kwargs):
        if not arg.is_Indexed:
            raise ValueError("Expected Indexed, got `%s` instead" % type(arg))
        if not is_integer(index) or index > 3:
            raise ValueError("Expected 0 <= index < 4")

        obj = Expr.__new__(cls, arg)
        obj._index = index

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self._index,)

    def __str__(self):
        return "%s.%s" % (self.base, self.sindex)

    __repr__ = __str__

    func = Pickable._rebuild

    def _sympystr(self, printer):
        return str(self)

    @property
    def base(self):
        return self.args[0]

    @property
    def arg(self):
        return self.base

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
    def function_access(self):
        return self.function.components[self.index]

    @property
    def indices(self):
        return self.base.indices

    @property
    def dtype(self):
        return self.function.dtype

    @cacheit
    def sort_key(self, order=None):
        # Ensure that the ComponentAccess is sorted as the base
        # Also ensure that e.g. `fg[x+1].x` appears before `fg[x+1].y`
        class_key, args, exp, coeff = self.base.sort_key(order=order)
        args = (len(args[1]) + 1, args[1] + (self.index,))
        return class_key, args, exp, coeff

    # Default assumptions correspond to those of the `base`
    for i in ('is_real', 'is_imaginary', 'is_commutative'):
        locals()[i] = property(lambda self, v=i: getattr(self.base, v))
