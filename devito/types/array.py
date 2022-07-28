from ctypes import POINTER, Structure, c_void_p, c_ulong
from math import ceil

import numpy as np
from cached_property import cached_property

from devito.parameters import configuration
from devito.tools import as_tuple, c_restrict_void_p, dtype_to_ctype
from devito.types.basic import AbstractFunction, IndexedData
from devito.types.utils import CtypesFactory

__all__ = ['Array', 'ArrayMapped', 'ArrayObject', 'PointerArray', 'IndexedArray']


class ArrayBasic(AbstractFunction):

    is_ArrayBasic = True

    @classmethod
    def __indices_setup__(cls, **kwargs):
        return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])

    @property
    def _C_name(self):
        if self._mem_stack:
            # No reason to distinguish between two different names, that is
            # the _C_name and the name -- just `self.name` is enough
            return self.name
        else:
            return super()._C_name

    @property
    def _C_aliases(self):
        return (self, self.indexed)

    @property
    def shape(self):
        return self.symbolic_shape

    shape_allocated = shape

    @cached_property
    def indexed(self):
        return IndexedArray(self.name, shape=self.shape, function=self.function)


class IndexedArray(IndexedData):

    @property
    def _C_aliases(self):
        return (self, self.function)


class Array(ArrayBasic):

    """
    Tensor symbol representing an array in symbolic equations.

    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code). An Array behaves similarly to
    a Function, but unlike a Function it carries no user data.

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
        'static'. Defaults to 'heap'. 'static' means a static array in a
        C/C++ sense and, therefore, implies 'stack'.
        Note: not all scopes make sense for a given space.
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
        assert self._scope in ['heap', 'stack', 'static']

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
        return self._scope == 'stack'

    @property
    def _mem_heap(self):
        return self._scope == 'heap'

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
    def _mem_heap(self):
        return False

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
