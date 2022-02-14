from ctypes import POINTER, Structure
from math import ceil

import numpy as np
from cached_property import cached_property
from cgen import Struct, Value

from devito.parameters import configuration
from devito.tools import as_tuple, ctypes_to_cstr, dtype_to_ctype
from devito.types.basic import AbstractFunction, IndexedData

__all__ = ['Array', 'ArrayObject', 'PointerArray']


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
    def _C_ctype(self):
        return POINTER(dtype_to_ctype(self.dtype))

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
    space : str, optional
        The memory space. Allowed values: 'local', 'mapped'.  Defaults to 'local'.
        Used to override `_mem_local` and `_mem_mapped`.
    scope : str, optional
        The scope in the given memory space. Allowed values: 'heap', 'stack'.
        Defaults to 'heap'. This may not have an impact on certain platforms.
    initvalue : array-like, optional
        The initial content of the Array. Must be None if `scope='heap'`.

    Warnings
    --------
    Arrays are created and managed directly by Devito (IOW, they are not
    expected to be used directly in user code).
    """

    is_Array = True

    def __new__(cls, *args, **kwargs):
        kwargs.update({'options': {'evaluate': False}})
        return AbstractFunction.__new__(cls, *args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super(Array, self).__init_finalize__(*args, **kwargs)

        self._space = kwargs.get('space', 'local')
        assert self._space in ['local', 'remote', 'mapped']

        self._scope = kwargs.get('scope', 'heap')
        assert self._scope in ['heap', 'stack']

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
    def _C_typename(self):
        return ctypes_to_cstr(self._C_ctype)

    @property
    def space(self):
        return self._space

    @property
    def scope(self):
        return self._scope

    @property
    def _mem_local(self):
        return self._space == 'local'

    @property
    def _mem_mapped(self):
        return self._space == 'mapped'

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

    # Pickling support
    _pickle_kwargs = AbstractFunction._pickle_kwargs + ['dimensions', 'space', 'scope']


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

    def __init_finalize__(self, *args, **kwargs):
        name = kwargs['name']
        fields = tuple(kwargs.pop('fields', ()))

        self._fields = fields
        self._pname = "t%s" % name

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __dtype_setup__(cls, **kwargs):
        return POINTER(type("t%s" % kwargs['name'],
                            (Structure,),
                            {'_fields_': cls.__pfields_setup__(**kwargs)}))

    @classmethod
    def __pfields_setup__(cls, **kwargs):
        return [(i._C_name, i._C_ctype) for i in kwargs.get('fields', [])]

    @cached_property
    def _C_typename(self):
        return ctypes_to_cstr(self.dtype)

    @cached_property
    def _C_typedata(self):
        if self._is_composite_dtype:
            return ctypes_to_cstr(self.dtype._type_)
        else:
            return self._C_typename

    @cached_property
    def _C_typedecl(self):
        if self._is_composite_dtype:
            return Struct(self.pname,
                          [Value(ctypes_to_cstr(j), i) for i, j in self.pfields])
        else:
            return None

    @property
    def _is_composite_dtype(self):
        return len(self.fields) > 0

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

    # Pickling support
    _pickle_kwargs = ArrayBasic._pickle_kwargs + ['dimensions', 'fields']
    _pickle_kwargs.remove('dtype')


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
    def _C_typename(self):
        return ctypes_to_cstr(POINTER(self._C_ctype))

    @property
    def dim(self):
        """Shortcut for self.dimensions[0]."""
        return self.dimensions[0]

    @property
    def array(self):
        return self._array

    # Pickling support
    _pickle_kwargs = ['name', 'dimensions', 'array']
