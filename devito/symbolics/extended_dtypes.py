import ctypes
import numpy as np

from devito.symbolics.extended_sympy import ReservedWord, Cast, CastStar, ValueLimit
from devito.tools import (Bunch, float2, float3, float4, double2, double3, double4,  # noqa
                          int2, int3, int4)

__all__ = ['cast_mapper', 'CustomType', 'limits_mapper',
           'INT', 'FLOAT', 'DOUBLE',  # noqa
           'VOID', 'NoDeclStruct', 'c_complex', 'c_double_complex',
           'c_half', 'c_half_p', 'Float16P']


limits_mapper = {
    np.int32: Bunch(min=ValueLimit('INT_MIN'), max=ValueLimit('INT_MAX')),
    np.int64: Bunch(min=ValueLimit('LONG_MIN'), max=ValueLimit('LONG_MAX')),
    np.float32: Bunch(min=-ValueLimit('FLT_MAX'), max=ValueLimit('FLT_MAX')),
    np.float64: Bunch(min=-ValueLimit('DBL_MAX'), max=ValueLimit('DBL_MAX')),
}


class NoDeclStruct(ctypes.Structure):
    """
    A ctypes.Structure that does not generate a struct definition.

    Some foreign types (e.g. complex) need to be passed to C/C++ as a struct
    that mimics an existing type, but the struct types themselves don't show
    up in the kernel, so we don't need to generate their definitions.
    """

    pass


class c_complex(NoDeclStruct):
    """Structure for passing complex float to C/C++"""

    _fields_ = [('real', ctypes.c_float), ('imag', ctypes.c_float)]

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


class c_double_complex(NoDeclStruct):
    """Structure for passing complex double to C/C++"""

    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


class c_half(ctypes.c_uint16):
    """Ctype for non-scalar half floats"""

    @classmethod
    def from_param(cls, val):
        return cls(np.float16(val).view(np.uint16))


class c_half_p(ctypes.POINTER(c_half)):
    """
    Ctype for half scalars; we can't directly pass _Float16 values so
    we use a pointer and dereference (see `passes.iet.dtypes`)
    """

    @classmethod
    def from_param(cls, val):
        arr = np.array(val, dtype=np.float16)
        return arr.ctypes.data_as(cls)


class Float16P(np.float16):
    """
    Dummy dtype for a scalar half value that has been mapped to a pointer.
    This is needed because we can't directly pass in the values; we map to
    pointers and dereference in the kernel. See `passes.iet.dtypes`.
    """

    pass


class CustomType(ReservedWord):
    pass


# Dynamically create INT, INT2, .... INTP, INT2P, ... FLOAT, ...
for base_name in ['int', 'float', 'double']:
    for i in ['', '2', '3', '4']:
        v = '%s%s' % (base_name, i)
        cls = type(v.upper(), (Cast,), {'_base_typ': v})
        globals()[cls.__name__] = cls

        clsp = type('%sP' % v.upper(), (CastStar,), {'base': cls})
        globals()[clsp.__name__] = clsp


def cast_mapper(arg):
    if len(arg) == 2 and arg[1] == '*':
        return lambda v, **kw: CastStar(arg[0], v, **kw)
    else:
        return lambda v, **kw: Cast(arg, v, **kw)


# Standard ones


class VOID(Cast):

    __rargs__ = ('base',)

    def __new__(cls, base, stars=None, **kwargs):
        dtype = type('void', (ctypes.c_int,), {})
        return super().__new__(cls, dtype, base, stars=stars, **kwargs)


class INT(Cast):

    __rargs__ = ('base',)

    def __new__(cls, base, stars=None, **kwargs):
        return super().__new__(cls, np.int32, base, stars=stars, **kwargs)


class FLOAT(Cast):

    __rargs__ = ('base',)

    def __new__(cls, base, stars=None, **kwargs):
        return super().__new__(cls, np.float32, base, stars=stars, **kwargs)


class DOUBLE(Cast):

    __rargs__ = ('base',)

    def __new__(cls, base, stars=None, **kwargs):
        return super().__new__(cls, np.float64, base, stars, **kwargs)
