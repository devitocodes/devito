import ctypes
import numpy as np

from devito.symbolics.extended_sympy import ReservedWord, Cast, CastStar, ValueLimit
from devito.tools import (Bunch, float2, float3, float4, double2, double3, double4,  # noqa
                          int2, int3, int4)

__all__ = ['cast_mapper', 'CustomType', 'limits_mapper', 'INT', 'FLOAT',
           'DOUBLE', 'VOID', 'NoDeclStruct', 'c_complex', 'c_double_complex',
           'c_half', 'c_half_p', 'Float16P']


limits_mapper = {
    np.int32: Bunch(min=ValueLimit('INT_MIN'), max=ValueLimit('INT_MAX')),
    np.int64: Bunch(min=ValueLimit('LONG_MIN'), max=ValueLimit('LONG_MAX')),
    np.float32: Bunch(min=-ValueLimit('FLT_MAX'), max=ValueLimit('FLT_MAX')),
    np.float64: Bunch(min=-ValueLimit('DBL_MAX'), max=ValueLimit('DBL_MAX')),
}


class NoDeclStruct(ctypes.Structure):
    """A ctypes.Structure that does not generate a struct definition"""

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
    Dummy dtype for a scalar float16 value that's been mapped to a pointer.
    This is needed because we can't directly pass in the values; we map to
    pointers and dereference in the kernel; see `passes.iet.dtypes`.
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


class CHAR(Cast):
    _base_typ = 'char'


class SHORT(Cast):
    _base_typ = 'short'


class USHORT(Cast):
    _base_typ = 'unsigned short'


class UCHAR(Cast):
    _base_typ = 'unsigned char'


class LONG(Cast):
    _base_typ = 'long'


class ULONG(Cast):
    _base_typ = 'unsigned long'


class CFLOAT(Cast):
    _base_typ = 'float'


class CDOUBLE(Cast):
    _base_typ = 'double'


class VOID(Cast):
    _base_typ = 'void'


class CHARP(CastStar):
    base = CHAR


class UCHARP(CastStar):
    base = UCHAR


class SHORTP(CastStar):
    base = SHORT


class USHORTP(CastStar):
    base = USHORT


class CFLOATP(CastStar):
    base = CFLOAT


class CDOUBLEP(CastStar):
    base = CDOUBLE


cast_mapper = {
    np.int8: CHAR,
    np.uint8: UCHAR,
    np.int16: SHORT,  # noqa
    np.uint16: USHORT,  # noqa
    int: INT,  # noqa
    np.int32: INT,  # noqa
    np.int64: LONG,
    np.uint64: ULONG,
    np.float32: FLOAT,  # noqa
    float: DOUBLE,  # noqa
    np.float64: DOUBLE,  # noqa

    (np.int8, '*'): CHARP,
    (np.uint8, '*'): UCHARP,
    (int, '*'): INTP,  # noqa
    (np.uint16, '*'): USHORTP,  # noqa
    (np.int16, '*'): SHORTP,  # noqa
    (np.int32, '*'): INTP,  # noqa
    (np.int64, '*'): INTP,  # noqa
    (np.float32, '*'): FLOATP,  # noqa
    (float, '*'): DOUBLEP,  # noqa
    (np.float64, '*'): DOUBLEP,  # noqa
}

for base_name in ['int', 'float', 'double']:
    for i in [2, 3, 4]:
        v = '%s%d' % (base_name, i)
        cls = locals()[v]
        cast_mapper[cls] = locals()[v.upper()]
        cast_mapper[(cls, '*')] = locals()['%sP' % v.upper()]
