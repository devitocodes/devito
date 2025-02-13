import ctypes
import numpy as np

from devito.symbolics.extended_sympy import ReservedWord, Cast, ValueLimit
from devito.tools import (Bunch, float2, float3, float4, double2, double3, double4,  # noqa
                          int2, int3, int4, ctypes_vector_mapper)

__all__ = ['cast_mapper', 'CustomType', 'limits_mapper', 'INT', 'FLOAT', 'BaseCast',  # noqa
           'DOUBLE', 'VOID', 'NoDeclStruct', 'c_complex', 'c_double_complex']


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

    _base_dtype = True

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


class c_double_complex(NoDeclStruct):
    """Structure for passing complex double to C/C++"""

    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    _base_dtype = True

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


ctypes_vector_mapper.update({np.complex64: c_complex,
                             np.complex128: c_double_complex})


class CustomType(ReservedWord):
    pass


def cast_mapper(arg):
    try:
        assert len(arg) == 2 and arg[1] == '*'
        return lambda v, dtype=None, **kw: Cast(v, dtype=arg[0], stars=arg[1], **kw)
    except (AssertionError, TypeError):
        return lambda v, dtype=None, **kw: Cast(v, dtype=arg, **kw)


FLOAT = cast_mapper(np.float32)
DOUBLE = cast_mapper(np.float64)
ULONG = cast_mapper(np.uint64)
UINTP = cast_mapper((np.uint32, '*'))


# Standard ones, needed as class for e.g. single dispatch
class BaseCast(Cast):

    def __new__(cls, base, stars=None, **kwargs):
        kwargs['dtype'] = cls._dtype
        return super().__new__(cls, base, stars=stars, **kwargs)


class VOID(BaseCast):

    _dtype = 'void'


class INT(BaseCast):

    _dtype = np.int32


# Dynamically create INT, INT2, .... INTP, INT2P, ... FLOAT, ...
for base_name in ['int', 'float', 'double']:
    for i in ['2', '3', '4']:
        v = '%s%s' % (base_name, i)
        globals()[v.upper()] = cast_mapper(v)
        globals()[f'{v.upper()}P'] = cast_mapper((v, '*'))
