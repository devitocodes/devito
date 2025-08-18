import ctypes
import numpy as np

from devito.symbolics.extended_sympy import ReservedWord, Cast, ValueLimit
from devito.tools import (Bunch, float2, float3, float4, double2, double3, double4,  # noqa
                          int2, int3, int4, ctypes_vector_mapper)
from devito.tools.dtypes_lowering import dtype_mapper

__all__ = ['cast', 'CustomType', 'limits_mapper', 'INT', 'FLOAT', 'BaseCast',  # noqa
           'DOUBLE', 'VOID', 'NoDeclStruct', 'c_complex', 'c_double_complex',
           'LONG']


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
    """
    Structure for passing complex float to C/C++
    """

    _fields_ = [('real', ctypes.c_float), ('imag', ctypes.c_float)]

    _base_dtype = True

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


class c_double_complex(NoDeclStruct):
    """
    Structure for passing complex double to C/C++
    """

    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    _base_dtype = True

    @classmethod
    def from_param(cls, val):
        return cls(val.real, val.imag)


ctypes_vector_mapper.update({np.complex64: c_complex,
                             np.complex128: c_double_complex})


class CustomType(ReservedWord):
    pass


def cast(casttype, stars=None):
    return lambda v, dtype=None, **kw: Cast(v, dtype=casttype, stars=stars, **kw)


ULONG = cast(np.uint64)
UINTP = cast(np.uint32, '*')
LONG = cast(np.int64)


# Standard ones, needed as class for e.g. single dispatch
class BaseCast(Cast):

    def __new__(cls, base, stars=None, **kwargs):
        kwargs['dtype'] = cls._dtype
        return super().__new__(cls, base, stars=stars, **kwargs)


class VOID(BaseCast):

    _dtype = 'void'


# Dynamically create INT, INT2, .... INTP, INT2P, ... FLOAT, ...
for (base_name, dtype) in dtype_mapper.items():
    name = base_name.upper()
    globals()[name] = type(name, (BaseCast,), {'_dtype': dtype})
    for i in ['2', '3', '4']:
        v = '%s%s' % (base_name, i)
        globals()[v.upper()] = cast(v)
        globals()[f'{v.upper()}P'] = cast(v, '*')
