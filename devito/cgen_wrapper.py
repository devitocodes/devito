from cgen import *
import ctypes


def convert_dtype_to_ctype(dtype):
    conversion_dict = {'int64': ctypes.c_int64, 'float64': ctypes.c_float}
    return conversion_dict[str(dtype)]
