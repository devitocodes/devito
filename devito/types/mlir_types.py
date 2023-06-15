import ctypes
import numpy as np

f32 = ctypes.c_float
f64 = ctypes.c_double

i32 = ctypes.c_int32
i64 = ctypes.c_int64

index  = ctypes.c_size_t

ptr_of = ctypes.POINTER

def memref_of_type_and_rank(dtype, rank: int):
    class Memref(ctypes.Structure):
        """ Matches memref<?x?xf23> """
        _fields_ = [
            ('ptr', ptr_of(dtype)),
            ('aligned', ptr_of(dtype)),
            ('offset', index),
            ('size', index * rank),
            ('stride', index * rank)
        ]
    return Memref

def make_memref_f32_struct_from_np(data: np.ndarray):
    rank = len(data.shape)
    memref_t = memref_of_type_and_rank(f32, rank)
    data_ptr = data.ctypes.data_as(ptr_of(f32))

    return memref_t(
        data_ptr,
        data_ptr,
        0,
        (ctypes.c_size_t * rank)(data.shape),
        (ctypes.c_size_t * rank)([1] * rank)
    )

    




