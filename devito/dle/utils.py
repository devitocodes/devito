import cpuinfo
import numpy as np

import cgen as c

"""
Compiler-specific language
"""
complang_ALL = {
    'IntelCompiler': {'ignore-deps': c.Pragma('ivdep'),
                      'ntstores': c.Pragma('vector nontemporal'),
                      'storefence': c.Statement('_mm_sfence()'),
                      'noinline': c.Pragma('noinline')}
}
complang_ALL['IntelKNLCompiler'] = complang_ALL['IntelCompiler']

"""
SIMD generic info
"""
simdinfo = {
    # Sizes in bytes of a vector register
    'sse': 16, 'see4_2': 16,
    'avx': 32, 'avx2': 32,
    'avx512f': 64
}


def get_simd_flag():
    """Retrieve the best SIMD flag on the current architecture."""
    if get_simd_flag.flag is None:
        ordered_known = ('sse', 'sse4_2', 'avx', 'avx2', 'avx512f')
        flags = cpuinfo.get_cpu_info().get('flags')
        if not flags:
            return None
        for i in reversed(ordered_known):
            if i in flags:
                get_simd_flag.flag = i
                return i
    else:
        # "Cached" because calls to cpuingo are expensive
        return get_simd_flag.flag
get_simd_flag.flag = None  # noqa


def get_simd_items(dtype):
    """Determine the number of items of type ``dtype`` that can fit in a SIMD
    register on the current architecture."""

    simd_size = simdinfo[get_simd_flag()]
    assert simd_size % np.dtype(dtype).itemsize == 0
    return int(simd_size / np.dtype(dtype).itemsize)
