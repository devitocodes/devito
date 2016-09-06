import ctypes
import math

import cpuinfo
import numpy as np
from sympy import symbols


def flatten(l):
    return [item for sublist in l for item in sublist]


def filter_ordered(elements):
    """Filter elements in a list while preserving order"""
    seen = set()
    return [e for e in elements if not (e in seen or seen.add(e))]


def convert_dtype_to_ctype(dtype):
    """Maps numpy types to C types.

    :param dtype: A Python numpy type of int32, float32, int64 or float64
    :returns: Corresponding C type
    """
    conversion_dict = {np.int32: ctypes.c_int, np.float32: ctypes.c_float,
                       np.int64: ctypes.c_int64, np.float64: ctypes.c_double}

    return conversion_dict[dtype]


def sympy_find(expr, term, repl):
    """Change all terms from function notation to array notation.

    Finds all terms of the form term(x1, x2, x3)
    and changes them to repl[x1, x2, x3]. i.e. changes from
    function notation to array notation. It also reorders the indices
    x1, x2, x3 so that the time index comes first.

    :param expr: The expression to be processed
    :param term: The pattern to be replaced
    :param repl: The replacing pattern
    :returns: The changed expression
    """

    t = symbols("t")

    if type(expr) == term:
        args_wo_t = [x for x in expr.args if x != t and t not in x.args]
        args_t = [x for x in expr.args if x == t or t in x.args]
        expr = repl[tuple(args_t + args_wo_t)]

    if hasattr(expr, "args"):
        for a in expr.args:
            expr = expr.subs(a, sympy_find(a, term, repl))

    return expr


def aligned(a, alignment=16):
    """Function to align the memmory

    :param a: The given memory
    :param alignment: Granularity of alignment, 16 bytes by default
    :returns: Reference to the start of the aligned memory
    """
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)

    assert (aa.ctypes.data % alignment) == 0

    return aa


def get_optimal_block_size(shape, load_c):
    """Gets optimal block size based on architecture

    :param shape: list - shape of the data buffer
    :param load_c: int - load count
    :return: optimal block size

    Assuming no prefetching, square block will give the most cache reuse.
    We then take the cache/core divide by the size of the inner most dimension in which
    we do not block. This gives us the X*Y block space, of which we take the square root
     to get the size of our blocks.

    ((C size / cores) / (4 * length inner most * kernel loads)
    """
    cache_s = int(cpuinfo.get_cpu_info()['l2_cache_size'].split(' ')[0])  # cache size
    core_c = cpuinfo.get_cpu_info()['count']  # number of cores

    optimal_b_size = math.sqrt(
        ((1000 * cache_s) / core_c) / (4 * shape[len(shape) - 1] * load_c))
    return int(math.ceil(optimal_b_size))
