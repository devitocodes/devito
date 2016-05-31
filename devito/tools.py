import ctypes
from sympy import symbols
import numpy as np


def convert_dtype_to_ctype(dtype):
    conversion_dict = {'int64': ctypes.c_int64, 'float64': ctypes.c_float}
    return conversion_dict[str(dtype)]


def sympy_find(expr, term, repl):
    """ This function finds all terms of the form term(x1, x2, x3)
    and changes them to repl[x1, x2, x3]. i.e. changes from
    function notation to array notation. It also reorders the indices
    x1, x2, x3 so that the time index comes first.
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
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)
    assert (aa.ctypes.data % alignment) == 0
    return aa
