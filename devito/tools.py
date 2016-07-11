import os
import stat
import shutil
import ctypes
import numpy as np
from sympy import symbols


def convert_dtype_to_ctype(dtype):
    conversion_dict = {np.int32: ctypes.c_int, np.float32: ctypes.c_float,
                       np.int64: ctypes.c_int64, np.float64: ctypes.c_double}
    return conversion_dict[dtype]


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


def clean_folder(folder_path):
    """Helper method. Deletes all files and folders in the specified directory

    Args:
        folder_path(str): full path to the folder where we want to delete everything (use with care)
    """

    if not os.path.isdir(folder_path):  # returns if folder does not exist
        return

    try:
        for the_file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, the_file)

            if os.path.isfile(file_path):  # removes all files
                os.unlink(file_path)
            elif os.path.isdir(file_path):  # removes all dirs
                shutil.rmtree(file_path)

    except Exception as e:
        print "Failed to clean %s\n%s" % (folder_path, e)


def set_x_permission(file_path):
    """Helper method. Sets os executable permission for a given file
    file_path(str): full path to the file that we want to set x permission
    """
    if not os.path.isfile(file_path):
        return

    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)
