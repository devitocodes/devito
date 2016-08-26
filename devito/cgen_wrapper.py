import ctypes

from cgen import *


def convert_dtype_to_ctype(dtype):
    """Method used to convert Python types into C types

    :param dtype: A Python type
    :returns: Corresponding C type
    """
    conversion_dict = {'int64': ctypes.c_int64, 'float64': ctypes.c_float}
    return conversion_dict[str(dtype)]


class PrintStatement(Statement):
    """Class representing a C printf statement

    :param args: List of strings to be printed
    """
    def __init__(self, *args):
        args = list(args)
        args[0] = args[0].replace("\n", "\\n")
        args[0] = "\"%s\"" % args[0]
        arglist = ", ".join(args)
        super(PrintStatement, self).__init__("printf(%s)" % arglist)
