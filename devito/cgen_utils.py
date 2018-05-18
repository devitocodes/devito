from collections import OrderedDict

import numpy as np
import cgen as c

from mpmath.libmp import prec_to_dps, to_str
from sympy import Function
from sympy.printing.ccode import C99CodePrinter


class Allocator(object):

    """
    Generate C strings to declare pointers, allocate and free memory.
    """

    def __init__(self):
        self.heap = OrderedDict()
        self.stack = OrderedDict()

    def push_stack(self, scope, obj):
        """
        Generate a cgen statement that allocates ``obj`` on the stack.
        """
        shape = "".join("[%s]" % ccode(i) for i in obj.symbolic_shape)
        alignment = "__attribute__((aligned(64)))"
        handle = self.stack.setdefault(scope, OrderedDict())
        handle[obj] = c.POD(obj.dtype, "%s%s %s" % (obj.name, shape, alignment))

    def push_heap(self, obj):
        """
        Generate cgen objects to declare, allocate memory, and free memory for
        ``obj``, of type :class:`Array`.
        """
        if obj in self.heap:
            return

        decl = "(*%s)%s" % (obj.name, "".join("[%s]" % i for i in obj.symbolic_shape[1:]))
        decl = c.Value(c.dtype_to_ctype(obj.dtype), decl)

        shape = "".join("[%s]" % i for i in obj.symbolic_shape)
        alloc = "posix_memalign((void**)&%s, 64, sizeof(%s%s))"
        alloc = alloc % (obj.name, c.dtype_to_ctype(obj.dtype), shape)
        alloc = c.Statement(alloc)

        free = c.Statement('free(%s)' % obj.name)

        self.heap[obj] = (decl, alloc, free)

    @property
    def onstack(self):
        return [(k, v.values()) for k, v in self.stack.items()]

    @property
    def onheap(self):
        return self.heap.values()


# Utils to print C strings

class CodePrinter(C99CodePrinter):

    custom_functions = {'INT': '(int)', 'FLOAT': '(float)', 'DOUBLE': '(double)'}

    """Decorator for sympy.printing.ccode.CCodePrinter.

    :param settings: A dictionary containing relevant settings
    """
    def __init__(self, dtype=np.float32, settings={}):
        self.dtype = dtype
        C99CodePrinter.__init__(self, settings)
        self.known_functions.update(self.custom_functions)

    def _print_CondEq(self, expr):
        return "%s == %s" % (self._print(expr.lhs), self._print(expr.rhs))

    def _print_Indexed(self, expr):
        """Print field as C style multidimensional array

        :param expr: An indexed expression

        e.g. U[t,x,y,z] -> U[t][x][y][z]

        :returns: The resulting string
        """
        output = self._print(expr.base.label) \
            + ''.join(['[' + self._print(x) + ']' for x in expr.indices])

        return output

    def _print_Rational(self, expr):
        """Print fractional number as float/float

        :param expr: A rational number

        (default was long double/long double)

        :returns: The resulting code as a string
        """
        # This method and _print_Float below forcefully add a F to any
        # literals generated in code. This forces all float literals
        # to be 32-bit floats.
        # http://en.cppreference.com/w/cpp/language/floating_literal
        p, q = int(expr.p), int(expr.q)
        if self.dtype == np.float64:
            return '%d.0/%d.0' % (p, q)
        else:
            return '%d.0F/%d.0F' % (p, q)

    def _print_Mod(self, expr):
        """Print mod using % operator in C++

        :param expr: The expression in which a C++ % operator is inserted
        :returns: The resulting code as a string
        """
        args = map(ccode, expr.args)
        args = ['('+x+')' for x in args]

        result = '%'.join(args)
        return result

    def _print_Float(self, expr):
        """Always printing floating point numbers in scientific notation

        :param expr: A floating point number
        :returns: The resulting code as a string
        """
        prec = expr._prec

        if prec < 5:
            dps = 0
        else:
            dps = prec_to_dps(expr._prec)
        if self._settings["full_prec"] is True:
            strip = False
        elif self._settings["full_prec"] is False:
            strip = True
        elif self._settings["full_prec"] == "auto":
            strip = self._print_level > 1

        rv = to_str(expr._mpf_, dps, strip_zeros=strip, max_fixed=-2, min_fixed=2)

        if rv.startswith('-.0'):
            rv = '-0.' + rv[3:]
        elif rv.startswith('.0'):
            rv = '0.' + rv[2:]

        if self.dtype == np.float32:
            rv = rv + 'F'
        return rv

    def _print_FrozenExpr(self, expr):
        return self._print(expr.args[0])

    def _print_FunctionFromPointer(self, expr):
        indices = [self._print(i) for i in expr.params]
        return "%s->%s(%s)" % (expr.pointer, expr.function, ', '.join(indices))

    def _print_ListInitializer(self, expr):
        return "{%s}" % ', '.join([self._print(i) for i in expr.params])

    def _print_IntDiv(self, expr):
        return str(expr)


def ccode(expr, dtype=np.float32, **settings):
    """Generate C++ code from an expression calling CodePrinter class

    :param expr: The expression
    :param settings: A dictionary of settings for code printing
    :returns: The resulting code as a string. If it fails, then it returns the expr
    """
    return CodePrinter(dtype=dtype, settings=settings).doprint(expr, None)


blankline = c.Line("")
printmark = lambda i: c.Line('printf("Here: %s\\n"); fflush(stdout);' % i)
printvar = lambda i: c.Statement('printf("%s=%%s\\n", %s); fflush(stdout);' % (i, i))
INT = Function('INT')
FLOAT = Function('FLOAT')
DOUBLE = Function('DOUBLE')

cast_mapper = {np.float32: FLOAT, float: DOUBLE, np.float64: DOUBLE}
