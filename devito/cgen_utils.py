from collections import OrderedDict

import cgen as c
from mpmath.libmp import prec_to_dps, to_str
from sympy import Eq
from sympy.printing.ccode import CCodePrinter


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
        ``obj``, of type :class:`SymbolicData`.
        """
        if obj in self.heap:
            return

        decl = "(*%s)%s" % (obj.name,
                            "".join("[%s]" % i.symbolic_size for i in obj.indices[1:]))
        decl = c.Value(c.dtype_to_ctype(obj.dtype), decl)

        shape = "".join("[%s]" % i.symbolic_size for i in obj.indices)
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

class CodePrinter(CCodePrinter):
    """Decorator for sympy.printing.ccode.CCodePrinter.

    :param settings: A dictionary containing relevant settings
    """
    def __init__(self, settings={}):
        CCodePrinter.__init__(self, settings)
        custom_functions = {
            'INT': '(int)',
            'FLOAT': '(float)'
        }
        self.known_functions.update(custom_functions)

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

        return '%d.0F/%d.0F' % (p, q)  # float precision by default

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

        return rv + 'F'

    def _print_FrozenExpr(self, expr):
        return self._print(expr.args[0])


def ccode(expr, **settings):
    """Generate C++ code from an expression calling CodePrinter class

    :param expr: The expression
    :param settings: A dictionary of settings for code printing
    :returns: The resulting code as a string. If it fails, then it returns the expr
    """
    if isinstance(expr, Eq):
        return ccode_eq(expr)
    try:
        return CodePrinter(settings).doprint(expr, None)
    except:
        return expr


def ccode_eq(eq, **settings):
    """Generate C++ assignment from an equation assigning RHS to LHS

    :param eq: The equation
    :param settings: A dictionary of settings for code printing
    :returns: The resulting code as a string
    """
    return CodePrinter(settings).doprint(eq.lhs, None) \
        + ' = ' + CodePrinter(settings).doprint(eq.rhs, None)


blankline = c.Line("")
