"""
Utilities to turn SymPy objects into C strings.
"""

import numpy as np
import sympy

from mpmath.libmp import prec_to_dps, to_str
from packaging.version import Version
from sympy.logic.boolalg import BooleanFunction
from sympy.printing.precedence import PRECEDENCE_VALUES, precedence
from sympy.printing.c import C99CodePrinter

from devito.arch.compiler import AOMPCompiler

__all__ = ['ccode']


class CodePrinter(C99CodePrinter):

    """
    Decorator for sympy.printing.ccode.CCodePrinter.

    Parameters
    ----------
    settings : dict
        Options for code printing.
    """
    _default_settings = {'compiler': None, 'dtype': np.float32,
                         **C99CodePrinter._default_settings}

    @property
    def dtype(self):
        return self._settings['dtype']

    @property
    def compiler(self):
        return self._settings['compiler']

    def parenthesize(self, item, level, strict=False):
        if isinstance(item, BooleanFunction):
            return "(%s)" % self._print(item)
        return super().parenthesize(item, level, strict=strict)

    def _print_Function(self, expr):
        # There exist no unknown Functions
        if expr.func.__name__ not in self.known_functions:
            self.known_functions[expr.func.__name__] = expr.func.__name__
        return super()._print_Function(expr)

    def _print_CondEq(self, expr):
        return "%s == %s" % (self._print(expr.lhs), self._print(expr.rhs))

    def _print_Indexed(self, expr):
        """
        Print an Indexed as a C-like multidimensional array.

        Examples
        --------
        U[t,x,y,z] -> U[t][x][y][z]
        """
        inds = ''.join(['[' + self._print(x) + ']' for x in expr.indices])
        return '%s%s' % (self._print(expr.base.label), inds)

    def _print_FIndexed(self, expr):
        """
        Print an FIndexed, that is a special Indexed, as a C-like multiarguments function.

        Examples
        --------
        U[t,x,y,z] -> U(t,x,y,z)
        """
        inds = ', '.join(self._print(x) for x in expr.indices)
        return '%s(%s)' % (self._print(expr.base.label), inds)

    def _print_Rational(self, expr):
        """Print a Rational as a C-like float/float division."""
        # This method and _print_Float below forcefully add a F to any
        # literals generated in code. This forces all float literals
        # to be 32-bit floats.
        # http://en.cppreference.com/w/cpp/language/floating_literal
        p, q = int(expr.p), int(expr.q)
        if self.dtype == np.float64:
            return '%d.0/%d.0' % (p, q)
        else:
            return '%d.0F/%d.0F' % (p, q)

    def _print_Pow(self, expr):
        # Need to override because of issue #1627
        # E.g., (Pow(h_x, -1) AND h_x.dtype == np.float32) => 1.0F/h_x
        try:
            if expr.exp == -1 and self.dtype == np.float32:
                PREC = precedence(expr)
                return '1.0F/%s' % self.parenthesize(expr.base, PREC)
        except AttributeError:
            pass
        return super()._print_Pow(expr)

    def _print_Mod(self, expr):
        """Print a Mod as a C-like %-based operation."""
        args = ['(%s)' % self._print(a) for a in expr.args]
        return '%'.join(args)

    def _print_Min(self, expr):
        """Print Min using devito defined header Min"""
        func = 'MIN' if has_integer_args(*expr.args) else 'fmin'
        return "%s(%s)" % (func, self._print(expr.args)[1:-1])

    def _print_Max(self, expr):
        """Print Max using devito defined header Max"""
        func = 'MAX' if has_integer_args(*expr.args) else 'fmax'
        return "%s(%s)" % (func, self._print(expr.args)[1:-1])

    def _print_Abs(self, expr):
        """Print an absolute value. Use `abs` if can infer it is an Integer"""
        # AOMPCC errors with abs, always use fabs
        if isinstance(self.compiler, AOMPCompiler):
            return "fabs(%s)" % self._print(expr.args[0])
        # Check if argument is an integer
        func = "abs" if has_integer_args(*expr.args[0].args) else "fabs"
        return "%s(%s)" % (func, self._print(expr.args[0]))

    def _print_Add(self, expr, order=None):
        """"
        Print an addition.
        """
        terms = self._as_ordered_terms(expr, order=order)

        PREC = precedence(expr)
        l = []
        for term in terms:
            t = self._print(term)
            if precedence(term) < PREC:
                l.extend(["+", "(%s)" % t])
            elif t.startswith('-'):
                l.extend(["-", t[1:]])
            else:
                l.extend(["+", t])
        sign = l.pop(0)
        if sign == '+':
            sign = ""
        return sign + ' '.join(l)

    def _print_Float(self, expr):
        """Print a Float in C-like scientific notation."""
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

    def _print_Differentiable(self, expr):
        return "(%s)" % self._print(expr._expr)

    _print_EvalDerivative = C99CodePrinter._print_Add

    def _print_CallFromPointer(self, expr):
        indices = [self._print(i) for i in expr.params]
        return "%s->%s(%s)" % (expr.pointer, expr.call, ', '.join(indices))

    def _print_FieldFromPointer(self, expr):
        return "%s->%s" % (expr.pointer, expr.field)

    def _print_FieldFromComposite(self, expr):
        return "%s.%s" % (expr.pointer, expr.field)

    def _print_ListInitializer(self, expr):
        return "{%s}" % ', '.join([self._print(i) for i in expr.params])

    def _print_IndexedPointer(self, expr):
        return "%s%s" % (expr.base, ''.join('[%s]' % self._print(i) for i in expr.index))

    def _print_IntDiv(self, expr):
        lhs = self._print(expr.lhs)
        if not expr.lhs.is_Atom:
            lhs = '(%s)' % (lhs)
        rhs = self._print(expr.rhs)
        PREC = precedence(expr)
        return self.parenthesize("%s / %s" % (lhs, rhs), PREC)

    def _print_InlineIf(self, expr):
        cond = self._print(expr.cond)
        true_expr = self._print(expr.true_expr)
        false_expr = self._print(expr.false_expr)
        PREC = precedence(expr)
        return self.parenthesize("(%s) ? %s : %s" % (cond, true_expr, false_expr), PREC)

    def _print_UnaryOp(self, expr):
        if expr.base.is_Symbol:
            return "%s%s" % (expr._op, self._print(expr.base))
        else:
            return "%s(%s)" % (expr._op, self._print(expr.base))

    def _print_ComponentAccess(self, expr):
        return "%s.%s" % (self._print(expr.base), expr.sindex)

    def _print_TrigonometricFunction(self, expr):
        func_name = str(expr.func)
        if self.dtype == np.float32:
            func_name += 'f'
        return '%s(%s)' % (func_name, self._print(*expr.args))

    def _print_Fallback(self, expr):
        return expr.__str__()

    _print_DefFunction = _print_Fallback
    _print_MacroArgument = _print_Fallback
    _print_IndexedBase = _print_Fallback
    _print_IndexSum = _print_Fallback
    _print_Keyword = _print_Fallback
    _print_Basic = _print_Fallback


# Always parenthesize IntDiv and InlineIf within expressions
PRECEDENCE_VALUES['IntDiv'] = 1
PRECEDENCE_VALUES['InlineIf'] = 1


def ccode(expr, **settings):
    """Generate C++ code from an expression.

    Parameters
    ----------
    expr : expr-like
        The expression to be printed.
    settings : dict
        Options for code printing.

    Returns
    -------
    str
        The resulting code as a C++ string. If something went south, returns
        the input ``expr`` itself.
    """
    return CodePrinter(settings=settings).doprint(expr, None)


# Sympy 1.11 has introduced a bug in `_print_Add`, so we enforce here
# to always use the correct one from our printer
if Version(sympy.__version__) >= Version("1.11"):
    setattr(sympy.printing.str.StrPrinter, '_print_Add', CodePrinter._print_Add)


# Check arguements type
def has_integer_args(*args):
    """
    Check if expression is Integer.
    Used to choose the function printed in the c-code
    """
    if len(args) == 0:
        return False

    if len(args) == 1:
        try:
            return np.issubdtype(args[0].dtype, np.integer)
        except AttributeError:
            return args[0].is_integer

    res = True
    for a in args:
        try:
            if len(a.args) > 0:
                res = res and has_integer_args(*a.args)
            else:
                res = res and has_integer_args(a)
        except AttributeError:
            res = res and has_integer_args(a)
    return res
