"""
Utilities to turn SymPy objects into C strings.
"""
import numpy as np
import sympy

from mpmath.libmp import prec_to_dps, to_str
from packaging.version import Version

from sympy.core import S
from sympy.core.numbers import equal_valued, Float
from sympy.printing.codeprinter import CodePrinter
from sympy.logic.boolalg import BooleanFunction
from sympy.printing.precedence import PRECEDENCE_VALUES, precedence

from devito import configuration
from devito.arch.compiler import AOMPCompiler
from devito.symbolics.inspection import has_integer_args, sympy_dtype
from devito.symbolics.queries import q_leaf
from devito.types.basic import AbstractFunction
from devito.tools import ctypes_to_cstr, dtype_to_ctype, ctypes_vector_mapper

__all__ = ['BasePrinter', 'ccode']


class BasePrinter(CodePrinter):

    """
    Decorator for sympy.printing.ccode.CCodePrinter.

    Parameters
    ----------
    settings : dict
        Options for code printing.
    """
    _default_settings = {'compiler': None, 'dtype': np.float32,
                         **CodePrinter._default_settings}

    _func_prefix = {}
    _func_literals = {}
    _prec_literals = {np.float32: 'F', np.complex64: 'F'}

    _qualifiers_mapper = {
        'is_const': 'const',
        'is_volatile': 'volatile',
        '_mem_constant': 'static',
        '_mem_shared': '',
    }

    _restrict_keyword = 'restrict'

    _includes = []
    _namespaces = []
    _headers = [('_POSIX_C_SOURCE', '200809L')]

    @property
    def dtype(self):
        try:
            return self._settings['dtype'].nptype
        except AttributeError:
            return self._settings['dtype']

    @property
    def compiler(self):
        return self._settings['compiler'] or configuration['compiler']

    def doprint(self, expr, assign_to=None):
        """
        The sympy code printer does a lot of extra things we do not need
        as we handle all of it in the compiler so we directly default to `_print`.
        """
        return self._print(expr)

    def _prec(self, expr):
        dtype = sympy_dtype(expr, default=self.dtype)
        if dtype is None or np.issubdtype(dtype, np.integer):
            if any(isinstance(i, Float) for i in expr.atoms()):
                try:
                    return np.promote_types(self.dtype, np.float32).type
                except np.exceptions.DTypePromotionError:
                    # Corner cases, e.g. Void, cannot (shouldn't) be promoted
                    return self.dtype
            else:
                return dtype or self.dtype
        else:
            return dtype or self.dtype

    def prec_literal(self, expr):
        return self._prec_literals.get(self._prec(expr), '')

    def func_literal(self, expr):
        return self._func_literals.get(self._prec(expr), '')

    def func_prefix(self, expr, mfunc=False):
        prefix = self._func_prefix.get(self._prec(expr), '')
        if mfunc:
            return prefix
        elif prefix == 'f':
            return ''
        else:
            return prefix

    def parenthesize(self, item, level, strict=False):
        if isinstance(item, BooleanFunction):
            return f"({self._print(item)})"
        return super().parenthesize(item, level, strict=strict)

    def _print_PyCPointerType(self, expr):
        ctype = f'{self._print_type(expr._type_)}'
        if ctype.endswith('*'):
            return f'{ctype}*'
        else:
            return f'{ctype} *'

    def _print_type(self, expr):
        try:
            expr = dtype_to_ctype(expr)
        except TypeError:
            pass
        try:
            return self.type_mappings[expr]
        except KeyError:
            return ctypes_to_cstr(expr)

    def _print_VoidDType(self, expr):
        return ctypes_vector_mapper[expr].__name__

    def _print_Function(self, expr):
        if isinstance(expr, AbstractFunction):
            return str(expr)
        else:
            if expr.func.__name__ not in self.known_functions:
                self.known_functions[expr.func.__name__] = expr.func.__name__
            return super()._print_Function(expr)

    def _print_CondEq(self, expr):
        return f"{self._print(expr.lhs)} == {self._print(expr.rhs)}"

    def _print_Indexed(self, expr):
        """
        Print an Indexed as a C-like multidimensional array.

        Examples
        --------
        U[t,x,y,z] -> U[t][x][y][z]
        """
        inds = ''.join(['[' + self._print(x) + ']' for x in expr.indices])
        return f'{self._print(expr.base.label)}{inds}'

    def _print_FIndexed(self, expr):
        """
        Print an FIndexed, that is a special Indexed, as a C-like multiarguments function.

        Examples
        --------
        U[t,x,y,z] -> U(t,x,y,z)
        """
        inds = ', '.join(self._print(x) for x in expr.indices)
        try:
            label = expr.accessor.label
        except AttributeError:
            label = expr.base.label
        return f'{self._print(label)}({inds})'

    def _print_Rational(self, expr):
        """Print a Rational as a C-like float/float division."""
        # This method and _print_Float below forcefully add a F to any
        # literals generated in code. This forces all float literals
        # to be 32-bit floats.
        # http://en.cppreference.com/w/cpp/language/floating_literal
        p, q = int(expr.p), int(expr.q)
        prec = self.prec_literal(expr)
        return f'{p}.0{prec}/{q}.0{prec}'

    def _print_math_func(self, expr, nest=False, known=None):
        cls = type(expr)
        name = cls.__name__

        try:
            cname = self.known_functions[name]
        except KeyError:
            return super()._print_math_func(expr, nest=nest, known=known)

        if cname not in self._prec_funcs:
            return super()._print_math_func(expr, nest=nest, known=known)

        cname = f'{self.func_prefix(expr)}{cname}{self.func_literal(expr)}'

        if nest and len(expr.args) > 2:
            args = ', '.join([self._print(expr.args[0]),
                              self._print_math_func(cls(*expr.args[1:]))])
        else:
            args = ', '.join([self._print(arg) for arg in expr.args])

        return f'{self._ns}{cname}({args})'

    def _print_Pow(self, expr):
        # Completely reimplement `_print_Pow` from sympy, since it doesn't
        # correctly handle precision
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self.func_literal(expr)
        base = self._print(expr.base)
        if equal_valued(expr.exp, -1):
            return self._print_Float(Float(1.0)) + '/' + \
                self.parenthesize(expr.base, PREC)
        elif equal_valued(expr.exp, 0.5):
            return f'{self._ns}sqrt{suffix}({base})'
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return f'{self._ns}cbrt{suffix}({base})'
        else:
            return f'{self._ns}pow{suffix}({base}, {self._print(expr.exp)})'

    def _print_SafeInv(self, expr):
        """Print a SafeInv as a C-like division with a check for zero."""
        base = self._print(expr.base)
        val = self._print(expr.val)
        return f'SAFEINV({val}, {base})'

    def _print_Mod(self, expr):
        """Print a Mod as a C-like %-based operation."""
        args = [f'({self._print(a)})' for a in expr.args]
        return '%'.join(args)

    def _print_Mul(self, expr):
        args = [a for a in expr.args if a != -1]
        neg = (len(expr.args) - len(args)) % 2

        if len(args) > 1:
            term = super()._print_Mul(expr.func(*args, evaluate=False))
        else:
            term = self.parenthesize(args[0], precedence(expr))

        if neg:
            return f'-{term}'
        else:
            return term

    def _print_fmath_func(self, name, expr):
        args = ",".join([self._print(i) for i in expr.args])
        func = f'{self.func_prefix(expr, mfunc=True)}{name}{self.func_literal(expr)}'
        return f"{self._ns}{func}({args})"

    def _print_Min(self, expr):
        if len(expr.args) > 2:
            return self._print_Min(expr.func(expr.args[0],
                                             expr.func(*expr.args[1:]),
                                             evaluate=False))
        elif has_integer_args(*expr.args) and len(expr.args) == 2:
            return f"MIN({self._print(expr.args)[1:-1]})"
        else:
            return self._print_fmath_func('min', expr)

    def _print_Max(self, expr):
        if len(expr.args) > 2:
            return self._print_Max(expr.func(expr.args[0],
                                             expr.func(*expr.args[1:]),
                                             evaluate=False))
        elif has_integer_args(*expr.args) and len(expr.args) == 2:
            return f"MAX({self._print(expr.args)[1:-1]})"
        else:
            return self._print_fmath_func('max', expr)

    def _print_Abs(self, expr):
        """Print an absolute value. Use `abs` if can infer it is an Integer"""
        # Unary function, single argument
        arg = expr.args[0]
        # AOMPCC errors with abs, always use fabs
        if isinstance(self.compiler, AOMPCompiler) and \
           not np.issubdtype(self._prec(expr), np.integer):
            return f"fabs({self._print(arg)})"
        return self._print_fmath_func('abs', expr)

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
                l.extend(["+", f"({t})"])
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
            rv = "-0." + rv[3:]
        elif rv.startswith('.0'):
            rv = "0." + rv[2:]

        # Remove trailing zero except first one to avoid 1. instead of 1.0
        if 'e' not in rv:
            rv = rv.rstrip('0') + "0"

        return f'{rv}{self.prec_literal(expr)}'

    def _print_Differentiable(self, expr):
        return f"({self._print(expr._expr)})"

    _print_EvalDerivative = _print_Add

    def _print_CallFromPointer(self, expr):
        indices = [self._print(i) for i in expr.params]
        return f"{expr.pointer}->{expr.call}({', '.join(indices)})"

    def _print_CallFromComposite(self, expr):
        indices = [self._print(i) for i in expr.params]
        return f"{expr.pointer}.{expr.call}({', '.join(indices)})"

    def _print_FieldFromPointer(self, expr):
        return f"{expr.pointer}->{expr.field}"

    def _print_FieldFromComposite(self, expr):
        return f"{expr.pointer}.{expr.field}"

    def _print_ListInitializer(self, expr):
        return f"{{{', '.join(self._print(i) for i in expr.params)}}}"

    def _print_IndexedPointer(self, expr):
        return f"{expr.base}{''.join(f'[{self._print(i)}]' for i in expr.index)}"

    def _print_IntDiv(self, expr):
        lhs = self._print(expr.lhs)
        if not expr.lhs.is_Atom:
            lhs = f"({lhs})"
        rhs = self._print(expr.rhs)
        PREC = precedence(expr)
        return self.parenthesize(f"{lhs} / {rhs}", PREC)

    def _print_InlineIf(self, expr):
        cond = self._print(expr.cond)
        true_expr = self._print(expr.true_expr)
        false_expr = self._print(expr.false_expr)
        PREC = precedence(expr)
        return self.parenthesize(f"({cond}) ? {true_expr} : {false_expr}", PREC)

    def _print_UnaryOp(self, expr, op=None, parenthesize=False):
        op = op or expr._op
        base = self._print(expr.base)
        if not q_leaf(expr.base) or parenthesize:
            base = f'({base})'
        return f'{op}{base}'

    def _print_Cast(self, expr):
        cast = f'({self._print(expr._C_ctype)}{self._print(expr.stars)})'
        return self._print_UnaryOp(expr, op=cast, parenthesize=not expr.base.is_Atom)

    def _print_ComponentAccess(self, expr):
        return f"{self._print(expr.base)}.{expr.sindex}"

    def _print_DefFunction(self, expr):
        arguments = [self._print(i) for i in expr.arguments]
        if expr.template:
            ctemplate = ','.join([str(i) for i in expr.template])
            template = f'<{ctemplate}>'
        else:
            template = ''
        args = ','.join(arguments)
        return f"{expr.name}{template}({args})"

    def _print_SizeOf(self, expr):
        return f'sizeof({self._print(expr.intype)}{self._print(expr.stars)})'

    def _print_MathFunction(self, expr):
        return f"{self._ns}{self._print_DefFunction(expr)}"

    def _print_Fallback(self, expr):
        return expr.__str__()

    _print_Namespace = _print_Fallback
    _print_Rvalue = _print_Fallback
    _print_MacroArgument = _print_Fallback
    _print_IndexedBase = _print_Fallback
    _print_IndexSum = _print_Fallback
    _print_ReservedWord = _print_Fallback
    _print_Basic = _print_Fallback
    _print_VectorAccess = _print_Fallback


# Lifted from SymPy so that we go through our own `_print_math_func`
for k in ('exp log sin cos tan ceiling floor').split():
    setattr(BasePrinter, f'_print_{k}', BasePrinter._print_math_func)


# Always parenthesize IntDiv and InlineIf within expressions
PRECEDENCE_VALUES['IntDiv'] = 1
PRECEDENCE_VALUES['InlineIf'] = 1


# Sympy 1.11 has introduced a bug in `_print_Add`, so we enforce here
# to always use the correct one from our printer
if Version(sympy.__version__) >= Version("1.11"):
    setattr(sympy.printing.str.StrPrinter, '_print_Add', BasePrinter._print_Add)


def ccode(expr, printer=None, **settings):
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
    if printer is None:
        from devito.passes.iet.languages.C import CPrinter
        printer = CPrinter
    return printer(settings=settings).doprint(expr, None)
