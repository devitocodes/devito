"""
Extended SymPy hierarchy.
"""

import numpy as np
import sympy
from sympy import Expr, Integer, Float, Function, Symbol
from sympy.core.basic import _aresame
from sympy.functions.elementary.trigonometric import TrigonometricFunction

from devito.tools import Pickable, as_tuple

__all__ = ['FrozenExpr', 'Eq', 'CondEq', 'CondNe', 'Mul', 'Add', 'Pow', 'IntDiv',
           'FunctionFromPointer', 'FieldFromPointer', 'FieldFromComposite',
           'ListInitializer', 'Byref', 'IndexedPointer', 'Macro', 'Literal',
           'taylor_sin', 'taylor_cos', 'bhaskara_sin', 'bhaskara_cos',
           'INT', 'FLOAT', 'DOUBLE', 'FLOOR', 'cast_mapper']


class FrozenExpr(Expr):

    """
    Use FrozenExpr in place of sympy.Expr to make sure than an e an expression
    is no longer transformable; that is, standard manipulations such as
    xreplace, collect, expand, ... have no effect, thus building a new
    expression identical to self.

    Notes
    -----
    At the moment, only xreplace is overridded (to prevent unpicking factorizations)
    """

    def xreplace(self, rule):
        if self in rule:
            return rule[self]
        elif rule:
            args = []
            for a in self.args:
                try:
                    args.append(a.xreplace(rule))
                except AttributeError:
                    args.append(a)
            args = tuple(args)
            if not _aresame(args, self.args):
                return self.func(*args, evaluate=False)
        return self

    def evalf(self, *args, **kwargs):
        return self


class Eq(sympy.Eq, FrozenExpr):

    """A customized version of sympy.Eq which suppresses evaluation."""

    def __new__(cls, *args, **kwargs):
        return sympy.Eq.__new__(cls, *args, evaluate=False)


class CondEq(sympy.Eq, FrozenExpr):

    """
    A customized version of sympy.Eq representing a conditional equality.
    It suppresses evaluation.
    """

    def __new__(cls, *args, **kwargs):
        return sympy.Eq.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self

    @property
    def negated(self):
        return CondNe(*self.args, evaluate=False)


class CondNe(sympy.Ne, FrozenExpr):

    """
    A customized version of sympy.Ne representing a conditional inequality.
    It suppresses evaluation.
    """

    def __new__(cls, *args, **kwargs):
        return sympy.Ne.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self

    @property
    def negated(self):
        return CondEq(*self.args, evaluate=False)


class Mul(sympy.Mul, FrozenExpr):
    def __new__(cls, *args, **kwargs):
        return sympy.Mul.__new__(cls, *args, evaluate=False)


class Add(sympy.Add, FrozenExpr):
    def __new__(cls, *args, **kwargs):
        return sympy.Add.__new__(cls, *args, evaluate=False)


class Pow(sympy.Pow, FrozenExpr):
    def __new__(cls, *args, **kwargs):
        return sympy.Pow.__new__(cls, *args, evaluate=False)


class IntDiv(sympy.Expr):

    """
    A support type for integer division. Should only be used by the compiler
    for code generation purposes (i.e., not for symbolic manipulation).
    This works around the annoying way SymPy represents integer division,
    namely as a ``Mul`` between the numerator and the reciprocal of the
    denominator (e.g., ``a*3.args -> (a, 1/3)), which ends up generating
    "weird" C code.
    """

    is_Atom = True

    def __new__(cls, lhs, rhs, params=None):
        try:
            rhs = Integer(rhs)
            if rhs == 0:
                raise ValueError("Cannot divide by 0")
            elif rhs == 1:
                return lhs
        except TypeError:
            # We must be sure the symbolic RHS is of type int
            if not hasattr(rhs, 'dtype'):
                raise ValueError("Symbolic RHS `%s` lacks dtype" % rhs)
            if not issubclass(rhs.dtype, np.integer):
                raise ValueError("Symbolic RHS `%s` must be of type `int`, found "
                                 "`%s` instead" % (rhs, rhs.dtype))
        obj = sympy.Expr.__new__(cls, lhs, rhs)
        obj.lhs = lhs
        obj.rhs = rhs
        return obj

    def __str__(self):
        return "%s / %s" % (self.lhs, self.rhs)

    __repr__ = __str__


class FunctionFromPointer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``pointer->function(params)``.
    """

    def __new__(cls, function, pointer, params=None):
        args = []
        if isinstance(pointer, str):
            pointer = Symbol(pointer)
        args.append(pointer)
        if isinstance(function, FunctionFromPointer):
            args.append(function)
        elif not isinstance(function, str):
            raise ValueError("`function` must be FunctionFromPointer or str")
        _params = []
        for p in as_tuple(params):
            if isinstance(p, str):
                _params.append(Symbol(p))
            elif not isinstance(p, Expr):
                raise ValueError("`params` must be an iterable of Expr or str")
            else:
                _params.append(p)
        args.extend(_params)
        obj = sympy.Expr.__new__(cls, *args)
        obj.function = function
        obj.pointer = pointer
        obj.params = tuple(_params)
        return obj

    def __str__(self):
        return '%s->%s(%s)' % (self.pointer, self.function,
                               ", ".join(str(i) for i in as_tuple(self.params)))

    __repr__ = __str__

    def _hashable_content(self):
        return super(FunctionFromPointer, self)._hashable_content() +\
            (self.function, self.pointer) + self.params

    @property
    def base(self):
        if isinstance(self.pointer, FunctionFromPointer):
            # FunctionFromPointer may be nested
            return self.pointer.base
        else:
            return self.pointer

    # Pickling support
    _pickle_args = ['function', 'pointer']
    _pickle_kwargs = ['params']
    __reduce_ex__ = Pickable.__reduce_ex__


class FieldFromPointer(FunctionFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``pointer->field``.
    """

    def __new__(cls, field, pointer):
        return FunctionFromPointer.__new__(cls, field, pointer)

    def __str__(self):
        return '%s->%s' % (self.pointer, self.field)

    @property
    def field(self):
        return self.function

    # Our __new__ cannot accept the params argument
    _pickle_kwargs = []

    __repr__ = __str__


class FieldFromComposite(FunctionFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``composite.field``,
    where ``composite`` is a struct/union/...
    """

    def __new__(cls, field, composite):
        return FunctionFromPointer.__new__(cls, field, composite)

    def __str__(self):
        return '%s.%s' % (self.composite, self.field)

    @property
    def field(self):
        return self.function

    @property
    def composite(self):
        return self.pointer

    # Our __new__ cannot accept the params argument
    _pickle_kwargs = []

    __repr__ = __str__


class ListInitializer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C++ list initializer notation ``{a, b, ...}``.
    """

    def __new__(cls, params):
        args = []
        for p in as_tuple(params):
            if isinstance(p, str):
                args.append(Symbol(p))
            elif isinstance(p, int):
                args.append(p)
            elif not isinstance(p, Expr):
                raise ValueError("`params` must be an iterable of Expr or str")
            else:
                args.append(p)
        obj = sympy.Expr.__new__(cls, *args)
        obj.params = tuple(args)
        return obj

    def __str__(self):
        return "{%s}" % ", ".join(str(i) for i in self.params)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['params']
    __reduce_ex__ = Pickable.__reduce_ex__


class Byref(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``&symbol``.
    """

    def __new__(cls, base):
        if isinstance(base, str):
            base = Symbol(base)
        elif not isinstance(base, sympy.Expr):
            raise ValueError("`base` must be sympy.Expr or str")
        obj = sympy.Expr.__new__(cls, base)
        obj._base = base
        return obj

    @property
    def base(self):
        return self._base

    def __str__(self):
        if self.base.is_Symbol:
            return "&%s" % self.base
        else:
            return "&(%s)" % self.base

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['base']
    __reduce_ex__ = Pickable.__reduce_ex__


class IndexedPointer(sympy.Expr):

    """
    Symbolic representation of the C notation ``symbol[...]``

    Unlike a sympy.Indexed, an IndexedPointer accepts, as base, objects that
    are not necessarily a Symbol or an IndexedBase, such as a FieldFromPointer.
    """

    def __new__(cls, base, index):
        if isinstance(base, (str, sympy.IndexedBase, sympy.Symbol)):
            return sympy.Indexed(base, index)
        elif not isinstance(base, sympy.Basic):
            raise ValueError("`base` must be of type sympy.Basic")
        obj = sympy.Expr.__new__(cls, base)
        obj._base = base
        obj._index = as_tuple(index)
        return obj

    @property
    def base(self):
        return self._base

    @property
    def index(self):
        return self._index

    def __str__(self):
        return "%s%s" % (self.base, ''.join('[%s]' % i for i in self.index))

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['base', 'index']
    __reduce_ex__ = Pickable.__reduce_ex__


class Macro(sympy.Symbol):

    """
    Symbolic representation of a C macro.
    """
    pass


class Literal(sympy.Symbol):

    """
    Symbolic representation of a Literal element.
    """
    pass


INT = Function('INT')
FLOAT = Function('FLOAT')
DOUBLE = Function('DOUBLE')
FLOOR = Function('floor')

cast_mapper = {np.float32: FLOAT, float: DOUBLE, np.float64: DOUBLE}


class taylor_sin(TrigonometricFunction):

    """
    Approximation of the sine function using a Taylor polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return eval_taylor_sin(arg)


class taylor_cos(TrigonometricFunction):

    """
    Approximation of the cosine function using a Taylor polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 1.0 if arg == 0.0 else eval_taylor_cos(arg)


class bhaskara_sin(TrigonometricFunction):

    """
    Approximation of the sine function using a Bhaskara polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return eval_bhaskara_sin(arg)


class bhaskara_cos(TrigonometricFunction):

    """
    Approximation of the cosine function using a Bhaskara polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 1.0 if arg == 0.0 else eval_bhaskara_sin(arg + 1.5708)


# Utils

def eval_bhaskara_sin(expr):
    return 16.0*expr*(3.1416-abs(expr))/(49.3483-4.0*abs(expr)*(3.1416-abs(expr)))


def eval_taylor_sin(expr):
    v = expr + Mul(-1/6.0,
                   Mul(expr, expr, expr, evaluate=False),
                   1.0 + Mul(Mul(expr, expr, evaluate=False), -0.05, evaluate=False),
                   evaluate=False)
    try:
        Float(expr)
        return v.doit()
    except (TypeError, ValueError):
        return v


def eval_taylor_cos(expr):
    v = 1.0 + Mul(-0.5,
                  Mul(expr, expr, evaluate=False),
                  1.0 + Mul(expr, expr, -1/12.0, evaluate=False),
                  evaluate=False)
    try:
        Float(expr)
        return v.doit()
    except (TypeError, ValueError):
        return v
