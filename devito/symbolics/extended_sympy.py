"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr, Integer, Symbol
from sympy.core.basic import _aresame

from devito.tools import Pickable, as_tuple

__all__ = ['FrozenExpr', 'Eq', 'CondEq', 'CondNe', 'IntDiv', 'Add', 'Mul',
           'FunctionFromPointer', 'FieldFromPointer', 'FieldFromComposite',
           'ListInitializer', 'Byref', 'Macro']


class FrozenExpr(Expr):

    """
    Use :class:`FrozenExpr` in place of :class:`sympy.Expr` to make sure than
    an expression is no longer transformable; that is, standard manipulations
    such as xreplace, collect, expand, ... have no effect, thus building a
    new expression identical to self.

    :Notes:

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


class Eq(sympy.Eq, FrozenExpr):

    """A customized version of :class:`sympy.Eq` which suppresses evaluation."""

    def __new__(cls, *args, **kwargs):
        return sympy.Eq.__new__(cls, *args, evaluate=False)


class CondEq(sympy.Eq, FrozenExpr):
    """A customized version of :class:`sympy.Eq` representing a conditional
    equality. It suppresses evaluation."""

    def __new__(cls, *args, **kwargs):
        return sympy.Eq.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self


class CondNe(sympy.Ne, FrozenExpr):
    """A customized version of :class:`sympy.Ne` representing a conditional
    inequality. It suppresses evaluation."""

    def __new__(cls, *args, **kwargs):
        return sympy.Ne.__new__(cls, *args, evaluate=False)

    @property
    def canonical(self):
        return self


class Mul(sympy.Mul, FrozenExpr):
    pass


class Add(sympy.Add, FrozenExpr):
    pass


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
        rhs = Integer(rhs)
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


class Byref(sympy.Symbol, Pickable):

    """
    Symbolic representation of the C++ notation ``&symbol``.
    """

    def __new__(cls, name):
        return sympy.Symbol.__new__(cls, name)

    def __str__(self):
        return "&%s" % self.name

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['name']
    __reduce_ex__ = Pickable.__reduce_ex__


class Macro(sympy.Symbol):
    """
    Symbolic representation of a C++ macro.
    """
    pass
