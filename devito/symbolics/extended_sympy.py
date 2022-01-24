"""
Extended SymPy hierarchy.
"""

import numpy as np
import sympy
from sympy import Expr, Integer, Function, Symbol, sympify

from devito.symbolics.printer import ccode
from devito.tools import Pickable, as_tuple, is_integer

__all__ = ['CondEq', 'CondNe', 'IntDiv', 'CallFromPointer', 'FieldFromPointer',
           'FieldFromComposite', 'ListInitializer', 'Byref', 'IndexedPointer', 'Cast',
           'DefFunction', 'InlineIf', 'Macro', 'MacroArgument', 'Literal', 'Deref',
           'INT', 'FLOAT', 'DOUBLE', 'VOID', 'CEIL', 'FLOOR', 'MAX', 'MIN',
           'SizeOf', 'rfunc', 'cast_mapper']


class CondEq(sympy.Eq):

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


class CondNe(sympy.Ne):

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

    is_commutative = True

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


class BasicWrapperMixin(object):

    """
    Abstract mixin class for objects wrapping types.Basic objects.
    """

    @property
    def base(self):
        """
        The wrapped object.
        """
        raise NotImplementedError

    @property
    def function(self):
        """
        The underlying function.
        """
        return self.base.function

    @property
    def dtype(self):
        """
        The wrapped object data type.
        """
        return self.function.dtype


class CallFromPointer(sympy.Expr, Pickable, BasicWrapperMixin):

    """
    Symbolic representation of the C notation ``pointer->call(params)``.
    """

    def __new__(cls, call, pointer, params=None):
        args = []
        if isinstance(pointer, str):
            pointer = Symbol(pointer)
        args.append(pointer)
        if isinstance(call, (DefFunction, CallFromPointer)):
            args.append(call)
        elif not isinstance(call, str):
            raise ValueError("`call` must be CallFromPointer or str")
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
        obj.call = call
        obj.pointer = pointer
        obj.params = tuple(_params)
        return obj

    def __str__(self):
        return '%s->%s(%s)' % (self.pointer, self.call,
                               ", ".join(str(i) for i in as_tuple(self.params)))

    __repr__ = __str__

    def _hashable_content(self):
        return super(CallFromPointer, self)._hashable_content() +\
            (self.call, self.pointer) + self.params

    @property
    def base(self):
        if isinstance(self.pointer, CallFromPointer):
            # CallFromPointer may be nested
            return self.pointer.base
        else:
            return self.pointer

    # Pickling support
    _pickle_args = ['call', 'pointer']
    _pickle_kwargs = ['params']
    __reduce_ex__ = Pickable.__reduce_ex__


class FieldFromPointer(CallFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``pointer->field``.
    """

    def __new__(cls, field, pointer):
        return CallFromPointer.__new__(cls, field, pointer)

    def __str__(self):
        return '%s->%s' % (self.pointer, self.field)

    @property
    def field(self):
        return self.call

    # Our __new__ cannot accept the params argument
    _pickle_kwargs = []

    __repr__ = __str__


class FieldFromComposite(CallFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``composite.field``,
    where ``composite`` is a struct/union/...
    """

    def __new__(cls, field, composite):
        return CallFromPointer.__new__(cls, field, composite)

    def __str__(self):
        return '%s.%s' % (self.composite, self.field)

    @property
    def field(self):
        return self.call

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
            elif is_integer(p):
                args.append(Integer(p))
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


class UnaryOp(sympy.Expr, Pickable):

    """
    Symbolic representation of a unary C operator.
    """

    _op = ''

    def __new__(cls, base, **kwargs):
        try:
            # If an AbstractFunction, pull the underlying Symbol
            base = base.indexed.label
        except AttributeError:
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

    @property
    def free_symbols(self):
        return self.base.free_symbols

    def __str__(self):
        if self.base.is_Symbol:
            return "%s%s" % (self._op, ccode(self.base))
        else:
            return "%s(%s)" % (self._op, ccode(self.base))

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['base']
    __reduce_ex__ = Pickable.__reduce_ex__


class Byref(UnaryOp):

    """
    Symbolic representation of the C notation `&expr`.
    """

    _op = '&'


class Deref(UnaryOp):

    """
    Symbolic representation of the C notation `*expr`.
    """

    _op = '*'


class Cast(UnaryOp):

    """
    Symbolic representation of the C notation `(type)expr`.
    """

    _base_typ = ''

    def __new__(cls, base, stars=None, **kwargs):
        obj = super().__new__(cls, base)
        obj._stars = stars
        return obj

    @property
    def stars(self):
        return self._stars

    @property
    def typ(self):
        return '%s%s' % (self._base_typ, self.stars or '')

    @property
    def _op(self):
        return '(%s)' % self.typ

    # Pickling support
    _pickle_kwargs = ['stars']


class IndexedPointer(sympy.Expr, Pickable, BasicWrapperMixin):

    """
    Symbolic representation of the C notation ``symbol[...]``

    Unlike a sympy.Indexed, an IndexedPointer accepts, as base, objects that
    are not necessarily a Symbol or an IndexedBase, such as a FieldFromPointer.
    """

    def __new__(cls, base, index):
        try:
            # If an AbstractFunction, pull the underlying Symbol
            base = base.indexed.label
        except AttributeError:
            if not isinstance(base, sympy.Basic):
                raise ValueError("`base` must be of type sympy.Basic")

        index = tuple(sympify(i) for i in as_tuple(index))

        obj = sympy.Expr.__new__(cls, base, *index)
        obj._base = base
        obj._index = index

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


class DefFunction(Function, Pickable):

    """
    A definitely-defined sympy.Function, to work around:

        https://github.com/sympy/sympy/issues/4297
    """

    is_Atom = True

    def __new__(cls, name, arguments=None):
        arguments = as_tuple(arguments)
        obj = Function.__new__(cls, name, *arguments)
        obj._name = name
        obj._arguments = arguments
        return obj

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    @property
    def free_symbols(self):
        return set().union(*[i.free_symbols for i in self.arguments
                             if isinstance(i, Expr)])

    def __str__(self):
        return "%s(%s)" % (self.name, ', '.join(str(i) for i in self.arguments))

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['name']
    _pickle_kwargs = ['arguments']
    __reduce_ex__ = Pickable.__reduce_ex__


class InlineIf(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``(C) ? a : b``.
    """

    def __new__(cls, cond, true_expr, false_expr):
        if not isinstance(cond, sympy.core.relational.Relational):
            raise ValueError("`cond` must be of type sympy.relational.Relational")
        if not isinstance(true_expr, sympy.Expr):
            raise ValueError("`true_expr` must be of type sympy.Expr")
        if not isinstance(false_expr, sympy.Expr):
            raise ValueError("`false_expr` must be of type sympy.Expr")
        obj = sympy.Expr.__new__(cls, cond, true_expr, false_expr)
        obj._cond = cond
        obj._true_expr = true_expr
        obj._false_expr = false_expr
        return obj

    @property
    def cond(self):
        return self._cond

    @property
    def true_expr(self):
        return self._true_expr

    @property
    def false_expr(self):
        return self._false_expr

    def __str__(self):
        return "(%s) ? %s : %s" % (self.cond, self.true_expr, self.false_expr)

    __repr__ = __str__

    # Pickling support
    _pickle_args = ['cond', 'true_expr', 'false_expr']
    __reduce_ex__ = Pickable.__reduce_ex__


class Macro(sympy.Symbol):

    """
    Symbolic representation of a C macro.
    """
    pass


class MacroArgument(sympy.Symbol):

    """
    Symbolic representation of a C macro.
    """

    def __str__(self):
        return "(%s)" % self.name

    __repr__ = __str__


class Literal(sympy.Symbol):

    """
    Symbolic representation of a Literal element.
    """
    pass


# Shortcuts (mostly for retrocompatibility)

class INT(Cast):
    _base_typ = 'int'


class FLOAT(Cast):
    _base_typ = 'float'


class DOUBLE(Cast):
    _base_typ = 'double'


class VOID(Cast):
    _base_typ = 'void'


class CastStar(object):

    base = None

    def __new__(cls, base=''):
        return cls.base(base, '*')


class INTP(CastStar):
    base = INT


class FLOATP(CastStar):
    base = FLOAT


class DOUBLEP(CastStar):
    base = DOUBLE


# Some other utility functions

CEIL = Function('ceil')
FLOOR = Function('floor')
MAX = Function('MAX')
MIN = Function('MIN')

# DefFunction, unlike sympy.Function, generates e.g. `sizeof(float)`, not `sizeof(float_)`
SizeOf = lambda *args: DefFunction('sizeof', tuple(args))


def rfunc(func, item, *args):
    """
    A utility function that recursively generates 'func' nested relations.

    Examples
    ----------
    >> rfunc(min, [a, b, c, d])
    MIN(a, MIN(b, MIN(c, d)))

    >> rfunc(max, [a, b, c, d])
    MAX(a, MAX(b, MAX(c, d)))
    """

    assert func in rfunc_mapper
    rf = rfunc_mapper[func]

    if len(args) == 0:
        return item
    else:
        return rf(item, rfunc(func, *args))


cast_mapper = {
    int: INT,
    np.int32: INT,
    np.int64: INT,
    np.float32: FLOAT,
    float: DOUBLE,
    np.float64: DOUBLE,

    (int, '*'): INTP,
    (np.int32, '*'): INTP,
    (np.int64, '*'): INTP,
    (np.float32, '*'): FLOATP,
    (float, '*'): DOUBLEP,
    (np.float64, '*'): DOUBLEP
}

rfunc_mapper = {
    min: MIN,
    max: MAX,
}
