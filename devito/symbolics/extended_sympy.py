"""
Extended SymPy hierarchy.
"""
import re

import numpy as np
import sympy
from sympy import Expr, Function, Number, Tuple, cacheit, sympify
from sympy.core.decorators import call_highest_priority

from devito.finite_differences.elementary import Min, Max
from devito.tools import (Pickable, Bunch, as_tuple, is_integer, float2,  # noqa
                          float3, float4, double2, double3, double4, int2, int3,
                          int4, dtype_to_ctype, ctypes_to_cstr, ctypes_vector_mapper,
                          ctypes_to_cstr)
from devito.types import Symbol
from devito.types.basic import Basic

__all__ = ['CondEq', 'CondNe', 'IntDiv', 'CallFromPointer',  # noqa
           'CallFromComposite', 'FieldFromPointer', 'FieldFromComposite',
           'ListInitializer', 'Byref', 'IndexedPointer', 'Cast', 'DefFunction',
           'MathFunction', 'InlineIf', 'ReservedWord', 'Keyword', 'String',
           'Macro', 'Class', 'MacroArgument', 'Deref', 'Namespace',
           'Rvalue', 'Null', 'SizeOf', 'rfunc', 'BasicWrapperMixin', 'ValueLimit',
           'VectorAccess']


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

    # Set the operator priority higher than SymPy (10.0) to force the overridden
    # operators to be used
    _op_priority = sympy.Expr._op_priority + 1.

    def __new__(cls, lhs, rhs, params=None):
        if rhs == 0:
            raise ValueError("Cannot divide by 0")
        elif rhs == 1 or rhs is None:
            return lhs

        if not is_integer(rhs):
            # Perhaps it's a symbolic RHS -- but we wanna be sure it's of type int
            if not hasattr(rhs, 'dtype'):
                raise ValueError(f"Symbolic RHS `{rhs}` lacks dtype")
            if not issubclass(rhs.dtype, np.integer):
                raise ValueError(f"Symbolic RHS `{rhs}` must be of type `int`, found "
                                 f"`{rhs.dtype}` instead")
        rhs = sympify(rhs)

        obj = sympy.Expr.__new__(cls, lhs, rhs)

        obj.lhs = lhs
        obj.rhs = rhs

        return obj

    def __str__(self):
        return f"IntDiv({self.lhs}, {self.rhs})"

    __repr__ = __str__

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        if other is self.rhs:
            # a*(i/a) => i
            return self.lhs
        return super().__mul__(other)


class BasicWrapperMixin:

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

    @property
    def is_commutative(self):
        """
        Overridden SymPy assumption -- now based on the wrapped object dtype.
        """
        try:
            return issubclass(self.dtype, np.number)
        except TypeError:
            return self.dtype in ctypes_vector_mapper

    def _sympystr(self, printer):
        return str(self)


class CallFromPointer(sympy.Expr, Pickable, BasicWrapperMixin):

    """
    Symbolic representation of the C notation ``pointer->call(params)``.
    """

    __rargs__ = ('call', 'pointer')
    __rkwargs__ = ('params',)

    def __new__(cls, call, pointer, params=None, **kwargs):
        if isinstance(pointer, str):
            pointer = Symbol(pointer)
        if isinstance(call, str):
            call = Symbol(call)
        elif not isinstance(call, Basic):
            raise ValueError("`call` must be a `devito.Basic` or a type "
                             "with compatible interface")
        _params = []
        for p in as_tuple(params):
            if isinstance(p, str):
                _params.append(Symbol(p))
            elif isinstance(p, Expr):
                _params.append(p)
            else:
                try:
                    _params.append(Number(p))
                except TypeError:
                    raise ValueError("`params` must be Expr, numbers or str")
        params = Tuple(*_params)

        obj = sympy.Expr.__new__(cls, call, pointer, params)
        obj.call = call
        obj.pointer = pointer
        obj.params = params

        return obj

    def __str__(self):
        params = ", ".join(str(i) for i in as_tuple(self.params))
        return f'{self.pointer}->{self.call}({params})'

    __repr__ = __str__

    def _hashable_content(self):
        return super()._hashable_content() + (self.call, self.pointer) + self.params

    @property
    def base(self):
        if isinstance(self.pointer, CallFromPointer):
            # CallFromPointer may be nested
            return self.pointer.base
        else:
            return self.pointer

    @property
    def bound_symbols(self):
        return {self.call}

    @property
    def free_symbols(self):
        return super().free_symbols - self.bound_symbols

    is_commutative = BasicWrapperMixin.is_commutative

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class CallFromComposite(CallFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``composite.call(params)``.
    """

    def __str__(self):
        params = ", ".join(str(i) for i in as_tuple(self.params))
        return f'{self.pointer}.{self.call}({params})'

    __repr__ = __str__


class FieldFromPointer(CallFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``pointer->field``.
    """

    __rkwargs__ = ()

    def __new__(cls, field, pointer, *args, **kwargs):
        return CallFromPointer.__new__(cls, field, pointer)

    def __str__(self):
        return f'{self.pointer}->{self.field}'

    @property
    def field(self):
        return self.call

    __repr__ = __str__


class FieldFromComposite(CallFromPointer, Pickable):

    """
    Symbolic representation of the C notation ``composite.field``,
    where ``composite`` is a struct/union/...
    """

    __rkwargs__ = ()

    def __new__(cls, field, composite, *args, **kwargs):
        return CallFromPointer.__new__(cls, field, composite)

    def __str__(self):
        return f'{self.composite}.{self.field}'

    @property
    def field(self):
        return self.call

    @property
    def composite(self):
        return self.pointer

    __repr__ = __str__


class ListInitializer(sympy.Expr, Pickable):

    """
    Symbolic representation of the C++ list initializer notation ``{a, b, ...}``.
    """

    __rargs__ = ('params',)
    __rkwargs__ = ('dtype',)

    def __new__(cls, params, dtype=None):
        args = []
        for p in as_tuple(params):
            try:
                args.append(sympify(p))
            except sympy.SympifyError:
                raise ValueError(f"Illegal param `{p}`")
        obj = sympy.Expr.__new__(cls, *args)

        obj.params = tuple(args)
        obj.dtype = dtype

        return obj

    def __str__(self):
        return f"{{{', '.join(str(i) for i in self.params)}}}"

    __repr__ = __str__

    @property
    def is_numeric(self):
        return all(i.is_Number for i in self.params)

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class UnaryOp(sympy.Expr, Pickable, BasicWrapperMixin):

    """
    Symbolic representation of a unary C operator.
    """

    _op = ''

    __rargs__ = ('base',)

    def __new__(cls, base, **kwargs):
        try:
            # If an AbstractFunction, pull the underlying Symbol
            base = base.indexed.label
        except AttributeError:
            if isinstance(base, str):
                base = Symbol(base)
            else:
                # Fallback: go plain sympy
                base = sympify(base)

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
            return f'{self._op}{self.base}'
        else:
            return f'{self._op}({self.base})'

    __repr__ = __str__

    # Pickling support
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

    __rargs__ = ('base', )
    __rkwargs__ = ('dtype', 'stars', 'reinterpret')

    def __new__(cls, base, dtype=None, stars=None, reinterpret=False, **kwargs):
        try:
            if issubclass(dtype, np.generic) and sympify(base).is_Number:
                base = sympify(dtype(base))
        except TypeError:
            # E.g. void
            pass

        dtype, stars = cls._process_dtype(dtype, stars)

        obj = super().__new__(cls, base)
        obj._stars = stars or ''
        obj._dtype = dtype
        obj._reinterpret = reinterpret

        return obj

    @classmethod
    def _process_dtype(cls, dtype, stars):
        if not isinstance(dtype, str) or stars is not None:
            return dtype, stars

        # String dtype, e.g. "float", "int*", "foo**"
        match = re.fullmatch(r'(\w+)\s*(\*+)?', dtype)
        if match:
            dtype = match.group(1)
            stars = match.group(2) or ''
            return dtype, stars
        else:
            return dtype, stars

    def _hashable_content(self):
        return super()._hashable_content() + (self._stars,)

    func = Pickable._rebuild

    @property
    def stars(self):
        return self._stars

    @property
    def dtype(self):
        return self._dtype

    @property
    def reinterpret(self):
        return self._reinterpret

    @property
    def _C_ctype(self):
        ctype = ctypes_vector_mapper.get(self.dtype, self.dtype)
        try:
            ctype = dtype_to_ctype(ctype)
        except TypeError:
            pass
        return ctype

    @property
    def _op(self):
        return f'({ctypes_to_cstr(self._C_ctype)}{self.stars})'

    def __str__(self):
        return f"{self._op}{self.base}"


class IndexedPointer(sympy.Expr, Pickable, BasicWrapperMixin):

    """
    Symbolic representation of the C notation ``symbol[...]``

    Unlike a sympy.Indexed, an IndexedPointer accepts, as base, objects that
    are not necessarily a Symbol or an IndexedBase, such as a FieldFromPointer.
    """

    __rargs__ = ('base', 'index')

    def __new__(cls, base, index, **kwargs):
        try:
            # If an AbstractFunction, pull the underlying Symbol
            base = base.indexed.label
        except AttributeError:
            if not isinstance(base, sympy.Basic):
                raise ValueError("`base` must be of type sympy.Basic")

        index = Tuple(*[sympify(i) for i in as_tuple(index)])

        obj = sympy.Expr.__new__(cls, base, index)
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
        indices = ''.join(f'[{i}]' for i in self.index)
        return f"{self.base}{indices}"

    __repr__ = __str__

    is_commutative = BasicWrapperMixin.is_commutative

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class ReservedWord(sympy.Atom, Pickable):

    """
    A `ReservedWord` carries a value that has special meaning in the
    generated code. A ReservedWord, for example, may be a keyword of the
    underlying language syntax or a special name reserved via typedefs
    or macros.

    A ReservedWord cannot be used as an identifier and has no underlying
    symbols associated. Hence, a ReservedWord is an "atomic" object in
    a SymPy sense.
    """

    __rargs__ = ('value',)

    def __new__(cls, value, **kwargs):
        if not isinstance(value, str):
            raise TypeError(f"Expected str, got `{type(value)}`")
        obj = sympy.Atom.__new__(cls, **kwargs)
        obj.value = value

        return obj

    def __str__(self):
        return self.value

    __repr__ = __str__

    def _hashable_content(self):
        return (self.value,)

    def _sympystr(self, printer):
        return str(self)

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Keyword(ReservedWord):
    pass


class String(ReservedWord):
    pass


class Macro(ReservedWord):
    pass


class Class(ReservedWord):

    def __str__(self):
        return f"class {self.value}"

    __repr__ = __str__


class MacroArgument(sympy.Symbol):

    def __str__(self):
        return f"({self.name})"

    __repr__ = __str__


class ValueLimit(ReservedWord, sympy.Expr):

    """
    Symbolic representation of the so called limits macros, which provide the
    minimum and maximum limits for various types, such as INT_MIN, INT_MAX etc.
    """

    pass


class DefFunction(Function, Pickable):

    """
    A definitely-defined sympy.Function, to work around:

        https://github.com/sympy/sympy/issues/4297
    """

    __rargs__ = ('name', 'arguments', 'template')

    def __new__(cls, name, arguments=None, template=None, **kwargs):
        if isinstance(name, str):
            name = Keyword(name)

        _arguments = []
        for i in as_tuple(arguments):
            if isinstance(i, str):
                # Make sure there's no cast to sympy.Symbol underneath
                # We don't know what `i` is exactly, because the caller won't
                # tell us, but we're just better off with ReservedWord
                _arguments.append(ReservedWord(i))
            else:
                _arguments.append(i)

        _template = []
        for i in as_tuple(template):
            if isinstance(i, str):
                # Same story as above
                _template.append(ReservedWord(i))
            else:
                _template.append(i)

        args = [name]
        args.append(Tuple(*_arguments))
        if _template:
            args.append(Tuple(*_template))

        obj = Function.__new__(cls, *args)
        obj._name = name
        obj._arguments = tuple(_arguments)
        obj._template = tuple(_template)

        return obj

    def _eval_is_commutative(self):
        # DefFunction defaults to commutative
        return True

    @property
    def name(self):
        return self._name

    @property
    def arguments(self):
        return self._arguments

    @property
    def template(self):
        return self._template

    def __str__(self):
        if self.template:
            template = f"<{','.join(str(i) for i in self.template)}>"
        else:
            template = ''
        arguments = ', '.join(str(i) for i in self.arguments)
        return f"{self.name}{template}({arguments})"

    __repr__ = __str__

    def _sympystr(self, printer):
        return str(self)

    func = Pickable._rebuild

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class MathFunction(DefFunction):

    # Supposed to involve real operands
    is_commutative = True


class InlineIf(sympy.Expr, Pickable):

    """
    Symbolic representation of the C notation ``(C) ? a : b``.
    """

    __rargs__ = ('cond', 'true_expr', 'false_expr')

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
        return f"({self.cond}) ? {self.true_expr} : {self.false_expr}"

    __repr__ = __str__

    # Pickling support
    __reduce_ex__ = Pickable.__reduce_ex__


class Namespace(sympy.Expr, Pickable):

    """
    Symbolic representation of a C++ namespace `ns0::ns1::...`.
    """

    __rargs__ = ('items',)

    def __new__(cls, items, **kwargs):
        normalized_items = []
        for i in as_tuple(items):
            if isinstance(i, str):
                normalized_items.append(ReservedWord(i))
            elif isinstance(i, ReservedWord):
                normalized_items.append(i)
            else:
                raise ValueError("`items` must be iterable of str or ReservedWord")

        obj = sympy.Expr.__new__(cls)
        obj._items = tuple(items)

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + self.items

    @property
    def items(self):
        return self._items

    def __str__(self):
        return "::".join(str(i) for i in self.items)

    __repr__ = __str__


class Rvalue(sympy.Expr, Pickable):

    """
    A generic C++ rvalue, that is a value that occupies a temporary location in
    memory.
    """

    __rargs__ = ('expr',)
    __rkwargs__ = ('namespace', 'init')

    def __new__(cls, expr, namespace=None, init=None):
        args = [expr]
        if namespace is not None:
            args.append(namespace)
        if init is not None:
            args.append(init)

        obj = sympy.Expr.__new__(cls, *args)

        obj._expr = expr
        obj._namespace = namespace
        obj._init = init

        return obj

    @property
    def expr(self):
        return self._expr

    @property
    def namespace(self):
        return self._namespace

    @property
    def init(self):
        return self._init

    def __str__(self):
        rvalue = str(self.expr)
        if self.namespace:
            rvalue = f"{self.namespace}::{rvalue}"
        if self.init:
            rvalue = f"{rvalue}{self.init}"
        return rvalue

    __repr__ = __str__


class VectorAccess(Expr, Pickable, BasicWrapperMixin):

    """
    Represent a vector access operation at high-level.
    """

    def __new__(cls, *args, **kwargs):
        return Expr.__new__(cls, *args)

    def __str__(self):
        return f"VL<{self.base}>"

    __repr__ = __str__

    func = Pickable._rebuild

    @property
    def base(self):
        return self.args[0]

    @property
    def indices(self):
        return self.base.indices

    @cacheit
    def sort_key(self, order=None):
        # Ensure that the VectorAccess is sorted as the base
        return self.base.sort_key(order=order)


# Some other utility objects
Null = Macro('NULL')


# DefFunction, unlike sympy.Function, generates e.g. `sizeof(float)`, not `sizeof(float_)`
class SizeOf(DefFunction):

    __rargs__ = ('intype', 'stars')

    def __new__(cls, intype, stars=None, **kwargs):
        stars = stars or ''
        if not isinstance(intype, (str, ReservedWord)):
            ctype = dtype_to_ctype(intype)
            for k, v in ctypes_vector_mapper.items():
                if ctype is v:
                    intype = k
                    break
            else:
                intype = ctypes_to_cstr(ctype)

        newobj = super().__new__(cls, 'sizeof', arguments=f'{intype}{stars}', **kwargs)
        newobj.stars = stars
        newobj.intype = intype

        return newobj

    @property
    def args(self):
        return super().args[1]


def rfunc(func, item, *args):
    """
    A utility function that recursively generates 'func' nested relations.

    Examples
    ----------
    >> rfunc(min, [a, b, c, d])
    Min(a, Min(b, Min(c, d)))

    >> rfunc(max, [a, b, c, d])
    Max(a, Max(b, Max(c, d)))
    """

    assert func in rfunc_mapper
    rf = rfunc_mapper[func]

    if len(args) == 0:
        return item
    else:
        return rf(item, rfunc(func, *args), evaluate=False)


rfunc_mapper = {
    min: Min,
    max: Max,
}
