from functools import singledispatch

import numpy as np
from sympy import (Function, Indexed, Integer, Mul, Number,
                   Pow, S, Symbol, Tuple)
from sympy.core.numbers import ImaginaryUnit

from devito.finite_differences import Derivative
from devito.finite_differences.differentiable import IndexDerivative
from devito.logger import warning
from devito.symbolics.extended_dtypes import INT
from devito.symbolics.extended_sympy import (CallFromPointer, Cast,
                                             DefFunction, ReservedWord)
from devito.symbolics.queries import q_routine
from devito.tools import as_tuple, prod
from devito.tools.dtypes_lowering import infer_dtype

__all__ = ['compare_ops', 'estimate_cost', 'has_integer_args', 'sympy_dtype']


def compare_ops(e1, e2):
    """
    True if the two input expressions perform the same arithmetic operations
    over the same input "operands", False otherwise.

    An operand here is anything that can appear as a leaf in a SymPy
    expression, but in the case of an Indexed only the labels are compared,
    while the indices are ignored.

    Examples
    --------
    >>> from devito import Dimension, Grid, Function
    >>> grid = Grid(shape=(4,))
    >>> x = grid.dimensions[0]
    >>> y = Dimension(name='y')
    >>> u = Function(name='u', grid=grid)
    >>> v = Function(name='v', grid=grid)
    >>> compare_ops(u[x] + u[x+1], u[x] + u[x-1])
    True
    >>> compare_ops(u[x] + u[x+1], u[x] - u[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] * u[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] + v[x+1])
    False
    >>> compare_ops(u[x] + u[x+1], u[x] + u[y+10])
    True
    """
    if type(e1) is type(e2) and len(e1.args) == len(e2.args):
        if e1.is_Atom:
            return True if e1 == e2 else False
        elif isinstance(e1, IndexDerivative) and isinstance(e2, IndexDerivative):
            if e1.mapper == e2.mapper:
                return compare_ops(e1.expr, e2.expr)
            else:
                return False
        elif e1.is_Indexed and e2.is_Indexed:
            return True if e1.base == e2.base else False
        else:
            for a1, a2 in zip(e1.args, e2.args):
                if not compare_ops(a1, a2):
                    return False
            return True
    else:
        return False


def estimate_cost(exprs, estimate=False):
    """
    Estimate the operation count of an expression.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions for which the operation count is calculated.
    estimate : bool, optional
        Defaults to False; if True, the following rules are applied in order:
            * An operation involving only `Constant`'s counts as 1 ops.
            * Trascendental functions (e.g., cos, sin, ...) count as 50 ops.
            * Divisions (powers with a negative exponent) count as 25 ops.
            * Powers with integer exponent `n>0` count as n-1 ops (as if
              it were a chain of multiplications).
    """
    try:
        # Is it a plain symbol/array ?
        if exprs.is_Atom or exprs.is_Indexed or exprs.is_AbstractFunction:
            return 0
    except AttributeError:
        pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        seen = {}
        flops = 0
        for expr in as_tuple(exprs):
            # TODO: this if-then should be part of singledispatch too, but because
            # of the annoying symbolics/ vs types/ structuring, we can't do that
            # because of circular imports...
            if expr.is_Equality:
                e = expr.rhs
                if expr.is_Reduction:
                    flops += 1
            else:
                e = expr

            flops += _estimate_cost(e, estimate, seen)[0]

        return flops
    except:
        warning("Cannot estimate cost of `%s`" % str(exprs))
        return 0


estimate_values = {
    'elementary': 100,
    'pow': 50,
    'SafeInv': 10,
    'div': 5,
    'Abs': 5,
    'floor': 1,
    'ceil': 1,
}


def dont_count_if_seen(func):
    """
    This decorator is used to avoid counting the same expression multiple
    times. This is necessary because the same expression may appear multiple
    times in the same expression tree or even across different expressions.
    """
    def wrapper(expr, estimate, seen):
        try:
            _, flags = seen[expr]
            flops = 0
        except KeyError:
            flops, flags = seen[expr] = func(expr, estimate, seen)
        return flops, flags
    return wrapper


@singledispatch
def _estimate_cost(expr, estimate, seen):
    # Retval: flops (int), flag (bool)
    # The flag tells wether it's an integer expression (implying flops==0) or not
    if not expr.args:
        return 0, False
    flops, flags = zip(*[_estimate_cost(a, estimate, seen) for a in expr.args])
    flops = sum(flops)
    if all(flags):
        # `expr` is an operation involving integer operands only
        # NOTE: one of the operands may contain, internally, non-integer
        # operations, e.g. the `a*b` in `2 + INT(a*b)`
        return flops, True
    else:
        return flops + (len(expr.args) - 1), False


@_estimate_cost.register(Tuple)
@_estimate_cost.register(CallFromPointer)
def _(expr, estimate, seen):
    try:
        flops, flags = zip(*[_estimate_cost(a, estimate, seen) for a in expr.args])
    except ValueError:
        flops, flags = [], []
    return sum(flops), all(flags)


@_estimate_cost.register(Integer)
def _(expr, estimate, seen):
    return 0, True


@_estimate_cost.register(ImaginaryUnit)
@_estimate_cost.register(Number)
@_estimate_cost.register(ReservedWord)
def _(expr, estimate, seen):
    return 0, False


@_estimate_cost.register(Symbol)
@_estimate_cost.register(Indexed)
def _(expr, estimate, seen):
    try:
        if issubclass(expr.dtype, np.integer):
            return 0, True
    except:
        pass
    return 0, False


@_estimate_cost.register(Mul)
def _(expr, estimate, seen):
    flops, flags = _estimate_cost.registry[object](expr, estimate, seen)
    if {S.One, S.NegativeOne}.intersection(expr.args):
        flops -= 1
    if ImaginaryUnit in expr.args:
        flops *= 2
    return flops, flags


@_estimate_cost.register(INT)
def _(expr, estimate, seen):
    return _estimate_cost(expr.base, estimate, seen)[0], True


@_estimate_cost.register(Cast)
def _(expr, estimate, seen):
    return _estimate_cost(expr.base, estimate, seen)[0], False


@_estimate_cost.register(Function)
def _(expr, estimate, seen):
    if q_routine(expr):
        flops, _ = zip(*[_estimate_cost(a, estimate, seen) for a in expr.args])
        flops = sum(flops)
        if isinstance(expr, DefFunction):
            # Bypass user-defined or language-specific functions
            flops += 0
        elif estimate:
            try:
                flops += estimate_values[type(expr).__name__]
            except KeyError:
                flops += estimate_values['elementary']
        else:
            flops += 1
    else:
        flops = 0
    return flops, False


@_estimate_cost.register(Pow)
def _(expr, estimate, seen):
    flops, _ = zip(*[_estimate_cost(a, estimate, seen) for a in expr.args])
    flops = sum(flops)
    if estimate:
        if expr.exp.is_Number:
            if expr.exp < 0:
                flops += estimate_values['div']
            elif expr.exp == 0 or expr.exp == 1:
                flops += 0
            elif expr.exp.is_Integer:
                # Natural pows a**b are estimated as b-1 Muls
                flops += int(expr.exp) - 1
            else:
                flops += estimate_values['pow']
        else:
            flops += estimate_values['pow']
    else:
        flops += 1
    return flops, False


@_estimate_cost.register(Derivative)
@dont_count_if_seen
def _(expr, estimate, seen):
    return _estimate_cost(expr._evaluate(expand=False), estimate, seen)


@_estimate_cost.register(IndexDerivative)
@dont_count_if_seen
def _(expr, estimate, seen):
    flops, _ = _estimate_cost(expr.expr, estimate, seen)

    # It's an increment
    flops += 1

    # To be multiplied by the number of points this index sum implicitly
    # iterates over
    flops *= prod(i._size for i in expr.dimensions)

    return flops, False


def has_integer_args(*args):
    """
    True if all `args` are of integer type, False otherwise.
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
            if isinstance(a, INT):
                res = res and True
            elif len(a.args) > 0:
                res = res and has_integer_args(*a.args)
            else:
                res = res and has_integer_args(a)
        except AttributeError:
            res = res and has_integer_args(a)
    return res


def sympy_dtype(expr, base=None, default=None, smin=None):
    """
    Infer the dtype of the expression.
    """
    if expr is None:
        return default

    dtypes = {base} - {None}
    for i in expr.free_symbols:
        try:
            dtypes.add(i.dtype)
        except AttributeError:
            pass

    dtype = infer_dtype(dtypes)

    # Promote if we missed complex number, i.e f + I
    is_im = np.issubdtype(dtype, np.complexfloating)
    if expr.has(ImaginaryUnit) and not is_im:
        if dtype is None:
            dtype = default or np.complex64
        else:
            dtype = np.promote_types(dtype, np.complex64).type

    if smin is not None and not np.issubdtype(dtype, np.integer):
        dtype = np.promote_types(dtype, smin).type

    return dtype
