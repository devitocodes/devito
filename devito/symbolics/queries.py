from sympy import Eq, IndexedBase, Mod, S, diff, nan

from devito.symbolics.extended_sympy import (FieldFromComposite, FieldFromPointer,
                                             IndexedPointer, IntDiv)
from devito.tools import as_tuple, is_integer
from devito.types.basic import AbstractFunction
from devito.types.constant import Constant
from devito.types.dimension import Dimension
from devito.types.object import AbstractObject


__all__ = ['q_leaf', 'q_indexed', 'q_terminal', 'q_function', 'q_routine',
           'q_terminalop', 'q_indirect', 'q_constant', 'q_affine', 'q_linear',
           'q_identity', 'q_symbol', 'q_multivar', 'q_monoaffine', 'q_dimension',
           'q_positive', 'q_negative']


# The following SymPy objects are considered tree leaves:
#
# * Number
# * Symbol
# * Indexed
extra_leaves = (FieldFromPointer, FieldFromComposite, IndexedBase, AbstractObject)


def q_symbol(expr):
    try:
        return expr.is_Symbol
    except AttributeError:
        return False


def q_leaf(expr):
    return expr.is_Atom or expr.is_Indexed or isinstance(expr, extra_leaves)


def q_indexed(expr):
    return expr.is_Indexed


def q_function(expr):
    from devito.types.dense import DiscreteFunction
    return isinstance(expr, DiscreteFunction)


def q_terminal(expr):
    return (expr.is_Symbol or
            expr.is_Indexed or
            isinstance(expr, extra_leaves + (IndexedPointer,)))


def q_routine(expr):
    return expr.is_Function and not isinstance(expr, AbstractFunction)


def q_terminalop(expr, depth=0):
    assert depth >= 0

    if depth > 0:
        return all(q_leaf(a) or q_terminalop(a, depth-1) for a in expr.args)

    if expr.is_Function:
        return True
    elif expr.is_Add or expr.is_Mul:
        for a in expr.args:
            if a.is_Pow:
                elems = a.args
            else:
                elems = [a]
            if any(not q_leaf(i) for i in elems):
                return False
        return True
    elif expr.is_Pow:
        return all(q_leaf(i) for i in expr.args)
    else:
        return False


def q_indirect(expr):
    """
    Return True if ``indexed`` has indirect accesses, False otherwise.

    :Examples:

    a[i] --> False
    a[b[i]] --> True
    """
    from devito.symbolics.search import retrieve_indexed
    if not expr.is_Indexed:
        return False
    return any(retrieve_indexed(i) for i in expr.indices)


def q_multivar(expr, vars):
    """
    Return True if at least two variables in ``vars`` appear in ``expr``,
    False otherwise.
    """
    # The vast majority of calls here provide incredibly simple single variable
    # functions, so if there are < 2 free symbols we return immediately
    if not len(expr.free_symbols) > 1:
        return False
    return len(set(as_tuple(vars)) & expr.free_symbols) >= 2


def q_constant(expr):
    """
    Return True if ``expr`` is a constant, possibly symbolic, value, False otherwise.
    Examples of non-constants are expressions containing Dimensions.
    """
    if is_integer(expr):
        return True
    for i in expr.free_symbols:
        try:
            if not i.is_const:
                return False
        except AttributeError:
            return False
    return True


def q_affine(expr, vars):
    """
    Return True if ``expr`` is (separately) affine in the variables ``vars``,
    False otherwise.

    Notes
    -----
    Exploits:

        https://stackoverflow.com/questions/36283548\
        /check-if-an-equation-is-linear-for-a-specific-set-of-variables/
    """
    vars = as_tuple(vars)
    free_symbols = expr.free_symbols

    # At this point, `expr` is (separately) affine in the `vars` variables
    # if all non-mixed second order derivatives are identically zero.
    for x in vars:
        if expr is x:
            continue

        if x not in free_symbols:
            # At this point the only hope is that `expr` is constant
            return q_constant(expr)

        # The vast majority of calls here are incredibly simple tests
        # like q_affine(x+1, [x]).  Catch these quickly and
        # explicitly, instead of calling the very slow function `diff`.
        if expr.is_Add and len(expr.args) == 2:
            if expr.args[1] is x and expr.args[0].is_Number:
                continue
            if expr.args[0] is x and expr.args[1].is_Number:
                continue

        try:
            if diff(expr, x) is nan or not Eq(diff(expr, x, x), 0):
                return False
        except TypeError:
            return False

    return True


def q_monoaffine(expr, x, vars):
    """
    Return True if ``expr`` is a single variable function which is affine in ``x`` ,
    False otherwise.
    """
    if q_multivar(expr, vars):
        return False
    return q_affine(expr, x)


def q_linear(expr, vars):
    """
    Return True if ``expr`` is (separately) linear in the variables ``vars``,
    False otherwise.
    """
    return q_affine(expr, vars) and all(not i.is_Number for i in expr.args + (expr,))


def q_identity(expr, var):
    """
    Return True if ``expr`` is the identity function in ``var``, modulo a constant
    (that is, a function affine in ``var`` in which the value of the coefficient of
    ``var`` is 1), False otherwise.

    Examples
    ========
    3x -> False
    3x + 1 -> False
    x + 2 -> True
    """
    return len(as_tuple(var)) == 1 and q_affine(expr, var) and (expr - var).is_Number


def q_positive(expr):
    """
    Return True if `expr` is definitely positive, False otherwise.

    Notes
    -----
    Consider a Relational in the form `X > 0`. Sometimes SymPy is either
    unable to evaluate it (and thus return True/False) or it takes too long
    to simplify `X` to produce an answer. This function, on the other hand,
    is quick and focuses only on the cases of interest for Devito.
    If False is returned, then `expr` may or may not be positive; IOW,
    False simply means that it was not possible to determine an answer
    to the query `X > 0`.
    """
    if not expr.is_Add:
        return False

    def case0(integer, maybe_mul):
        # E.g., 2 + x % p
        if not (integer.is_Integer and integer > 0):
            return False

        if maybe_mul.is_Mul and len(maybe_mul.args) == 2:
            sign, mod = maybe_mul.args
            if sign != -1:
                return False
        else:
            sign = 1
            mod = maybe_mul

        if not isinstance(mod, Mod):
            return False

        dividend, divisor = mod.args
        if not (divisor.is_Integer and divisor > 0):
            return False

        # At this point we are in the form `X {+,-} (Y % p)`
        # * if '+', then it's the sum of two positive numbers, so we're good
        if sign == 1:
            return True

        # * if '-', instead, it only remains to ensure that X >= p
        if integer >= divisor:
            return True

    def case1(*args):
        # E.g., p0 - x / p1 + x

        assert len(args) == 3

        found = [None, None, None]
        for i in args:
            if i.is_Integer and i >= 0:
                found[0] = i
            elif isinstance(i, Constant):
                # TODO: here we should rather check for the value of .nonnegative, but
                # this would introduce a requirement on the overarching apps, so for now
                # we just run this isinstance(..., Constant) check. In theory it's not
                # enough, but in practice it ism because there's only one known way to
                # get deep down to this point, and such a Constant would definitely be
                # positive (i.e, the often called "factor" used for time subsampling)
                found[0] = i
            elif (i.is_Mul and len(i.args) == 2 and i.args[0] == -1 and
                  isinstance(i.args[1], IntDiv)):
                found[1] = i.args[1]
            elif isinstance(i, IntDiv):
                found[1] = i
            elif q_dimension(i):
                found[2] = i

        if any(i is None for i in found):
            return False

        p0, intdiv, x1 = found
        x0, p1 = intdiv.args

        if x0 is not x1:
            return False
        if not isinstance(p1, Constant):
            # TODO: Same considerations above about Constant apply
            return False

        # At this point we are in the form `X {+,-} X / p0 + p1`, where
        # `X`, `p0`, and `p1` are definitely positive; since `X > X / p0`,
        # definitely the answer is True
        return True

    if len(expr.args) == 2:
        return case0(*expr.args) or case1(S.Zero, *expr.args)

    elif len(expr.args) == 3:
        return case1(*expr.args)

    return False


def q_negative(expr):
    return q_positive(-expr)


def q_dimension(expr):
    """
    Return True if ``expr`` is a dimension, False otherwise.
    """
    return isinstance(expr, Dimension)
