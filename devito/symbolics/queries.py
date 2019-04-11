from sympy import Eq, diff, cos, sin, nan

from devito.tools import as_tuple, is_integer


__all__ = ['q_leaf', 'q_indexed', 'q_terminal', 'q_trigonometry', 'q_op',
           'q_terminalop', 'q_sum_of_product', 'q_indirect', 'q_timedimension',
           'q_constant', 'q_affine', 'q_linear', 'q_identity', 'q_inc', 'q_scalar',
           'q_multivar', 'q_monoaffine', 'iq_timeinvariant', 'iq_timevarying']


"""
The q_* functions are to be applied directly to expression objects.
The iq_* functions return functions to be applied to expressions objects
('iq' stands for 'indirect query')

The following SymPy objects are considered as tree leaves: ::

    * Number
    * Symbol
    * Indexed
"""


def q_scalar(expr):
    return expr.is_Number or expr.is_Symbol


def q_leaf(expr):
    return expr.is_Number or expr.is_Symbol or expr.is_Indexed


def q_indexed(expr):
    return expr.is_Indexed


def q_function(expr):
    from devito.types.dense import DiscreteFunction
    return isinstance(expr, DiscreteFunction)


def q_terminal(expr):
    return expr.is_Symbol or expr.is_Indexed


def q_trigonometry(expr):
    return expr.is_Function and expr.func in [sin, cos]


def q_op(expr):
    return expr.is_Add or expr.is_Mul or expr.is_Function


def q_terminalop(expr):
    from devito.symbolics.manipulation import as_symbol
    if not q_op(expr):
        return False
    else:
        for a in expr.args:
            try:
                as_symbol(a)
            except TypeError:
                return False
        return True


def q_sum_of_product(expr):
    return q_leaf(expr) or q_terminalop(expr) or all(q_terminalop(i) for i in expr.args)


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


def q_timedimension(expr):
    from devito.types import Dimension
    return isinstance(expr, Dimension) and expr.is_Time


def q_inc(expr):
    try:
        return expr.is_Increment
    except AttributeError:
        return False


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

    Readapted from: https://stackoverflow.com/questions/36283548\
        /check-if-an-equation-is-linear-for-a-specific-set-of-variables/
    """
    vars = as_tuple(vars)
    # If any `vars` does not appear in `expr`, the only possibility
    # for `expr` to be affine is that it's a constant function
    if any(x not in expr.atoms() for x in vars):
        return q_constant(expr)
    # At this point, `expr` is (separately) affine in the `vars` variables
    # if all non-mixed second order derivatives are identically zero.
    for x in vars:
        # The vast majority of calls here are incredibly simple tests
        # like q_affine(x+1, [x]).  Catch these quickly and
        # explicitly, instead of calling the very slow function `diff`.
        if expr == x:
            continue
        if expr.is_Add and len(expr.args) == 2:
            if expr.args[0] == x and expr.args[1].is_Number:
                continue
            if expr.args[1] == x and expr.args[0].is_Number:
                continue

        try:
            if diff(expr, x) == nan or not Eq(diff(expr, x, x), 0):
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
    return q_affine(expr, vars) and all(not i.is_Number for i in expr.args)


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


def iq_timeinvariant(graph):
    return lambda e: not e.is_Number and graph.time_invariant(e)


def iq_timevarying(graph):
    return lambda e: e.is_Number or not graph.time_invariant(e)
