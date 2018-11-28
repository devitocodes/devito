from sympy import Eq, diff, cos, sin, nan

from devito.tools import as_tuple


__all__ = ['q_leaf', 'q_indexed', 'q_terminal', 'q_trigonometry', 'q_op',
           'q_terminalop', 'q_sum_of_product', 'q_indirect', 'q_timedimension',
           'q_affine', 'q_linear', 'q_identity', 'q_inc', 'q_scalar',
           'iq_timeinvariant', 'iq_timevarying']


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
    from devito.function import TensorFunction
    return isinstance(expr, TensorFunction)


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
    from devito.dimension import Dimension
    return isinstance(expr, Dimension) and expr.is_Time


def q_inc(expr):
    try:
        return expr.is_Increment
    except AttributeError:
        return False


def q_affine(expr, vars):
    """
    Return True if ``expr`` is (separately) affine in the variables ``vars``,
    False otherwise.

    Readapted from: https://stackoverflow.com/questions/36283548\
        /check-if-an-equation-is-linear-for-a-specific-set-of-variables/
    """
    # A function is (separately) affine in a given set of variables if all
    # non-mixed second order derivatives are identically zero.
    for x in as_tuple(vars):
        if x not in expr.atoms():
            return False

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
