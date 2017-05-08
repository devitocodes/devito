from sympy import Indexed, cos, sin


"""
The q_* functions are to be applied directly to expression objects.
The iq_* functions return functions to be applied to expressions objects.

Function names are usually self-explanatory of what the queries achieves,
otherwise a docstring is provided.
"""


def q_leaf(expr):
    """
    The DSE interprets the following SymPy objects as tree leaves: ::

        * Number
        * Symbol
        * Indexed
    """
    return expr.is_Number or expr.is_Symbol or q_indexed(expr)


def q_indexed(expr):
    return isinstance(expr, Indexed)


def q_trigonometry(expr):
    return expr.is_Function and expr.func in [sin, cos]


def q_op(expr):
    return expr.is_Add or expr.is_Mul or expr.is_Function


def q_terminalop(expr):
    from devito.dse.inspection import as_symbol
    if not q_op(expr):
        return False
    else:
        for a in expr.args:
            try:
                as_symbol(a)
            except TypeError:
                return False
        return True


def q_indirect(expr):
    """
    Return True if ``indexed`` has indirect accesses, False otherwise.

    Examples
    ========
    a[i] --> False
    a[b[i]] --> True
    """
    from devito.dse.search import retrieve_indexed

    if not q_indexed(expr):
        return False
    return any(retrieve_indexed(i) for i in expr.indices)


def iq_timeinvariant(graph):
    return lambda e: not e.is_Number and graph.time_invariant(e)


def iq_timevarying(graph):
    return lambda e: e.is_Number or not graph.time_invariant(e)
