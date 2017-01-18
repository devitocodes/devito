"""
Routines to construct new SymPy expressions transforming the provided input.
"""

from sympy import Indexed, S

from devito.dse.extended_sympy import Add, Mul


def unevaluate_arithmetic(expr):
    """
    Reconstruct ``expr`` turning all :class:`sympy.Mul` and :class:`sympy.Add`
    into, respectively, :class:`devito.Mul` and :class:`devito.Add`.
    """
    if expr.is_Float:
        return expr.func(*expr.atoms())
    elif isinstance(expr, Indexed):
        return expr.func(*expr.args)
    elif expr.is_Symbol:
        return expr.func(expr.name)
    elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
        return expr.func()
    elif expr.is_Atom:
        return expr.func(*expr.atoms())
    elif expr.is_Add:
        rebuilt_args = [unevaluate_arithmetic(e) for e in expr.args]
        return Add(*rebuilt_args, evaluate=False)
    elif expr.is_Mul:
        rebuilt_args = [unevaluate_arithmetic(e) for e in expr.args]
        return Mul(*rebuilt_args, evaluate=False)
    else:
        return expr.func(*[unevaluate_arithmetic(e) for e in expr.args])


def flip_indices(expr, rule):
    """
    Construct a new ``expr'`` from ``expr`` such that all indices are shifted as
    established by ``rule``.

    For example: ::

        (rule=(x, y)) a[i][j+2] + b[j][i] --> a[x][y] + b[x][y]
    """

    def run(expr, flipped):
        if expr.is_Float:
            return expr.func(*expr.atoms())
        elif isinstance(expr, Indexed):
            flipped.add(expr.indices)
            return Indexed(expr.base, *rule)
        elif expr.is_Symbol:
            return expr.func(expr.name)
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func()
        elif expr.is_Atom:
            return expr.func(*expr.atoms())
        else:
            return expr.func(*[run(e, flipped) for e in expr.args], evaluate=False)

    flipped = set()
    handle = run(expr, flipped)
    return handle, flipped


def rxreplace(exprs, mapper):
    """
    Apply Sympy's xreplace recursively.
    """

    replaced = []
    for i in exprs:
        old, new = i, i.xreplace(mapper)
        while new != old:
            old, new = new, new.xreplace(mapper)
        replaced.append(new)
    return replaced
