"""
Routines to construct new SymPy expressions transforming the provided input.
"""

from sympy import Indexed, IndexedBase, S

from devito.dse.extended_sympy import Add, Eq, Mul
from devito.dse.graph import temporaries_graph
from devito.interfaces import DenseData

__all__ = ['unevaluate_arithmetic', 'flip_indices', 'rxreplace',
           'promote_scalar_expressions']


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
    elif expr.is_Equality:
        rebuilt_args = [unevaluate_arithmetic(e) for e in expr.args]
        return Eq(*rebuilt_args, evaluate=False)
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


def promote_scalar_expressions(exprs, shape, indices):
    """
    Transform a collection of scalar expressions into tensor expressions.
    """
    processed = []

    # Fist promote the LHS
    graph = temporaries_graph(exprs)
    mapper = {}
    for k, v in graph.items():
        if v.is_scalar:
            # Create a new function symbol
            data = DenseData(name=k.name, shape=shape, dimensions=indices)
            indexed = Indexed(data.indexed, *indices)
            mapper[k] = indexed
            processed.append(Eq(indexed, v.rhs))
        else:
            processed.append(Eq(k, v.rhs))

    # Propagate the transformed LHS through the expressions
    processed = [Eq(n.lhs, n.rhs.xreplace(mapper)) for n in processed]

    return processed
