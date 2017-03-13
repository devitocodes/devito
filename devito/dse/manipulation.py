"""
Routines to construct new SymPy expressions transforming the provided input.
"""

from collections import OrderedDict

from sympy import Indexed, S, collect, collect_const, flatten

from devito.dse.extended_sympy import Add, Eq, Mul
from devito.dse.graph import temporaries_graph
from devito.interfaces import TensorFunction

__all__ = ['collect_nested', 'unevaluate_arithmetic', 'flip_indices',
           'replace_invariants', 'rxreplace', 'promote_scalar_expressions']


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


def promote_scalar_expressions(exprs, shape, indices, onstack):
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
            data = TensorFunction(name=k.name, shape=shape,
                                  dimensions=indices, onstack=onstack)
            indexed = Indexed(data.indexed, *indices)
            mapper[k] = indexed
            processed.append(Eq(indexed, v.rhs))
        else:
            processed.append(Eq(k, v.rhs))

    # Propagate the transformed LHS through the expressions
    processed = [Eq(n.lhs, n.rhs.xreplace(mapper)) for n in processed]

    return processed


def collect_nested(expr):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Float:
            return expr.func(*expr.atoms()), [expr]
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), []
        elif expr.is_Symbol:
            return expr.func(expr.name), [expr]
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), [expr]
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), []
        elif expr.is_Add:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])

            w_numbers = [i for i in rebuilt if any(j.is_Number for j in i.args)]
            wo_numbers = [i for i in rebuilt if i not in w_numbers]

            w_numbers = collect_const(expr.func(*w_numbers))
            wo_numbers = expr.func(*wo_numbers)

            if wo_numbers:
                for i in flatten(candidates):
                    wo_numbers = collect(wo_numbers, i)

            rebuilt = expr.func(w_numbers, wo_numbers)
            return rebuilt, []
        elif expr.is_Mul:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            rebuilt = collect_const(expr.func(*rebuilt))
            return rebuilt, flatten(candidates)
        else:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*rebuilt), flatten(candidates)

    return run(expr)[0]


def replace_invariants(expr, make, invariant=lambda e: e, cm=lambda e: True):
    """
    Replace all sub-expressions of ``expr`` such that ``invariant(expr) == True``
    with a temporary created through ``make(expr)``. A sub-expression ``e``
    within ``expr`` is not visited if ``cm(e) == False``.
    """

    def run(expr, root, mapper):
        # Return semantic: (rebuilt expr, True <==> invariant)

        if expr.is_Float:
            return expr.func(*expr.atoms()), True
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), True
        elif expr.is_Symbol:
            return expr.func(expr.name), invariant(expr)
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), True
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), invariant(expr)
        elif expr.is_Equality:
            handle, flag = run(expr.rhs, expr.rhs, mapper)
            return expr.func(expr.lhs, handle, evaluate=False), flag
        else:
            children = [run(a, root, mapper) for a in expr.args]
            invs = [a for a, flag in children if flag]
            varying = [a for a, _ in children if a not in invs]
            if not invs:
                # Nothing is time-invariant
                return (expr.func(*varying, evaluate=False), False)
            elif len(invs) == len(children):
                # Everything is time-invariant
                if expr == root:
                    if cm(expr):
                        temporary = make(mapper)
                        mapper[temporary] = expr.func(*invs, evaluate=False)
                        return temporary, True
                    else:
                        return expr.func(*invs, evaluate=False), False
                else:
                    # Go look for longer expressions first
                    return expr.func(*invs, evaluate=False), True
            else:
                # Some children are time-invariant, but expr is time-dependent
                if cm(expr) and len(invs) > 1:
                    temporary = make(mapper)
                    mapper[temporary] = expr.func(*invs, evaluate=False)
                    return expr.func(*(varying + [temporary]), evaluate=False), False
                else:
                    return expr.func(*(varying + invs), evaluate=False), False

    mapper = OrderedDict()
    handle, flag = run(expr, expr, mapper)
    return handle, flag, mapper
