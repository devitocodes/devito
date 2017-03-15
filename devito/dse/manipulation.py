"""
Routines to construct new SymPy expressions transforming the provided input.
"""

from collections import OrderedDict

from sympy import Indexed, S, collect, collect_const, flatten

from devito.dse.extended_sympy import Add, Eq, Mul
from devito.dse.graph import temporaries_graph
from devito.interfaces import TensorFunction
from devito.tools import as_tuple

__all__ = ['collect_nested', 'unevaluate_arithmetic', 'xreplace_constrained',
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


def xreplace_constrained(exprs, make, rule, cm=lambda e: True, repeat=False):
    """
    As opposed to ``xreplace``, which replaces all objects specified in a mapper,
    this function replaces all objects satisfying two criteria: ::

        * The "matching rule" -- a function returning True if a node within ``expr``
            satisfies a given property, and as such should be replaced;
        * A "cost model" -- a function triggering replacement only if a certain
            cost (e.g., operation count) is exceeded. This function is optional.

    Note that there is not necessarily a relationship between the set of nodes
    for which the matching rule returns True and those nodes passing the cost
    model check. It might happen for example that, given the expression ``a + b``,
    all of ``a``, ``b``, and ``a + b`` satisfy the matching rule, but only
    ``a + b`` satisfies the cost model.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param make: A function to construct symbols used for replacement.
                 The function takes as input an integer ID; ID is computed internally
                 and used as a unique identifier for the newly constructed symbol.
    :param rule: The matching rule (a lambda function).
    :param cm: The cost model (a lambda function, optional).
    :param repeat: Repeatedly apply ``xreplace`` until no more replacements are
                   possible (optional, defaults to False).
    """

    processed = []

    def replace(expr):
        if cm(expr):
            temporary = make(replace.c)
            processed.append(Eq(temporary, expr))
            replace.c += 1
            return temporary
        else:
            return expr
    replace.c = 0  # Unique identifier for new temporaries

    def run(expr):
        if expr.is_Float:
            return expr.func(*expr.atoms()), rule(expr)
        elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
            return expr.func(), rule(expr)
        elif expr.is_Symbol:
            return expr.func(expr.name), rule(expr)
        elif expr.is_Atom:
            return expr.func(*expr.atoms()), rule(expr)
        elif isinstance(expr, Indexed):
            return expr.func(*expr.args), rule(expr)
        elif expr.is_Pow:
            base, flag = run(expr.base)
            return expr.func(base, expr.exp), flag
        else:
            children = [run(a) for a in expr.args]
            matching = [a for a, flag in children if flag]
            other = [a for a, _ in children if a not in matching]
            if matching:
                matched = expr.func(*matching)
                if len(matching) == len(children) and rule(expr):
                    # Go look for longer expressions first
                    return matched, True
                elif rule(matched):
                    # Replace what I can replace, then give up
                    return expr.func(*(other + [replace(matched)])), False
                else:
                    # Replace flagged children, then give up
                    return expr.func(*(other + [replace(e) for e in matching])), False
            return expr.func(*other), False

    for expr in as_tuple(exprs):
        root = expr.rhs if expr.is_Equality else expr

        while True:
            handle, _ = run(root)
            if repeat and handle != root:
                root = handle
            else:
                rebuilt = expr.func(expr.lhs, handle) if expr.is_Equality else handle
                processed.append(rebuilt)
                break

    return processed
