"""
Routines to construct new SymPy expressions transforming the provided input.
"""

from collections import Iterable, OrderedDict

from sympy import collect, collect_const, flatten

from devito.dse.extended_sympy import Add, Eq, Mul
from devito.dse.inspection import count, estimate_cost, retrieve_indexed
from devito.dse.graph import temporaries_graph
from devito.dse.queries import q_indexed, q_op
from devito.interfaces import Indexed, TensorFunction
from devito.tools import as_tuple

__all__ = ['collect_nested', 'common_subexprs_elimination', 'freeze_expression',
           'xreplace_constrained', 'xreplace_indices', 'promote_scalar_expressions']


def freeze_expression(expr):
    """
    Reconstruct ``expr`` turning all :class:`sympy.Mul` and :class:`sympy.Add`
    into, respectively, :class:`devito.Mul` and :class:`devito.Add`.
    """
    if expr.is_Atom or q_indexed(expr):
        return expr
    elif expr.is_Add:
        rebuilt_args = [freeze_expression(e) for e in expr.args]
        return Add(*rebuilt_args, evaluate=False)
    elif expr.is_Mul:
        rebuilt_args = [freeze_expression(e) for e in expr.args]
        return Mul(*rebuilt_args, evaluate=False)
    elif expr.is_Equality:
        rebuilt_args = [freeze_expression(e) for e in expr.args]
        return Eq(*rebuilt_args, evaluate=False)
    else:
        return expr.func(*[freeze_expression(e) for e in expr.args])


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


def collect_nested(expr, aggressive=False):
    """
    Collect terms appearing in expr, checking all levels of the expression tree.

    :param expr: the expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Number or expr.is_Symbol:
            return expr, [expr]
        elif q_indexed(expr) or expr.is_Atom:
            return expr, []
        elif expr.is_Add:
            rebuilt, candidates = zip(*[run(arg) for arg in expr.args])

            w_numbers = [i for i in rebuilt if any(j.is_Number for j in i.args)]
            wo_numbers = [i for i in rebuilt if i not in w_numbers]

            w_numbers = collect_const(expr.func(*w_numbers))
            wo_numbers = expr.func(*wo_numbers)

            if aggressive is True and wo_numbers:
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


def xreplace_constrained(exprs, make, rule, costmodel=lambda e: True, repeat=False):
    """
    Unlike ``xreplace``, which replaces all objects specified in a mapper,
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
                 and used as a unique identifier for the constructed symbols.
    :param rule: The matching rule (a lambda function).
    :param costmodel: The cost model (a lambda function, optional).
    :param repeat: Repeatedly apply ``xreplace`` until no more replacements are
                   possible (optional, defaults to False).
    """

    found = OrderedDict()
    rebuilt = []

    def replace(expr):
        temporary = found.get(expr)
        if temporary:
            return temporary
        else:
            temporary = make(replace.c)
            found[expr] = temporary
            replace.c += 1
            return temporary
    replace.c = 0  # Unique identifier for new temporaries

    def run(expr):
        if expr.is_Atom or q_indexed(expr):
            return expr, rule(expr)
        elif expr.is_Pow:
            base, flag = run(expr.base)
            return expr.func(base, expr.exp, evaluate=False), flag
        else:
            children = [run(a) for a in expr.args]
            matching = [a for a, flag in children if flag]
            other = [a for a, _ in children if a not in matching]
            if matching:
                matched = expr.func(*matching, evaluate=False)
                if len(matching) == len(children) and rule(expr):
                    # Go look for longer expressions first
                    return matched, True
                elif rule(matched) and costmodel(matched):
                    # Replace what I can replace, then give up
                    rebuilt = expr.func(*(other + [replace(matched)]), evaluate=False)
                    return rebuilt, False
                else:
                    # Replace flagged children, then give up
                    replaced = [replace(e) for e in matching if costmodel(e)]
                    unreplaced = [e for e in matching if not costmodel(e)]
                    rebuilt = expr.func(*(other + replaced + unreplaced), evaluate=False)
                    return rebuilt, False
            return expr.func(*other, evaluate=False), False

    # Process the provided expressions
    for expr in as_tuple(exprs):
        assert expr.is_Equality
        root = expr.rhs

        while True:
            ret, _ = run(root)
            if repeat and ret != root:
                root = ret
            else:
                rebuilt.append(expr.func(expr.lhs, ret))
                break

    # Post-process the output
    found = [Eq(v, k) for k, v in found.items()]

    return found + rebuilt, found


def xreplace_indices(exprs, mapper, candidates=None, only_rhs=False):
    """
    Create new expressions from ``exprs``, by replacing all index variables
    specified in mapper appearing as a tensor index. Only tensors whose symbolic
    name appears in ``candidates`` are considered if ``candidates`` is not None.
    """
    get = lambda i: i.rhs if only_rhs is True else i
    handle = flatten(retrieve_indexed(get(i)) for i in as_tuple(exprs))
    if candidates is not None:
        handle = [i for i in handle if i.base.label in candidates]
    mapper = dict(zip(handle, [i.xreplace(mapper) for i in handle]))
    replaced = [i.xreplace(mapper) for i in as_tuple(exprs)]
    return replaced if isinstance(exprs, Iterable) else replaced[0]


def common_subexprs_elimination(exprs, make, mode='default'):
    """
    Perform common subexpressions elimination.

    Note: the output is not guranteed to be topologically sorted.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param make: A function to construct symbols used for replacement.
                 The function takes as input an integer ID; ID is computed internally
                 and used as a unique identifier for the constructed symbols.
    """

    # Note: not defaulting to SymPy's CSE() function for three reasons:
    # - it also captures array index access functions (eg, i+1 in A[i+1] and B[i+1]);
    # - it sometimes "captures too much", losing factorization opportunities;
    # - very slow
    # TODO: a second "sympy" mode will be provided, relying on SymPy's CSE() but
    # also ensuring some sort of post-processing
    assert mode == 'default'  # Only supported mode ATM

    processed = list(exprs)
    mapped = []
    while True:
        # Detect redundancies
        counted = count(mapped + processed, q_op).items()
        targets = OrderedDict([(k, estimate_cost(k)) for k, v in counted if v > 1])
        if not targets:
            break

        # Create temporaries
        hit = max(targets.values())
        picked = [k for k, v in targets.items() if v == hit]
        mapper = OrderedDict([(e, make(len(mapped) + i)) for i, e in enumerate(picked)])

        # Apply repleacements
        processed = [e.xreplace(mapper) for e in processed]
        mapped = [e.xreplace(mapper) for e in mapped]
        mapped = [Eq(v, k) for k, v in reversed(list(mapper.items()))] + mapped

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # Simply renumber the temporaries in ascending order
    mapper = {i.lhs: j.lhs for i, j in zip(mapped, reversed(mapped))}
    processed = [e.xreplace(mapper) for e in processed]

    # Some temporaries may be droppable at this point
    processed = compact_temporaries(processed)

    return processed


def compact_temporaries(exprs):
    """
    Drop temporaries consisting of single symbols.
    """
    g = temporaries_graph(exprs)

    mapper = {list(v.reads)[0]: k for k, v in g.items() if v.is_dead}

    processed = []
    for k, v in g.items():
        if k in mapper:
            processed.append(Eq(mapper[k], v.rhs))
        elif not v.is_dead:
            processed.append(v.xreplace(mapper))

    return processed
