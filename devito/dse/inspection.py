import numpy as np
from sympy import (Indexed, Function, Symbol,
                   count_ops, flatten, lambdify, preorder_traversal)

from devito.dimension import t
from devito.interfaces import SymbolicData
from devito.logger import warning
from devito.tools import flatten

__all__ = ['indexify', 'retrieve_dimensions', 'retrieve_dtype', 'retrieve_symbols',
           'retrieve_shape', 'terminals', 'tolambda']


def terminals(expr, discard_indexed=False):
    """
    Return all Indexed and Symbols in a SymPy expression.
    """

    indexed = retrieve_indexed(expr)

    # Use '.name' for quickly checking uniqueness
    junk = flatten([i.free_symbols for i in indexed])
    junk = [i.name for i in junk]

    symbols = {i for i in expr.free_symbols if i.name not in junk}

    if discard_indexed:
        return symbols
    else:
        indexed.update(symbols)
        return indexed


def collect_aliases(exprs):
    """
    Determine all expressions in ``exprs`` that alias to the same expression.

    An expression A aliases an expression B if both A and B apply the same
    operations to the same input operands, with the possibility for
    :class:`Indexed` to index into locations at a fixed constant offset in
    each dimension.

    For example: ::

        exprs = (a[i+1] + b[i+1], a[i+1] + b[j+1], a[i] + c[i],
                 a[i+2] - b[i+2], a[i+2] + b[i], a[i-1] + b[i-1])

    The following expressions in ``exprs`` alias to ``a[i] + b[i]``: ::

        ``(a[i+1] + b[i+1], a[i-1] + b[i-1])``

    Whereas the following do not: ::

        ``a[i+1] + b[j+1]``: because at least one index differs
        ``a[i] + c[i]``: because at least one of the operands differs
        ``a[i+2] - b[i+2]``: because at least one operation differs
        ``a[i+2] + b[i]``: because there are two offsets (+2 and +0)
    """

    def check_ofs(e):
        return len(set([i.indices for i in retrieve_indexed(e)])) <= 1

    def compare_ops(e1, e2):
        if type(e1) == type(e2) and len(e1.args) == len(e2.args):
            if e1.is_Atom:
                return True if e1 == e2 else False
            elif isinstance(e1, Indexed) and isinstance(e2, Indexed):
                return True if e1.base == e2.base else False
            else:
                for a1, a2 in zip(e1.args, e2.args):
                    if not compare_ops(a1, a2):
                        return False
                return True
        else:
            return False

    def compare(e1, e2):
        return compare_ops(e1, e2) and check_ofs(e1) and check_ofs(e2)

    found = {}
    clusters = []
    unseen = list(exprs)
    while unseen:
        handle = unseen[0]
        alias = []
        for e in list(unseen):
            if compare(handle, e):
                alias.append(e)
                unseen.remove(e)
        if alias:
            cluster = tuple(alias)
            for e in alias:
                found[e] = cluster
            clusters.append(cluster)
        else:
            unseen.remove(handle)
            found[handle] = ()

    return found, clusters


def estimate_cost(handle):
    try:
        # Is it a plain SymPy object ?
        iter(handle)
    except TypeError:
        handle = [handle]
    try:
        # Is it a dict ?
        handle = handle.values()
    except AttributeError:
        try:
            # Must be a list of dicts then
            handle = flatten([i.values() for i in handle])
        except AttributeError:
            pass
    try:
        # At this point it must be a list of SymPy objects
        # We don't count non floating point operations
        handle = [i.rhs if i.is_Equality else i for i in handle]
        total_ops = sum(count_ops(i.args) for i in handle)
        non_flops = sum(count_ops(i.find(Indexed)) for i in handle)
        return total_ops - non_flops
    except:
        warning("Cannot estimate cost of %s" % str(handle))


def retrieve_dimensions(expr):
    """
    Collect all function dimensions used in a sympy expression.
    """
    dimensions = []

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            dimensions += [i for i in e.indices if i not in dimensions]

    return dimensions


def retrieve_symbols(expr):
    """
    Collect defined and undefined symbols used in a sympy expression.

    Defined symbols are functions that have an associated :class
    SymbolicData: object, or dimensions that are known to the devito
    engine. Undefined symbols are generic `sympy.Function` or
    `sympy.Symbol` objects that need to be substituted before generating
    operator C code.
    """
    defined = set()
    undefined = set()

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            defined.add(e.func(*e.indices))
        elif isinstance(e, Function):
            undefined.add(e)
        elif isinstance(e, Symbol):
            undefined.add(e)

    return list(defined), list(undefined)


def retrieve_dtype(expr):
    """
    Try to infer the data type of an expression.
    """
    dtypes = [e.dtype for e in preorder_traversal(expr) if hasattr(e, 'dtype')]
    return np.find_common_type(dtypes, [])


def retrieve_shape(expr):
    indexed = set([e for e in preorder_traversal(expr) if isinstance(e, Indexed)])
    if not indexed:
        return ()
    indexed = sorted(indexed, key=lambda s: len(s.indices), reverse=True)
    indices = [flatten([j.free_symbols for j in i.indices]) for i in indexed]
    assert all(set(indices[0]).issuperset(set(i)) for i in indices)
    return tuple(indices[0])


def retrieve_indexed(expr):
    """
    Find the free terms in an expression. This is much quicker than the more general
    SymPy's find.
    """

    if isinstance(expr, Indexed):
        return {expr}
    else:
        found = set()
        for a in expr.args:
            found.update(retrieve_indexed(a))
        return found


def is_time_invariant(expr, graph=None):
    """
    Check if expr is time invariant. A temporaries graph may be provided
    to determine whether any of the symbols involved in the evaluation
    of expr are time-dependent. If a symbol in expr does not appear in the
    graph, then time invariance is inferred from its shape.
    """
    graph = graph or {}

    if t in expr.free_symbols:
        return False
    elif expr in graph:
        return graph[expr].is_time_invariant

    if expr.is_Equality:
        to_visit = [expr.rhs]
    else:
        to_visit = [expr]

    while to_visit:
        handle = to_visit.pop()
        for i in retrieve_indexed(handle):
            if t in i.free_symbols:
                return False
        temporaries = [i for i in handle.free_symbols if i in graph]
        for i in temporaries:
            to_visit.append(graph[i].rhs)

    return True


def indexify(expr):
    """
    Convert functions into indexed matrix accesses in sympy expression.

    :param expr: sympy function expression to be converted.
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def tolambda(exprs):
    """
    Tranform an expression into a lambda.

    :param exprs: an expression or a list of expressions.
    """
    exprs = exprs if isinstance(exprs, list) else [exprs]

    lambdas = []

    for expr in exprs:
        terms = free_terms(expr.rhs)
        term_symbols = [Symbol("i%d" % i) for i in range(len(terms))]

        # Substitute IndexedBase references to simple variables
        # lambdify doesn't support IndexedBase references in expressions
        tolambdify = expr.rhs.subs(dict(zip(terms, term_symbols)))
        lambdified = lambdify(term_symbols, tolambdify)
        lambdas.append((lambdified, terms))

    return lambdas
