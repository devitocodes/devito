from collections import OrderedDict

from sympy import (Function, Indexed, Number, Symbol, cos, lambdify,
                   preorder_traversal, sin)

from devito.dimension import Dimension, t
from devito.dse.search import retrieve_indexed, retrieve_ops, search
from devito.logger import warning
from devito.tools import flatten

__all__ = ['estimate_cost', 'estimate_memory', 'indexify', 'as_symbol',
           'terminals', 'tolambda']


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


def count(exprs, query):
    """
    Return a mapper ``{(k, v)}`` where ``k`` is a sub-expression in ``exprs``
    matching ``query`` and ``v`` is the number of its occurrences.
    """
    mapper = OrderedDict()
    for expr in exprs:
        found = search(expr, query, 'all', 'bfs')
        for i in found:
            mapper.setdefault(i, 0)
            mapper[i] += 1
    return mapper


def estimate_cost(handle, estimate_functions=False):
    """Estimate the operation count of ``handle``.

    :param handle: a SymPy expression or an iterator of SymPy expressions.
    :param estimate_functions: approximate the operation count of known
                               functions (eg, sin, cos).
    """
    external_functions = {sin: 50, cos: 50}
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
        # We don't use SymPy's count_ops because we do not count integer arithmetic
        # (e.g., array index functions such as i+1 in A[i+1])
        # Also, the routine below is *much* faster than count_ops
        handle = [i.rhs if i.is_Equality else i for i in handle]
        operations = flatten(retrieve_ops(i) for i in handle)
        flops = 0
        for op in operations:
            if op.is_Function:
                if estimate_functions:
                    flops += external_functions.get(op.__class__, 1)
                else:
                    flops += 1
            else:
                flops += len(op.args) - (1 + sum(True for i in op.args if i.is_Integer))
        return flops
    except:
        warning("Cannot estimate cost of %s" % str(handle))


def estimate_memory(handle, mode='realistic'):
    """Estimate the number of memory reads and writes.

    :param handle: a SymPy expression or an iterator of SymPy expressions.
    :param mode: There are multiple ways of computing the estimate: ::

        * ideal: also known as "compulsory traffic", which is the minimum
            number of read/writes to be performed (ie, models an infinite cache).
        * ideal_with_stores: like ideal, but a data item which is both read
            and written is counted twice (ie both load and store are counted).
        * realistic: assume that all datasets, even the time-independent ones,
            need to be re-read at each time iteration.
    """
    assert mode in ['ideal', 'ideal_with_stores', 'realistic']

    def access(symbol):
        assert isinstance(symbol, Indexed)
        # Irregular accesses (eg A[B[i]]) are counted as compulsory traffic
        if any(i.atoms(Indexed) for i in symbol.indices):
            return symbol
        else:
            return symbol.base

    try:
        # Is it a plain SymPy object ?
        iter(handle)
    except TypeError:
        handle = [handle]

    if mode in ['ideal', 'ideal_with_stores']:
        filter = lambda s: t in s.atoms()
    else:
        filter = lambda s: s
    reads = set(flatten([retrieve_indexed(e.rhs) for e in handle]))
    writes = set(flatten([retrieve_indexed(e.lhs) for e in handle]))
    reads = set([access(s) for s in reads if filter(s)])
    writes = set([access(s) for s in writes if filter(s)])
    if mode == 'ideal':
        return len(set(reads) | set(writes))
    else:
        return len(reads) + len(writes)


def as_symbol(expr):
    """
    Extract the "main" symbol from a SymPy object.
    """
    try:
        return Number(expr)
    except (TypeError, ValueError):
        pass
    if isinstance(expr, str):
        return Symbol(expr)
    elif isinstance(expr, Dimension):
        return Symbol(expr.name)
    elif expr.is_Symbol:
        return expr
    elif isinstance(expr, Indexed):
        return expr.base.label
    elif isinstance(expr, Function):
        return Symbol(expr.name)
    else:
        raise TypeError("Cannot extract symbol from type %s" % type(expr))


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
        terms = retrieve_indexed(expr.rhs)
        term_symbols = [Symbol("i%d" % i) for i in range(len(terms))]

        # Substitute IndexedBase references to simple variables
        # lambdify doesn't support IndexedBase references in expressions
        tolambdify = expr.rhs.subs(dict(zip(terms, term_symbols)))
        lambdified = lambdify(term_symbols, tolambdify)
        lambdas.append((lambdified, terms))

    return lambdas
