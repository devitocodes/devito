from devito.interfaces import SymbolicData
from devito.logger import warning

from devito.dse.extended_sympy import Add, Mul

import numpy as np
from sympy import (Indexed, Function, Symbol,
                   count_ops, flatten, lambdify, preorder_traversal)

__all__ = ['indexify', 'retrieve_dimensions', 'retrieve_dtype', 'retrieve_symbols',
           'terminals', 'tolambda']


def free_terms(expr):
    """
    Find the free terms in an expression.
    """
    found = []

    for term in expr.args:
        if isinstance(term, Indexed):
            found.append(term)
        else:
            found += free_terms(term)

    return found


def terminals(expr, discard_indexed=False):
    indexed = list(expr.find(Indexed))

    # To be discarded
    junk = flatten(i.atoms() for i in indexed)

    symbols = list(expr.find(Symbol))
    symbols = [i for i in symbols if i not in junk]

    if discard_indexed:
        return set(symbols)
    else:
        return set(indexed + symbols)


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
            handle = flatten(i.values() for i in handle)
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
