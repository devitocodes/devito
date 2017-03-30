from collections import OrderedDict, namedtuple

import numpy as np
from sympy import (Function, Indexed, Number, Symbol, cos, lambdify,
                   preorder_traversal, sin)

from devito.dimension import Dimension, t
from devito.dse.search import retrieve_indexed, retrieve_ops, search
from devito.dse.queries import q_indirect
from devito.interfaces import SymbolicData
from devito.logger import warning
from devito.tools import SetOrderedDict, flatten

__all__ = ['estimate_cost', 'estimate_memory', 'indexify', 'retrieve_dimensions',
           'retrieve_dtype', 'retrieve_symbols', 'retrieve_shape', 'as_symbol',
           'stencil', 'terminals', 'tolambda']


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


def stencil(expr):
    """
    Return the stencil of ``expr`` as an OrderedDict from encountered dimensions
    to integer points (the "neighboring" points accessed).
    """
    assert expr.is_Equality

    offsets = SetOrderedDict()

    indexed = list(retrieve_indexed(expr.lhs))
    indexed += list(retrieve_indexed(expr.rhs))
    indexed += flatten([retrieve_indexed(i) for i in e.indices] for e in indexed)
    for e in indexed:
        for a in e.indices:
            if isinstance(a, Dimension):
                offsets[a].update([0])
            d = None
            off = []
            for idx in a.args:
                if isinstance(idx, Dimension):
                    d = idx
                elif idx.is_integer:
                    off += [idx]
            if d is not None:
                offsets[d].update(off)
    return offsets


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

    AliasInfo = namedtuple('AliasInfo', 'aliased offsets')

    cache = {e: retrieve_indexed(e, mode='all') for e in exprs}

    def find_translation(e1, e2):
        # Example:
        # e1 = A[i,j] + A[i,j+1]
        # e2 = A[i+1,j] + A[i+1,j+1]
        # Compute the pairwise offsets translation, that is:
        # d=[(1,0), (1,0)]
        # e1 and e2 alias each other iff the translation is uniform across
        # their symbols ((1, 0) in this example)
        translations = set()
        for indexed1, indexed2 in zip(cache[e1], cache[e2]):
            if q_indirect(indexed1) or q_indirect(indexed2):
                return ()
            translation = []
            dimensions = indexed1.base.function.indices
            for i1, i2, d in zip(indexed1.indices, indexed2.indices, dimensions):
                stride = i2 - i1
                if stride.is_Number:
                    translation.append((d, stride))
                else:
                    return ()
            translations.add(tuple(translation))
        return () if len(translations) != 1 else translations.pop()

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
        return find_translation(e1, e2) if compare_ops(e1, e2) else ()

    aliases = OrderedDict()
    mapper = OrderedDict()
    unseen = list(exprs)
    while unseen:
        handle = unseen[0]
        alias = OrderedDict()
        for e in list(unseen):
            translation = compare(handle, e)
            if translation:
                alias[e] = translation
                unseen.remove(e)
        if alias:
            for e in alias:
                mapper[e] = alias
            # ``handle`` represents the group origin, ie the expression with
            # respect to which all translations have been computed
            # ``offsets`` is a summary of the translations w.r.t. the origin
            v = [SetOrderedDict([(k, {v}) for k, v in i]) for i in alias.values()]
            offsets = SetOrderedDict.union(*v)
            aliases[handle] = AliasInfo(alias.keys(), offsets)
        else:
            unseen.remove(handle)
            mapper[handle] = OrderedDict()

    return mapper, aliases


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
