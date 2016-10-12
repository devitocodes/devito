"""
The Devito symbolic engine is built on top of SymPy and provides two
classes of functions:
- for inspection of expressions
- for (in-place) manipulation of expressions
- for creation of new objects given some expressions
All exposed functions are prefixed with 'dse' (devito symbolic engine)
"""

import numpy as np

from sympy import *

from devito.dimension import t, x, y, z
from devito.interfaces import SymbolicData

__all__ = ['dse_dimensions', 'dse_symbols', 'dse_dtype', 'dse_indexify',
           'dse_cse', 'dse_tolambda']


# Inspection

def dse_dimensions(expr):
    """
    Collect all function dimensions used in a sympy expression.
    """
    dimensions = []

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            dimensions += [i for i in e.indices if i not in dimensions]

    return dimensions


def dse_symbols(expr):
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


def dse_dtype(expr):
    """
    Try to infer the data type of an expression.
    """
    dtypes = [e.dtype for e in preorder_traversal(expr) if hasattr(e, 'dtype')]
    return np.find_common_type(dtypes, [])


# Manipulation

def dse_indexify(expr):
    """
    Convert functions into indexed matrix accesses in sympy expression.

    :param expr: sympy function expression to be converted.
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def dse_cse(expr):
    """
    Perform common subexpression elimination on sympy expressions.

    :param expr: sympy equation or list of equations on which CSE is performed.

    :return: A list of the resulting equations after performing CSE
    """
    expr = expr if isinstance(expr, list) else [expr]

    temps, stencils = cse(expr, numbered_symbols("temp"))

    # Restores the LHS
    for i in range(len(expr)):
        stencils[i] = Eq(expr[i].lhs, stencils[i].rhs)

    to_revert = {}
    to_keep = []

    # Restores IndexedBases if they are collected by CSE and
    # reverts changes to simple index operations (eg: t - 1)
    for temp, value in temps:
        if isinstance(value, IndexedBase):
            to_revert[temp] = value
        elif isinstance(value, Indexed):
            to_revert[temp] = value
        elif isinstance(value, Add) and not \
                set([t, x, y, z]).isdisjoint(set(value.args)):
            to_revert[temp] = value
        else:
            to_keep.append((temp, value))

    # Restores the IndexedBases and the Indexes in the assignments to revert
    for temp, value in to_revert.items():
        s_dict = {}
        for arg in preorder_traversal(value):
            if isinstance(arg, Indexed):
                new_indices = []
                for index in arg.indices:
                    if index in to_revert:
                        new_indices.append(to_revert[index])
                    else:
                        new_indices.append(index)
                if arg.base.label in to_revert:
                    s_dict[arg] = Indexed(to_revert[value.base.label], *new_indices)
        to_revert[temp] = value.xreplace(s_dict)

    subs_dict = {}

    # Builds a dictionary of the replacements
    for expr in stencils + [assign for temp, assign in to_keep]:
        for arg in preorder_traversal(expr):
            if isinstance(arg, Indexed):
                new_indices = []
                for index in arg.indices:
                    if index in to_revert:
                        new_indices.append(to_revert[index])
                    else:
                        new_indices.append(index)
                if arg.base.label in to_revert:
                    subs_dict[arg] = Indexed(to_revert[arg.base.label], *new_indices)
                elif tuple(new_indices) != arg.indices:
                    subs_dict[arg] = Indexed(arg.base, *new_indices)
            if arg in to_revert:
                subs_dict[arg] = to_revert[arg]

    stencils = [stencil.xreplace(subs_dict) for stencil in stencils]

    to_keep = [Eq(temp[0], temp[1].xreplace(subs_dict)) for temp in to_keep]

    # If the RHS of a temporary variable is the LHS of a stencil,
    # update the value of the temporary variable after the stencil

    new_stencils = []

    for stencil in stencils:
        new_stencils.append(stencil)

        for temp in to_keep:
            if stencil.lhs in preorder_traversal(temp.rhs):
                new_stencils.append(temp)
                break

    return to_keep + new_stencils


# Creation

def dse_tolambda(exprs):
    """
    Tranform an expression into a lambda.

    :param exprs: an expression or a list of expressions.
    """
    exprs = exprs if isinstance(exprs, list) else [exprs]

    lambdas = []

    for expr in exprs:
        terms = free_terms(expr.rhs)
        term_symbols = [symbols("i%d" % i) for i in range(len(terms))]

        # Substitute IndexedBase references to simple variables
        # lambdify doesn't support IndexedBase references in expressions
        tolambdify = expr.rhs.subs(dict(zip(terms, term_symbols)))
        lambdified = lambdify(term_symbols, tolambdify)
        lambdas.append((lambdified, terms))

    return lambdas


# Utilities

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
