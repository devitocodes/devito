import ctypes
import numpy as np

from sympy import (Add, collect, Eq, Function, Indexed, IndexedBase, Symbol, cse,
                   preorder_traversal, symbols, Wild)
from sympy.utilities.iterables import numbered_symbols

from devito.interfaces import SymbolicData
from devito.dimension import t, x, y, z

__all__ = ['flatten', 'filter_ordered', 'convert_dtype_to_ctype',
           'sympy_find', 'aligned', 'expr_symbols', 'expr_dimensions',
           'expr_indexify', 'expr_cse']


def flatten(l):
    return [item for sublist in l for item in sublist]


def filter_ordered(elements):
    """Filter elements in a list while preserving order"""
    seen = set()
    return [e for e in elements if not (e in seen or seen.add(e))]


def convert_dtype_to_ctype(dtype):
    """Maps numpy types to C types.

    :param dtype: A Python numpy type of int32, float32, int64 or float64
    :returns: Corresponding C type
    """
    conversion_dict = {np.int32: ctypes.c_int, np.float32: ctypes.c_float,
                       np.int64: ctypes.c_int64, np.float64: ctypes.c_double}

    return conversion_dict[dtype]


def sympy_find(expr, term, repl):
    """Change all terms from function notation to array notation.

    Finds all terms of the form term(x1, x2, x3)
    and changes them to repl[x1, x2, x3]. i.e. changes from
    function notation to array notation. It also reorders the indices
    x1, x2, x3 so that the time index comes first.

    :param expr: The expression to be processed
    :param term: The pattern to be replaced
    :param repl: The replacing pattern
    :returns: The changed expression
    """

    _t = symbols("t")

    if type(expr) == term:
        args_wo_t = [i for i in expr.args if i != _t and _t not in i.args]
        args_t = [i for i in expr.args if i == _t or _t in i.args]
        expr = repl[tuple(args_t + args_wo_t)]

    if hasattr(expr, "args"):
        for a in expr.args:
            expr = expr.subs(a, sympy_find(a, term, repl))

    return expr


def aligned(a, alignment=16):
    """Function to align the memmory

    :param a: The given memory
    :param alignment: Granularity of alignment, 16 bytes by default
    :returns: Reference to the start of the aligned memory
    """
    if (a.ctypes.data % alignment) == 0:
        return a

    extra = alignment / a.itemsize
    buf = np.empty(a.size + extra, dtype=a.dtype)
    ofs = (-buf.ctypes.data % alignment) / a.itemsize
    aa = buf[ofs:ofs+a.size].reshape(a.shape)
    np.copyto(aa, a)

    assert (aa.ctypes.data % alignment) == 0

    return aa


def expr_dimensions(expr):
    """Collects all function dimensions used in a sympy expression"""
    dimensions = []

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            dimensions += e.indices

    return list(set(dimensions))


def expr_symbols(expr):
    """Collects defined and undefined symbols used in a sympy expression

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


def expr_indexify(expr):
    """Convert functions into indexed matrix accesses in sympy expression

    :param expr: SymPy function expression to be converted
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def expr_cse(expr):
    """Performs common subexpression elimination on expressions

    :param expr: Sympy equation or list of equations on which CSE needs to be performed

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
        elif isinstance(value, Add) and not set([t, x, y, z]).isdisjoint(set(value.args)):
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
    w = Wild('w1')
    for stencil in stencils:
        new_stencils.append(Eq(stencil.lhs, collect(collect(stencil.rhs, w), [temp.lhs for temp in to_keep])))
        print(stencil)
        print([temp.lhs for temp in to_keep])
        for temp in to_keep:
            if stencil.lhs in preorder_traversal(temp.rhs):
                new_stencils.append(temp)
                break

    return to_keep + new_stencils
