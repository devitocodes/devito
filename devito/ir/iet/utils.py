import numpy as np

from devito.ir.iet import FindSections, FindSymbols
from devito.symbolics import Keyword, Macro
from devito.tools import filter_ordered
from devito.types import Global

__all__ = ['filter_iterations', 'retrieve_iteration_tree', 'derive_parameters',
           'maybe_alias']


class IterationTree(tuple):

    """
    Represent a sequence of nested Iterations.
    """

    @property
    def root(self):
        return self[0] if self else None

    @property
    def inner(self):
        return self[-1] if self else None

    @property
    def dimensions(self):
        return [i.dim for i in self]

    def __repr__(self):
        return "IterationTree%s" % super().__repr__()

    def __getitem__(self, key):
        ret = super().__getitem__(key)
        return IterationTree(ret) if isinstance(key, slice) else ret


def retrieve_iteration_tree(node, mode='normal'):
    """
    A list with all Iteration sub-trees within an IET.

    Examples
    --------
    Given the Iteration tree:

        .. code-block:: c

           Iteration i
             expr0
             Iteration j
               Iteration k
                 expr1
             Iteration p
               expr2

    Return the list: ::

        [(Iteration i, Iteration j, Iteration k), (Iteration i, Iteration p)]

    Parameters
    ----------
    iet : Node
        The searched Iteration/Expression tree.
    mode : str, optional
        - ``normal``
        - ``superset``: Iteration trees that are subset of larger iteration trees
                        are dropped.
    """
    assert mode in ('normal', 'superset')

    trees = [IterationTree(i) for i in FindSections().visit(node) if i]
    if mode == 'normal':
        return trees
    else:
        found = []
        for i in trees:
            if any(set(i).issubset(set(j)) for j in trees if i != j):
                continue
            found.append(i)
        return found


def filter_iterations(tree, key=lambda i: i):
    """
    Return the first sub-sequence of consecutive Iterations such that
    ``key(iteration)`` is True.
    """
    filtered = []
    for i in tree:
        if key(i):
            filtered.append(i)
        elif len(filtered) > 0:
            break
    return filtered


def derive_parameters(iet, drop_locals=False, ordering='default'):
    """
    Derive all input parameters (function call arguments) from an IET
    by collecting all symbols not defined in the tree itself.
    """
    assert ordering in ('default', 'canonical')

    # Extract all candidate parameters
    candidates = FindSymbols().visit(iet)

    # Symbols, Objects, etc, become input parameters as well
    basics = FindSymbols('basics').visit(iet)
    candidates.extend(i.function for i in basics)

    # Filter off duplicates (e.g., `x_size` is extracted by both calls to
    # FindSymbols)
    candidates = filter_ordered(candidates)

    # Filter off symbols which are defined somewhere within `iet`
    defines = [s.name for s in FindSymbols('defines').visit(iet)]
    parameters = [s for s in candidates if s.name not in defines]

    # Drop globally-visible objects
    parameters = [p for p in parameters
                  if not isinstance(p, (Global, Keyword, Macro))]
    # Drop (to be) locally declared objects as well as global objects
    parameters = [p for p in parameters
                  if not (p._mem_internal_eager or p._mem_constant)]

    # Maybe filter out all other compiler-generated objects
    if drop_locals:
        parameters = [p for p in parameters if not p.is_LocalType]

    # NOTE: This is requested by the caller when the parameters are used to
    # construct Callables whose signature only depends on the object types,
    # rather than on their name
    # TODO: It should maybe be done systematically... but it's gonna change a huge
    # amount of tests and examples; plus, it might break compatibility those
    # using devito as a library-generator to be embedded within legacy codes
    if ordering == 'canonical':
        parameters = sorted(parameters, key=lambda p: str(type(p)))

    return parameters


def maybe_alias(obj, candidate):
    """
    True if `candidate` can act as an alias for `obj`, False otherwise.
    """
    if obj is candidate:
        return True

    # Names are unique throughout compilation, so this is another case we can handle
    # straightforwardly. It might happen that we have an alias used in a subroutine
    # with different type qualifiers (e.g., const vs not const, volatile vs not
    # volatile), but if the names match, they definitely represent the same
    # logical object
    if obj.name == candidate.name:
        return True

    if obj.is_AbstractFunction:
        if not candidate.is_AbstractFunction:
            # Obv
            return False

        # E.g. TimeFunction vs SparseFunction -> False
        if type(obj).__base__ is not type(candidate).__base__:
            return False

        # TODO: At some point we may need to introduce some logic here, but we'll
        # also need to introduce something like __eq_weak__ that compares most of
        # the __rkwargs__ except for e.g. the name

    return False


def has_dtype(iet, dtype):
    """
    Check if the given IET has at least one symbol with the given dtype or
    dtype kind.
    """
    for f in FindSymbols().visit(iet):
        try:
            # Check if the dtype matches exactly (dtype input)
            # or matches the generic kind (dtype generic input)
            if np.issubdtype(f.dtype, dtype) or f.dtype == dtype:
                return True
        except TypeError:
            continue
    else:
        return False
