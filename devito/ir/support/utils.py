from collections import defaultdict, namedtuple
from itertools import product

from devito.finite_differences import IndexDerivative
from devito.symbolics import (CallFromPointer, retrieve_indexed, retrieve_terminals,
                              search)
from devito.tools import DefaultOrderedDict, as_tuple, flatten, filter_sorted, split
from devito.types import (Dimension, DimensionTuple, Indirection, ModuloDimension,
                          StencilDimension)

__all__ = ['AccessMode', 'Stencil', 'IMask', 'detect_accesses', 'detect_io',
           'pull_dims', 'unbounded', 'minimum', 'maximum', 'minmax_index',
           'extrema', 'erange']


class AccessMode:

    """
    A descriptor for access modes (read, write, ...).
    """

    _modes = ('R', 'W', 'R/W', 'RR', 'WR', 'NA')

    def __init__(self, is_read=False, is_write=False, mode=None):
        if mode is None:
            assert isinstance(is_read, bool) and isinstance(is_write, bool)
            if is_read and is_write:
                mode = 'R/W'
            elif is_read:
                mode = 'R'
            elif is_write:
                mode = 'W'
            else:
                mode = 'NA'

        assert mode in self._modes
        self.mode = mode

    def __repr__(self):
        return self.mode

    def __eq__(self, other):
        return isinstance(other, AccessMode) and self.mode == other.mode

    @property
    def is_read(self):
        return self.mode in ('R', 'R/W', 'RR')

    @property
    def is_write(self):
        return self.mode in ('W', 'R/W', 'WR')

    @property
    def is_read_only(self):
        return self.is_read and not self.is_write

    @property
    def is_write_only(self):
        return self.is_write and not self.is_read

    @property
    def is_read_write(self):
        return self.is_read and self.is_write

    @property
    def is_read_reduction(self):
        return self.mode == 'RR'

    @property
    def is_write_reduction(self):
        return self.mode == 'WR'

    @property
    def is_reduction(self):
        return self.is_read_reduction or self.is_write_reduction


class Stencil(DefaultOrderedDict):

    """
    A mapping between Dimensions and symbolic expressions representing the
    points of the stencil.

    Typically the values are just integers.

    Parameters
    ----------
    entries : iterable of 2-tuples, optional
        The Stencil entries.
    """

    def __init__(self, items=None):
        # Normalize input
        items = [(d, set(as_tuple(v))) for d, v in as_tuple(items)]

        super().__init__(set, items)

    @classmethod
    def union(cls, *dicts):
        """
        Compute the union of a collection of Stencils.
        """
        output = Stencil()
        for i in dicts:
            for k, v in i.items():
                output[k] |= v
        return output


class IMask(DimensionTuple):

    """
    A mapper from Dimensions to data points or ranges.
    """

    pass


def detect_accesses(exprs):
    """
    Return a mapper `M : F -> S`, where F are Functions appearing in `exprs`
    and S are Stencils. `M[f]` represents all data accesses to `f` within
    `exprs`. Also map `M[None]` to all Dimensions used in `exprs` as plain
    symbols, rather than as array indices.
    """
    # Compute M : F -> S
    mapper = defaultdict(Stencil)
    for e in retrieve_indexed(exprs, deep=True):
        f = e.function

        for a, d0 in zip(e.indices, f.dimensions):
            if isinstance(a, Indirection):
                a = a.mapped

            if isinstance(a, ModuloDimension) and a.parent.is_Stepping:
                # Explicitly unfold SteppingDimensions-induced ModuloDimensions
                mapper[f][a.root].update([a.offset - a.root])

            elif isinstance(a, Dimension):
                mapper[f][a].update([0])

            elif a.is_Add:
                dims = {i for i in a.free_symbols if isinstance(i, Dimension)}

                if not dims:
                    continue
                elif len(dims) > 1:
                    # There are two reasons we may end up here, 1) indirect
                    # accesses (e.g., a[b[x, y] + 1, y]) or 2) as a result of
                    # skewing-based optimizations, such as time skewing (e.g.,
                    # `x - time + 1`) or CIRE rotation (e.g., `x + xx - 4`)
                    d, others = split(dims, lambda i: d0 in i._defines)

                    if any(i.is_Indexed for i in a.args) or len(d) != 1:
                        # Case 1) -- with indirect accesses there's not much we can infer
                        continue
                    else:
                        # Case 2)
                        d, = d
                        _, o = split(others, lambda i: i.is_Custom)
                        off = sum(i for i in a.args if i.is_integer or i.free_symbols & o)
                else:
                    d, = dims

                    # At this point, typically, the offset will be an integer.
                    # In some cases though it could be an expression, e.g.
                    # `db0 + time_m - 1` (from CustomDimensions due to buffering)
                    # or `x + o_x` (from MPI routines) or `time - ns` (from
                    # guarded accesses to TimeFunctions) or ... In all these cases,
                    # what really matters is the integer part of the offset, as
                    # any other symbols may resolve to zero at runtime, which is
                    # the base case scenario we fallback to
                    off = sum(i for i in a.args if i.is_integer)

                    # NOTE: `d in a.args` is too restrictive because of guarded
                    # accesses such as `time / factor - 1`
                    assert d in a.free_symbols

                if (d.is_Custom or d.is_Default) and d.symbolic_size.is_integer:
                    # Explicitly unfold Default and CustomDimensions
                    mapper[f][d].update(range(off, d.symbolic_size + off))
                else:
                    mapper[f][d].add(off)

    # Compute M[None]
    other_dims = set()
    for e in as_tuple(exprs):
        other_dims.update(i for i in e.free_symbols if isinstance(i, Dimension))
        try:
            other_dims.update(e.implicit_dims or {})
        except AttributeError:
            # Not a types.Eq
            pass
    other_dims = filter_sorted(other_dims)
    mapper[None] = Stencil([(i, 0) for i in other_dims])

    return mapper


def detect_io(exprs, relax=False):
    """
    ``{exprs} -> ({reads}, {writes})``

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        The searched expressions.
    relax : bool, optional
        If False, as by default, collect all Input objects, such as
        Constants and Functions. Otherwise, also collect AbstractFunctions.
    """
    exprs = as_tuple(exprs)
    if relax is False:
        rule = lambda i: i.is_Input
    else:
        rule = lambda i: i.is_Input or i.is_AbstractFunction

    # Don't forget the nasty case with indirections on the LHS:
    # >>> u[t, a[x]] = f[x]  -> (reads={a, f}, writes={u})

    roots = []
    for i in exprs:
        try:
            roots.append(i.rhs)
            roots.extend(list(i.lhs.indices))
            roots.extend(list(i.conditionals.values()))
        except AttributeError:
            # E.g., CallFromPointer
            roots.append(i)

    reads = []
    terminals = flatten(retrieve_terminals(i, deep=True) for i in roots)
    for i in terminals:
        candidates = set(i.free_symbols)
        try:
            candidates.update({i.function})
        except AttributeError:
            pass
        for j in candidates:
            try:
                if rule(j):
                    reads.append(j)
            except AttributeError:
                pass

    writes = []
    for i in exprs:
        try:
            f = i.lhs.function
        except AttributeError:
            continue
        try:
            if rule(f):
                writes.append(f)
        except AttributeError:
            # We only end up here after complex IET transformations which make
            # use of composite types
            assert isinstance(i.lhs, CallFromPointer)
            f = i.lhs.base.function
            if rule(f):
                writes.append(f)

    return filter_sorted(reads), filter_sorted(writes)


def pull_dims(exprs, flag=True):
    """
    Extract all Dimensions from one or more expressions. If `flag=True`
    (default), all of the ancestor and descendant Dimensions are extracted
    as well.
    """
    dims = set()
    for e in as_tuple(exprs):
        dims.update({i for i in e.free_symbols if isinstance(i, Dimension)})
    if flag:
        return set().union(*[d._defines for d in dims])
    else:
        return dims


# *** Utility functions for expressions that potentially contain StencilDimensions

def unbounded(expr):
    """
    Retrieve all unbounded Dimensions in `expr`.
    """
    # At the moment we only have logic to retrieve unbounded StencilDimensions,
    # but in the future this might change
    bound = set().union(*[i.dimensions for i in search(expr, IndexDerivative)])
    sdims = search(expr, StencilDimension, mode='unique', deep=True)

    return sdims - bound


Extrema = namedtuple('Extrema', 'm M')


def _relational(expr, callback, udims=None):
    """
    Helper for `minimum`, `maximum`, and potential future utilities that share
    a significant chunk of logic.
    """
    if not udims:
        udims = unbounded(expr)

    # Resolution rule 1: StencilDimensions
    sdims = [d for d in udims if d.is_Stencil]
    if not sdims:
        return expr
    mapper = {e: callback(e) for e in sdims}

    return expr.subs(mapper)


def minimum(expr, udims=None, ispace=None):
    """
    Substitute the unbounded Dimensions in `expr` with their minimum point.

    Unbounded Dimensions whose possible minimum value is not known are ignored.
    """
    def callback(sd):
        try:
            return sd._min + ispace[sd].lower
        except (TypeError, KeyError):
            return sd._min

    return _relational(expr, callback, udims)


def maximum(expr, udims=None, ispace=None):
    """
    Substitute the unbounded Dimensions in `expr` with their maximum point.

    Unbounded Dimensions whose possible maximum value is not known are ignored.
    """
    def callback(sd):
        try:
            return sd._max + ispace[sd].upper
        except (TypeError, KeyError):
            return sd._max

    return _relational(expr, callback, udims)


def extrema(expr, ispace=None):
    """
    The minimum and maximum extrema assumed by `expr` once the unbounded
    Dimensions are resolved.
    """
    return Extrema(minimum(expr, ispace=ispace), maximum(expr, ispace=ispace))


def minmax_index(expr, d):
    """
    Return the minimum and maximum indices along the `d` Dimension
    among all Indexeds in `expr`.
    """
    indices = set()
    for i in retrieve_indexed(expr):
        try:
            indices.add(i.indices[d])
        except KeyError:
            pass

    return Extrema(min(minimum(i) for i in indices),
                   max(maximum(i) for i in indices))


def erange(expr):
    """
    All possible values that `expr` can assume once its unbounded Dimensions
    are resolved.
    """
    udims = unbounded(expr)
    if not udims:
        return (expr,)

    sdims = [d for d in udims if d.is_Stencil]
    ranges = [i.range for i in sdims]
    mappers = [dict(zip(sdims, i)) for i in product(*ranges)]

    return tuple(expr.subs(m) for m in mappers)
