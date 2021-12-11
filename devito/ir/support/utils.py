from collections import OrderedDict, defaultdict

from devito.ir.support.guards import BaseGuardBoundNext
from devito.ir.support.space import DataSpace, Forward, Interval, IntervalGroup
from devito.ir.support.stencil import Stencil
from devito.symbolics import CallFromPointer, retrieve_indexed, retrieve_terminals
from devito.tools import as_tuple, flatten, filter_sorted, is_integer
from devito.types import Dimension, ModuloDimension

__all__ = ['detect_accesses', 'derive_dspace', 'detect_io']


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
        v = mapper[e.function]

        for a in e.indices:
            if isinstance(a, ModuloDimension) and a.parent.is_Stepping:
                # Explicitly unfold SteppingDimensions-induced ModuloDimensions
                v[a.root].update([a.offset - a.root])
            elif isinstance(a, Dimension):
                v[a].update([0])
            elif a.is_Add:
                d = None
                off = 0
                for i in a.args:
                    if isinstance(i, Dimension):
                        d = i
                    else:
                        off += i
                if d is not None and is_integer(off):
                    v[d].update([off])

    # Compute M[None]
    other_dims = set()
    for e in as_tuple(exprs):
        other_dims.update(i for i in e.free_symbols if isinstance(i, Dimension))
        other_dims.update(e.implicit_dims)
    mapper[None] = Stencil([(i, 0) for i in other_dims])

    return mapper


def derive_dspace(exprs, guards=None):
    """
    Construct a DataSpace from a collection of `exprs`.
    """
    accesses = detect_accesses(exprs)

    # Construct the `parts` of the DataSpace, that is a projection of the data
    # space for each Function appearing in `exprs`
    parts = {}
    for f, v in accesses.items():
        if f is None:
            continue

        intervals = [Interval(d, min(offs), max(offs)) for d, offs in v.items()]
        intervals = IntervalGroup(intervals)

        # E.g., `xs -> x -> x0_blk0 -> x` or `t0 -> t -> time`
        intervals = intervals.promote(lambda d: d.is_SubIterator)

        # Special case: if the factor of a ConditionalDimension has value 1,
        # then we can safely resort to the parent's Interval
        intervals = intervals.promote(lambda d: d.is_Conditional and d.factor == 1)

        parts[f] = intervals

    # If the bound of a Dimension is explicitely guarded, then we should
    # shrink the `parts` accordingly
    for d, v in (guards or {}).items():
        ret = v.find(BaseGuardBoundNext)
        assert len(ret) <= 1
        if len(ret) == 1:
            if ret.pop().direction is Forward:
                parts = {f: i.translate(d, 0, -1) for f, i in parts.items()}
            else:
                parts = {f: i.translate(d, 1, 0) for f, i in parts.items()}

    # Determine the Dimensions requiring shifted min/max points to avoid OOB accesses
    oobs = set()
    for f, intervals in parts.items():
        for i in intervals:
            if i.dim.is_Sub:
                d = i.dim.parent
            else:
                d = i.dim
            try:
                if i.lower < 0 or \
                   i.upper > f._size_nodomain[d].left + f._size_halo[d].right:
                    # It'd mean trying to access a point before the
                    # left halo (test0) or after the right halo (test1)
                    oobs.update(d._defines)
            except (KeyError, TypeError):
                # Unable to detect presence of OOB accesses (e.g., `d` not in
                # `f._size_halo`, that is typical of indirect accesses `A[B[i]]`)
                pass

    # Construct the `intervals` of the DataSpace, that is a global,
    # Dimension-centric view of the data space
    v = IntervalGroup.generate('union', *parts.values())
    intervals = [i if i.dim in oobs else i.zero() for i in v]

    return DataSpace(intervals, parts)


def detect_io(exprs, relax=False):
    """
    ``{exprs} -> ({reads}, {writes})``

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        The searched expressions.
    relax : bool, optional
        If False, as by default, collect only Constants and Functions.
        Otherwise, collect any Basic object.
    """
    exprs = as_tuple(exprs)
    if relax is False:
        rule = lambda i: i.is_Input
    else:
        rule = lambda i: i.is_Scalar or i.is_Tensor

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
