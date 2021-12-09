from collections import OrderedDict, defaultdict

from devito.ir.support.guards import GuardBound
from devito.ir.support.space import DataSpace, Interval, IntervalGroup
from devito.ir.support.stencil import Stencil
from devito.symbolics import CallFromPointer, retrieve_indexed, retrieve_terminals
from devito.tools import as_tuple, flatten, filter_sorted
from devito.types import Dimension, ModuloDimension

__all__ = ['detect_accesses', 'detect_oobs', 'build_iterators', 'build_intervals',
           'derive_dspace', 'detect_io']


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
            else:
                d = None
                off = []
                for i in a.args:
                    if isinstance(i, Dimension):
                        d = i
                    elif i.is_integer:
                        off += [int(i)]
                if d is not None:
                    v[d].update(off or [0])

    # Compute M[None]
    other_dims = set()
    for e in as_tuple(exprs):
        other_dims.update(i for i in e.free_symbols if isinstance(i, Dimension))
        other_dims.update(e.implicit_dims)
    mapper[None] = Stencil([(i, 0) for i in other_dims])

    return mapper


def detect_oobs(mapper):
    """
    Given M as produced by :func:`detect_accesses`, return the set of
    Dimensions that *cannot* be iterated over the entire computational
    domain, to avoid out-of-bounds (OOB) accesses.
    """
    found = set()
    for f, stencil in mapper.items():
        if f is None or not (f.is_DiscreteFunction or f.is_Array):
            continue
        for d, v in stencil.items():
            p = d.parent if d.is_Sub else d
            try:
                test0 = min(v) < 0
                test1 = max(v) > f._size_nodomain[p].left + f._size_halo[p].right
                if test0 or test1:
                    # It'd mean trying to access a point before the
                    # left padding (test0) or after the right halo (test1)
                    found.add(p)
            except KeyError:
                # Unable to detect presence of OOB accesses
                # (/p/ not in /f._size_halo/, typical of indirect
                # accesses such as A[B[i]])
                pass
    return found | set().union(*[i._defines for i in found if i.is_Derived])


def build_iterators(dimensions):
    """
    Given `C` a collection of Dimensions, return a mapper `M' : D -> V`, where
    D are the non-derived Dimensions in C, while V the DerivedDimensions. Thus,
    `M'[d]` provides the sub-iterators along the Dimension `d`.
    """
    iterators = OrderedDict()
    for d in dimensions:
        if d.is_Stepping or d.is_Incr:
            values = iterators.setdefault(d.root, [])
            if d not in values:
                values.append(d)
        elif d.is_Conditional:
            iterators.setdefault(d.root, [])
        else:
            iterators.setdefault(d, [])
    return {d: tuple(v) for d, v in iterators.items()}


def build_intervals(stencil):
    """
    Given a Stencil, return an iterable of Intervals, one
    for each Dimension in the stencil.
    """
    mapper = defaultdict(set)
    for d, offs in stencil.items():
        if d.is_Conditional or d.is_Incr:
            mapper[d.parent].update(offs)
        elif d.is_NonlinearDerived:
            mapper[d.root].update(offs)
        else:
            mapper[d].update(offs)
    return [Interval(d, min(offs), max(offs)) for d, offs in mapper.items()]


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

        intervals = IntervalGroup(build_intervals(v))

        # E.g., relax `t` as `time`, but do *not* relax `xi` as `x`
        cond = lambda d: d.is_Derived and not d.is_Sub
        intervals = intervals.promote(cond)

        parts[f] = intervals

    # If the bound of a Dimension is explicitely guarded, then we should
    # shrink the `parts` accordingly
    for d, v in (guards or {}).items():
        if v.find(GuardBound):
            parts = {f: i.zero(d) for f, i in parts.items()}

    # Construct the `intervals` of the DataSpace, that is a global,
    # Dimension-centric view of the data space
    oobs = detect_oobs(accesses)
    v = build_intervals(Stencil.union(*accesses.values()))
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
