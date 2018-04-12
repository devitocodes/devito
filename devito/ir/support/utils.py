from collections import OrderedDict, defaultdict
from itertools import groupby

from devito.dimension import Dimension
from devito.ir.support.basic import Access, Scope
from devito.ir.support.space import Interval, Backward, Forward, Any
from devito.ir.support.stencil import Stencil
from devito.symbolics import retrieve_indexed, retrieve_terminals
from devito.tools import as_tuple, flatten, filter_sorted

__all__ = ['detect_accesses', 'detect_oobs', 'build_intervals',
           'detect_flow_directions', 'force_directions', 'group_expressions',
           'align_accesses', 'detect_io']


def detect_accesses(expr):
    """
    Return a mapper ``M : F -> S``, where F are :class:`Function`s appearing
    in ``expr`` and S are :class:`Stencil`s. ``M[f]`` represents all data accesses
    to ``f`` within ``expr``.
    """
    mapper = defaultdict(Stencil)
    for e in retrieve_indexed(expr, mode='all', deep=True):
        f = e.base.function
        for a in e.indices:
            if isinstance(a, Dimension):
                mapper[f][a].update([0])
            d = None
            off = []
            for i in a.args:
                if isinstance(i, Dimension):
                    d = i
                elif i.is_integer:
                    off += [int(i)]
            if d is not None:
                mapper[f][d].update(off or [0])
    return mapper


def detect_oobs(mapper):
    """
    Given M as produced by :func:`detect_accesses`, return the set of
    :class:`Dimension`s that cannot be iterated over for the entire
    computational domain, to avoid out-of-bounds (OOB) accesses.
    """
    found = set()
    for f, stencil in mapper.items():
        if not f.is_TensorFunction:
            continue
        for d, v in stencil.items():
            p = d.parent if d.is_Sub else d
            try:
                if min(v) < 0 or max(v) > sum(f._offset_domain[p]):
                    found.add(p)
            except KeyError:
                # Unable to detect presence of OOB accesses
                # (/p/ not in /f._offset_domain/, typical of indirect
                # accesses such as A[B[i]])
                pass
    return found | {i.parent for i in found if i.is_Derived}


def build_intervals(mapper):
    """
    Given M as produced by :func:`detect_accesses`, return: ::

        * An iterable of :class:`Interval`s, representing the data items
          accessed in each :class:`Dimension` in M;
        * A dictionary of ``iterators``, suitable to build an
          :class:`IterationSpace`.
    """
    iterators = OrderedDict()
    stencil = Stencil.union(*mapper.values())
    for i in stencil.dimensions:
        if i.is_NonlinearDerived:
            iterators.setdefault(i.parent, []).append(stencil.entry(i))
        else:
            iterators.setdefault(i, [])

    intervals = []
    for k, v in iterators.items():
        offs = set.union(set(stencil.get(k)), *[i.ofs for i in v])
        intervals.append(Interval(k, min(offs), max(offs)))

    return intervals, iterators


def align_accesses(expr, reverse=False):
    """
    ``expr -> expr'``, with ``expr'`` semantically equivalent to ``expr``, but
    with data accesses aligned to the computational domain. If the optional flag
    ``reverse`` is passed as True (defaults False), then the reverse operation
    takes place; that is, assuming ``expr`` was aligned to the computational
    domain, ``expr'`` gets aligned back to the first allocated entry.
    """
    shift = lambda i: (-i if reverse is True else i)
    mapper = {}
    for indexed in retrieve_indexed(expr):
        f = indexed.base.function
        if not f.is_TensorFunction:
            continue
        subs = {i: i + shift(j.left) for i, j in zip(indexed.indices, f._offset_domain)}
        mapper[indexed] = indexed.xreplace(subs)
    return expr.xreplace(mapper)


def detect_flow_directions(exprs):
    """
    Return a mapper from :class:`Dimension`s to iterables of
    :class:`IterationDirection`s representing the theoretically necessary
    directions to evaluate ``exprs`` so that the information "naturally
    flows" from an iteration to another.
    """
    exprs = as_tuple(exprs)

    writes = [Access(i.lhs, 'W') for i in exprs]
    reads = flatten(retrieve_indexed(i.rhs, mode='all') for i in exprs)
    reads = [Access(i, 'R') for i in reads]

    # Determine indexed-wise direction by looking at the vector distance
    mapper = defaultdict(set)
    for w in writes:
        for r in reads:
            if r.name != w.name:
                continue
            dimensions = [d for d in w.aindices if d is not None]
            if not dimensions:
                continue
            for d in dimensions:
                distance = None
                for i in d._defines:
                    try:
                        distance = w.distance(r, i, view=i)
                    except TypeError:
                        pass
                try:
                    if distance > 0:
                        mapper[d].add(Forward)
                        break
                    elif distance < 0:
                        mapper[d].add(Backward)
                        break
                    else:
                        mapper[d].add(Any)
                except TypeError:
                    # Nothing can be deduced
                    mapper[d].add(Any)
                    break
            # Remainder
            for d in dimensions[dimensions.index(d) + 1:]:
                mapper[d].add(Any)

    # Add in any encountered Dimension
    mapper.update({d: {Any} for d in flatten(i.aindices for i in reads + writes)
                   if d is not None and d not in mapper})

    # Add in derived-dimensions parents, in case they haven't been detected yet
    mapper.update({k.parent: set(v) for k, v in mapper.items()
                   if k.is_Derived and mapper.get(k.parent, {Any}) == {Any}})

    return mapper


def force_directions(mapper, key):
    """
    Return a mapper ``M : D -> I`` where D is the set of :class:`Dimension`s
    found in the input mapper ``M' : D -> {I}``, while I = {Any, Backward,
    Forward} (i.e., the set of possible :class:`IterationDirection`s).

    The iteration direction is chosen so that the information "naturally flows"
    from an iteration to another (i.e., to generate "flow" or "read-after-write"
    dependencies).

    In the case of a clash (e.g., both Forward and Backward should be used
    for a given dimension in order to have a flow dependence), the function
    ``key : D -> I`` is used to pick one value.
    """
    mapper = {k: set(v) for k, v in mapper.items()}
    clashes = set(k for k, v in mapper.items() if len(v - {Any}) > 1)
    directions = {}
    for k, v in mapper.items():
        if len(v) == 1:
            directions[k] = v.pop()
        elif len(v) == 2:
            try:
                v.remove(Any)
                directions[k] = v.pop()
            except KeyError:
                assert k in clashes
                directions[k] = key(k)
        else:
            assert k in clashes
            directions[k] = key(k)

    # Derived dimensions enforce a direction on the parent
    for k, v1 in list(directions.items()):
        if k.is_Derived and directions.get(k.parent) == Any:
            directions[k.parent] = v1

    return directions, clashes


def group_expressions(exprs):
    """``{exprs} -> ({exprs'}, {exprs''}, ...)`` where: ::

        * There are data dependences within exprs' and within exprs'';
        * There are *no* data dependencies across exprs' and exprs''.
    """
    # Partion based on data dependences
    mapper = OrderedDict()
    ngroups = 0
    for e1 in exprs:
        if e1 in mapper:
            # Optimization: we know already that a group for `e1` has been found
            continue
        found = False
        for e2 in exprs:
            if e1 is e2:
                continue
            elif Scope([e1, e2]).has_dep:
                v = mapper.get(e1, mapper.get(e2))
                if v is None:
                    ngroups += 1
                    v = ngroups
                mapper[e1] = mapper[e2] = v
                found = True
        if not found:
            ngroups += 1
            mapper[e1] = ngroups

    # Reorder to match input ordering
    groups = []
    data = sorted(mapper, key=lambda i: mapper[i])
    for k, g in groupby(data, key=lambda i: mapper[i]):
        groups.append(tuple(sorted(g, key=lambda i: exprs.index(i))))

    # Sanity check
    assert max(mapper.values()) == len(groups)

    return tuple(groups)


def detect_io(exprs, relax=False):
    """``{exprs} -> ({reads}, {writes})

    :param exprs: The expressions inspected.
    :param relax: (Optional) if False, as by default, collect only
                  :class:`Constant`s and :class:`Function`s. Otherwise,
                  collect any :class:`Basic`s.
    """
    exprs = as_tuple(exprs)
    if relax is False:
        rule = lambda i: i.is_Input
    else:
        rule = lambda i: i.is_Scalar or i.is_Tensor

    reads = []
    for i in flatten(retrieve_terminals(i, deep=True) for i in exprs):
        candidates = i.free_symbols
        try:
            candidates.update({i.base.function})
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
            f = i.lhs.base.function
        except AttributeError:
            continue
        if rule(f):
            writes.append(f)

    return filter_sorted(reads), filter_sorted(writes)
