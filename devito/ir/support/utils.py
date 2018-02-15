from collections import OrderedDict, defaultdict
from itertools import groupby

from devito.dimension import Dimension
from devito.ir.support.basic import Access, Scope
from devito.ir.support.space import Interval, Backward, Forward, Any
from devito.ir.support.stencil import Stencil
from devito.symbolics import retrieve_indexed
from devito.tools import as_tuple, flatten

__all__ = ['compute_intervals', 'detect_flow_directions', 'compute_directions',
           'force_directions', 'group_expressions']


def compute_intervals(expr):
    """Return an iterable of :class:`Interval`s representing the data items
    accessed by the :class:`sympy.Eq` ``expr``."""
    # Deep retrieval of indexed objects in /expr/
    indexeds = retrieve_indexed(expr, mode='all')
    indexeds += flatten([retrieve_indexed(i) for i in e.indices] for e in indexeds)

    # Detect the indexeds' offsets along each dimension
    stencil = Stencil()
    for e in indexeds:
        for a in e.indices:
            if isinstance(a, Dimension):
                stencil[a].update([0])
            d = None
            off = [0]
            for i in a.args:
                if isinstance(i, Dimension):
                    d = i
                elif i.is_integer:
                    off += [int(i)]
            if d is not None:
                stencil[d].update(off)

    # Determine intervals and their iterators
    iterators = OrderedDict()
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


def detect_flow_directions(exprs):
    """Return a mapper from :class:`Dimension`s to iterables of
    :class:`IterationDirection`s representing the theoretically necessary
    directions to evaluate ``exprs`` so that the information "naturally
    flows" from an iteration to another."""
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
            for d in dimensions:
                try:
                    if w.distance(r, d) > 0:
                        mapper[d].add(Forward)
                        break
                    elif w.distance(r, d) < 0:
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

    # Add in stepping dimensions (just in case they haven't been detected yet)
    # note: stepping dimensions may force a direction on the parent
    assert all(v == {Any} or mapper.get(k.parent, v) in [v, {Any}]
               for k, v in mapper.items() if k.is_Stepping)
    mapper.update({k.parent: set(v) for k, v in mapper.items()
                   if k.is_Stepping and mapper.get(k.parent) == {Any}})

    # Add in derived dimensions parents
    mapper.update({k.parent: set(v) for k, v in mapper.items()
                   if k.is_Derived and k.parent not in mapper})

    return mapper


def compute_directions(exprs, key):
    """
    Return a mapper ``M : D -> I`` where D is the set of :class:`Dimension`s
    found in the input expressions ``exprs``, while I = {Any, Backward,
    Forward} (i.e., the set of possible :class:`IterationDirection`s).

    The iteration direction is chosen so that the information "naturally flows"
    from an iteration to another (i.e., to generate "flow" or "read-after-write"
    dependencies).

    In the case of a clash (e.g., both Forward and Backward should be used
    for a given dimension in order to have a flow dependence), the function
    ``key : D -> I`` is used to pick one value.
    """
    mapper = detect_flow_directions(exprs)
    return force_directions(mapper, key)


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

    return directions, clashes


def group_expressions(exprs):
    """``{exprs} -> ({exprs'}, {exprs''}, ...)`` where: ::

        * There are data dependences within exprs' and within exprs'';
        * There are *no* data dependencies across exprs' and exprs''.
    """
    # Partion based on data dependences
    mapper = OrderedDict()
    ngroups = 0
    for i, e1 in enumerate(exprs):
        if e1 in mapper:
            continue
        found = False
        for e2 in exprs[i+1:]:
            if Scope([e1, e2]).has_dep:
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
