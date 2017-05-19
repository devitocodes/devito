import itertools
from collections import OrderedDict, namedtuple

from sympy import Indexed

from devito.dse.search import retrieve_indexed
from devito.dse.queries import q_indirect
from devito.stencil import Stencil

__all__ = ['collect_aliases']


def collect_aliases(exprs):
    """
    Determine groups of aliasing expressions in ``exprs``.

    An expression A aliases an expression B if both A and B apply the same
    operations to the same input operands, with the possibility for indexed objects
    to index into locations at a fixed constant offset in each dimension.

    For example: ::

        exprs = (a[i+1] + b[i+1], a[i+1] + b[j+1], a[i] + c[i],
                 a[i+2] - b[i+2], a[i+2] + b[i], a[i-1] + b[i-1])

    The following expressions in ``exprs`` alias to ``a[i] + b[i]``: ::

        a[i+1] + b[i+1] : same operands and operations, distance along i = 1
        a[i-1] + b[i-1] : same operands and operations, distance along i = -1

    Whereas the following do not: ::

        a[i+1] + b[j+1] : because at least one index differs
        a[i] + c[i] : because at least one of the operands differs
        a[i+2] - b[i+2] : because at least one operation differs
        a[i+2] + b[i] : because distance along ``i`` differ (+2 and +0)
    """

    ExprData = namedtuple('ExprData', 'dimensions offsets')

    # Discard expressions that surely won't alias to anything
    candidates = OrderedDict()
    for expr in exprs:
        indexeds = retrieve_indexed(expr, mode='all')
        if indexeds and not any(q_indirect(i) for i in indexeds):
            handle = calculate_offsets(indexeds)
            if handle:
                candidates[expr.rhs] = ExprData(*handle)

    aliases = OrderedDict()
    mapper = OrderedDict()
    unseen = list(candidates)
    while unseen:
        # Find aliasing expressions
        handle = unseen.pop(0)
        group = [handle]
        for e in list(unseen):
            if compare(handle, e) and\
                    is_translated(candidates[handle].offsets, candidates[e].offsets):
                group.append(e)
                unseen.remove(e)
        mapper.update([(i, group) for i in group])
        if len(group) == 1:
            continue

        # Create an alias for the group of aliasing expressions, as well as
        # any metadata needed by the caller
        offsets = [tuple(candidates[e].offsets) for e in group]
        COM, distances = calculate_COM(offsets)
        alias = create_alias(handle, COM)

        aliases[alias] = Alias(alias, group, distances, candidates[handle].dimensions)

    # Heuristically attempt to relax the aliases offsets
    # to maximize the likelyhood of loop fusion
    grouped = OrderedDict()
    for i in aliases.values():
        grouped.setdefault(i.dimensions, []).append(i)
    for dimensions, group in grouped.items():
        ideal_anti_stencil = Stencil.union(*[i.anti_stencil for i in group])
        for i in group:
            if i.anti_stencil.subtract(ideal_anti_stencil).empty:
                aliases[i.alias] = i.relax(ideal_anti_stencil)

    return mapper, aliases


# Helpers

def create_alias(expr, offsets):
    """
    Create an aliasing expression of ``expr`` by replacing the offsets of each
    indexed object in ``expr`` with the new values in ``offsets``. ``offsets``
    is an ordered sequence of tuples with as many elements as the number of
    indexed objects in ``expr``.
    """
    indexeds = retrieve_indexed(expr, mode='all')
    assert len(indexeds) == len(offsets)

    mapper = {}
    for indexed, ofs in zip(indexeds, offsets):
        base = indexed.base
        dimensions = base.function.indices
        assert len(dimensions) == len(ofs)
        mapper[indexed] = indexed.func(base, *[sum(i) for i in zip(dimensions, ofs)])

    return expr.xreplace(mapper)


def calculate_COM(offsets):
    """
    Determine the centre of mass (COM) in a collection of offsets.

    The COM minimizes the average distance along each dimension amongst the
    values in ``offsets``.

    Also return the distance of each element in ``offsets`` from the COM.
    """
    COM = []
    for ofs in zip(*offsets):
        handle = []
        for i in zip(*ofs):
            strides = sorted(set(i))
            # Heuristic:
            # - middle point if odd number of values, or
            # - zero-offset if present, or
            # - middle point rounded to the bottom
            index = int((len(strides) - 1) / 2)
            if (len(strides) - 1) % 2 == 0:
                handle.append(strides[index])
            elif 0 in strides:
                handle.append(0)
            else:
                handle.append(strides[index])
        COM.append(tuple(handle))

    distances = []
    for ofs in offsets:
        handle = distance(COM, ofs)
        assert len(handle) == 1
        distances.append(handle.pop())

    return COM, distances


def calculate_offsets(indexeds):
    """
    Return a list of tuples, with one tuple for each indexed object appearing
    in ``indexeds``. A tuple represents the offsets from the origin (0, 0, ..., 0)
    along each dimension. All objects must use the same dimensions, in the same
    order; otherwise, ``None`` is returned.

    For example, given: ::

        indexeds = [A[i,j,k], B[i,j+2,k+3]]

    Return: ::

        [(0, 0, 0), (0, 2, 3)]
    """
    processed = []
    reference = indexeds[0].base.function.indices
    for indexed in indexeds:
        dimensions = indexed.base.function.indices
        if dimensions != reference:
            return None
        handle = []
        for d, i in zip(dimensions, indexed.indices):
            offset = i - d
            if offset.is_Number:
                handle.append(offset)
            else:
                return None
        processed.append(tuple(handle))
    return tuple(reference), processed


def distance(ofs1, ofs2):
    """
    Determine the distance of ``ofs2`` from ``ofs1``.
    """
    assert len(ofs1) == len(ofs2)
    handle = set()
    for o1, o2 in zip(ofs1, ofs2):
        assert len(o1) == len(o2)
        handle.add(tuple(i2 - i1 for i1, i2 in zip(o1, o2)))
    return handle


def is_translated(ofs1, ofs2):
    """
    Return True if ``ofs2`` is translated w.r.t. to ``ofs1``, False otherwise.

    For example: ::

        e1 = A[i,j] + A[i,j+1]
        e2 = A[i+1,j] + A[i+1,j+1]

    ``ofs1`` would be [(0, 0), (0, 1)], while ``ofs2`` would be [(1, 0), (1,1)], so
    ``e2`` is translated w.r.t. ``e1`` by ``(1, 0)``, and True is returned.
    """
    return len(distance(ofs1, ofs2)) == 1


def compare(e1, e2):
    """
    Return True if the two expressions e1 and e2 alias each other, False otherwise.
    """
    if type(e1) == type(e2) and len(e1.args) == len(e2.args):
        if e1.is_Atom:
            return True if e1 == e2 else False
        elif isinstance(e1, Indexed) and isinstance(e2, Indexed):
            return True if e1.base == e2.base else False
        else:
            for a1, a2 in zip(e1.args, e2.args):
                if not compare(a1, a2):
                    return False
            return True
    else:
        return False


class Alias(object):

    """
    Map an expression (the so called "alias") to a set of aliasing expressions.
    For each aliasing expression, the distance from the alias along each dimension
    is tracked.
    """

    def __init__(self, alias, aliased, distances, dimensions, ghost_offsets=None):
        assert len(aliased) == len(distances)
        assert all(len(i) == len(dimensions) for i in distances)
        self.alias = alias
        self.aliased = aliased
        self.distances = distances
        self.dimensions = dimensions
        self._ghost_offsets = ghost_offsets or []

    @property
    def anti_stencil(self):
        handle = Stencil()
        for d, i in zip(self.dimensions, zip(*self.distances)):
            handle[d].update(set(i))
        for d, i in zip(self.dimensions, zip(*self._ghost_offsets)):
            handle[d].update(set(i))
        return handle

    @property
    def with_distance(self):
        return [(i, OrderedDict(zip(self.dimensions, j)))
                for i, j in zip(self.aliased, self.distances)]

    def relax(self, distances):
        return Alias(self.alias, self.aliased, self.distances, self.dimensions,
                     self._ghost_offsets + list(itertools.product(*distances.values())))
