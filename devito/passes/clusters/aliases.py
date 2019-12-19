from collections import OrderedDict, namedtuple

from sympy import Indexed
import numpy as np

from devito.ir.support import IterationInstance, LabeledVector, Stencil
from devito.symbolics import retrieve_indexed

__all__ = ['collect']


def collect(exprs):
    """
    Determine groups of aliasing expressions in ``exprs``.

    An expression A aliases an expression B if both A and B perform the same
    arithmetic operations over the same input operands, with the possibility for
    Indexeds to access locations at a fixed constant offset in each Dimension.

    For example: ::

        exprs = (a[i+1] + b[i+1],
                 a[i+1] + b[j+1],
                 a[i] + c[i],
                 a[i+2] - b[i+2],
                 a[i+2] + b[i],
                 a[i-1] + b[i-1])

    The following expressions in ``exprs`` alias to ``a[i] + b[i]``: ::

        a[i+1] + b[i+1] : same operands and operations, distance along i = 1
        a[i-1] + b[i-1] : same operands and operations, distance along i = -1

    Whereas the following do not: ::

        a[i+1] + b[j+1] : because at least one index differs
        a[i] + c[i] : because at least one of the operands differs
        a[i+2] - b[i+2] : because at least one operation differs
        a[i+2] + b[i] : because distance along ``i`` differ (+2 and +0)
    """
    # Determine the potential aliases
    candidates = []
    for expr in exprs:
        candidate = analyze(expr)
        if candidate is not None:
            candidates.append(candidate)

    # Group together the aliasing expressions (ultimately build an Alias for each
    # group of aliasing expressions)
    aliases = Aliases()
    unseen = list(candidates)
    while unseen:
        c = unseen.pop(0)

        # Find aliasing expressions
        group = [c]
        for i in list(unseen):
            if compare_ops(c.expr, i.expr) and is_translated(c, i):
                group.append(i)
                unseen.remove(i)

        # Try creating a basis spanning the aliasing expressions' iteration vectors
        try:
            COM, distances = calculate_COM(group)
        except ValueError:
            # Ignore these aliasing expressions and move on
            continue

        # Create an alias expression centering `c`'s Indexeds at the COM
        subs = {i: i.function[[x + v.fromlabel(x, 0) for x in b]]
                for i, b, v in zip(c.indexeds, c.bases, COM)}
        alias = c.expr.xreplace(subs)
        aliased = [i.expr for i in group]

        aliases[alias] = Alias(alias, aliased, distances)

    # Heuristically attempt to relax the Aliases offsets to maximize the
    # likelyhood of loop fusion
    groups = OrderedDict()
    for i in aliases.values():
        groups.setdefault(i.dimensions, []).append(i)
    for group in groups.values():
        ideal_anti_stencil = Stencil.union(*[i.anti_stencil for i in group])
        for i in group:
            if i.anti_stencil.subtract(ideal_anti_stencil).empty:
                aliases[i.alias] = i.relax(ideal_anti_stencil)

    return aliases


# Helpers

Candidate = namedtuple('Candidate', 'expr indexeds bases offsets')


def analyze(expr):
    """
    Determine whether ``expr`` is a potential Alias and collect relevant metadata.

    A necessary condition is that all Indexeds in ``expr`` are affine in the
    access Dimensions so that the access offsets (or "strides") can be derived.
    For example, given the following Indexeds: ::

        A[i, j, k], B[i, j+2, k+3], C[i-1, j+4]

    All of the access functions all affine in ``i, j, k``, and the offsets are: ::

        (0, 0, 0), (0, 2, 3), (-1, 4)
    """
    # No way if writing to a tensor or an increment
    if expr.lhs.is_Indexed or expr.is_Increment:
        return

    indexeds = retrieve_indexed(expr.rhs)
    if not indexeds:
        return

    bases = []
    offsets = []
    for i in indexeds:
        ii = IterationInstance(i)

        # There must not be irregular accesses, otherwise we won't be able to
        # calculate the offsets
        if ii.is_irregular:
            return

        # Since `ii` is regular (and therefore affine), it is guaranteed that `ai`
        # below won't be None
        base = []
        offset = []
        for e, ai in zip(ii, ii.aindices):
            if e.is_Number:
                base.append(e)
            else:
                base.append(ai)
                offset.append((ai, e - ai))
        bases.append(tuple(base))
        offsets.append(LabeledVector(offset))

    return Candidate(expr.rhs, indexeds, bases, offsets)


def compare_ops(e1, e2):
    """
    Return True if the two expressions ``e1`` and ``e2`` perform the same arithmetic
    operations over the same input operands, False otherwise.
    """
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


def is_translated(c1, c2):
    """
    Given two potential aliases ``c1`` and ``c2``, return True if ``c1``
    is translated w.r.t. ``c2``, False otherwise.

    For example: ::

        c1 = A[i,j] + A[i,j+1]
        c2 = A[i+1,j] + A[i+1,j+1]

    ``c1``'s Toffsets are ``{i: [0, 0], j: [0, 1]}``, while ``c2``'s Toffsets are
    ``{i: [1, 1], j: [0, 1]}``. Then, ``c2`` is translated w.r.t. ``c1`` by
    ``(1, 0)``, and True is returned.
    """
    assert len(c1.offsets) == len(c2.offsets)

    # Transpose `offsets` so that
    # offsets = [{x: 2, y: 0}, {x: 1, y: 3}] => {x: [2, 1], y: [0, 3]}
    Toffsets1 = LabeledVector.transpose(*c1.offsets)
    Toffsets2 = LabeledVector.transpose(*c2.offsets)

    return all(len(set(i - j)) == 1 for (_, i), (_, j) in zip(Toffsets1, Toffsets2))


def calculate_COM(group):
    """
    Determine a centre of mass (COM) for a group of definitely aliasing expressions,
    which is a set of bases spanning all iteration vectors.

    Return the COM as well as the vector distance of each aliasing expression from
    the COM.
    """
    # Find the COM
    COM = []
    for ofs in zip(*[i.offsets for i in group]):
        Tofs = LabeledVector.transpose(*ofs)
        entries = []
        for k, v in Tofs:
            try:
                entries.append((k, int(np.mean(v, dtype=int))))
            except TypeError:
                # At least an element in `v` has symbolic components. Even though
                # `analyze` guarantees that no accesses can be irregular, a symbol
                # might still be present as long as it's constant (i.e., known to
                # be never written to). For example: `A[t, x_m + 2, y, z]`
                # At this point, the only chance we have is that the symbolic entry
                # is identical across all elements in `v`
                if len(set(v)) == 1:
                    entries.append((k, v[0]))
                else:
                    raise ValueError
        COM.append(LabeledVector(entries))

    # Calculate the distance from the COM
    distances = []
    for i in group:
        assert len(COM) == len(i.offsets)
        distance = [o.distance(c) for o, c in zip(i.offsets, COM)]
        distance = [(l, set(i)) for l, i in LabeledVector.transpose(*distance)]
        # The distance of each Indexed from the COM must be uniform across all Indexeds
        if any(len(i) != 1 for l, i in distance):
            raise ValueError
        distances.append(LabeledVector([(l, i.pop()) for l, i in distance]))

    return COM, distances


class Aliases(OrderedDict):

    def get(self, key):
        ret = super(Aliases, self).get(key)
        if ret is not None:
            return ret.aliased
        for v in self.values():
            if key in v.aliased:
                return v.aliased
        return []


class Alias(object):

    """
    Map an expression (the so called "alias") to a set of aliasing expressions.
    For each aliasing expression, the distance from the Alias along each Dimension
    is tracked.
    """

    def __init__(self, alias, aliased=None, distances=None, ghost_offsets=None):
        self.alias = alias
        self.aliased = aliased or []
        self.distances = distances or []
        self.ghost_offsets = ghost_offsets or Stencil()

        assert len(self.aliased) == len(self.distances)

        # Transposed distances
        self.Tdistances = LabeledVector.transpose(*distances)

    @property
    def dimensions(self):
        return tuple(i for i, _ in self.Tdistances)

    @property
    def anti_stencil(self):
        ret = Stencil()
        for k, v in self.Tdistances:
            ret[k].update(set(v))
        for k, v in self.ghost_offsets.items():
            ret[k].update(v)
        return ret

    @property
    def diameter(self):
        """
        The min/max distance along each Dimension for this Alias.
        """
        return OrderedDict((k, (min(v), max(v))) for k, v in self.Tdistances)

    @property
    def relaxed_diameter(self):
        """
        Return a map telling the min/max offsets in each Dimension for this Alias.
        The extremes are potentially larger than those provided by ``self.diameter``,
        as here we're also taking into account the ghost offsets.
        """
        return OrderedDict((k, (min(v), max(v))) for k, v in self.anti_stencil.items())

    @property
    def with_distance(self):
        """Return a tuple associating each aliased expression with its distance from
        ``self.alias``."""
        return tuple(zip(self.aliased, self.distances))

    def add(self, aliased, distance):
        aliased = self.aliased + [aliased]
        distances = self.distances + [distance]
        return Alias(self.alias, aliased, distances, self.ghost_offsets)

    def relax(self, stencil):
        ghost_offsets = stencil.add(self.ghost_offsets)
        return Alias(self.alias, self.aliased, self.distances, ghost_offsets)
