from collections import OrderedDict, namedtuple

from sympy import Eq

from devito.exceptions import StencilOperationError
from devito.dimension import Dimension
from devito.symbolics import retrieve_indexed, retrieve_terminals
from devito.tools import DefaultOrderedDict, flatten, partial_order


class Stencil(DefaultOrderedDict):

    """
    A Stencil is a mapping from :class:`Dimension` symbols to the set of integer
    offsets used with it in expressions (the "neighboring points accessed").

    This also include zero offsets.

    The mapping is ordered based on the order in which dimensions are encountered
    (if extracted from expressions) or inserted.

    Note: Expressions must have been indexified for a Stencil to be computed.
    """

    def __init__(self, *args):
        """
        Initialize the Stencil.

        :param args: A Stencil may be created in several ways: ::

            * From a SymPy equation, or
            * A list of elements of type: ::
                * SymPy equation, or
                * StencilEntry, or
                * 2-tuple (Dimension, set) -- raw initialization
        """
        processed = []
        for i in args:
            if isinstance(i, Eq):
                processed.extend(self.extract(i).items())
            else:
                for j in i:
                    if isinstance(j, StencilEntry):
                        processed.append((j.dim, set(j.ofs)))
                    elif isinstance(j, tuple) and len(j) == 2:
                        entry = StencilEntry(*j)  # Type checking
                        processed.append((entry.dim, set(entry.ofs)))
                    else:
                        raise RuntimeError('Cannot construct a Stencil for %s' % str(j))
        super(Stencil, self).__init__(set, processed)

    @classmethod
    def extract(cls, expr):
        """
        Compute the stencil of ``expr``.
        """
        assert expr.is_Equality

        # Collect all indexed objects appearing in /expr/
        terminals = retrieve_terminals(expr, mode='all')
        indexeds = [i for i in terminals if i.is_Indexed]
        indexeds += flatten([retrieve_indexed(i) for i in e.indices] for e in indexeds)

        # Enforce deterministic dimension ordering...
        dims = OrderedDict()
        for e in terminals:
            if isinstance(e, Dimension):
                dims[(e,)] = e
            elif e.is_Indexed:
                d = []
                for a in e.indices:
                    found = [i for i in a.free_symbols if isinstance(i, Dimension)]
                    d.extend([i for i in found if i not in d])
                dims[tuple(d)] = e
        # ... giving higher priority to TimeFunction objects; time always go first
        dims = sorted(list(dims),
                      key=lambda i: not (isinstance(dims[i], Dimension) or
                                         dims[i].base.function.is_TimeFunction))
        stencil = Stencil([(i, set()) for i in partial_order(dims)])

        # Determine the points accessed along each dimension
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
                        off += [i]
                if d is not None:
                    stencil[d].update(off)

        return stencil

    @classmethod
    def union(cls, *dicts):
        """
        Compute the union of an iterable of :class:`Stencil` objects.
        """
        output = Stencil()
        for i in dicts:
            for k, v in i.items():
                output[k] |= v
        return output

    @property
    def frozen(self):
        return Stencil([(k, frozenset(v)) for k, v in self.items()])

    @property
    def empty(self):
        return all(len(i) == 0 for i in self.values())

    @property
    def dimensions(self):
        return list(self.keys())

    @property
    def entries(self):
        return tuple(StencilEntry(k, frozenset(v)) for k, v in self.items())

    @property
    def diameter(self):
        return {k: abs(max(v) - min(v)) for k, v in self.items()}

    def null(self):
        """
        Return the null Stencil of ``self``.

        Examples:

        self = {i: {-1, 0, 1}, j: {-2, -1, 0, 1, 2}}
        self.null() >> {i: {0}, j: {0}}
        """
        return Stencil([(i, set([0])) for i in self.dimensions])

    def section(self, d):
        """
        Return a view of the Stencil in which the Dimensions in ``d`` have been
        dropped.
        """
        output = Stencil()
        for k, v in self.items():
            if k not in d:
                output[k] = set(v)
        return output

    def subtract(self, o):
        """
        Compute the set difference of each Dimension in self with the corresponding
        Dimension in ``o``.
        """
        output = Stencil()
        for k, v in self.items():
            output[k] = set(v)
            if k in o:
                output[k] -= o[k]
        return output

    def add(self, o):
        """
        Compute the set union of each Dimension in self with the corresponding
        Dimension in ``o``.
        """
        output = Stencil()
        for k, v in self.items():
            output[k] = set(v)
            if k in o:
                output[k] |= o[k]
        return output

    def rshift(self, m):
        """
        Right-shift the Dimensions ``d`` of ``self`` appearing in the mapper ``m``
        by the constant quantity ``m[d]``.
        """
        return Stencil([(k, set([i - m.get(k, 0) for i in v])) for k, v in self.items()])

    def split(self, ds=None):
        """
        Split ``self`` into two Stencils, one with the negative axis, and one
        with the positive axis. If ``ds`` is provided, the split occurs only
        along the Dimensions listed in ``ds``.
        """
        ds = ds or self.dimensions
        negative, positive = Stencil(), Stencil()
        for k, v in self.items():
            if k in ds:
                negative[k] = {i for i in v if i < 0}
                positive[k] = {i for i in v if i > 0}
        return negative, positive

    def anti(self, o):
        """
        Compute the anti-Stencil of ``self`` constrained by ``o``.

        Examples:

        Assuming one single dimension (omitted for brevity)

        self = {-3, -2, -1, 0, 1, 2, 3}
        o = {-3, -2, -1, 0, 1, 2, 3}
        self.anti(o) >> {}

        self = {-3, -2, -1, 0, 1, 2, 3}
        o = {-2, -1, 0, 1}
        self.anti(o) >> {-1, 0, 1, 2}

        self = {-1, 0, 1}
        o = {-2, -1, 0, 1, 2}
        self.anti(o) >> {-1, 0, 1}
        """
        if any(not o[i].issuperset(self[i]) for i in o.dimensions if i in self):
            raise StencilOperationError

        diff = o.subtract(self)
        n, p = diff.split()
        n = n.rshift({i: min(self[i]) for i in self})
        p = p.rshift({i: max(self[i]) for i in self})
        union = Stencil.union(*[n, o.null(), p])

        return union

    def get(self, k, v=None):
        obj = super(Stencil, self).get(k, v)
        return frozenset([0]) if obj is None else obj

    def prefix(self, o):
        """
        Return the common prefix of ``self`` and ``o`` as a new Stencil.
        """
        output = Stencil()
        for (k1, v1), (k2, v2) in zip(self.items(), o.items()):
            if k1 == k2 and v1 == v2:
                output[k1] = set(v1)
            else:
                break
        return output

    def copy(self):
        """
        Return a deep copy of the Stencil.
        """
        return Stencil(self.entries)

    def replace(self, mapper):
        """
        Return a new Stencil in which a key ``k`` (dimension) appearing in
        ``mapper``  is replaced by ``mapper[k]``. The original order is therefore
        unchangend, but a new dictionary is produced with potentially different
        keys.
        """
        return Stencil([(k if k not in mapper else mapper[k], set(v))
                        for k, v in self.items()])

    def __eq__(self, other):
        return self.entries == other.entries

    def __ne__(self, other):
        return not self.__eq__(other)

    def __setitem__(self, key, val):
        entry = StencilEntry(key, val)  # Type checking
        super(Stencil, self).__setitem__(entry.dim, entry.ofs)


def retrieve_offsets(stencils):
    """
    Return a mapper from :class:`Dimension`s to the min/max integer offsets
    within ``stencils``.
    """
    offs = Stencil.union(*stencils)
    mapper = {d: v for d, v in offs.diameter.items()}
    mapper.update({d.parent: v for d, v in mapper.items() if d.is_Stepping})
    return mapper


StencilEntry = namedtuple('StencilEntry', 'dim ofs')
