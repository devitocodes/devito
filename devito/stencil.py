from collections import OrderedDict, namedtuple

from sympy import Eq

from devito.dse.search import retrieve_indexed
from devito.dimension import Dimension
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

            * A single SymPy equation, or
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
                        processed.append((j.dim, j.ofs))
                    elif isinstance(j, tuple) and len(j) == 2:
                        entry = StencilEntry(*j)  # Type checking
                        processed.append((entry.dim, entry.ofs))
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
        indexed = list(retrieve_indexed(expr.lhs))
        indexed += list(retrieve_indexed(expr.rhs))
        indexed += flatten([retrieve_indexed(i) for i in e.indices] for e in indexed)

        # Enforce deterministic ordering
        dims = []
        for e in indexed:
            d = []
            for a in e.indices:
                found = [idx for idx in a.free_symbols if isinstance(idx, Dimension)]
                d.extend([idx for idx in found if idx not in d])
            dims.append(tuple(d))
        stencil = Stencil([(d, set()) for d in partial_order(dims)])

        # Determine the points accessed along each dimension
        for e in indexed:
            for a in e.indices:
                if isinstance(a, Dimension):
                    stencil[a].update([0])
                d = None
                off = []
                for idx in a.args:
                    if isinstance(idx, Dimension):
                        d = idx
                    elif idx.is_integer:
                        off += [idx]
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
        return self.keys()

    @property
    def entries(self):
        processed = OrderedDict()
        for k, v in self.items():
            if k.is_Buffered and self.get(k.parent, v) == v:
                # Ignore a BufferedDimension if identical to its parent
                processed[k.parent] = v
            else:
                processed[k] = v
        return tuple(StencilEntry(k, v) for k, v in processed.items())

    def subtract(self, o):
        """
        Compute the set difference of each Dimension in self with the corresponding
        Dimension in ``o``.
        """
        output = Stencil()
        for k, v in self.items():
            output[k] = v
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
            output[k] = v
            if k in o:
                output[k] |= o[k]
        return output

    def get(self, k, v=None):
        obj = super(Stencil, self).get(k, v)
        return frozenset([0]) if obj is None else obj

    def __setitem__(self, key, val):
        entry = StencilEntry(key, val)  # Type checking
        super(Stencil, self).__setitem__(entry.dim, entry.ofs)


StencilEntry = namedtuple('StencilEntry', 'dim ofs')
