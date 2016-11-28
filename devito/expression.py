from __future__ import absolute_import

from collections import defaultdict
from operator import attrgetter

import cgen
from sympy import Eq, IndexedBase, preorder_traversal

from devito.codeprinter import ccode
from devito.dimension import Dimension
from devito.interfaces import SymbolicData
from devito.symbolics import dse_indexify, terminals
from devito.tools import filter_ordered

__all__ = ['Expression']


class Expression(object):
    """Class encpasulating a single stencil expressions"""

    def __init__(self, stencil):
        assert isinstance(stencil, Eq)
        self.stencil = stencil

        self.dimensions = []
        self.functions = []
        # Traverse stencil to determine meta information
        for e in preorder_traversal(self.stencil):
            if isinstance(e, SymbolicData):
                self.dimensions += list(e.indices)
                self.functions += [e]
            if isinstance(e, IndexedBase):
                self.dimensions += list(e.function.indices)
                self.functions += [e.function]
        # Filter collected dimensions and functions
        self.dimensions = filter_ordered(self.dimensions)
        self.functions = filter_ordered(self.functions)

    def __repr__(self):
        return "Expression<%s = %s>" % (self.stencil.lhs, self.stencil.rhs)

    def substitute(self, substitutions):
        """Apply substitutions to the expression stencil

        :param substitutions: Dict containing the substitutions to apply to
                              the stored loop stencils.
        """
        self.stencil = self.stencil.xreplace(substitutions)

    @property
    def ccode(self):
        return cgen.Assign(ccode(self.stencil.lhs), ccode(self.stencil.rhs))

    @property
    def signature(self):
        """List of data objects used by the expression

        :returns: List of unique data objects required by the expression
        """
        return filter_ordered([f for f in self.functions],
                              key=attrgetter('name'))

    def indexify(self):
        """Convert stencil expression to "indexed" format"""
        self.stencil = dse_indexify(self.stencil)

    @property
    def index_offsets(self):
        """Collect all non-zero offsets used with each index in a map

        Note: This assumes we have indexified the stencil expression."""
        offsets = defaultdict(list)
        for e in terminals(self.stencil):
            for a in e.indices:
                d = None
                off = []
                for idx in a.args:
                    if isinstance(idx, Dimension):
                        d = idx
                    else:
                        off += [idx]
                if d is not None:
                    offsets[d] += off
        return offsets
