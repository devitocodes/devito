from sympy import Eq, preorder_traversal, IndexedBase
from devito.codeprinter import ccode
import cgen as c

__all__ = ['Expression']


class Expression(object):
    """Class encpasulating a single stencil expressions"""

    def __init__(self, stencil):
        assert isinstance(stencil, Eq)
        self.stencil = stencil

        # Traverse stencil to determine dimensions
        self.dimensions = []
        for e in preorder_traversal(self.stencil.lhs):
            if isinstance(e, IndexedBase):
                self.dimensions += list(e.function.indices)
        # Filter collected dimensions while preserving order
        seen = set()
        self.dimensions = [d for d in self.dimensions
                           if not (d in seen or seen.add(d))]

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
        return c.Assign(ccode(self.stencil.lhs), ccode(self.stencil.rhs))
