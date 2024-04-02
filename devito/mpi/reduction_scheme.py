import sympy

from devito.tools import Reconstructable

__all__ = ['DistributedReduction']


class DistributedReduction(sympy.Function, Reconstructable):

    """
    A SymPy object representing a distributed Reduction.
    """

    __rargs__ = ('var',)
    __rkwargs__ = ('op', 'grid')

    def __new__(cls, var, op=None, grid=None, **kwargs):
        obj = sympy.Function.__new__(cls, var, **kwargs)
        obj.op = op
        obj.grid = grid
        return obj

    def __repr__(self):
        return "DistributedReduction(%s,%s)" % (self.var, self.op)

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    def _hashable_content(self):
        return (self.op, self.grid)

    def __eq__(self, other):
        return (isinstance(other, DistributedReduction) and
                self.var == other.var and
                self.op == other.op and
                self.grid == other.grid)

    func = Reconstructable._rebuild

    @property
    def var(self):
        return self.args[0]
