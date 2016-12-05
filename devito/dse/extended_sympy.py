"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr
from sympy.core.basic import _aresame


class UnevaluatedExpr(Expr):

    """
    Use :class:`UnevaluatedExpr` in place of :class:`sympy.Expr` to prevent
    xreplace from unpicking factorizations.
    """

    def xreplace(self, rule):
        if self in rule:
            return rule[self]
        elif rule:
            args = []
            for a in self.args:
                try:
                    args.append(a.xreplace(rule))
                except AttributeError:
                    args.append(a)
            args = tuple(args)
            if not _aresame(args, self.args):
                return self.func(*args, evaluate=False)
        return self


class Mul(sympy.Mul, UnevaluatedExpr):
    pass


class Add(sympy.Add, UnevaluatedExpr):
    pass
