"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr, Float
from sympy.core.basic import _aresame
from sympy.functions.elementary.trigonometric import TrigonometricFunction

__all__ = ['UnevaluatedExpr', 'Eq', 'Mul', 'Add', 'NaturalMod', 'taylor_sin',
           'taylor_cos', 'bhaskara_sin', 'bhaskara_cos']


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


class Eq(sympy.Eq, UnevaluatedExpr):
    pass


class Mul(sympy.Mul, UnevaluatedExpr):
    pass


class Add(sympy.Add, UnevaluatedExpr):
    pass


class NaturalMod(sympy.Mod):
    pass


class taylor_sin(TrigonometricFunction):

    """
    Approximation of the sine function using a Taylor polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return eval_taylor_sin(arg)


class taylor_cos(TrigonometricFunction):

    """
    Approximation of the cosine function using a Taylor polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 1.0 if arg == 0.0 else eval_taylor_cos(arg + 1.5708)


class bhaskara_sin(TrigonometricFunction):

    """
    Approximation of the sine function using a Bhaskara polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return eval_bhaskara_sin(arg)


class bhaskara_cos(TrigonometricFunction):

    """
    Approximation of the cosine function using a Bhaskara polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 1.0 if arg == 0.0 else eval_bhaskara_sin(arg + 1.5708)


# Utils

def eval_bhaskara_sin(expr):
    return 16.0*expr*(3.1416-abs(expr))/(49.3483-4.0*abs(expr)*(3.1416-abs(expr)))


def eval_taylor_sin(expr):
    v = expr + Mul(-1/6.0,
                   Mul(expr, expr, expr, evaluate=False),
                   1.0 + Mul(Mul(expr, expr, evaluate=False), -0.05, evaluate=False),
                   evaluate=False)
    try:
        Float(expr)
        return v.doit()
    except (TypeError, ValueError):
        return v


def eval_taylor_cos(expr):
    v = 1.0 + Mul(-0.5,
                  Mul(expr, expr, evaluate=False),
                  1.0 + Mul(expr, expr, -1/12.0, evaluate=False),
                  evaluate=False)
    try:
        Float(expr)
        return v.doit()
    except (TypeError, ValueError):
        return v
