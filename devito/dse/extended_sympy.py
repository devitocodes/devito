"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr
from sympy.core.basic import _aresame
from sympy.functions.elementary.trigonometric import TrigonometricFunction


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


class taylor_sin(TrigonometricFunction):

    """
    Approximation of the sine function using a Taylor polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 0.0 if arg == 0.0 else eval_taylor_sin(arg)


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
        return 0.0 if arg == 0.0 else eval_bhaskara_sin(arg)


class bhaskara_cos(TrigonometricFunction):

    """
    Approximation of the cosine function using a Bhaskara polynomial.
    """

    @classmethod
    def eval(cls, arg):
        return 1.0 if arg == 0.0 else eval_bhaskara_sin(arg + 1.5708)


# Utils

def eval_bhaskara_sin(angle):
    return 16.0*angle*(3.1416-abs(angle))/(49.3483-4.0*abs(angle)*(3.1416-abs(angle)))


def eval_taylor_sin(angle):
    return angle-(angle*angle*angle/6.0*(1.0-angle*angle/20.0))


def eval_taylor_cos(angle):
    return 1.0-.5*angle*angle*(1.0-angle*angle/12.0)
