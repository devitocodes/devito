"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr, Float, Indexed, S
from sympy.core.basic import _aresame, Basic
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
    except TypeError, ValueError:
        return v


def eval_taylor_cos(expr):
    expr_square = Mul(expr, expr, evaluate=False)
    b =  1.0-.5*expr_square*(1.0-expr_square/12.0)
    v = 1.0 + Mul(-0.5,
                  Mul(expr, expr, evaluate=False),
                  1.0 + Mul(expr, expr, -1/12.0, evaluate=False),
                  evaluate=False)
    try:
        Float(expr)
        return b, v.doit()
    except TypeError, ValueError:
        return v


def unevaluate_arithmetic(expr):
    """
    Reconstruct ``expr`` turning all :class:`sympy.Mul` and :class:`sympy.Add`
    into, respectively, :class:`devito.Mul` and :class:`devito.Add`.
    """
    if expr.is_Float:
        return expr.func(*expr.atoms())
    elif isinstance(expr, Indexed):
        return expr.func(*expr.args)
    elif expr.is_Symbol:
        return expr.func(expr.name)
    elif expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
        return expr.func()
    elif expr.is_Atom:
        return expr.func(*expr.atoms())
    elif expr.is_Add:
        rebuilt_args = [unevaluate_arithmetic(e) for e in expr.args]
        return Add(*rebuilt_args, evaluate=False)
    elif expr.is_Mul:
        rebuilt_args = [unevaluate_arithmetic(e) for e in expr.args]
        return Mul(*rebuilt_args, evaluate=False)
    else:
        return expr.func(*[unevaluate_arithmetic(e) for e in expr.args])
