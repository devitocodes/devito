"""
Extended SymPy hierarchy.
"""

import sympy
from sympy import Expr, Float
from sympy.core.basic import _aresame
from sympy.functions.elementary.trigonometric import TrigonometricFunction

__all__ = ['FrozenExpr', 'Eq', 'Mul', 'Add', 'FunctionFromPointer',
           'taylor_sin', 'taylor_cos', 'bhaskara_sin', 'bhaskara_cos']


class FrozenExpr(Expr):

    """
    Use :class:`FrozenExpr` in place of :class:`sympy.Expr` to make sure than
    an expression is no longer transformable; that is, standard manipulations
    such as xreplace, collect, expand, ... have no effect, thus building a
    new expression identical to self.

    :Notes:

    At the moment, only xreplace is overridded (to prevent unpicking factorizations)
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


class Eq(sympy.Eq, FrozenExpr):

    """
    A customized version of :class:`sympy.Eq` which suppresses
    evaluation.
    """

    def __new__(cls, *args, **kwargs):
        kwargs['evaluate'] = False
        obj = sympy.Eq.__new__(cls, *args, **kwargs)
        return obj


class Mul(sympy.Mul, FrozenExpr):
    pass


class Add(sympy.Add, FrozenExpr):
    pass


class FunctionFromPointer(sympy.Symbol):

    def __new__(cls, function, pointer, params=None):
        obj = sympy.Symbol.__new__(cls, '%s->%s' % (pointer, function))
        obj.function = function
        obj.pointer = pointer
        obj.params = params or ()
        return obj

    def __str__(self):
        return "%s->%s(%s)" % (self.pointer, self.function,
                               ', '.join([str(i) for i in self.params]))

    __repr__ = __str__


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
        return 1.0 if arg == 0.0 else eval_taylor_cos(arg)


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
