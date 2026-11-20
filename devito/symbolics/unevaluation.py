import sympy
from sympy import sympify

__all__ = ['Add', 'Mod', 'Mul', 'OrderedAdd', 'OrderedMul', 'Pow', 'ceiling', 'floor']


class UnevaluableMixin:

    def __new__(cls, *args, evaluate=None, **kwargs):
        return cls.__base__.__new__(cls, *args, evaluate=False, **kwargs)


class Add(sympy.Add, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class Mod(sympy.Mod, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class Mul(sympy.Mul, UnevaluableMixin):
    def __new__(cls, *args, evaluate=None, **kwargs):
        # A Mul will always be zero if any of its arguments are zero
        if any(sympify(arg).is_zero for arg in args):
            return sympy.S.Zero

        return super().__new__(cls, *args, evaluate=False, **kwargs)


class Pow(sympy.Pow, UnevaluableMixin):

    def __new__(cls, base, exp, evaluate=None, **kwargs):
        if base == 1:
            # Otherwise we might get trapped within vicious recursion inside
            # SymPy each time it tries to perform a simplification via
            # `as_numer_denom`, since the `denom` turns into a Pow itself
            # such as `1**c`
            return sympy.S.One
        else:
            return cls.__base__.__new__(cls, base, exp, evaluate=False, **kwargs)


class ceiling(sympy.ceiling, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class floor(sympy.floor, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class OrderedAdd(Add):

    """An addition with an explicit operand order for printing."""

    def __new__(cls, *args, **kwargs):
        return cls._from_args(args)

    def as_ordered_terms(self, order=None, data=False):
        if data:
            return super().as_ordered_terms(order=order, data=data)
        return list(self.args)


class OrderedMul(Mul):

    """A product with an explicit factor order for printing."""

    def __new__(cls, *args, **kwargs):
        return cls._from_args(args)

    def as_ordered_factors(self, order=None):
        return list(self.args)
