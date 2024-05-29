import sympy

__all__ = ['Add', 'Mul', 'Pow']


class UnevaluableMixin:

    def __new__(cls, *args, evaluate=None, **kwargs):
        return cls.__base__.__new__(cls, *args, evaluate=False, **kwargs)


class Add(sympy.Add, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class Mul(sympy.Mul, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


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
