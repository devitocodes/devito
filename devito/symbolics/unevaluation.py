import sympy

__all__ = ['Add', 'Mul', 'Pow']


class UnevaluableMixin(object):

    def __new__(cls, *args, evaluate=None, **kwargs):
        return cls.__base__.__new__(cls, *args, evaluate=False, **kwargs)


class Add(sympy.Add, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class Mul(sympy.Mul, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__


class Pow(sympy.Pow, UnevaluableMixin):
    __new__ = UnevaluableMixin.__new__
