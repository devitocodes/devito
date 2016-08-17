from sympy import Symbol

__all__ = ['Dimension']


class Dimension(Symbol):
    """Index object that represents iteration spaces"""

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.size = kwargs.get('size', None)
        return newobj

    def __str__(self):
        return self.name
