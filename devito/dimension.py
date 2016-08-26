from sympy import Symbol

__all__ = ['Dimension', 'x', 'y', 'z', 't']


class Dimension(Symbol):
    """Index object that represents iteration spaces"""

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.size = kwargs.get('size', None)
        return newobj

    def __str__(self):
        return self.name


# Set of default dimensions for space and time
x = Dimension('x')
y = Dimension('y')
z = Dimension('z')
t = Dimension('t')
