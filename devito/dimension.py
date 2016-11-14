from sympy import Symbol

__all__ = ['Dimension', 'x', 'y', 'z', 't', 'p']


class Dimension(Symbol):
    """Index object that represents iteration spaces"""

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.size = kwargs.get('size', None)
        newobj._count = 0
        return newobj

    def __str__(self):
        return self.name

    def get_varname(self):
        """Generates a new variables name based on an internal counter"""
        name = "%s%d" % (self.name, self._count)
        self._count += 1
        return name


# Set of default dimensions for space and time
x = Dimension('x')
y = Dimension('y')
z = Dimension('z')
t = Dimension('t')
p = Dimension('p')
