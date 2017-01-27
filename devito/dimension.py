import cgen

import numpy as np
from sympy import Symbol


__all__ = ['Dimension', 'x', 'y', 'z', 't', 'p']


class Dimension(Symbol):
    """Index object that represents a problem dimension and thus
    defines a potential iteration space.

    :param size: Optional, size of the array dimension.
    :param buffered: Optional, boolean flag indicating whether to
                     buffer variables when iterating this dimension.
    """

    def __new__(cls, name, **kwargs):
        newobj = Symbol.__new__(cls, name)
        newobj.size = kwargs.get('size', None)
        newobj.buffered = kwargs.get('buffered', None)
        newobj._count = 0
        return newobj

    def __str__(self):
        return self.name

    def get_varname(self):
        """Generates a new variables name based on an internal counter"""
        name = "%s%d" % (self.name, self._count)
        self._count += 1
        return name

    @property
    def ccode(self):
        """C-level variable name of this dimension"""
        return "%s_size" % self.name if self.size is None else "%d" % self.size

    @property
    def decl(self):
        """Variable declaration for C-level kernel headers"""
        return cgen.Value("const int", self.ccode)

    @property
    def dtype(self):
        """The data type of the iteration variable"""
        return np.int32


# Set of default dimensions for space and time
x = Dimension('x')
y = Dimension('y')
z = Dimension('z')
t = Dimension('t')
p = Dimension('p')
