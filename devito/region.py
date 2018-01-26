__all__ = ['DOMAIN', 'INTERIOR']


class Region(object):
    """
    A region of the computational domain over which a :class:`Function` is
    discretized.
    """

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return self._name

    def __eq__(self, other):
        return isinstance(other, Region) and self._name == other._name


DOMAIN = Region('DOMAIN')
"""
Represent the physical domain of the PDE; that is, domain = boundary + interior
"""

INTERIOR = Region('INTERIOR')
"""
Represent the physical interior domain of the PDE; that is, PDE boundaries are
not included.
"""
