"""User API to specify equations."""

import sympy

__all__ = ['Eq', 'Inc', 'DOMAIN', 'INTERIOR']


class Eq(sympy.Eq):

    """
    A :class:`sympy.Eq` that accepts the additional keyword parameter ``region``.

    The ``region``, an object of type :class:`Region`, may be used to restrict
    the execution of the equation to a sub-domain.
    """

    is_Increment = False

    def __new__(cls, *args, **kwargs):
        kwargs['evaluate'] = False
        region = kwargs.pop('region', DOMAIN)
        obj = sympy.Eq.__new__(cls, *args, **kwargs)
        obj._region = region
        return obj

    def xreplace(self, rules):
        return self.func(self.lhs.xreplace(rules), self.rhs.xreplace(rules),
                         region=self._region)


class Inc(Eq):

    """
    A :class:`Eq` performing a linear increment.
    """

    is_Increment = True


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
