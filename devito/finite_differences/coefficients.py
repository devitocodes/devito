import sympy

from cached_property import cached_property

from devito.tools import filter_ordered, flatten

__all__ = ['Coefficients']


class Coefficients(object):
    """
    Devito class for users to define custom finite difference weights.
    """

    def __init__(self, *args, **kwargs):
            
        Coefficients.data = args
        
        print(Coefficients.data)
