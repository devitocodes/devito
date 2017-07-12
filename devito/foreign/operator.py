from __future__ import absolute_import

import devito.operator as operator

__all__ = ['Operator']


class Operator(operator.Operator):
    """
    A special :class:`Operator` for use outside of Python.
    """

    def arguments(self, *args, **kwargs):
        arguments, _ = super(Operator, self).arguments(*args, **kwargs)
        return arguments.items()
