from __future__ import absolute_import

from collections import OrderedDict
import devito.operator as operator

__all__ = ['Operator']


class Operator(operator.Operator):
    """
    A special :class:`Operator` for use outside of Python.
    """

    def arguments(self, *args, **kwargs):
        arguments = super(Operator, self).arguments(*args, **kwargs)
        return arguments.items()

    def _default_args(self):
        defaults = OrderedDict()
        for p in self.parameters:
            if p.is_ScalarArgument:
                defaults[p.name] = p.value
            else:
                defaults[p.name] = None
        return defaults
