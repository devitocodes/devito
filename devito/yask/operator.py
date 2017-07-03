from __future__ import absolute_import

from devito.operator import OperatorRunnable

__all__ = ['OperatorYask']


class OperatorYask(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through YASK.
    """

    def __init__(self, expressions, **kwargs):
        kwargs['dle'] = 'basic'
        super(OperatorYask, self).__init__(expressions, **kwargs)

    def _specialize(self, nodes, elemental_functions):
        return nodes, elemental_functions

    def _autotune(self, arguments):
        """No-op, as YASK does its own auto-tuning."""
        return arguments
