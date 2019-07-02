from devito.logger import warning
from devito.operator import Operator

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    _default_includes = Operator._default_includes + ['stdio.h']

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        return iet
