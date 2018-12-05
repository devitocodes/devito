from devito.operator import OperatorRunnable
from devito.logger import warning

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
        A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        return iet
