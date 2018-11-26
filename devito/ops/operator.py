from devito.operator import OperatorRunnable

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        raise NotImplementedError
