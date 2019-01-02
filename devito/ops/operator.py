from devito.operator import Operator

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def _specialize_iet(self, iet, **kwargs):
        raise NotImplementedError
