from devito.ir.iet import Call, List
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

        ops_init = Call("ops_init", [0, 0, 1])
        ops_exit = Call("ops_exit")

        return List(body=[ops_init, iet, ops_exit])
