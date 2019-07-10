from devito.ir.iet import Call, List
from devito.logger import warning
from devito.operator import Operator
from devito.types.basic import FunctionPointer

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    _default_includes = Operator._default_includes + ['stdio.h']

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        ops_init = Call("ops_init", [0, 0, 2])
        ops_timing = Call("ops_timing_output", [FunctionPointer("stdout")])
        ops_exit = Call("ops_exit")

        return List(body=[ops_init, iet, ops_timing, ops_exit])
