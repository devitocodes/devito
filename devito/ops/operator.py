from devito.ir.iet import Call, List
from devito.logger import warning
from devito.symbolics import Literal
from devito.operator import Operator

from devito.ops.utils import namespace

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        ops_partition = Call(namespace['ops_partition'], Literal('""'))

        body = [ops_partition, iet]

        return List(body=body)
