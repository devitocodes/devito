from devito.ir.iet import List
from devito.logger import warning
from devito.operator import Operator

from devito.ops.types import String
from devito.ops.utils import namespace
from devito.ops.nodes import Call

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        ops_partition = Call(namespace['ops_partition'], String(''))

        body = [ops_partition, iet]

        return List(body=body)
