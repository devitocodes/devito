from numpy import float32

from devito import Dimension
from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees
from devito.ir.iet.nodes import Iteration, Call
from devito.ir.iet.visitors import FindNodes
from devito.ir.iet import Transformer, List, MetaCall
from devito.types import Array
from devito.symbolics import Macro
from devito.logger import warning

from devito.ops.utils import namespace
from devito.ops.transformer import opsit

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
        A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        for _, trees in (find_offloadable_trees(iet).items()):

            # Generate OPS kernels
            kernels = opsit(trees)

            # Mark the kernels as calls.
            for index, kernel in enumerate(kernels):
                self._func_table[namespace['ops-kernel'](index)] = MetaCall(kernel, True)

        return iet
