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

    _default_includes = OperatorRunnable._default_includes + \
        ['ops_seq.h', 'ops_lib_cpp.h']

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        # Define ops_init call.
        ops_init_Object = Call(name=namespace['call-ops_init'],
                               params=[0, Macro('NULL'), 1])
        # Define ops_exit call.
        ops_exit_Object = Call(name=namespace['call-ops_exit'],
                               params=[])

        # Inserts both ops init and exit into the iet.
        iterationInitial = FindNodes(Iteration).visit(iet)[0]
        iet = Transformer({iterationInitial: List(
            body=[ops_init_Object, iterationInitial, ops_exit_Object])}).visit(iet)

        for _, trees in (find_offloadable_trees(iet).items()):
            node = trees[0].root
            iterations = [i for i in FindNodes(Iteration).visit(node)]

            # Generate OPS kernels
            kernels = opsit(trees)

            # Mark the kernels as calls.
            for index, kernel in enumerate(kernels):
                self._func_table[namespace['ops-kernel'](index)] = MetaCall(kernel, True)

            # Creates ops_par_loop
            # FIXME: this is just temporary, it will be replaced.
            ops_parLoop_Object = Call(name=namespace['call-ops_par_loop'],
                                      params=[2,
                                              Array(name='u',
                                                    dimensions=[Dimension(name='t0')],
                                                    dtype=float32)])

            # Replace the iteration for the ops_par_loop call.
            mapper = {iterations[0]: ops_parLoop_Object}
            iet = Transformer(mapper).visit(iet)

        return iet
