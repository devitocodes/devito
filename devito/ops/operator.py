from devito.logger import warning
from devito.ir.iet import iet_insert_casts, iet_insert_decls, find_affine_trees
from devito.ir.iet.nodes import MetaCall
from devito.operator import Operator
from devito.ops.nodes import OPSKernel
from devito.ops.transformer import opsit

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def _specialize_iet(self, iet, **kwargs):
        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            callable_kernel, par_loop_call_block = opsit(trees, n)

            self._func_table[callable_kernel.name] = MetaCall(callable_kernel, True)

        warning("The OPS backend is still work-in-progress")

        return iet

    def _finalize(self, iet, parameters):
        iet = iet_insert_decls(iet, parameters)
        iet = iet_insert_casts(iet, parameters)

        # Now do the same to each ElementalFunction
        for k, (root, local) in list(self._func_table.items()):
            if local:
                body = iet_insert_decls(root.body, root.parameters)
                if not isinstance(root, OPSKernel):
                    body = iet_insert_casts(body, root.parameters)
                self._func_table[k] = MetaCall(root._rebuild(body=body), True)

        return iet
