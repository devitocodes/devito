from sympy import Function

from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees
from devito.ir.iet import FindNodes, Expression

from devito.ops.utils import namespace

from devito.ops.types import OPSGridObject


__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        # print(iet)
        for (section, tree) in (find_offloadable_trees(iet).items()):
            node = tree[0].root
            # print(node)
            expressions = [i.expr for i in FindNodes(Expression).visit(node)]
            ops_data = []        
            for e in expressions:

                """
                First things first. We need to create an `OPS grid` at top of all. 
                For a (x,y) spatial domain, that'd be something like: 
                "ops_block grid = ops_decl_block(1, "grid");"

                We will start considering that each Devito's `Operator` will have 
                only one `offloadable_tree`, only one `Grid`, and only one `Expression`.
                """
                ops_data.append(OPSGridObject(namespace['name-ops_grid'], 
                                Function(namespace['call-ops_block'])
                                    (e.grid.dim, namespace['name-ops_grid'])))


        for od in ops_data:
            print(od)

        # raise NotImplementedError


