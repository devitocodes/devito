from sympy import Function

from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees
from devito.ir.iet.nodes import Expression, Iteration, Call
from devito.ir.iet.visitors import FindNodes
from devito.symbolics.extended_sympy import ListInitializer

from devito.ops.utils import namespace
from devito.ops.types import OPSDeclObject


__all__ = ['Operator']

"""
We will start considering that each `Operator` will have 
only one `offloadable_tree`, only one `Grid`, and only one `Expression`.
"""
class Operator(OperatorRunnable):

    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        ops_data = []
        for (section, tree) in (find_offloadable_trees(iet).items()):
            node = tree[0].root
            expressions = [e.expr for e in FindNodes(Expression).visit(node)]
            iterations = [i for i in FindNodes(Iteration).visit(node)]

            """
            First things first.
            """
            ops_init_Object = Call(name=namespace['call-ops_init'], 
                                    params=(0, 'NULL', 1))

            """
            We need to create an `OPS grid`. 
            For a (x,y) spatial domain, that'd be something like: 
            "ops_block grid = ops_decl_block(2, "grid");"
            """
            ops_grid_Object = OPSDeclObject(dtype = namespace['type-ops_block'],
                                            name = namespace['name-ops_grid'], 
                                            value = Function(namespace['call-ops_block'])
                                                    (expressions[0].grid.dim,
                                                     namespace['name-ops_grid']))

            limits = []
            for i in iterations:
                limits.append(str(i.limits[1]))

            ops_size_Object = ListInitializer(limits)
            ops_base_Object = ListInitializer(['0','0'])
            ops_negBound_Object = ListInitializer(['-1','-1'])
            ops_posBound_Object = ListInitializer(['1','1'])

            ops_dat_Object = OPSDeclObject(dtype = namespace['type-ops_dat'],
                                            name = namespace['name-ops_dat'](''), 
                                            value = Function(namespace['call-ops_dat'])
                                                    (ops_grid_Object,
                                                     1,
                                                     ops_size_Object,
                                                     ops_base_Object,
                                                     ops_negBound_Object,
                                                     ops_posBound_Object,
                                                     self.input[0],
                                                     'double',
                                                     namespace['name-ops_dat']('')))


            """
            Last but not least.
            """
            ops_exit_Object = Call(name=namespace['call-ops_exit'], 
                                   params=None)
                
                
            ops_data.append(ops_init_Object)
            ops_data.append(ops_grid_Object)
            ops_data.append(ops_dat_Object)
            ops_data.append(ops_exit_Object)

        # Temporary. 
        # for od in ops_data:
        #     print(od)

        # raise NotImplementedError


