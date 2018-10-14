from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees

from devito.ir.iet import FindNodes, Expression, Call, Element, List
from devito.tools import filter_ordered, flatten

from collections import OrderedDict
import cgen as c

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        # print(iet)
        grid_decl = build_grid_decl(iet)
        # print(grid_decl)

        iet = List(body=grid_decl + [iet])
        print(iet)


def build_grid_decl(iet):
    """
    First things first. We need to declare an `OPS grid` at top of all. 
    For a (x,y) spatial domain, that'd be something like: 
    "ops_block grid = ops_decl_block(2, "grid");"

    We will start considering that each Devito's `Operator` will have 
    only one `offloadable_tree`, only one `Grid`, and only one `Expression`.
    """

    for (section, tree) in (find_offloadable_trees(iet).items()):
        node = tree[0].root
        # print(node)
        expressions = [i.expr for i in FindNodes(Expression).visit(node)]
        grid_decls = []        
        for e in expressions:
            # print(e)
            grid_dim = e.grid.dim
            # print(grid_dim)
            grid_name = namespace['name-ops_grid']
            grid_type = namespace['type-ops_block']
            grid_call = namespace['call-ops_block']

            grid_decl = LocalExpression(Eq(Object(...), sympy.Function(...))

            grid_decl = Element(c.Statement("%(gt)s %(gn)s = %(gc)s (%(gd)s, \"%(gns)s\")" % 
                      {'gt': grid_type,
                       'gn': grid_name,
                       'gc': grid_call,
                       'gd': grid_dim,
                       'gns': '{}'.format(grid_name)}))

            grid_decls.append(grid_decl)

        return grid_decls


namespace = OrderedDict()
namespace['name-ops_grid'] = 'grid'
namespace['type-ops_block'] = 'ops_block'
namespace['call-ops_block'] = 'ops_decl_block'



