from devito import Eq
from devito.ir.equations import ClusterizedEq
from devito.ir.iet import Call, List, Expression, find_affine_trees, Transformer
from devito.logger import warning
from devito.operator import Operator
from devito.symbolics import Literal

from devito.ops.transformer import opsit
from devito.ops.types import OpsBlock
from devito.ops.utils import namespace

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def __init__(self, *args, **kwargs):
        self._ops_kernels = []
        super().__init__(*args, **kwargs)

    def _specialize_iet(self, iet, **kwargs):
        warning("The OPS backend is still work-in-progress")

        ops_init = Call(namespace['ops_init'], [0, 0, 2])
        ops_partition = Call(namespace['ops_partition'], Literal('""'))
        ops_exit = Call(namespace['ops_exit'])

        ops_block = OpsBlock('block')

        dims = []
        for section, trees in find_affine_trees(iet).items():
            dims.append(len(trees[0].dimensions))

        pre_time_loop = []
        mapper = {}
        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            pre_loop, ops_call_par_loop, ops_kernel = opsit(trees, n, ops_block)
            pre_time_loop.extend(pre_loop)
            self._ops_kernels.append(ops_kernel)
            mapper[trees[0].root] = ops_call_par_loop
            mapper.update({i.root: mapper.get(i.root) for i in trees})  # Drop trees

        iet = Transformer(mapper).visit(iet)

        assert (d == dims[0] for d in dims), \
            "The OPS backend currently assumes that all kernels \
            have the same number of dimensions"

        ops_block_init = Expression(ClusterizedEq(Eq(
            ops_block,
            namespace['ops_decl_block'](
                dims[0],
                Literal('"block"')
            )
        )))

        self._headers.append(namespace['ops_define_dimension'](dims[0]))
        self._includes.append('stdio.h')

        body = [ops_init, ops_block_init, *pre_time_loop, ops_partition, iet, ops_exit]

        return List(body=body)
