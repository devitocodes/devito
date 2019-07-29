from devito import Eq
from devito.ir.equations import ClusterizedEq
from devito.ir.iet import Call, List, Expression, find_affine_trees
from devito.ir.iet.visitors import FindSymbols
from devito.logger import warning
from devito.operator import Operator
from devito.symbolics import Literal

from devito.ops.transformer import create_ops_dat, opsit
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
        to_dat = set()
        for section, trees in find_affine_trees(iet).items():
            dims.append(len(trees[0].dimensions))
            symbols = set(FindSymbols('symbolics').visit(trees[0].root))
            symbols -= set(FindSymbols('defines').visit(trees[0].root))
            to_dat |= symbols

        name_to_ops_dat = {}
        pre_time_loop = []
        for f in to_dat:
            if f.is_Constant:
                continue

            pre_time_loop.extend(create_ops_dat(f, name_to_ops_dat, ops_block))

        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            pre_loop, ops_kernel = opsit(trees, n)

            pre_time_loop.extend(pre_loop)
            self._ops_kernels.append(ops_kernel)

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
