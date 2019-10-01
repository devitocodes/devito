from devito import Eq, TimeFunction
from devito.ir.equations import ClusterizedEq
from devito.ir.iet import (Call, Expression, find_affine_trees,
                           List, retrieve_iteration_tree)
from devito.ir.iet.visitors import FindSymbols, Transformer
from devito.logger import warning
from devito.operator import Operator
from devito.symbolics import Literal
from devito.tools import filter_sorted

from devito.ops import ops_configuration
from devito.ops.transformer import create_ops_dat, create_ops_fetch, opsit
from devito.ops.types import OpsBlock
from devito.ops.utils import namespace

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    _default_headers = Operator._default_headers + ['#define restrict __restrict']

    def __init__(self, *args, **kwargs):
        self._ops_kernels = []
        super(OperatorOPS, self).__init__(*args, **kwargs)
        self._compiler = ops_configuration['compiler'].copy()

    def _specialize_iet(self, iet, **kwargs):
        warning("The OPS backend is still work-in-progress")

        # If there is no iteration tree, then there is no loop to be optimized using OPS.
        iteration_tree = retrieve_iteration_tree(iet, mode='normal')
        if not len(iteration_tree):
            return iet
        time_upper_bound = iteration_tree[0].dimensions[TimeFunction._time_position]\
            .extreme_max

        ops_init = Call(namespace['ops_init'], [0, 0, 2])
        ops_partition = Call(namespace['ops_partition'], Literal('""'))
        ops_exit = Call(namespace['ops_exit'])

        # Extract all symbols that need to be converted to ops_dat
        dims = []
        to_dat = set()
        for section, trees in find_affine_trees(iet).items():
            dims.append(len(trees[0].dimensions))
            symbols = set(FindSymbols('symbolics').visit(trees[0].root))
            symbols -= set(FindSymbols('defines').visit(trees[0].root))
            to_dat |= symbols

        # Create the OPS block for this problem
        ops_block = OpsBlock('block')
        ops_block_init = Expression(ClusterizedEq(Eq(
            ops_block,
            namespace['ops_decl_block'](
                dims[0],
                Literal('"block"')
            )
        )))

        # To ensure deterministic code generation we order the datasets to
        # be generated (since a set is an unordered collection)
        to_dat = filter_sorted(to_dat)

        name_to_ops_dat = {}
        pre_time_loop = []
        after_time_loop = []
        for f in to_dat:
            if f.is_Constant:
                continue

            pre_time_loop.extend(create_ops_dat(f, name_to_ops_dat, ops_block))
            # To return the result to Devito, it is necessary to copy the data
            # from the dat object back to the CPU memory.
            after_time_loop.extend(create_ops_fetch(f, name_to_ops_dat, time_upper_bound))

        # Generate ops kernels for each offloadable iteration tree
        mapper = {}
        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            pre_loop, ops_kernel, ops_par_loop_call = opsit(
                trees, n, name_to_ops_dat, ops_block, dims[0]
            )

            pre_time_loop.extend(pre_loop)
            self._ops_kernels.append(ops_kernel)
            mapper[trees[0].root] = ops_par_loop_call
            mapper.update({i.root: mapper.get(i.root) for i in trees})  # Drop trees

        iet = Transformer(mapper).visit(iet)

        assert (d == dims[0] for d in dims), \
            "The OPS backend currently assumes that all kernels \
            have the same number of dimensions"

        self._headers.append(namespace['ops_define_dimension'](dims[0]))
        self._includes.extend(['stdio.h', 'ops_seq.h'])

        body = [ops_init, ops_block_init, *pre_time_loop,
                ops_partition, iet, *after_time_loop, ops_exit]

        return List(body=body)

    @property
    def hcode(self):
        return ''.join(str(kernel) for kernel in self._ops_kernels)

    def _compile(self):
        self._includes.append('%s.h' % self._soname)
        if self._lib is None:
            self._compiler.jit_compile(self._soname, str(self.ccode), str(self.hcode))
