from devito.ir.iet import Call, List, find_affine_trees
from devito.logger import warning
from devito.operator import Operator
from devito.symbolics import Literal

from devito.ops.utils import namespace

__all__ = ['OperatorOPS']


class OperatorOPS(Operator):

    """
    A special Operator generating and executing OPS code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _specialize_iet(self, iet, **kwargs):

        warning("The OPS backend is still work-in-progress")

        ops_init = Call(namespace['ops_init'], [0, 0, 2])
        ops_partition = Call(namespace['ops_partition'], Literal('""'))
        ops_exit = Call(namespace['ops_exit'])

        dims = []
        for n, (section, trees) in enumerate(find_affine_trees(iet).items()):
            dims.append(len(trees[n].dimensions))

        assert (d == dims[0] for d in dims), \
            "The OPS backend currently assumes that all kernels \
            have the same number of dimensions"

        self._headers.append(namespace['ops-define-dimension'](dims[0]))
        self._includes.append('stdio.h')

        body = [ops_init, ops_partition, iet, ops_exit]

        return List(body=body)
