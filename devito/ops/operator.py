from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        for n, (section, trees) in enumerate(find_offloadable_trees(iet).items()):
            print(trees[0].root)
