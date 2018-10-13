from devito.operator import OperatorRunnable
from devito.ir.iet.utils import find_offloadable_trees

__all__ = ['Operator']


class Operator(OperatorRunnable):
    """
    A special :class:`OperatorCore` to JIT-compile and run operators through OPS.
    """

    def _specialize_iet(self, iet, **kwargs):
        print('\n\tTHIS IS OUR IET SO FAR:\n>>>>>')
        print(iet)
        print('<<<<< :)\n')
        for n, (section, trees) in enumerate(find_offloadable_trees(iet).items()):
            print('Number of offloadable trees within the above IET: {}\n'.format(n))
            print(trees[0].root)

        iet manipuation


        return new_iet