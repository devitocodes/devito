from devito import Eq, Grid, TimeFunction, Operator
from devito.archinfo import get_gpu_info
from devito.ir.iet import retrieve_iteration_tree

from conftest import skipif

pytestmark = skipif(['nodevice'], whole_module=True)


class TestGPUInfo(object):

    def test_get_gpu_info(self):
        info = get_gpu_info()
        assert 'tesla' in info['architecture'].lower()


class TestCodeGeneration(object):

    def test_maxpar_option(self):
        """
        Make sure the `cire-maxpar` option is set to True by default.
        """
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=2)

        eq = Eq(u.forward, u.dy.dy)

        op = Operator(eq)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][0] is trees[1][0]
        assert trees[0][1] is not trees[1][1]
