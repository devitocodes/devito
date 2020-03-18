from conftest import skipif
from devito import Grid, TimeFunction, Eq, Operator, switchconfig
from devito.ir.iet import retrieve_iteration_tree

pytestmark = skipif(['yask', 'ops'])


class TestCodeGeneration(object):

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'acc parallel loop collapse(3)'
        assert op.body[1].header[0].value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].footer[0].contents[0].value ==\
            ('acc exit data copyout(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].footer[0].contents[1].value ==\
            ('acc exit data delete(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')


class TestOperator(object):

    from test_gpu_openmp import TestOperator as TestGPUOpenMPOperator

    @switchconfig(platform='nvidiaX', language='openacc')
    def test_op_apply(self):
        self.TestGPUOpenMPOperator().test_op_apply()

    @switchconfig(platform='nvidiaX', language='openacc')
    def test_iso_ac(self):
        self.TestGPUOpenMPOperator().test_iso_ac()
