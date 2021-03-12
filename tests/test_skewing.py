import pytest
from functools import reduce
from operator import mul

from sympy import Add, cos, sin, sqrt  # noqa
import numpy as np

from devito.core.autotuning import options  # noqa
from devito import (NODE, Eq, Inc, Constant, Function, TimeFunction, SparseTimeFunction,  # noqa
                    Dimension, SubDimension, Grid, Operator, norm, grad, div, dimensions,
                    switchconfig, configuration, centered, first_derivative, solve,
                    transpose)
from devito.ir import (Expression, Iteration, FindNodes,
                       retrieve_iteration_tree, Call, Conditional)
from devito.tools import as_tuple


def get_blocksizes(op, opt, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if opt[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}


def _new_operator2(shape, time_order, blockshape=None, opt=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, opt=opt)

    blocksizes = get_blocksizes(op, opt, grid, blockshape)
    op(infield=infield, outfield=outfield, t=10, **blocksizes)

    return outfield, op


def _new_operator3(shape, blockshape0=None, blockshape1=None, opt=None):
    blockshape0 = as_tuple(blockshape0)
    blockshape1 = as_tuple(blockshape1)

    grid = Grid(shape=shape, extent=shape, dtype=np.float64)

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=(2, 2, 2))
    u.data[0, :] = np.linspace(-1, 1, reduce(mul, shape)).reshape(shape)

    # Derive the stencil according to devito conventions
    op = Operator(Eq(u.forward, 0.5 * u.laplace + u), opt=opt)

    blocksizes0 = get_blocksizes(op, opt, grid, blockshape0, 0)
    blocksizes1 = get_blocksizes(op, opt, grid, blockshape1, 1)
    op.apply(u=u, t=10, **blocksizes0, **blocksizes1)

    return u.data[1, :], op


class TestAutotuningWithSkewing(object):

    """
    This class contains tests mainly inherited from test_autotuning.py
    Aims to test interoperability for
    """
    @switchconfig(log_level='DEBUG')
    @pytest.mark.parametrize("shape,expected", [
        ((30, 30), 13),
        ((30, 30, 30), 17)
    ])
    def test_at_skewed_is_actually_working(self, shape, expected):
        """
        Check that autotuning and skewing interoperate,
        in both 2D and 3D operators.
        """
        grid = Grid(shape=shape)
        f = TimeFunction(name='f', grid=grid)

        eqn = Eq(f.forward, f + 1)
        op = Operator(eqn, opt=('blocking', 'skewing',
                                {'openmp': False, 'blockinner': True}))

        # Run with whatever `configuration` says (by default, basic+preemptive)
        op(time_M=0, autotune=True)
        assert op._state['autotuning'][-1]['runs'] == 4
        assert op._state['autotuning'][-1]['tpr'] == options['squeezer'] + 1

        # Now try `aggressive` autotuning
        configuration['autotuning'] = 'aggressive'
        op(time_M=0, autotune=True)
        assert op._state['autotuning'][-1]['runs'] == expected
        assert op._state['autotuning'][-1]['tpr'] == options['squeezer'] + 1
        configuration['autotuning'] = configuration._defaults['autotuning']

        # Try again, but using the Operator API directly
        op(time_M=0, autotune='aggressive')
        assert op._state['autotuning'][-1]['runs'] == expected
        assert op._state['autotuning'][-1]['tpr'] == options['squeezer'] + 1

        # Similar to above
        op(time_M=0, autotune=('aggressive', 'preemptive'))
        assert op._state['autotuning'][-1]['runs'] == expected
        assert op._state['autotuning'][-1]['tpr'] == options['squeezer'] + 1

    @switchconfig(profiling='advanced')
    def test_mode_runtime_forward_w_skewing(self):
        """Test autotuning in runtime mode."""
        grid = Grid(shape=(96, 96, 96))
        f = TimeFunction(name='f', grid=grid)

        op = Operator(Eq(f.forward, f + 1.), opt=('blocking', 'skewing',
                                                  {'openmp': False}))
        summary = op.apply(time=100, autotune=('basic', 'runtime'))

        # AT is expected to have attempted 6 block shapes
        assert op._state['autotuning'][0]['runs'] == 6

        # AT is expected to have executed 30 timesteps
        assert summary[('section0', None)].itershapes[0][0] == 101-30
        assert np.all(f.data[0] == 100)
        assert np.all(f.data[1] == 101)

    @pytest.mark.parametrize('openmp, expected', [
        (False, 2), (True, 3)
    ])
    def test_mixed_blocking_w_skewing(self, openmp, expected):
        grid = Grid(shape=(96, 96, 96))
        f = TimeFunction(name='f', grid=grid)

        op = Operator(Eq(f.forward, f + 1.), opt=('blocking', 'skewing',
                                                  {'openmp': openmp}))
        op.apply(time=0, autotune=True)
        assert op._state['autotuning'][0]['runs'] == 6
        assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][0]['tuned']) == expected
        if openmp:
            assert 'nthreads' in op._state['autotuning'][0]['tuned']
        else:
            assert 'nthreads' not in op._state['autotuning'][0]['tuned']

    def test_skewed_tti_aggressive(self):
        from test_dse import TestTTI
        wave_solver = TestTTI().tti_operator(opt=('blocking', 'skewing'))
        op = wave_solver.op_fwd(kernel='centered')
        op.apply(time=0, autotune='aggressive')
        assert op._state['autotuning'][0]['runs'] == 30

    def test_multiple_skewed_blocking(self):
        """
        Test that if there are more than one skewed-blocked Iteration nests, then
        the autotuner works "incrementally" -- it starts determining the best block
        shape for the first Iteration nest, then it moves on to the second one,
        then the third, etc. IOW, the autotuner must not be attempting the
        cartesian product of all possible block shapes across the various
        blocked nests.
        """
        grid = Grid(shape=(96, 96, 96))

        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid)

        op = Operator([Eq(u.forward, u + 1), Eq(v.forward, u.forward.dx2 + v + 1)],
                      opt=('blocking', 'skewing', {'openmp': False}))

        # First of all, make sure there are indeed two different loop nests
        assert 'bf0' in op._func_table
        assert 'bf1' in op._func_table

        # 'basic' mode
        op.apply(time_M=0, autotune='basic')
        assert op._state['autotuning'][0]['runs'] == 12  # 6 for each Iteration nest
        assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][0]['tuned']) == 4

        # 'aggressive' mode
        op.apply(time_M=0, autotune='aggressive')
        assert op._state['autotuning'][1]['runs'] == 60
        assert op._state['autotuning'][1]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][1]['tuned']) == 4

        # With OpenMP, we tune over one more argument (`nthreads`), though the AT
        # will only attempt one value
        op = Operator([Eq(u.forward, u + 1), Eq(v.forward, u.forward.dx2 + v + 1)],
                      opt=('blocking', 'skewing', {'openmp': True}))
        op.apply(time_M=0, autotune='basic')
        assert op._state['autotuning'][0]['runs'] == 12
        assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][0]['tuned']) == 5

    def test_hierarchical_skewed_blocking(self):
        grid = Grid(shape=(64, 64, 64))

        u = TimeFunction(name='u', grid=grid, space_order=2)

        op = Operator(Eq(u.forward, u + 1), opt=('blocking', 'skewing',
                                                 {'openmp': False, 'blocklevels': 2}))

        # 'basic' mode
        op.apply(time_M=0, autotune='basic')
        assert op._state['autotuning'][0]['runs'] == 10
        assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][0]['tuned']) == 4

        # 'aggressive' mode
        op.apply(time_M=0, autotune='aggressive')
        assert op._state['autotuning'][1]['runs'] == 38
        assert op._state['autotuning'][1]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][1]['tuned']) == 4

    @switchconfig(platform='cpu64-dummy')  # To fix the core count
    def test_multiple_threads_w_skewing(self):
        """
        Test autotuning when different ``num_threads`` for a given OpenMP parallel
        region are attempted.
        """
        grid = Grid(shape=(96, 96, 96))

        v = TimeFunction(name='v', grid=grid)

        op = Operator(Eq(v.forward, v + 1), opt=('blocking', 'skewing',
                                                 {'openmp': True}))
        op.apply(time_M=0, autotune='max')
        assert op._state['autotuning'][0]['runs'] == 60  # Would be 30 with `aggressive`
        assert op._state['autotuning'][0]['tpr'] == options['squeezer'] + 1
        assert len(op._state['autotuning'][0]['tuned']) == 3


class TestDSEWithSkewing(object):
    """
    This class contains tests mainly inherited from test_dse.py
    Aims to test interoperability for skewing and prior dse passes
    """
    def test_time_dependent_split(self):
        grid = Grid(shape=(10, 10))
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=3)
        v = TimeFunction(name='v', grid=grid, time_order=2, space_order=0, save=3)

        # The second equation needs a full loop over x/y for u then
        # a full one over x.y for v
        eq = [Eq(u.forward, 2 + grid.time_dim),
              Eq(v.forward, u.forward.dx + u.forward.dy + 1)]
        op = Operator(eq, opt=('blocking', 'skewing',
                               {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        op()

        assert np.allclose(u.data[2, :, :], 3.0)
        assert np.allclose(v.data[1, 1:-1, 1:-1], 1.0)


class TestDLEWithSkewing(object):
    """
    This class contains tests mainly inherited from test_dle.py
    Aims to test interoperability for skewing and prior DLE passes
    """

    @pytest.mark.parametrize("blockinner,exp_calls,exp_iters", [
        (False, 4, 5),
        (True, 8, 6)
    ])
    def test_cache_blocking_structure(self, blockinner, exp_calls, exp_iters):
        # Check code structure
        _, op = _new_operator2((10, 31, 45), time_order=2,
                               opt=('blocking', 'skewing',
                                    {'blockinner': blockinner, 'par-collapse-ncores': 1}))
        calls = FindNodes(Call).visit(op)
        assert len(calls) == exp_calls
        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == exp_iters
        if blockinner:
            assert all(tree[i].dim.is_Incr for i in range(exp_iters))
        else:
            assert all(tree[i].dim.is_Incr for i in range(exp_iters-1))
            assert not tree[-1].dim.is_Incr

        # Check presence of openmp pragmas at the right place
        _, op = _new_operator2((10, 31, 45), time_order=2,
                               opt=('blocking', 'skewing',
                                    {'openmp': True, 'blockinner': blockinner,
                                     'par-collapse-ncores': 1}))
        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree.root.pragmas) == 1
        assert 'omp for' in tree.root.pragmas[0].value
        # Also, with omp parallelism enabled, the step increment must be != 0
        # to avoid omp segfaults at scheduling time (only certain omp implementations,
        # including Intel's)
        conditionals = FindNodes(Conditional).visit(op._func_table['bf0'].root)
        assert len(conditionals) == 1
        conds = conditionals[0].condition.args
        expected_guarded = tree[:2+blockinner]
        assert len(conds) == len(expected_guarded)
        assert all(i.lhs == j.step for i, j in zip(conds, expected_guarded))


class TestCodeGenSkew(object):

    '''
    Test code generation with skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z-time+1],u[t0,x-time+1,y-time+1,z-time+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)']),
    ])
    def test_skewed_bounds(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', 'skewing'))

        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1
        time_iter = time_iter[0]

        for i in ['bf0']:
            assert i in op._func_table
            iters = FindNodes(Iteration).visit(op._func_table[i].root)
            assert len(iters) == 5
            assert iters[0].dim.parent is x
            assert iters[1].dim.parent is y
            assert iters[4].dim is z
            assert iters[2].dim.parent is iters[0].dim
            assert iters[3].dim.parent is iters[1].dim

            assert (iters[2].symbolic_min == (iters[0].dim + time))
            assert (iters[2].symbolic_max == (iters[0].dim + time +
                                              iters[0].dim.symbolic_incr - 1))
            assert (iters[3].symbolic_min == (iters[1].dim + time))
            assert (iters[3].symbolic_max == (iters[1].dim + time +
                                              iters[1].dim.symbolic_incr - 1))

            assert (iters[4].symbolic_min == (iters[4].dim.symbolic_min + time))
            assert (iters[4].symbolic_max == (iters[4].dim.symbolic_max + time))
            skewed = [i.expr for i in FindNodes(Expression).visit(op._func_table[i].root)]
            assert str(skewed[0]).replace(' ', '') == expected
