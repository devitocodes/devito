import os
import sys
from subprocess import check_call

import pytest

from benchmarks.user.benchmark import run
from conftest import skipif
from devito import Eq, Function, Grid, Operator, TimeFunction, configuration, switchconfig
from devito.ir.iet import (
    Call, CallableBody, Conditional, Expression, FindNodes, Iteration, List, TimedList
)
from devito.operator.profiling import TimeLoopTimer
from devito.passes.iet.instrument import time_loop_region
from devito.types import Timer


@skipif('cpu64-icx')
@pytest.mark.parametrize('mode, problem, op', [
    ('run', 'acoustic', 'forward'), ('run', 'acoustic', 'adjoint'),
    ('run', 'acoustic', 'jacobian'), ('run', 'acoustic', 'jacobian_adjoint'),
    ('run', 'tti', 'forward'), ('run', 'elastic', 'forward'),
    ('run', 'viscoelastic', 'forward'), ('run', 'acoustic_sa', 'forward'),
    ('run', 'acoustic_sa', 'adjoint'), ('run', 'acoustic_sa', 'jacobian'),
    ('run', 'acoustic_sa', 'jacobian_adjoint'), ('run', 'tti', 'jacobian_adjoint')
])
def test_bench(mode, problem, op):
    """
    Test the Devito benchmark framework on various combinations of modes and problems.
    """

    tn = 4
    nx, ny, nz = 16, 16, 16

    if configuration['language'] == 'openmp':
        nthreads = int(os.environ.get('OMP_NUM_THREADS',
                                      configuration['platform'].cores_physical))
    else:
        nthreads = 1

    pyversion = sys.executable
    baseline = os.path.realpath(__file__).split("tests/test_benchmark.py")[0]
    benchpath = f'{baseline}benchmarks/user/benchmark.py'

    command_bench = [
        pyversion, benchpath, mode, '-P', problem,
        '-d', str(nx), str(ny), str(nz), '--tn', str(tn), '-op', op
    ]
    if mode == "bench":
        command_bench.extend(['-x', '1'])
    check_call(command_bench)

    dir_name = 'results/'

    base_filename = problem
    filename_suffix = '.json'
    arch = 'arch[unknown]'
    shape = f'shape[{nx}{ny}{nz}]'
    nbl = 'nbl[10]'
    t = f'tn[{tn}]'
    so = 'so[2]'
    to = 'to[2]'
    opt = 'opt[advanced]'
    at = 'at[aggressive]'
    nt = f'nt[{nthreads}]'
    mpi = 'mpi[False]'
    np = 'np[1]'
    rank = 'rank[0]'

    if mode == "bench":
        bench_corename = os.path.join('_'.join([base_filename, arch, shape, nbl, t,
                                      so, to, opt, at, nt, mpi, np, rank]))

        bench_filename = f"{dir_name}{bench_corename}{filename_suffix}"
        assert os.path.isfile(bench_filename)
    else:
        assert True


@pytest.mark.parallel(mode=2)
@switchconfig(profiling='advanced')
def test_run_mpi(mode):
    """
    Test the `run` mode over MPI, with all key arguments used.
    """
    kwargs = {
        'space_order': [4],
        'time_order': [2],
        'autotune': 'off',
        'block_shape': [],
        'shape': (16, 16, 16),
        'tn': 4,
        'warmup': False,
        'dump_summary': 'summary.txt',
        'dump_norms': 'norms.txt'
    }
    run('acoustic', **kwargs)


@skipif('noadvisor')
@switchconfig(profiling='advisor')
def test_advisor_profiling():
    """
    Test includes and compilation with `advisor`
    """
    grid = Grid(shape=(10, 10, 10))
    f = TimeFunction(name='f', grid=grid, space_order=2)

    eq0 = [Eq(f.forward, f.dx + 1.)]

    op = Operator(eq0)
    assert 'ittnotify.h' in op._includes
    op.apply(time_M=5)


@switchconfig(profiling='advanced')
def test_nosetup_time_loop():
    grid = Grid(shape=(32, 32))
    f = Function(name='f', grid=grid)
    u = TimeFunction(name='u', grid=grid, save=6)
    v = TimeFunction(name='v', grid=grid, save=6)
    op = Operator([Eq(f, 2), Eq(u.forward, u + f),
                   Eq(v.backward, v + u), Eq(f, f + 3)], opt='noop')

    outer, = [i for i in FindNodes(TimedList).visit(op) if i.name == 'time_loop']
    assert len([i for i in FindNodes(Iteration).visit(outer) if i.dim.is_Time]) == 2
    assert all(e.write is not f for e in FindNodes(Expression).visit(outer))
    if (wait := op._Target.langbb().get('device-wait')) is not None:
        assert str(outer.body[-1]) == str(wait)
    summary = op.apply(time_m=1, time_M=4)
    assert (u.data[5, 0, 0], v.data[0, 0, 0], f.data[0, 0]) == (8, 12, 5)
    timer = next(p for p in op.parameters if isinstance(p, Timer))
    assert isinstance(timer, TimeLoopTimer)
    assert summary.globals_nosetup_all.time == max(timer.total, 1e-6)


@pytest.mark.parallel(mode=2)
@pytest.mark.parametrize('stepping', [False, True])
@switchconfig(profiling='advanced')
def test_nosetup_rank_reduction(mode, stepping):
    grid = Grid(shape=(16, 16))
    if stepping:
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        eqns = [Eq(u.forward, u + 1),
                Eq(v.forward, v + u.forward, subdomain=grid.interior)]
    else:
        u = Function(name='u', grid=grid)
        eqns = [Eq(u, u + 1), Eq(u, u + 2, subdomain=grid.interior)]
    op = Operator(eqns, opt='noop')
    args = op.arguments(time_M=1) if stepping else op.arguments()
    timer = args['timers']._obj

    # Opposite ranks dominate the two sections. Use exact times to test the reduction.
    rank = grid.distributor.myrank
    timer.section0, timer.section1 = (8., 2.) if rank == 0 else (2., 8.)
    if stepping:
        timer.time_loop = 10. + rank
    summary = op._profiler.summary(args, op._dtype, reduce_over=20.)
    assert summary.globals['vanilla-nosetup'].time == (11. if stepping else 16.)
    if stepping:
        assert summary.globals_nosetup_all.gpointss == (2 * 16 * 16 / 1e9) / 11.


def test_nosetup_guarded_time_loops():
    t = Grid(shape=(8,)).time_dim
    first = Conditional(t.symbolic_max > 0, Iteration((), t, (0, 1, 1)))
    second = Iteration((), t, (2, 3, 1))
    middle = Call('between_steps')
    group = List(body=[Call('setup'), first, middle, second, Call('teardown')])

    scope, span = time_loop_region(CallableBody([Call('init'), group]))
    assert scope.body[span] == (first, middle, second)

    scope, span = time_loop_region(CallableBody(first))
    assert scope.body[span] == (first,)
