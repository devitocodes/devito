import os
import sys
from subprocess import check_call
from unittest.mock import patch

import pytest

from benchmarks.user.benchmark import run
from conftest import skipif
from devito import Eq, Function, Grid, Operator, TimeFunction, configuration, switchconfig
from devito.ir.iet import (
    Call, Callable, Conditional, EntryFunction, FindNodes, Iteration, List, Section,
    TimedList
)
from devito.ir.support import SymbolRegistry
from devito.operator.profiling import AdvancedProfiler
from devito.passes.iet.engine import Graph, iet_pass
from devito.passes.iet.instrument import instrument
from devito.passes.iet.languages.targets import CTarget


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


class TestTimeLoopProfiling:

    @switchconfig(profiling='advanced')
    def test_loop_span(self):
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=6)
        v = TimeFunction(name='v', grid=grid, save=6)

        eqs = [Eq(f, 2), Eq(u.forward, u + f), Eq(f, u[5, x, y]),
               Eq(v.backward, v + f), Eq(f, v[0, x, y])]
        op = Operator(eqs, opt=('noop', {'gpu-fit': (u, v)}))

        outer, = [i for i in FindNodes(TimedList).visit(op) if i.name == 'time_loop']
        assert FindNodes(Section).visit(outer) == FindNodes(Section).visit(op)[1:-1]
        wait = op._Target.langbb().get('device-wait')
        if wait is not None:
            assert str(outer.body[-1]) == str(wait)

        summary = op.apply(time_m=1, time_M=4)
        assert f.data[0, 0] == 32
        assert summary.globals_nosetup_all.time == max(outer.timer.total, 1e-6)

    @pytest.mark.parallel(mode=2)
    @pytest.mark.parametrize('stepping, expected', [(False, 16.), (True, 11.)])
    @switchconfig(profiling='advanced')
    def test_rank_reduction(self, mode, stepping, expected):
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions
        u = TimeFunction(name='u', grid=grid, save=3)
        v = Function(name='v', grid=grid)
        f = u if stepping else u[0, x, y]

        op = Operator([Eq(f, f + 1), Eq(v, v + f, subdomain=grid.interior)], opt='noop')
        args = op.arguments()
        timer = args['timers']._obj
        rank = grid.distributor.myrank

        # Different ranks dominate the two sections.
        timer.section0, timer.section1 = (8., 2.) if rank == 0 else (2., 8.)
        if stepping:
            timer.time_loop = 10. + rank

        summary = op._profiler.summary(args, op._dtype, reduce_over=20.)
        assert summary.globals['vanilla-nosetup'].time == expected
        if stepping:
            assert summary.globals_nosetup_all.gpointss == (3 * 8**2 / 1e9) / expected

    def test_guarded_helper(self):
        t = Grid(shape=(8,)).time_dim
        step = Callable('step', Section('section0', Call('work')), 'void')
        loop = Iteration(Call('step'), t, (0, 1, 1))
        guard = Conditional(t.symbolic_max > 0, loop)
        body = [List(body=[Call('setup'), guard]), List(body=[loop, Call('teardown')])]

        graph = Graph(EntryFunction('Kernel', body, 'int'))
        graph.efuncs[step.name] = step

        profiler = AdvancedProfiler('timers')
        profiler.analyze(step)

        @iet_pass
        def sections(iet, timer, **kwargs):
            return profiler.instrument(iet, timer), {}

        with patch('devito.passes.iet.instrument.instrument_sections', sections):
            instrument(graph, profiler=profiler, sregistry=SymbolRegistry(),
                       langbb=CTarget.langbb())

        outer, = FindNodes(TimedList).visit(graph.root)
        assert [type(i) for i in outer.body] == [Conditional, Iteration]
        assert all(i.arguments == (outer.timer,) for i in FindNodes(Call).visit(outer))
