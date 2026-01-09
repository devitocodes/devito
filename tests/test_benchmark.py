import os
import sys
from subprocess import check_call

import pytest

from benchmarks.user.benchmark import run
from conftest import skipif
from devito import Eq, Grid, Operator, TimeFunction, configuration, switchconfig


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
