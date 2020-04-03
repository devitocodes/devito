import pytest
import os

from devito import configuration
from subprocess import check_call


@pytest.mark.parametrize('mode', ['bench'])
@pytest.mark.parametrize('problem', ['acoustic', 'tti', 'elastic', 'viscoelastic'])
def test_bench(mode, problem):
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

    baseline = os.path.realpath(__file__).split("tests/test_benchmark.py")[0]

    command_bench = ['python', '%sbenchmarks/user/benchmark.py' % baseline, mode,
                     '-P', problem, '-d', '%d' % nx, '%d' % ny, '%d' % nz, '--tn',
                     '%d' % tn, '-x', '1']
    check_call(command_bench)

    dir_name = 'results/'

    base_filename = problem
    filename_suffix = '.json'
    arch = 'arch[unknown]'
    shape = 'shape[%d,%d,%d]' % (nx, ny, nz)
    nbl = 'nbl[10]'
    t = 'tn[%d]' % tn
    so = 'so[2]'
    to = 'to[2]'
    opt = 'opt[advanced]'
    at = 'at[aggressive]'
    nt = 'nt[%d]' % nthreads
    mpi = 'mpi[False]'
    np = 'np[1]'
    rank = 'rank[0]'

    bench_corename = os.path.join('_'.join([base_filename, arch, shape, nbl, t,
                                  so, to, opt, at, nt, mpi, np, rank]))

    bench_filename = "%s%s%s" % (dir_name, bench_corename, filename_suffix)
    assert os.path.isfile(bench_filename)
