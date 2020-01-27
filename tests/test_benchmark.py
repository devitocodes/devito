import pytest
import os

from conftest import skipif
from devito import configuration
from subprocess import check_call


pytestmark = skipif(['yask', 'ops'])


@pytest.mark.parametrize('mode', ['bench'])
@pytest.mark.parametrize('problem', ['acoustic', 'tti', 'elastic', 'viscoelastic'])
def test_bench(mode, problem):
    """
    Test the Devito benchmark framework on various combinations of modes and problems.
    """

    tn = 4
    nx, ny, nz = 16, 16, 16

    if configuration['openmp']:
        nthreads = configuration['platform'].cores_physical
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
    dse = 'dse[advanced]'
    dle = 'dle[advanced]'
    at = 'at[aggressive]'
    nt = 'nt[%d]' % nthreads
    mpi = 'mpi[False]'
    np = 'np[1]'
    rank = 'rank[0]'
    bkend = 'bkend[core]'

    bench_corename = os.path.join('_'.join([base_filename, arch, shape, nbl, t,
                                  so, to, dse, dle, at, nt, mpi, np, rank]))

    bench_filename = "%s%s%s" % (dir_name, bench_corename, filename_suffix)
    assert os.path.isfile(bench_filename)

    command_plot = ['python', '%sbenchmarks/user/benchmark.py' % baseline, 'plot',
                    '-P', problem, '-d', '%d' % nx, '%d' % ny, '%d' % nz, '--tn',
                    '%d' % tn, '--max-bw', '12.8', '--flop-ceil', '80', 'linpack']
    check_call(command_plot)

    plot_corename = os.path.join('_'.join([base_filename, shape, so, to, arch,
                                 bkend, at]))
    plot_filename = "%s%s" % (dir_name, plot_corename)

    assert os.path.isfile(plot_filename)
