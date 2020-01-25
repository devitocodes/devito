import pytest
from os import path
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
    if configuration['openmp']:
        nthreads = configuration['platform'].cores_physical
    else:
        nthreads = 1

    command_bench = ['python', 'benchmarks/user/benchmark.py', mode, '-P', problem,
                     '-d', '16', '16', '16', '--tn', '4', '-x', '1']
    check_call(command_bench)

    dir_name = 'results/'

    base_filename = problem
    filename_suffix = '.json'
    arch = 'arch[unknown]'
    shape = 'shape[16,16,16]'
    nbl = 'nbl[10]'
    tn = 'tn[4]'
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

    bench_corename = path.join('_'.join([base_filename, arch, shape, nbl, tn,
                               so, to, dse, dle, at, nt, mpi, np, rank]))

    bench_filename = "%s%s%s" % (dir_name, bench_corename, filename_suffix)
    assert path.isfile(bench_filename)

    command_plot = ['python', 'benchmarks/user/benchmark.py', 'plot', '-P', problem,
                    '-d', '16', '16', '16', '--tn', '4', '--max-bw', '12.8',
                    '--flop-ceil', '80', 'linpack']
    check_call(command_plot)

    plot_corename = path.join('_'.join([base_filename, shape, so, to, arch,
                              bkend, at]))
    plot_filename = "%s%s" % (dir_name, plot_corename)

    assert path.isfile(plot_filename)
