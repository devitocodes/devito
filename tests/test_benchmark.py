import pytest
from subprocess import CalledProcessError, check_call
from devito.exceptions import CompilationError


@pytest.mark.parametrize('mode', ['test', 'run', 'bench'])
@pytest.mark.parametrize('problem', ['acoustic', 'tti', 'elastic', 'viscoelastic'])
def test_bench(mode, problem):
    """
    Test combinations of modes and problems.
    """
    command = ['python', 'benchmarks/user/benchmark.py', mode, '-P', problem,
               '-d', '50', '50', '50', '--tn', '50']
    try:
        check_call(command)
    except CalledProcessError as e:
        raise CompilationError('Command "%s" return error status %d. '
                               'Unable to compile code.\n' %
                               (e.cmd, e.returncode))
