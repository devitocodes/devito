import os
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import gettempdir, mkdtemp
import shutil

import click

from devito.logger import info


@click.command()
# Required arguments
@click.option('--path', '-p', help='Absolute path to the Devito executable.',
              required=True)
@click.option('--output', '-o', help='A directory for storing profiling reports. '
                                     'The directory is created if it does not exist. '
                                     'If unspecified, reports are stored within '
                                     'the OS temporary directory')
# Optional arguments
@click.option('--name', '-n', help='A unique name identifying the run. '
                                   'If unspecified, a name is assigned joining '
                                   'the executable name with the options specified '
                                   'in --exec-args (if any).')
@click.option('--exec-args', type=click.UNPROCESSED,
              help='Arguments passed to the executable.')
@click.option('--advisor-home', help='Path to Intel Advisor. Defaults to /opt/intel'
                                     '/advisor, which is the directory in which '
                                     'Intel Compiler suite is installed.')
def run_with_advisor(path, output, name, exec_args, advisor_home):
    path = Path(path)
    check(path.is_file(), '%s not found' % path)
    check(path.suffix == '.py', '%s not a regular Python file' % path)

    # Create a directory to store the profiling report
    if name is None:
        name = path.stem
        if exec_args:
            name = "%s_%s" % (name, ''.join(exec_args))
    if output is None:
        output = Path(gettempdir()).joinpath('devito-profilings')
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = Path(output)
    output = Path(mkdtemp(dir=str(output), prefix="%s-" % name))

    # Devito must be told where to find Advisor, because it uses its C API
    if advisor_home:
        os.environ['ADVISOR_HOME'] = advisor_home
    else:
        os.environ['ADVISOR_HOME'] = '/opt/intel/advisor'

    # Tell Devito to instrument the generated code for Advisor
    os.environ['DEVITO_PROFILING'] = 'advisor'

    # Prevent NumPy from using threads, which otherwise leads to a deadlock when
    # used in combination with Advisor. This issue has been described at:
    #     `software.intel.com/en-us/forums/intel-advisor-xe/topic/780506`
    # Note: we should rather sniff the BLAS library used by NumPy, and set the
    # appropriate env var only
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Note: `Numaexpr`, used by NumPy, also employs threading, so we shall disable
    # it too via the corresponding env var. See:
    #     `stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading`  # noqa
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    advisor_command = [
        'advixe-cl',
        '-q',  # Silence advisor
        '-data-limit=500',
        '-collect survey',
        '-start-paused',  # The generated code will enable/disable Advisor on a loop basis
        '-project-dir', str(output),
        '-search-dir src:r=%s' % gettempdir(),  # Root directory where Devito stores the generated code  # noqa
        '-run-pass-thru=--no-altstack',  # Avoids `https://software.intel.com/en-us/vtune-amplifier-help-error-message-stack-size-is-too-small`  # noqa
        '-strategy ldconfig:notrace:notrace'  # Avoids `https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/779309`  # noqa
    ]
    py_command = ['python', str(path)] + exec_args.split()
    command = advisor_command + ['--'] + py_command

    check_call(command)
    info('Advisor data successfully stored in %s' % str(output))


def check(cond, msg):
    if not cond:
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run_with_advisor()
