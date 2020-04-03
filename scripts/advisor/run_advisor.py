import os
import sys
from pathlib import Path
from subprocess import check_call, check_output
from tempfile import gettempdir, mkdtemp
from contextlib import contextmanager

import click


@click.command()
# Required arguments
@click.option('--path', '-p', help='Absolute path to the Devito executable.',
              required=True)
# Optional arguments
@click.option('--exec-args', type=click.UNPROCESSED, default='',
              help='Arguments passed to the executable.')
@click.option('--output', '-o', help='A directory for storing profiling reports. '
                                     'The directory is created if it does not exist. '
                                     'If unspecified, reports are stored within '
                                     'a temporary directory.')
@click.option('--name', '-n', help='A unique name identifying the run. '
                                   'If unspecified, a name is generated joining '
                                   'the executable name with the options specified '
                                   'in --exec-args (if any).')
@click.option('--advisor-home', help='Path to Intel Advisor. Defaults to /opt/intel'
                                     '/advisor, the directory in which the Intel '
                                     'Compiler suite is installed by default.')
@click.option('--plot/--no-plot', default=True, help='Generate a roofline.')
def run_with_advisor(path, output, name, exec_args, advisor_home, plot):
    path = Path(path)
    check(path.is_file(), '%s not found' % path)
    check(path.suffix == '.py', '%s not a regular Python file' % path)

    # Create a directory to store the profiling report
    if name is None:
        name = path.stem
        if exec_args:
            name = "%s_%s" % (name, ''.join(exec_args.split()))
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

    # Intel Advisor 2018 must be available
    try:
        ret = check_output(['advixe-cl', '--version']).decode("utf-8")
    except FileNotFoundError:
        check(False, "Couldn't detect `advixe-cl` to run Intel Advisor.")
    # The 2018.3 release is the only one for which support is guaranteed
    if not any(ret.startswith(i) for i in supported_releases):
        log('Intel Advisor is available, but version `%s` does not appear '
            'among the supported ones `%s`, hence the behaviour is now undefined.'
            % (ret, supported_releases))

    # If Advisor is available, so is the Intel compiler
    os.environ['DEVITO_ARCH'] = 'intel'

    # Tell Devito to instrument the generated code for Advisor
    os.environ['DEVITO_PROFILING'] = 'advisor'

    # Devito Logging is disabled unless the user asks explicitly to see it
    logging = os.environ.get('DEVITO_LOGGING')
    if logging is None:
        os.environ['DEVITO_LOGGING'] = 'WARNING'

    with progress('Set up multi-threading environment'):
        # Roofline analyses only make sense with threading enabled
        os.environ['DEVITO_LANGUAGE'] = 'openmp'

        # We must be able to do thread pinning, otherwise any results would be
        # meaningless. Currently, we only support doing that via numactl
        try:
            ret = check_output(['numactl', '--show']).decode("utf-8")
            ret = dict(i.split(':') for i in ret.split('\n') if i)
            n_sockets = len(ret['cpubind'].split())
            n_cores = len(ret['physcpubind'].split())  # noqa
        except FileNotFoundError:
            check(False, "Couldn't detect `numactl`, necessary for thread pinning.")

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

    numactl_cmd = [
        'numactl',
        '--cpunodebind=0'
    ]
    advisor_cmd = [
        'advixe-cl',
        '-q',  # Silence advisor
        '-data-limit=500',
        '-project-dir', str(output),
        '-search-dir src:r=%s' % gettempdir(),  # Root directory where Devito stores the generated code  # noqa
    ]
    advisor_survey = [
        '-collect survey',
        '-start-paused',
        '-run-pass-thru=--no-altstack',  # Avoids `https://software.intel.com/en-us/vtune-amplifier-help-error-message-stack-size-is-too-small`  # noqa
        '-strategy ldconfig:notrace:notrace',  # Avoids `https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/779309`  # noqa
        '-start-paused',  # The generated code will enable/disable Advisor on a loop basis
    ]
    advisor_flops = [
        '-collect tripcounts',
        '-flop',
    ]
    py_cmd = ['python', str(path)] + exec_args.split()

    # To build a roofline with Advisor, we need to run two analyses back to
    # back, `survey` and `tripcounts`. These are preceded by a "pure" python
    # run to warmup the jit cache

    log('Starting Intel Advisor\'s `roofline` analysis for `%s`' % name)

    with progress('Performing `cache warm-up` run'):
        check(check_call(py_cmd) == 0, 'Failed!')

    with progress('Performing `survey` analysis'):
        cmd = numactl_cmd + ['--'] + advisor_cmd + advisor_survey + ['--'] + py_cmd
        check(check_call(cmd) == 0, 'Failed!')

    with progress('Performing `tripcounts` analysis'):
        cmd = numactl_cmd + ['--'] + advisor_cmd + advisor_flops + ['--'] + py_cmd
        check(check_call(cmd) == 0, 'Failed!')

    log('Storing `survey` and `tripcounts` data in `%s`' % str(output))

    # Finally, generate a roofline
    # TODO: Intel Advisor 2018 doesn't cope well with Python 3.5, so we rather use
    # the embedded advixe-python
    if plot:
        with progress('Generating roofline char for `%s`' % name):
            cmd = [
                'python2.7',
                'roofline.py',
                '--name %s' % name,
                '--project %s' % output,
                '--scale %f' % n_sockets
            ]
            check(check_call(cmd) == 0, 'Failed!')


supported_releases = [
    'Intel(R) Advisor 2018 Update 3'
]


def check(cond, msg):
    if not cond:
        err(msg)
        sys.exit(1)


def err(msg):
    print('\033[1;37;31m%s\033[0m' % msg)  # print in RED


def log(msg):
    print('\033[1;37;32m%s\033[0m' % msg)  # print in GREEN


@contextmanager
def progress(msg):
    print('\033[1;37;32m%s ... \033[0m' % msg, end='', flush=True)  # print in GREEN
    yield
    print('\033[1;37;32m%s\033[0m' % 'Done!')


if __name__ == '__main__':
    run_with_advisor()
