import datetime
import logging
import os
from pathlib import Path
from subprocess import check_output, PIPE, Popen
import sys
from tempfile import gettempdir, mkdtemp

import click


from benchmarks.user.advisor.advisor_logging import (check, log, progress,
                                                     log_process)


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
def run_with_advisor(path, output, name, exec_args):
    path = Path(path)
    check(path.is_file(), '%s not found' % path)
    check(path.suffix == '.py', '%s not a Python file' % path)

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
    if name is None:
        output = Path(mkdtemp(dir=str(output), prefix="%s-" % name))
    else:
        output = Path(output).joinpath(name)
        output.mkdir(parents=True, exist_ok=True)

    # Intel Advisor and Intel compilers must be available through either Intel Parallel
    # Studio or Intel oneAPI (currently tested versions include IPS 2020 Update 2 and
    # oneAPI 2021 beta08)
    try:
        ret = check_output(['advixe-cl', '--version']).decode("utf-8")
    except FileNotFoundError:
        check(False, "Error: Couldn't detect `advixe-cl` to run Intel Advisor.")

    try:
        ret = check_output(['icc', '--version']).decode("utf-8")
    except FileNotFoundError:
        check(False, "Error: Couldn't detect Intel Compiler (icc).")

    # All good, Intel compiler and advisor are available
    os.environ['DEVITO_ARCH'] = 'intel'

    # Tell Devito to instrument the generated code for Advisor
    os.environ['DEVITO_PROFILING'] = 'advisor'

    # Devito Logging is disabled unless the user asks explicitly to see it
    devito_logging = os.environ.get('DEVITO_LOGGING')
    if devito_logging is None:
        os.environ['DEVITO_LOGGING'] = 'WARNING'

    with progress('Setting up multi-threading environment'):
        # Roofline analyses are recommended with threading enabled
        os.environ['DEVITO_LANGUAGE'] = 'openmp'

        # Thread pinning is strongly recommended for reliable results.
        # This script is using numactl for this purpose. Users may want to set their
        # own pinning: https://hpc-wiki.info/hpc/Binding/Pinning
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

    # To build a roofline with Advisor, we need to run two analyses back to
    # back, `survey` and `tripcounts`.

    numactl_cmd = [
        'numactl',
        '--cpunodebind=0'
    ]
    advisor_cmd = [
        'advixe-cl',
        '-data-limit=500',
        '-project-dir', str(output),
        '-search-dir src:r=%s' % gettempdir(),  # Root directory where Devito stores the generated code  # noqa
    ]
    advisor_survey = [
        '-collect survey',
        '-run-pass-thru=--no-altstack',  # Avoids `https://software.intel.com/en-us/vtune-amplifier-help-error-message-stack-size-is-too-small`  # noqa
        '-run-pass-thru=-timestamp=sys',  # Avoids 'VTune Amplifier may detect which timer source to use incorrectly on Intel® Xeon® processor E5-XXXX processors (200287361)' # noqa
        '-strategy ldconfig:notrace:notrace',  # Avoids `https://software.intel.com/en-us/forums/intel-vtune-amplifier-xe/topic/779309`  # noqa
        '-start-paused',  # The generated code will enable/disable Advisor on a loop basis according to the decorated pragmas  # noqa
    ]
    advisor_flops = [
        '--collect=tripcounts',
        '--enable-cache-simulation', # Switch to '-enable-cache-simulation' for a CARM roofline model `https://software.intel.com/content/www/us/en/develop/articles/integrated-roofline-model-with-intel-advisor.html`  # noqa
        '--flop',
        '--stacks',
        '--collect=map',
        '-start-paused',
    ]
    py_cmd = [sys.executable, str(path)] + exec_args.split()

    # Before collecting the `survey` and `tripcounts` a "pure" python run to warmup the
    # jit cache is preceded

    log('Starting Intel Advisor\'s `roofline` analysis for `%s`' % name)
    dt = datetime.datetime.now()

    # Set up a file logger that will track the output of the advisor profiling
    advixe_logger = logging.getLogger('run_advisor_logger')
    advixe_logger.setLevel(logging.INFO)

    advixe_formatter = logging.Formatter('%(asctime)s: %(message)s')
    logger_datetime = '%d.%d.%d.%d.%d.%d' % (dt.year, dt.month,
                                             dt.day, dt.hour, dt.minute, dt.second)
    advixe_handler = logging.FileHandler('%s/%s_%s.log' % (output, name, logger_datetime))
    advixe_handler.setFormatter(advixe_formatter)
    advixe_logger.addHandler(advixe_handler)

    with progress('Performing `cache warm-up` run'):
        try:
            p_warm_up = Popen(py_cmd, stdout=PIPE, stderr=PIPE)
            log_process(p_warm_up, advixe_logger)
        except OSError:
            check(False, 'Failed!')

    with progress('Performing `survey` analysis'):
        cmd = numactl_cmd + ['--'] + advisor_cmd + advisor_survey + ['--'] + py_cmd
        try:
            p_survey = Popen(cmd, stdout=PIPE, stderr=PIPE)
            log_process(p_survey, advixe_logger)
        except OSError:
            check(False, 'Failed!')

    with progress('Performing `tripcounts` analysis'):
        cmd = numactl_cmd + ['--'] + advisor_cmd + advisor_flops + ['--'] + py_cmd
        try:
            p_tripcounts = Popen(cmd, stdout=PIPE, stderr=PIPE)
            log_process(p_tripcounts, advixe_logger)
        except OSError:
            check(False, 'Failed!')

    log('Storing `survey` and `tripcounts` data in `%s`' % str(output))
    log('To plot a roofline type: ')
    log('python3 roofline.py --name %s --project %s --scale %f'
        % (name, str(output), n_sockets))


if __name__ == '__main__':
    run_with_advisor()
