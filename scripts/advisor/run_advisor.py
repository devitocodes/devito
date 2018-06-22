import os
import sys
from pathlib import Path
from subprocess import DEVNULL, check_call
from tempfile import gettempdir, mkdtemp
from contextlib import contextmanager

# Required to generate a roofline
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import click


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


try:
    import advisor
except ImportError:
    err("Couldn't detect Intel Advisor on this system.")
    sys.exit(1)


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
@click.option('--exec-args', type=click.UNPROCESSED, default='',
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

    # Intel Advisor is available, and so must is the Intel compiler
    os.environ['DEVITO_ARCH'] = 'intel'

    # Tell Devito to instrument the generated code for Advisor
    os.environ['DEVITO_PROFILING'] = 'advisor'

    with progress('Set up multi-threading environment'):
        # Roofline analyses only make sense with threading enabled
        os.environ['DEVITO_OPENMP'] = '1'

        # We must be able to do thread pinning, otherwise any results would be
        # meaningless. Currently, we only support doing that via numactl
        try:
            errmsg = "Couldn't detect `numactl`, necessary for thread pinning."
            check(check_call(['numactl', '--show'], stdout=DEVNULL) == 0, errmsg)
        except:
            check(True, errmsg)

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
    roofline(name, output)


def roofline(name, output):
    """
    Generate a roofline for the Intel Advisor ``project``.

    This routine is partly extracted from the examples directory of Intel Advisor 2018;
    it has been tweaked to produce ad-hoc rooflines.
    """
    with progress('Generating roofline char for `%s`' % name):
        pd.options.display.max_rows = 20

        project = advisor.open_project(output)
        data = project.load(advisor.SURVEY)
        rows = [{col: row[col] for col in row} for row in data.bottomup]
        roofs = data.get_roofs()

        df = pd.DataFrame(rows).replace('', np.nan)
        # print(df[['self_arithmetic_intensity', 'self_gflops']].dropna())

        df.self_arithmetic_intensity = df.self_arithmetic_intensity.astype(float)
        df.self_gflops = df.self_gflops.astype(float)

        width = df.self_arithmetic_intensity.max() * 1.2

        fig, ax = plt.subplots()
        key = lambda roof: roof.bandwidth if 'bandwidth' not in roof.name.lower() else 0
        max_compute_roof = max(roofs, key=key)
        max_compute_bandwidth = max_compute_roof.bandwidth / math.pow(10, 9)  # as GByte/s

        for roof in roofs:
            # by default drawing multi-threaded roofs only
            if 'single-thread' not in roof.name:
                # memory roofs
                if 'bandwidth' in roof.name.lower():
                    bandwidth = roof.bandwidth / math.pow(10, 9) # as GByte/s
                    # y = banwidth * x
                    x1, x2 = 0, min(width, max_compute_bandwidth / bandwidth)
                    y1, y2 = 0, x2 * bandwidth
                    label = '{} {:.0f} GB/s'.format(roof.name, bandwidth)
                    ax.plot([x1, x2], [y1, y2], '-', label=label)

                # compute roofs
                else:
                    bandwidth = roof.bandwidth / math.pow(10, 9)  # as GFlOPS
                    x1, x2 = 0, width
                    y1, y2 = bandwidth, bandwidth
                    label = '{} {:.0f} GFLOPS'.format(roof.name, bandwidth)
                    ax.plot([x1, x2], [y1, y2], '-', label=label)

        # drawing points using the same ax
        ax.set_xscale('log', nonposx='clip')
        ax.set_yscale('log', nonposy='clip')
        ax.plot(df.self_arithmetic_intensity, df.self_gflops, 'o')

        plt.legend(loc='lower right', fancybox=True, prop={'size': 6})

        # saving the chart in PDF format
        plt.savefig('%s.pdf' % name)


if __name__ == '__main__':
    run_with_advisor()
