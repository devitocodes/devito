import os
import sys
from pathlib import Path
from subprocess import check_call

import click


@click.command()
# Required arguments
@click.option('--path', '-p', help='Absolute path to the Devito executable.',
              required=True)
@click.option('--output', '-o', help='A directory for storing profiling reports. '
                                     'The directory is created if it does not exist.',
              required=True)
# Optional arguments
@click.option('--exec-args', type=click.UNPROCESSED,
              help='Arguments passed to the executable.')
@click.option('--advisor-home', help='Path to Intel Advisor. Defaults to /opt/intel'
                                     '/advisor, which is the directory in which '
                                     'Intel Compiler suite is installed.')
def run_with_advisor(path, output, exec_args, advisor_home):
    path = Path(path)
    check(path.is_file(), '%s not found' % path)
    check(path.suffix == '.py', '%s not a regular Python file' % path)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if advisor_home:
        os.environ['ADVISOR_HOME'] = advisor_home
    else:
        os.environ['ADVISOR_HOME'] = '/opt/intel/advisor'

    os.environ['DEVITO_PROFILING'] = 'advisor'

    command = ['python', path.as_posix()]
    command.extend(exec_args.split())

    check_call(command)


def check(cond, msg):
    if not cond:
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run_with_advisor()
