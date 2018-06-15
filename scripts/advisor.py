from pathlib import Path
import sys

import click


@click.command()
@click.option('--path', '-p', help='Absolute path to the Devito executable.',
              required=True)
@click.option('--output', '-o', help='A directory for storing profiling reports. '
                                     'The directory is created if it does not exist.',
              required=True)
def run_with_advisor(path, output):
    path = Path(path)
    check(path.is_file(), '%s not found' % path)
    check(path.suffix == '.py', '%s not a regular Python file' % path)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)


def check(cond, msg):
    if not cond:
        print(msg, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    run_with_advisor()
