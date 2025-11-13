import sys
from contextlib import contextmanager


def check(cond, msg):
    if not cond:
        err(msg)
        sys.exit(1)


def err(msg):
    print(f'\033[1;37;31m{msg}\033[0m')  # print in RED


def log(msg):
    print(f'\033[1;37;32m{msg}\033[0m')  # print in GREEN


@contextmanager
def progress(msg):
    print(f'\033[1;37;32m{msg} ... \033[0m', end='', flush=True)  # print in GREEN
    yield
    print('\033[1;37;32m{}\033[0m'.format('Done!'))


def log_process(process, logger):
    output, errors = process.communicate()
    for output_line in output.splitlines():
        logger.info(output_line.decode('utf-8'))
    for error_line in errors.splitlines():
        logger.error(error_line.decode('utf-8'))

    if process.returncode != 0:
        check(False, 'Failed!')
