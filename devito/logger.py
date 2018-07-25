"""The Devito logger."""

import logging
import sys
from contextlib import contextmanager
from functools import wraps

from devito.parameters import configuration

__all__ = ('set_log_level', 'set_log_noperf', 'silencio', 'log',
           'log', 'warning', 'error', 'perf', 'perf_adv', 'dse', 'dse_warning',
           'dle', 'dle_warning',
           'RED', 'GREEN', 'BLUE')


logger = logging.getLogger('Devito')
_ch = logging.StreamHandler()
logger.addHandler(_ch)

# Add extra logging levels (note: INFO has value=20, WARNING has value=30)
DEBUG = logging.DEBUG
PERF = 19
YASK = 19
YASK_WARN = YASK
DSE = 18
DSE_WARN = 19
DLE = DSE
DLE_WARN = DSE_WARN
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(PERF, "PERF")
logging.addLevelName(YASK, "YASK")
logging.addLevelName(YASK_WARN, "YASK_WARN")
logging.addLevelName(DSE, "DSE")
logging.addLevelName(DSE_WARN, "DSE_WARN")
logging.addLevelName(DLE, "DLE")
logging.addLevelName(DLE_WARN, "DLE_WARN")

logger_registry = {
    'DEBUG': DEBUG,
    'PERF': PERF,
    'YASK': YASK,
    'YASK_WARN': YASK_WARN,
    'INFO': INFO,
    'DSE': DSE,
    'DSE_WARN': DSE_WARN,
    'DLE': DLE,
    'DLE_WARN': DLE_WARN,
    'WARNING': WARNING,
    'ERROR': ERROR,
    'CRITICAL': CRITICAL
}

NOCOLOR = '%s'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'

COLORS = {
    DEBUG: NOCOLOR,
    PERF: GREEN,
    YASK: GREEN,
    YASK_WARN: GREEN,
    INFO: NOCOLOR,
    DSE: NOCOLOR,
    DSE_WARN: BLUE,
    DLE: NOCOLOR,
    DLE_WARN: BLUE,
    WARNING: BLUE,
    ERROR: RED,
    CRITICAL: RED
}


def set_log_level(level):
    """
    Set the log level of the Devito logger.

    :param level: accepted values are: DEBUG, PERF, INFO, DSE, DSE_WARN,
                  DLE, DLE_WARN, WARNING, ERROR, CRITICAL
    """
    if level not in logger_registry:
        raise ValueError("Illegal logging level %s" % level)
    logger.setLevel(level)


def set_log_noperf():
    """Do not print performance-related messages."""
    logger.setLevel(WARNING)


configuration.add('log_level', 'INFO', list(logger_registry),
                  lambda i: set_log_level(i))


class silencio(object):

    """
    Decorator to temporarily change log levels.
    """

    def __init__(self, log_level='WARNING'):
        self.log_level = log_level

    def __call__(self, func, *args, **kwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            previous = configuration['log_level']
            configuration['log_level'] = self.log_level
            result = func(*args, **kwargs)
            configuration['log_level'] = previous
            return result
        return wrapper


def log(msg, level=INFO, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    :param msg: the message to be printed.
    :param level: accepted values are: DEBUG, PERF, INFO, DSE, DSE_WARN,
                  DLE, DLE_WARN, WARNING, ERROR, CRITICAL
    """
    assert level in [DEBUG, PERF, INFO, DSE, DSE_WARN, DLE, DLE_WARN,
                     WARNING, ERROR, CRITICAL]

    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    log(msg, INFO, *args, **kwargs)


def perf(msg, *args, **kwargs):
    log("Performance: %s" % msg, PERF, *args, **kwargs)


def perf_adv(msg, *args, **kwargs):
    log("Performance optimisation spotted: %s" % msg, PERF, *args, **kwargs)


def warning(msg, *args, **kwargs):
    log(msg, WARNING, *args, **kwargs)


def error(msg, *args, **kwargs):
    log(msg, ERROR, *args, **kwargs)


def debug(msg, *args, **kwargs):
    log(msg, DEBUG, *args, **kwargs)


def dse(msg, *args, **kwargs):
    log("DSE: %s" % msg, DSE, *args, **kwargs)


def dse_warning(msg, *args, **kwargs):
    log("DSE: %s" % msg, DSE_WARN, *args, **kwargs)


def dle(msg, *args, **kwargs):
    log("DLE: %s" % msg, DLE, *args, **kwargs)


def dle_warning(msg, *args, **kwargs):
    log("DLE: %s" % msg, DLE_WARN, *args, **kwargs)


def yask(msg, *args, **kwargs):
    log("YASK: %s" % msg, YASK, *args, **kwargs)


def yask_warning(msg, *args, **kwargs):
    log("YASK: %s" % msg, YASK_WARN, *args, **kwargs)


@contextmanager
def bar():
    log('='*89, INFO)
    yield
    log('='*89, INFO)
