"""The Devito logger."""

import logging
import sys
from contextlib import contextmanager

from devito.parameters import configuration

__all__ = ('set_log_level', 'set_log_noperf', 'log',
           'DEBUG', 'AUTOTUNER', 'YASK', 'YASK_WARN', 'INFO', 'DSE', 'DSE_WARN',
           'DLE', 'DLE_WARN', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'warning', 'error', 'info_at', 'dse', 'dse_warning',
           'dle', 'dle_warning',
           'RED', 'GREEN', 'BLUE')


logger = logging.getLogger('Devito')
_ch = logging.StreamHandler()
logger.addHandler(_ch)

# Add extra logging levels (note: INFO has value=20, WARNING has value=30)
DEBUG = logging.DEBUG
AUTOTUNER = 19
YASK = 19
YASK_WARN = YASK
DSE = 18
DLE = DSE
DSE_WARN = 19
DLE_WARN = DSE_WARN
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(AUTOTUNER, "AUTOTUNER")
logging.addLevelName(YASK, "YASK")
logging.addLevelName(YASK_WARN, "YASK_WARN")
logging.addLevelName(DSE, "DSE")
logging.addLevelName(DSE_WARN, "DSE_WARN")
logging.addLevelName(DSE, "DLE")
logging.addLevelName(DSE_WARN, "DLE_WARN")

logger_registry = {
    'DEBUG': DEBUG,
    'AUTOTUNER': AUTOTUNER,
    'YASK': YASK,
    'YASK_WARN': YASK_WARN,
    'INFO': INFO,
    'DSE': DSE,
    'DLE': DSE,
    'DSE_WARN': DSE_WARN,
    'DLE_WARN': DSE_WARN,
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
    AUTOTUNER: GREEN,
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

    :param level: accepted values are: DEBUG, INFO, AUTOTUNER, DSE, DSE_WARN,
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


def log(msg, level=INFO, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    :param msg: the message to be printed.
    :param level: accepted values are: DEBUG, INFO, AUTOTUNER, DSE, DSE_WARN,
                  DLE, DLE_WARN, WARNING, ERROR, CRITICAL
    """
    assert level in [DEBUG, INFO, AUTOTUNER, DSE, DSE_WARN, DLE, DLE_WARN,
                     WARNING, ERROR, CRITICAL]

    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    log(msg, INFO, *args, **kwargs)


def info_at(msg, *args, **kwargs):
    log("AutoTuner: %s" % msg, AUTOTUNER, *args, **kwargs)


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
