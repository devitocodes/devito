"""The Devito logger."""

import logging
import sys

__all__ = ('set_log_level', 'set_log_noperf', 'log',
           'DEBUG', 'INFO', 'AUTOTUNER', 'PERF_OK', 'PERF_WARN',
           'WARNING', 'ERROR', 'CRITICAL',
           'log', 'warning', 'error', 'info_at', 'perfok', 'perfbad',
           'RED', 'GREEN', 'BLUE')


logger = logging.getLogger('Devito')
_ch = logging.StreamHandler()
logger.addHandler(_ch)

# Add extra levels between INFO (value=20) and WARNING (value=30)
DEBUG = logging.DEBUG
INFO = logging.INFO
AUTOTUNER = 27
PERF_OK = 28
PERF_WARN = 29
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(AUTOTUNER, "AUTOTUNER")
logging.addLevelName(PERF_OK, "PERF_OK")
logging.addLevelName(PERF_WARN, "PERF_WARN")

logger.setLevel(INFO)

NOCOLOR = '%s'
RED = '\033[1;37;31m%s\033[0m'
BLUE = '\033[1;37;34m%s\033[0m'
GREEN = '\033[1;37;32m%s\033[0m'

COLORS = {
    DEBUG: RED,
    INFO: NOCOLOR,
    AUTOTUNER: GREEN,
    PERF_OK: GREEN,
    PERF_WARN: BLUE,
    WARNING: BLUE,
    ERROR: RED,
    CRITICAL: RED
}


def set_log_level(level):
    """
    Set the log level of the Devito logger.

    :param level: accepted values are: DEBUG, INFO, AUTOTUNER, PERF_OK, PERF_WARN,
                  WARNING, ERROR, CRITICAL
    """
    logger.setLevel(level)


def set_log_noperf():
    """Do not print performance-related messages."""
    logger.setLevel(WARNING)


def log(msg, level=INFO, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    :param msg: the message to be printed.
    :param level: accepted values are: DEBUG, INFO, AUTOTUNER, PERF_OK, PERF_WARN,
                  WARNING, ERROR, CRITICAL
    """
    assert level in [DEBUG, INFO, AUTOTUNER, PERF_OK, PERF_WARN,
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


def perfok(msg, *args, **kwargs):
    log(msg, PERF_OK, *args, **kwargs)


def perfbad(msg, *args, **kwargs):
    log(msg, PERF_WARN, *args, **kwargs)
