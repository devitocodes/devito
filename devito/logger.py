"""The Devito logger."""

import logging
import sys


__all__ = ('set_log_level', 'set_log_noperf', 'log',
           'DEBUG', 'INFO', 'PERF_OK', 'PERF_WARN', 'WARNING', 'ERROR', 'CRITICAL',
           'log', 'warning', 'error', 'RED', 'GREEN', 'BLUE')


logger = logging.getLogger('Devito')
_ch = logging.StreamHandler()
logger.addHandler(_ch)

# Add extra levels between INFO (value=20) and WARNING (value=30)
DEBUG = logging.DEBUG
INFO = logging.INFO
PERF_OK = 28
PERF_WARN = 29
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

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
    PERF_OK: GREEN,
    PERF_WARN: BLUE,
    WARNING: BLUE,
    ERROR: RED,
    CRITICAL: RED
}


def set_log_level(level):
    """
    Set the log level of the Devito logger.

    :param level: accepted values are: DEBUG, INFO, PERF_OK, PERF_WARN, WARNING,
                  ERROR, CRITICAL
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
    :param level: accepted values are: DEBUG, INFO, PERF_OK, PERF_WARN, WARNING,
                  ERROR, CRITICAL
    """
    assert level in [DEBUG, INFO, PERF_OK, PERF_WARN, WARNING, ERROR, CRITICAL]

    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    log(msg, WARNING, *args, **kwargs)


def error(msg, *args, **kwargs):
    log(msg, ERROR, *args, **kwargs)
