"""The Devito logger."""

import logging
import sys
from contextlib import contextmanager

__all__ = ('set_log_level', 'set_log_noperf', 'is_log_enabled_for',
           'log', 'warning', 'error', 'perf', 'perf_adv',
           'RED', 'GREEN', 'BLUE')


logger = logging.getLogger('Devito')
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

# Add extra logging levels (note: INFO has value=20, WARNING has value=30)
DEBUG = logging.DEBUG
PERF = 19
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(PERF, "PERF")

logger_registry = {
    'DEBUG': DEBUG,
    'PERF': PERF,
    'INFO': INFO,
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
    INFO: NOCOLOR,
    WARNING: BLUE,
    ERROR: RED,
    CRITICAL: RED
}


def _set_log_level(level):
    """
    Set the level of the Devito logger.
    """
    if level not in logger_registry:
        raise ValueError("Illegal logging level %s" % level)

    logger.setLevel(level)


def set_log_level(level, comm=None):
    """
    Set the level of the Devito logger.

    Parameters
    ----------
    level : int
        The logging level. Accepted values are: ``DEBUG, PERF, INFO, WARNING,
        ERROR, CRITICAL``.
    comm : MPI communicator, optional
        An MPI communicator the logger should be collective over. If provided, only
        rank-0 on that communicator will write to the registered handlers, other
        ranks will use a `logging.NullHandler`.  By default, ``comm`` is set
        to ``None``, so all ranks will use the default handlers.  This could be
        used, for example, if one wants to log to one file per rank.
    """
    from devito import configuration

    if comm is not None:
        if comm.rank != 0:
            logger.removeHandler(stream_handler)
            logger.addHandler(logging.NullHandler())

    # Triggers a callback to `_set_log_level`
    configuration['log-level'] = level


def set_log_noperf():
    """Do not print performance-related messages."""
    logger.setLevel(WARNING)


def is_log_enabled_for(level):
    """
    Wrapper around `logging.isEnabledFor`. Indicates if a message of severity
    level would be processed by this logger.
    """
    return logger.isEnabledFor(logger_registry[level])


def log(msg, level=INFO, *args, **kwargs):
    """
    Wrapper of the main Python's logging function. Print 'msg % args' with
    the severity 'level'.

    Parameters
    ----------
    msg : str
        The message to be printed.
    level : int
        The logging level. Accepted values are: ``DEBUG, PERF, INFO, WARNING,
        ERROR, CRITICAL``.
    """
    color = COLORS[level] if sys.stdout.isatty() and sys.stderr.isatty() else '%s'
    logger.log(level, color % msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    log(msg, INFO, *args, **kwargs)


def perf(msg, *args, **kwargs):
    log(msg, PERF, *args, **kwargs)


def perf_adv(msg, *args, **kwargs):
    log("Potential optimisation missed: %s" % msg, PERF, *args, **kwargs)


def warning(msg, *args, **kwargs):
    log(msg, WARNING, *args, **kwargs)


def error(msg, *args, **kwargs):
    log(msg, ERROR, *args, **kwargs)


def debug(msg, *args, **kwargs):
    log(msg, DEBUG, *args, **kwargs)


@contextmanager
def bar():
    log('='*89, INFO)
    yield
    log('='*89, INFO)
