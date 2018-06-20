import os
from pathlib import Path
from tempfile import gettempdir

__all__ = ['change_directory', 'make_tempdir']


class change_directory(object):
    """
    Context manager for changing the current working directory.

    Adapted from: ::

        https://stackoverflow.com/questions/431684/how-do-i-cd-in-python/
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def make_tempdir(prefix=None):
    """Create a temporary directory having a deterministic name. The directory
    is created within the default OS temporary directory."""
    if prefix is None:
        name = 'devito-uid%s' % os.getuid()
    else:
        name = 'devito-%s-uid%s' % (str(prefix), os.getuid())
    tmpdir = Path(gettempdir()).joinpath(name)
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir
