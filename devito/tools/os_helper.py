import os
from contextlib import suppress
from pathlib import Path
from tempfile import gettempdir, mkstemp

__all__ = ['atomic_write', 'change_directory', 'make_tempdir']


class change_directory:
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
        name = f'devito-uid{os.getuid()}'
    else:
        name = f'devito-{str(prefix)}-uid{os.getuid()}'
    tmpdir = Path(gettempdir()).joinpath(name)
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def atomic_write(path, data, mode='w'):
    """
    Write `data` to `path` through a private temporary file in the same
    directory, then rename it into place. The rename is atomic, so a concurrent
    reader never sees a partial file: it reads either the previous content or
    the new one, and a reader that already opened the previous file keeps
    reading that one in full. The temporary file is removed if the write fails.

    Note: the resulting file has the restrictive mode of the temporary file
    (0600, further restricted by the process umask).
    """
    path = Path(path)
    fd, tmpfile = mkstemp(dir=path.parent, prefix=f'{path.name}.', suffix='.tmp')
    try:
        try:
            f = os.fdopen(fd, mode)
        except BaseException:
            os.close(fd)
            raise
        with f:
            f.write(data)
        os.replace(tmpfile, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(tmpfile)
        raise
    return path
