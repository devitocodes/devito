import atexit
import os
import sys
from signal import SIGABRT, SIGINT, SIGSEGV, SIGTERM, signal
from tempfile import gettempdir

from devito.logger import warning


class MemmapManager():
    """Class for managing all memmap related settings"""
    # used to enable memmap as default
    _use_memmap = False
    # determine whether created files are deleted by default
    _delete_file = True
    # flag for registering exit func
    _registered = False
    # default directory to store memmap file
    _default_disk_path = os.path.join(gettempdir(), "devito_disk")
    # contains str name of all memmap file created
    _created_data = {}
    # unique id
    _id = 0
    # exit code used for normal exiting
    _default_exit_code = 0

    @staticmethod
    def set_memmap(memmap):
        """Call this method to set default value of memmap"""
        MemmapManager._use_memmap = memmap

    @staticmethod
    def set_delete_file(delete):
        """Call this method to set default flag contolling whether to delete
        crated files.
        """
        MemmapManager._delete_file = delete

    @staticmethod
    def set_default_disk_path(default_disk_path):
        """Call this method to change the default disk path for memmap"""
        MemmapManager._default_disk_path = default_disk_path

    @staticmethod
    def setup(data_self, *args, **kwargs):
        """This method is used to setup memmap parameters for data classes.

        :param name: Name of data
        :param memmap: Boolean indicates whether memmap is used. Optional
        :param disk_path: String indicates directory to create memmap file. Optional
        :param delete_file: Boolean indicates whether to delete created file. Optional

        Note: If memmap, disk_path or delete_file are not provided, the default values
        are used.
        """
        data_self.memmap = kwargs.get('memmap', MemmapManager._use_memmap)

        if data_self.memmap:
            disk_path = kwargs.get('disk_path', MemmapManager._default_disk_path)

            if not os.path.exists(disk_path):
                os.makedirs(disk_path)

            data_self.f = "%s/data_%s_%s" % (disk_path, kwargs.get('name'),
                                             str(MemmapManager._id))
            MemmapManager._id += 1
            data_self.delete_file = kwargs.get('delete_file', MemmapManager._delete_file)
            MemmapManager._created_data[data_self.f] = data_self.delete_file

            if not MemmapManager._registered:
                MemmapManager._register_remove_memmap_file_signal()
                MemmapManager._registered = True

    @staticmethod
    def _remove_memmap_file():
        """This method is used to clean up memmap file"""
        for f in MemmapManager._created_data:
            if MemmapManager._created_data[f]:
                try:
                    os.remove(f)
                except OSError:
                    warning("error removing %s it may be already removed, skipping", f)
            else:
                warning("file %s has been left", f)

    @staticmethod
    def _remove_memmap_file_on_signal(*args):
        """This method is used to clean memmap file on signal, internal method"""
        sys.exit(MemmapManager._default_exit_code)

    @staticmethod
    def _register_remove_memmap_file_signal():
        """This method is used to register clean up method for chosen signals"""
        atexit.register(MemmapManager._remove_memmap_file)

        for sig in (SIGABRT, SIGINT, SIGSEGV, SIGTERM):
            signal(sig, MemmapManager._remove_memmap_file_on_signal)
