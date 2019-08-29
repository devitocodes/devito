import os
# Import subprocess

from devito.parameters import configuration


__all__ = ['CompilerOPS']


class CompilerOPS(configuration['compiler'].__class__):
    def __init__(self, *args, **kwargs):
        self._ops_install_path = os.environ.get('OPS_INSTALL_PATH')
        if not self._ops_install_path:
            raise ValueError("Couldn't find OPS_INSTALL_PATH \
                environment variable, please check your OPS installation")
        super(CompilerOPS, self).__init__(*args, **kwargs)

    def create_files(self, soname, ccode, hcode):
        file_name = str(self.get_jit_dir().joinpath(soname))
        h_file = open("%s.h" % (file_name), "w")
        c_file = open("%s.c" % (file_name), "w")

        h_file.write(hcode)
        c_file.write(ccode)
        # Call from directory off the file with
        # run = self._ops_install_path +'/../ops_translator/c/ops.py'
        # subprocess.run(['python2',run,str(c_file)])
