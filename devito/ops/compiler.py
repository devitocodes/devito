import os
import subprocess
from devito.logger import warning
from devito.parameters import configuration


__all__ = ['CompilerOPS']


class CompilerOPS(configuration['compiler'].__class__):
    def __init__(self, *args, **kwargs):
        self._ops_install_path = os.environ.get('OPS_INSTALL_PATH')
        if not self._ops_install_path:
            warning("Couldn't find OPS_INSTALL_PATH \
                environment variable, please check your OPS installation")
        super(CompilerOPS, self).__init__(*args, **kwargs)

    def prepare_ops(self, soname, ccode, hcode):
        # Creating files
        file_name = str(self.get_jit_dir().joinpath(soname))
        h_file = open("%s.h" % (file_name), "w")
        c_file = open("%s.cpp" % (file_name), "w")

        c_file.write(ccode)
        h_file.write(hcode)

        c_file.close()
        h_file.close()

        # Calling OPS Translator
        translator = '%s/../ops_translator/c/ops.py' % (self._ops_install_path)
        subprocess.run([translator, c_file.name], cwd=self.get_jit_dir())
