from codepy.toolchain import GCCToolchain
from codepy.jit import extension_file_from_string
from os import environ, path, mkdir, getuid
from tempfile import gettempdir
import numpy.ctypeslib as npct


__all__ = ['get_tmp_dir', 'get_compiler_from_env',
           'jit_compile', 'load', 'jit_compile_and_load',
           'GNUCompiler']


class GNUCompiler(GCCToolchain):
    """Set of standard compiler flags for the GCC toolchain"""

    def __init__(self):
        self.cc = 'g++'
        self.ld = 'g++'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall']
        self.ldflags = ['-shared']
        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.defines = []
        self.undefines = []

    def __str__(self):
        return self.__class__.__name__


# Registry dict for deriving Compiler classes according to
# environment variable DEVITO_ARCH. Developers should add
# new compiler classes here and provide a description in
# the docstring of get_compiler_from_env().
compiler_registry = {
    'gcc': GNUCompiler, 'gnu': GNUCompiler,
}


def get_compiler_from_env():
    """Derive compiler class and settings from environment variables

    The key environment variable DEVITO_ARCH supports the following values:
     * 'gcc' or 'gnu' - (Default) Standard GNU compiler toolchain
    """
    key = environ.get('DEVITO_ARCH', 'gnu')
    return compiler_registry[key.lower()]()


def get_tmp_dir():
    "Return path to a devito-specific tmp directory"""
    tmpdir = path.join(gettempdir(), "devito-%s" % getuid())
    if not path.exists(tmpdir):
        mkdir(tmpdir)
    return tmpdir


def jit_compile(ccode, basename, compiler=GNUCompiler):
    """JIT compiles the given ccode and returns the lib filepath"""
    src_file = "%s.cpp" % basename
    lib_file = "%s.so" % basename
    print "%s: Compiling %s" % (compiler, src_file)
    extension_file_from_string(toolchain=compiler, ext_file=lib_file,
                               source_string=ccode, source_name=src_file)
    return lib_file


def load(basename):
    """Load a compiled library and return the ctypes object"""
    lib_file = "%s.so" % basename
    return npct.load_library(lib_file, '.')


def jit_compile_and_load(ccode, basename, compiler=GNUCompiler):
    """JIT compile the given ccode and return the loaded library"""
    jit_compile(ccode, basename, compiler=compiler)
    return load(basename)
