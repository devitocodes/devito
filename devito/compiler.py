from os import environ, getuid, mkdir, path
from tempfile import gettempdir

import numpy.ctypeslib as npct
from cgen import Pragma
from codepy.jit import extension_file_from_string
from codepy.toolchain import GCCToolchain

from devito.logger import log

__all__ = ['get_tmp_dir', 'get_compiler_from_env',
           'jit_compile', 'load', 'jit_compile_and_load',
           'GNUCompiler']


class Compiler(GCCToolchain):
    """Base class for all compiler classes.

    :param openmp: Boolean indicating if openmp is enabled. False by default

    The base class defaults all compiler specific settings to empty lists.
    Preset configurations can be built by inheriting from `Compiler` and setting
    specific flags to the desired values, eg.:

    def class MyCompiler(Compiler):
        def __init__(self):
            self.cc = 'mycompiler'
            self.cflags = ['list', 'of', 'all', 'compiler', 'flags']

    The flags that can be set are:
        * :data:`self.cc`
        * :data:`self.cflag`
        * :data:`self.ldflags`
        * :data:`self.include_dirs`
        * :data:`self.libraries`
        * :data:`self.library_dirs`
        * :data:`self.defines`
        * :data:`self.undefines`
        * :data:`self.pragma_ivdep`

    """
    def __init__(self, openmp=False):
        self.cc = 'unknown'
        self.ld = 'unknown'
        self.cflags = []
        self.ldflags = []
        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.defines = []
        self.undefines = []
        # Devito-specific flags and properties
        self.openmp = openmp
        self.pragma_ivdep = [Pragma('ivdep')]
        self.pragma_nontemporal = []
        self.pragma_aligned = "omp simd aligned"

    def __str__(self):
        return self.__class__.__name__


class GNUCompiler(Compiler):
    """Set of standard compiler flags for the GCC toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    def __init__(self, *args, **kwargs):
        super(GNUCompiler, self).__init__(*args, **kwargs)
        self.cc = 'g++'
        self.ld = 'g++'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall']
        self.ldflags = ['-shared']

        if self.openmp:
            self.ldflags += ['-fopenmp']
        self.pragma_ivdep = [Pragma('GCC ivdep')]


class ClangCompiler(Compiler):
    """Set of standard compiler flags for the clang toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Genrates warning if openmp is disabled.
    """

    def __init__(self, *args, **kwargs):
        super(ClangCompiler, self).__init__(*args, **kwargs)
        self.cc = 'clang'
        self.ld = 'clang'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall']
        self.ldflags = ['-shared']

        if self.openmp:
            log("WARNING: Disabling OpenMP because clang does not support it.")
            self.openmp = False


class IntelCompiler(Compiler):
    """Set of standard compiler flags for the Intel toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    def __init__(self, *args, **kwargs):
        super(IntelCompiler, self).__init__(*args, **kwargs)
        self.cc = 'icpc'
        self.ld = 'icpc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', "-xhost"]
        self.ldflags = ['-shared']

        if self.openmp:
            self.ldflags += ['-qopenmp']
        self.pragma_nontemporal = [Pragma('vector nontemporal')]


class IntelMICCompiler(Compiler):
    """Set of standard compiler flags for the IntelMIC toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Generates warning if openmp is disabled.
    """

    def __init__(self, *args, **kwargs):
        super(IntelMICCompiler, self).__init__(*args, **kwargs)
        self.cc = 'icpc'
        self.ld = 'icpc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', "-mmic"]
        self.ldflags = ['-shared']

        if self.openmp:
            self.ldflags += ['-qopenmp']
        else:
            log("WARNING: Running on Intel MIC without OpenMP is highly discouraged")
        self._mic = __import__('pymic')


class IntelKNLCompiler(Compiler):
    """Set of standard compiler flags for the clang toolchain"""

    def __init__(self, *args, **kwargs):
        super(IntelKNLCompiler, self).__init__(*args, **kwargs)
        self.cc = 'icpc'
        self.ld = 'icpc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', "-xMIC-AVX512"]
        self.ldflags = ['-shared']
        if self.openmp:
            self.ldflags += ['-qopenmp']
        else:
            log("WARNING: Running on Intel KNL without OpenMP is highly discouraged")

# Registry dict for deriving Compiler classes according to
# environment variable DEVITO_ARCH. Developers should add
# new compiler classes here and provide a description in
# the docstring of get_compiler_from_env().
compiler_registry = {
    'gcc': GNUCompiler, 'gnu': GNUCompiler,
    'clang': ClangCompiler, 'osx': ClangCompiler,
    'intel': IntelCompiler, 'icpc': IntelCompiler,
    'icc': IntelCompiler,
    'intel-mic': IntelMICCompiler, 'mic': IntelMICCompiler,
    'intel-knl': IntelKNLCompiler, 'knl': IntelKNLCompiler,
}


def get_compiler_from_env():
    """Derive compiler class and settings from environment variables

    :return: The compiler indicated by the environment variable.

    The key environment variable DEVITO_ARCH supports the following values:
     * 'gcc' or 'gnu' - (Default) Standard GNU compiler toolchain
     * 'clang' or 'osx' - Clang compiler toolchain for Mac OSX
     * 'intel' or 'icpc' - Intel compiler toolchain via icpc
     * 'intel-mic' or 'mic' - Intel MIC using offload mode via pymic

    Additionally, the variable DEVITO_OPENMP can be used to enable OpenMP
    parallelisation on by setting it to "1".
    """
    key = environ.get('DEVITO_ARCH', 'gnu')
    openmp = environ.get('DEVITO_OPENMP', "0") == "1"

    return compiler_registry[key.lower()](openmp=openmp)


def get_tmp_dir():
    """Function to get a temp directory.

    :return: Path to a devito-specific tmp directory
    """
    tmpdir = path.join(gettempdir(), "devito-%s" % getuid())

    if not path.exists(tmpdir):
        mkdir(tmpdir)

    return tmpdir


def jit_compile(ccode, basename, compiler=GNUCompiler):
    """JIT compiles the given ccode and returns the lib filepath.

    :param ccode: String of C source code.
    :param basename: The string used to name various files for this compilation.
    :param compiler: The toolchain used for compilation. GNUCompiler by default.
    :return: Path to compiled lib
    """
    src_file = "%s.cpp" % basename
    lib_file = "%s.so" % basename
    log("%s: Compiling %s" % (compiler, src_file))
    extension_file_from_string(toolchain=compiler, ext_file=lib_file,
                               source_string=ccode, source_name=src_file)

    return lib_file


def load(basename, compiler=GNUCompiler):
    """Load a compiled library

    :param basename: Name of the .so file.
    :param compiler: The toolchain used for compilation. GNUCompiler by default.
    :return: The loaded library.

    Note: If the provided compiler is of type `IntelMICCompiler`
    we utilise the `pymic` package to manage device streams.
    """
    lib_file = "%s.so" % basename

    if isinstance(compiler, IntelMICCompiler):
        compiler._device = compiler._mic.devices[0]
        compiler._stream = compiler._device.get_default_stream()

        return compiler._device.load_library(lib_file)

    return npct.load_library(lib_file, '.')


def jit_compile_and_load(ccode, basename, compiler=GNUCompiler):
    """JIT compile the given ccode

    :param ccode: String of C source code.
    :param basename: The string used to name various files for this compilation.
    :param compiler: The toolchain used for compilation. GNUCompiler by default.
    :return: The compiled library.
    """
    jit_compile(ccode, basename, compiler=compiler)

    return load(basename, compiler=compiler)
