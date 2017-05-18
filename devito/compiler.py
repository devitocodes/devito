from functools import partial
from hashlib import sha1
from os import environ, getuid, mkdir, path
from tempfile import gettempdir
from time import time
from sys import platform

import numpy.ctypeslib as npct
from cgen import Pragma
from codepy.jit import extension_file_from_string
from codepy.toolchain import GCCToolchain

from devito.logger import log

__all__ = ['get_tmp_dir', 'get_compiler_from_env', 'jit_compile', 'load',
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
        * :data:`self.src_ext`
        * :data:`self.lib_ext`
        * :data:`self.undefines`
        * :data:`self.pragma_ivdep`

    """

    fields = ['cc', 'ld']

    def __init__(self, openmp=False, **kwargs):
        self.cc = 'unknown'
        self.ld = 'unknown'
        self.cflags = []
        self.ldflags = []
        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.defines = []
        self.undefines = []
        self.src_ext = 'c'
        self.lib_ext = 'so'
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
        self.version = kwargs.get('version', None)
        self.cc = 'gcc' if self.version is None else 'gcc-%s' % self.version
        self.ld = 'gcc' if self.version is None else 'gcc-%s' % self.version
        self.cflags = ['-O3', '-g', '-march=native', '-fPIC', '-Wall', '-std=c99',
                       '-Wno-unused-result', '-Wno-unused-variable']
        self.ldflags = ['-shared']

        if self.openmp:
            self.ldflags += ['-fopenmp']
        self.pragma_ivdep = [Pragma('GCC ivdep')]


class GNUCompilerNoAVX(GNUCompiler):
    """Set of compiler flags for GCC but with AVX suppressed. This is
    a work around for a known gcc bug on MAC OS."

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    def __init__(self, *args, **kwargs):
        super(GNUCompilerNoAVX, self).__init__(*args, **kwargs)
        self.cflags = ['-O3', '-g', '-march=native', '-mno-avx', '-fPIC', '-Wall',
                       '-std=c99', '-Wno-unused-result', '-Wno-unused-variable']


class ClangCompiler(Compiler):
    """Set of standard compiler flags for the clang toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Genrates warning if openmp is disabled.
    """

    def __init__(self, *args, **kwargs):
        super(ClangCompiler, self).__init__(*args, **kwargs)
        self.cc = 'clang'
        self.ld = 'clang'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99']
        self.ldflags = ['-shared']
        self.lib_ext = 'dylib'

        if self.openmp:
            log("WARNING: Disabling OpenMP because clang does not support it.")
            self.openmp = False


class IntelCompiler(Compiler):
    """Set of standard compiler flags for the Intel toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    def __init__(self, *args, **kwargs):
        super(IntelCompiler, self).__init__(*args, **kwargs)
        self.cc = 'icc'
        self.ld = 'icc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99', "-xhost"]
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
        self.cc = 'icc'
        self.ld = 'icc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99', "-mmic"]
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
        self.cc = 'icc'
        self.ld = 'icc'
        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99', "-xMIC-AVX512"]
        self.ldflags = ['-shared']
        if self.openmp:
            self.ldflags += ['-qopenmp']
        else:
            log("WARNING: Running on Intel KNL without OpenMP is highly discouraged")


class CustomCompiler(Compiler):
    """Custom compiler based on standard environment flags

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Currently honours CC, CFLAGS, LD and LDFLAGS, with defaults similar
    to the default GNU settings. If DEVITO_ARCH is enabled, the OpenMP linker
    flags are read from OMP_LDFLAGS or otherwise default to ``-fopenmp``.
    """

    def __init__(self, *args, **kwargs):
        super(CustomCompiler, self).__init__(*args, **kwargs)
        self.cc = environ.get('CC', 'gcc')
        self.ld = environ.get('LD', 'gcc')
        default = '-O3 -g -march=native -fPIC -Wall -std=c99'
        self.cflags = environ.get('CFLAGS', default).split(' ')
        self.ldflags = environ.get('LDFLAGS', '-shared').split(' ')
        self.pragma_ivdep = [Pragma(environ.get('DEVITO_IVDEP', 'GCC ivdep'))]
        if self.openmp:
            self.ldflags += environ.get('OMP_LDFLAGS', '-fopenmp').split(' ')


# Registry dict for deriving Compiler classes according to
# environment variable DEVITO_ARCH. Developers should add
# new compiler classes here and provide a description in
# the docstring of get_compiler_from_env().
compiler_registry = {
    'gcc': GNUCompiler, 'gnu': GNUCompiler,
    'gcc-4.9': partial(GNUCompiler, version='4.9'),
    'gcc-5': partial(GNUCompiler, version='5'),
    'gcc-noavx': GNUCompilerNoAVX, 'gnu-noavx': GNUCompilerNoAVX,
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
     * 'gcc-4.9' - GNU compiler toolchain version 4.9
     * 'gcc-5' - GNU compiler toolchain version 5
     * 'clang' or 'osx' - Clang compiler toolchain for Mac OSX
     * 'intel' or 'icpc' - Intel compiler toolchain via icpc
     * 'intel-mic' or 'mic' - Intel MIC using offload mode via pymic

    Additionally, the variable DEVITO_OPENMP can be used to enable OpenMP
    parallelisation on by setting it to "1".
    """
    key = environ.get('DEVITO_ARCH', None)
    openmp = environ.get('DEVITO_OPENMP', "0") == "1"
    if key is None:
        return CustomCompiler(openmp=openmp)
    else:
        return compiler_registry[key.lower()](openmp=openmp)


def get_tmp_dir():
    """Function to get a temp directory.

    :return: Path to a devito-specific tmp directory
    """
    tmpdir = path.join(gettempdir(), "devito-%s" % getuid())

    if not path.exists(tmpdir):
        mkdir(tmpdir)

    return tmpdir


def load(basename, compiler=GNUCompiler):
    """Load a compiled library

    :param basename: Name of the .so file.
    :param compiler: The toolchain used for compilation. GNUCompiler by default.
    :return: The loaded library.

    Note: If the provided compiler is of type `IntelMICCompiler`
    we utilise the `pymic` package to manage device streams.
    """
    if isinstance(compiler, IntelMICCompiler):
        compiler._device = compiler._mic.devices[0]
        compiler._stream = compiler._device.get_default_stream()
        # The name with the extension is only used for MIC
        lib_file = "%s.%s" % (basename, compiler.lib_ext)
        return compiler._device.load_library(lib_file)

    return npct.load_library(basename, '.')


def jit_compile(ccode, compiler=GNUCompiler):
    """JIT compile the given ccode.

    :param ccode: String of C source code.
    :param compiler: The toolchain used for compilation. GNUCompiler by default.

    :return: The name of the compilation unit.
    """

    hash_key = sha1(str(ccode).encode()).hexdigest()
    basename = path.join(get_tmp_dir(), hash_key)

    src_file = "%s.%s" % (basename, compiler.src_ext)
    if platform == "linux" or platform == "linux2":
        lib_file = "%s.so" % basename
    elif platform == "darwin":
        lib_file = "%s.dylib" % basename
    elif platform == "win32" or platform == "win64":
        lib_file = "%s.dll" % basename

    tic = time()
    extension_file_from_string(toolchain=compiler, ext_file=lib_file,
                               source_string=ccode, source_name=src_file)
    toc = time()
    log("%s: compiled %s [%.2f s]" % (compiler, src_file, toc-tic))

    return basename
