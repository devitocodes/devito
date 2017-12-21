from functools import partial
from hashlib import sha1
from os import environ, path
from tempfile import mkdtemp
from time import time
from sys import platform
from distutils import version
import subprocess

import numpy.ctypeslib as npct
from codepy.jit import extension_file_from_string
from codepy.toolchain import GCCToolchain

from devito.exceptions import CompilationError
from devito.logger import log
from devito.parameters import configuration
from devito.tools import change_directory, sniff_compiler_version

__all__ = ['jit_compile', 'load', 'make', 'GNUCompiler']


class Compiler(GCCToolchain):
    """Base class for all compiler classes.

    The base class defaults all compiler specific settings to empty lists.
    Preset configurations can be built by inheriting from `Compiler` and setting
    specific flags to the desired values, eg.:

    def class MyCompiler(Compiler):
        def __init__(self):
            self.cc = 'mycompiler'
            self.cflags += ['list', 'of', 'all', 'compiler', 'flags']

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

    Two additional parameters may be passed.
    :param suffix: A string indicating a specific compiler version available on
                   the system. For example, assuming ``compiler=gcc`` and
                   ``suffix='4.9'``, then the ``gcc-4.9`` program will be used
                   to compile the generated code.
    :param cpp: Defaults to False. Pass True to set up for C++ compilation,
                instead of C compilation.
    """

    fields = {'cc', 'ld'}

    CC = 'unknown'
    CPP = 'unknown'

    def __init__(self, **kwargs):
        super(Compiler, self).__init__(**kwargs)

        self.suffix = kwargs.get('suffix')
        self.cc = self.CC if kwargs.get('cpp', False) is False else self.CPP
        self.cc = self.cc if self.suffix is None else ('%s-%s' % (self.cc, self.suffix))
        self.ld = self.cc  # Wanted by the superclass

        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99']
        self.ldflags = ['-shared']

        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.defines = []
        self.undefines = []

        self.src_ext = 'c' if kwargs.get('cpp', False) is False else 'cpp'
        self.lib_ext = 'so'

        if self.suffix is not None:
            try:
                self.version = version.StrictVersion(str(float(self.suffix)))
            except (TypeError, ValueError):
                self.version = version.LooseVersion(self.suffix)
        else:
            # Knowing the version may still be useful to pick supported flags
            self.version = sniff_compiler_version(self.CC)

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "DevitoJITCompiler[%s]" % self.__class__.__name__


class GNUCompiler(Compiler):
    """Set of standard compiler flags for the GCC toolchain."""

    CC = 'gcc'
    CPP = 'g++'

    def __init__(self, *args, **kwargs):
        super(GNUCompiler, self).__init__(*args, **kwargs)
        self.cflags += ['-march=native', '-Wno-unused-result', '-Wno-unused-variable',
                        '-Wno-unused-but-set-variable']
        try:
            if self.version >= version.StrictVersion("4.9.0"):
                # Append the openmp flag regardless of configuration['openmp'],
                # since GCC4.9 and later versions implement OpenMP 4.0, hence
                # they support `#pragma omp simd`
                self.ldflags += ['-fopenmp']
        except (TypeError, ValueError):
            if configuration['openmp']:
                self.ldflags += ['-fopenmp']


class GNUCompilerNoAVX(GNUCompiler):
    """Set of compiler flags for GCC but with AVX suppressed. This is
    a work around for a known gcc bug on MAC OS."

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    def __init__(self, *args, **kwargs):
        super(GNUCompilerNoAVX, self).__init__(*args, **kwargs)
        self.cflags += ['-mno-avx']


class ClangCompiler(Compiler):
    """Set of standard compiler flags for the clang toolchain

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Genrates warning if openmp is disabled.
    """

    CC = 'clang'
    CPP = 'clang++'

    def __init__(self, *args, **kwargs):
        super(ClangCompiler, self).__init__(*args, **kwargs)
        self.lib_ext = 'dylib'


class IntelCompiler(Compiler):
    """Set of standard compiler flags for the Intel toolchain.

    :param openmp: Boolean indicating if openmp is enabled. False by default
    """

    CC = 'icc'
    CPP = 'icpc'

    def __init__(self, *args, **kwargs):
        super(IntelCompiler, self).__init__(*args, **kwargs)
        self.cflags += ["-xhost"]
        if configuration['platform'] == 'skx':
            # Systematically use 512-bit vectors on skylake
            self.cflags += ["-qopt-zmm-usage=high"]
        try:
            if self.version >= version.StrictVersion("15.0.0"):
                # Append the openmp flag regardless of configuration['openmp'],
                # since icc15 and later versions implement OpenMP 4.0, hence
                # they support `#pragma omp simd`
                self.ldflags += ['-qopenmp']
        except (TypeError, ValueError):
            if configuration['openmp']:
                # Note: fopenmp, not qopenmp, is what is needed by icc versions < 15.0
                self.ldflags += ['-fopenmp']


class IntelKNLCompiler(IntelCompiler):
    """Set of standard compiler flags for the clang toolchain"""

    def __init__(self, *args, **kwargs):
        super(IntelKNLCompiler, self).__init__(*args, **kwargs)
        self.cflags += ["-xMIC-AVX512"]
        if not configuration['openmp']:
            log("WARNING: Running on Intel KNL without OpenMP is highly discouraged")


class CustomCompiler(Compiler):
    """Custom compiler based on standard environment flags

    :param openmp: Boolean indicating if openmp is enabled. False by default

    Note: Currently honours CC, CFLAGS and LDFLAGS, with defaults similar
    to the default GNU settings. If DEVITO_ARCH is enabled, the OpenMP linker
    flags are read from OMP_LDFLAGS or otherwise default to ``-fopenmp``.
    """

    CC = environ.get('CC', 'gcc')
    CPP = environ.get('CPP', 'g++')

    def __init__(self, *args, **kwargs):
        super(CustomCompiler, self).__init__(*args, **kwargs)
        default = '-O3 -g -march=native -fPIC -Wall -std=c99'
        self.cflags = environ.get('CFLAGS', default).split(' ')
        self.ldflags = environ.get('LDFLAGS', '-shared').split(' ')
        if configuration['openmp']:
            self.ldflags += environ.get('OMP_LDFLAGS', '-fopenmp').split(' ')


def get_tmp_dir():
    """Function to get a temp directory.

    :return: Path to a devito-specific tmp directory
    """
    global _devito_compiler_tmpdir
    try:
        path.exists(_devito_compiler_tmpdir)
    except:
        _devito_compiler_tmpdir = mkdtemp(prefix="devito-")

    return _devito_compiler_tmpdir


def load(basename, compiler):
    """Load a compiled library

    :param basename: Name of the .so file.
    :param compiler: The toolchain used for compilation.
    :return: The loaded library.
    """
    return npct.load_library(basename, '.')


def jit_compile(ccode, compiler):
    """JIT compile the given ccode.

    :param ccode: String of C source code.
    :param compiler: The toolchain used for compilation.

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
                               source_string=ccode, source_name=src_file,
                               debug=configuration['debug_compiler'])
    toc = time()
    log("%s: compiled %s [%.2f s]" % (compiler, src_file, toc-tic))

    return basename


def make(loc, args):
    """
    Invoke ``make`` command from within ``loc`` with arguments ``args``.
    """
    hash_key = sha1((loc + str(args)).encode()).hexdigest()
    logfile = path.join(get_tmp_dir(), "%s.log" % hash_key)
    errfile = path.join(get_tmp_dir(), "%s.err" % hash_key)

    with change_directory(loc):
        with open(logfile, "w") as log:
            with open(errfile, "w") as err:

                command = ['make'] + args
                log.write("Compilation command:\n")
                log.write(" ".join(command))
                log.write("\n\n")
                try:
                    subprocess.check_call(command, stderr=err, stdout=log)
                except subprocess.CalledProcessError as e:
                    raise CompilationError('Command "%s" return error status %d. '
                                           'Unable to compile code.\n'
                                           'Compile log in %s\n'
                                           'Compile errors in %s\n' %
                                           (e.cmd, e.returncode, logfile, errfile))


# Registry dict for deriving Compiler classes according to the environment variable
# DEVITO_ARCH. Developers should add new compiler classes here.
compiler_registry = {
    'custom': CustomCompiler,
    'gnu': GNUCompiler,
    'gcc': GNUCompiler,
    'gcc-noavx': GNUCompilerNoAVX,
    'gnu-noavx': GNUCompilerNoAVX,
    'clang': ClangCompiler,
    'osx': ClangCompiler,
    'intel': IntelCompiler,
    'icpc': IntelCompiler,
    'icc': IntelCompiler,
    'intel-knl': IntelKNLCompiler,
    'knl': IntelKNLCompiler,
}
compiler_registry.update({'gcc-%s' % i: partial(GNUCompiler, suffix=i)
                          for i in ['4.9', '5', '6', '7']})
