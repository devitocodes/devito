from functools import partial
from hashlib import sha1
from os import environ, path, makedirs
from packaging.version import Version
from subprocess import (DEVNULL, PIPE, CalledProcessError, check_output,
                        check_call, run)
import platform
import warnings
import time

import numpy.ctypeslib as npct
from codepy.jit import compile_from_string
from codepy.toolchain import (GCCToolchain,
                              call_capture_output as _call_capture_output)

from devito.arch import (AMDGPUX, Cpu64, AppleArm, NvidiaDevice, POWER8, POWER9,
                         Graviton, Cortex, IntelDevice, get_nvidia_cc, NvidiaArm,
                         check_cuda_runtime, get_m1_llvm_path)
from devito.exceptions import CompilationError
from devito.logger import debug, warning
from devito.parameters import configuration
from devito.tools import (as_list, change_directory, filter_ordered,
                          memoized_func, make_tempdir)

__all__ = ['sniff_mpi_distro', 'compiler_registry']


@memoized_func
def sniff_compiler_version(cc, allow_fail=False):
    """
    Detect the compiler version.

    Adapted from: ::

        https://github.com/OP2/PyOP2/
    """
    try:
        res = run([cc, "--version"], stdout=PIPE, stderr=DEVNULL)
        ver = res.stdout.decode("utf-8")
        if not ver:
            return Version("0")
    except UnicodeDecodeError:
        return Version("0")
    except OSError:
        if allow_fail:
            return Version("0")
        else:
            raise RuntimeError(f"The `{cc}` compiler isn't available on this system")

    ver = ver.strip()
    if ver.startswith("gcc"):
        compiler = "gcc"
    elif ver.startswith("clang"):
        compiler = "clang"
    elif ver.startswith("Apple LLVM"):
        compiler = "clang"
    elif ver.startswith("Homebrew clang"):
        compiler = "clang"
    elif ver.startswith("Intel"):
        compiler = "icx"
    elif ver.startswith("icc"):
        compiler = "icc"
    elif ver.startswith("icx"):
        compiler = "icx"
    elif ver.startswith("pgcc"):
        compiler = "pgcc"
    elif ver.startswith("nvc++"):
        compiler = "nvc"
    elif ver.startswith("cray"):
        compiler = "cray"
    else:
        compiler = "unknown"

    ver = Version("0")
    if compiler in ["gcc", "icc", "icx", "nvc"]:
        try:
            # gcc-7 series only spits out patch level on dumpfullversion.
            res = run([cc, "-dumpfullversion"], stdout=PIPE, stderr=DEVNULL)
            ver = res.stdout.decode("utf-8")
            ver = '.'.join(ver.strip().split('.')[:3])
            if not ver:
                res = run([cc, "-dumpversion"], stdout=PIPE, stderr=DEVNULL)
                ver = res.stdout.decode("utf-8")
                ver = '.'.join(ver.strip().split('.')[:3])
                if not ver:
                    return Version("0")
            # Sanitize bad icx formatting
            ver = ver.replace("+git", "").replace("git", "")
            ver = Version(ver)
        except UnicodeDecodeError:
            pass

    # Pure integer versions (e.g., ggc5, rather than gcc5.0) need special handling
    try:
        ver = Version(float(ver))
    except TypeError:
        pass

    return ver


@memoized_func
def sniff_mpi_distro(mpiexec):
    """
    Detect the MPI version.
    """
    try:
        ver = check_output([mpiexec, "--version"]).decode("utf-8")
        if "open-mpi" in ver:
            return 'OpenMPI'
        elif "HYDRA" in ver:
            return 'MPICH'
        elif "Intel(R) MPI" in ver:
            return 'IntelMPI'
    except (CalledProcessError, UnicodeDecodeError):
        pass
    return 'unknown'


@memoized_func
def sniff_mpi_flags(mpicc='mpicc'):
    mpi_distro = sniff_mpi_distro('mpiexec')
    if mpi_distro != 'OpenMPI':
        raise NotImplementedError("Unable to detect MPI compile and link flags")

    # OpenMPI's CC wrapper, namely mpicc, takes the --showme argument to find out
    # the flags used for compiling and linking
    compile_flags = check_output(['mpicc', "--showme:compile"]).decode("utf-8")
    link_flags = check_output(['mpicc', "--showme:link"]).decode("utf-8")

    return compile_flags.split(), link_flags.split()


@memoized_func
def call_capture_output(cmd):
    """
    Memoize calls to codepy's `call_capture_output` to avoid leaking memory due
    to some prefork/subprocess voodoo.
    """
    return _call_capture_output(cmd)


class Compiler(GCCToolchain):
    """
    Base class for all compiler classes.

    The base class defaults all compiler specific settings to empty lists.
    Preset configurations can be built by inheriting from `Compiler` and setting
    specific flags to the desired values, e.g.:

    .. code-block:: python
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
        * :data:`self.so_ext`
        * :data:`self.undefines`

    Parameters
    ----------
    suffix : str, optional
        The JIT compiler version to be used. For example, assuming ``CC=gcc`` and
        ``suffix='4.9'``, the ``gcc-4.9`` will be used as JIT compiler.
    cpp : bool, optional, default=False
        If True, JIT compile using a C++ compiler.
    mpi : bool, optional, default=False
        If True, JIT compile using an MPI compiler.
    platform : Platform, optional
        The target Platform on which the JIT compiler will be used.
    """

    fields = {'cc', 'ld'}
    linker_opt = '-Wl,'
    _default_cpp = False
    _cxxstd = 'c++14'
    _cstd = 'c99'

    def __init__(self, **kwargs):
        maybe_name = kwargs.pop('name', self.__class__.__name__)
        if isinstance(maybe_name, Compiler):
            self._name = maybe_name.name
        else:
            self._name = maybe_name

        super().__init__(**kwargs)

        self.__lookup_cmds__()
        self._cpp = kwargs.get('cpp', self._default_cpp)

        self.suffix = kwargs.get('suffix')
        if not kwargs.get('mpi'):
            self.cc = self.CC if self._cpp is False else self.CXX
            self.cc = self.cc if self.suffix is None else f'{self.cc}-{self.suffix}'
        else:
            self.cc = self.MPICC if self._cpp is False else self.MPICXX
        self.ld = self.cc  # Wanted by the superclass

        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', f'-std={self.std}']
        self.ldflags = ['-shared']

        self.include_dirs = []
        self.libraries = ['m']
        self.library_dirs = []
        self.defines = []
        self.undefines = []

        self.src_ext = 'c' if self._cpp is False else 'cpp'

        if platform.system() == "Linux":
            self.so_ext = '.so'
        elif platform.system() == "Darwin":
            self.so_ext = '.dylib'
        elif platform.system() == "Windows":
            self.so_ext = '.dll'
        else:
            raise NotImplementedError(f"Unsupported platform {platform}")

        self.__init_finalize__(**kwargs)

    def __init_finalize__(self, **kwargs):
        pass

    def __new_with__(self, **kwargs):
        """
        Create a new Compiler from an existing one, inherenting from it
        the flags that are not specified via ``kwargs``.
        """
        return self.__class__(name=self.name, suffix=kwargs.pop('suffix', self.suffix),
                              mpi=kwargs.pop('mpi', configuration['mpi']),
                              **kwargs)

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        if self.suffix is not None:
            try:
                version = Version(str(float(self.suffix)))
            except (TypeError, ValueError):
                version = Version(self.suffix)
        else:
            # Knowing the version may still be useful to pick supported flags
            allow_fail = isinstance(self, CustomCompiler)
            version = sniff_compiler_version(self.CC, allow_fail=allow_fail)

        return version

    @property
    def std(self):
        return self._cxxstd if self._cpp else self._cstd

    def get_version(self):
        result, stdout, stderr = call_capture_output((self.cc, "--version"))
        if result != 0:
            raise RuntimeError(f"version query failed: {stderr}")
        return stdout

    def get_jit_dir(self):
        """A deterministic temporary directory for jit-compiled objects."""
        return make_tempdir('jitcache')

    def get_codepy_dir(self):
        """A deterministic temporary directory for the codepy cache."""
        return make_tempdir('codepy')

    def load(self, soname):
        """
        Load a compiled shared object.

        Parameters
        ----------
        soname : str
            Name of the .so file (w/o the suffix).

        Returns
        -------
        obj
            The loaded shared object.
        """
        return npct.load_library(str(self.get_jit_dir().joinpath(soname)), '.')

    def save(self, soname, binary):
        """
        Store a binary into a file within a temporary directory.

        Parameters
        ----------
        soname : str
            Name of the .so file (w/o the suffix).
        binary : obj
            The binary data.
        """
        sofile = self.get_jit_dir().joinpath(soname).with_suffix(self.so_ext)
        if sofile.is_file():
            debug(f"{self}: `{sofile.name}` was not saved in `{self.get_jit_dir()}`"
                  " as it already exists")
        else:
            makedirs(self.get_jit_dir(), exist_ok=True)
            with open(str(sofile), 'wb') as f:
                f.write(binary)
            debug(f"{self}: `{sofile.name}` successfully saved in `{self.get_jit_dir()}`")

    def make(self, loc, args):
        """Invoke the ``make`` command from within ``loc`` with arguments ``args``."""
        hash_key = sha1((loc + str(args)).encode()).hexdigest()
        logfile = path.join(self.get_jit_dir(), f"{hash_key}.log")
        errfile = path.join(self.get_jit_dir(), f"{hash_key}.err")

        with change_directory(loc):
            with open(logfile, "w") as lf:
                with open(errfile, "w") as ef:

                    command = ['make'] + args
                    lf.write("Compilation command:\n")
                    lf.write(" ".join(command))
                    lf.write("\n\n")
                    try:
                        check_call(command, stderr=ef, stdout=lf)
                    except CalledProcessError as e:
                        raise CompilationError(f'Command "{e.cmd}" return error status'
                                               f'{e.returncode}. '
                                               f'Unable to compile code.\n'
                                               f'Compile log in {logfile}\n'
                                               f'Compile errors in {errfile}\n')
        debug(f"Make <{' '.join(args)}>")

    def _cmdline(self, files, object=False):
        """
        Sanitize command line to remove all shell string escape such as
        mpicc/mpicxx would add, e.g., `-Wl\\,-rpath,/path/to/lib`.
        """
        cc_line = super()._cmdline(files, object=object)
        return [s.replace('\\', '') for s in cc_line]

    def jit_compile(self, soname, code):
        """
        JIT compile some source code given as a string.

        This function relies upon codepy's ``compile_from_string``, which performs
        caching of compilation units and avoids potential race conditions due to
        multiple processing trying to compile the same object.

        Parameters
        ----------
        soname : str
            Name of the .so file (w/o the suffix).
        code : str
            The source code to be JIT compiled.
        """
        target = str(self.get_jit_dir().joinpath(soname))
        src_file = f"{target}.{self.src_ext}"

        cache_dir = self.get_codepy_dir().joinpath(soname[:7])
        if configuration['jit-backdoor'] is False:
            # Typically we end up here
            # Make a suite of cache directories based on the soname
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Warning: dropping `code` on the floor in favor to whatever is written
            # within `src_file`
            try:
                with open(src_file, 'r') as f:
                    code = f.read()
                    code = f'{code}/* Backdoor edit at {time.ctime()}*/ \n'
                # Bypass the devito JIT cache
                # Note: can't simply use Python's `mkdtemp()` as, with MPI, different
                # ranks would end up creating different cache dirs
                cache_dir = cache_dir.joinpath('jit-backdoor')
                cache_dir.mkdir(parents=True, exist_ok=True)
            except FileNotFoundError:
                raise ValueError(f"Trying to use the JIT backdoor for `{src_file}`, but "
                                 "the file isn't present")

        # Should the compilation command be emitted?
        debug = configuration['log-level'] == 'DEBUG'

        # Spinlock in case of MPI
        sleep_delay = 0 if configuration['mpi'] else 1

        # `catch_warnings` suppresses codepy complaining that it's taking
        # too long to acquire the cache lock. This warning can only appear
        # in a multiprocess session, typically (but not necessarily) when
        # many processes are frequently attempting jit-compilation (e.g.,
        # when running the test suite in parallel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, _, recompiled = compile_from_string(self, target, code, src_file,
                                                      cache_dir=cache_dir, debug=debug,
                                                      sleep_delay=sleep_delay)

        return recompiled, src_file

    def __lookup_cmds__(self):
        self.CC = 'unknown'
        self.CXX = 'unknown'
        self.MPICC = 'unknown'
        self.MPICXX = 'unknown'

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"JITCompiler[{self.__class__.__name__}]"

    def __getstate__(self):
        # The superclass would otherwise only return a subset of attributes
        return self.__dict__

    def add_include_dirs(self, dirs):
        self.include_dirs = filter_ordered(self.include_dirs + as_list(dirs))

    def add_library_dirs(self, dirs, rpath=False):
        self.library_dirs = filter_ordered(self.library_dirs + as_list(dirs))
        if rpath:
            # Add rpath flag to embed library dir
            for d in as_list(dirs):
                self.ldflags.append(f'{self.linker_opt}-rpath,{d}')

    def add_libraries(self, libs):
        self.libraries = filter_ordered(self.libraries + as_list(libs))

    def add_ldflags(self, flags):
        self.ldflags = filter_ordered(self.ldflags + as_list(flags))


class GNUCompiler(Compiler):

    def __init_finalize__(self, **kwargs):
        platform = kwargs.pop('platform', configuration['platform'])

        self.cflags += ['-Wno-unused-result',
                        '-Wno-unused-variable', '-Wno-unused-but-set-variable']

        if configuration['safe-math']:
            self.cflags.append('-fno-unsafe-math-optimizations')
        else:
            self.cflags.append('-ffast-math')

        if platform.isa == 'avx512':
            if self.version >= Version("8.0.0"):
                # The default is `=256` because avx512 slows down the CPU frequency;
                # however, we empirically found that stencils generally benefit
                # from `=512`
                self.cflags.append('-mprefer-vector-width=512')
            else:
                # Unsupported on earlier versions
                pass

        if platform in [POWER8, POWER9]:
            # -march isn't supported on power architectures, is -mtune needed?
            self.cflags = ['-mcpu=native'] + self.cflags
        elif isinstance(platform, (Graviton, NvidiaArm)):
            self.cflags = [f'-mcpu={platform.march}'] + self.cflags
        elif isinstance(platform, Cortex):
            self.cflags += [f'-march={platform.march}']
            self.cflags += [f'-mtune={platform.mtune}']
        else:
            self.cflags = ['-march=native'] + self.cflags

        language = kwargs.pop('language', configuration['language'])
        try:
            if self.version >= Version("4.9.0"):
                # Append the openmp flag regardless of the `language` value,
                # since GCC4.9 and later versions implement OpenMP 4.0, hence
                # they support `#pragma omp simd`
                self.ldflags += ['-fopenmp']
        except (TypeError, ValueError):
            if 'openmp' in language:
                self.ldflags += ['-fopenmp']

    def __lookup_cmds__(self):
        self.CC = 'gcc'
        self.CXX = 'g++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


class ArmCompiler(GNUCompiler):
    pass


class ClangCompiler(Compiler):

    def __init_finalize__(self, **kwargs):

        self.cflags += ['-Wno-unused-result', '-Wno-unused-variable']
        if not configuration['safe-math']:
            self.cflags.append('-ffast-math')
        self.cflags.append('-fdenormal-fp-math=ieee')

        language = kwargs.pop('language', configuration['language'])
        platform = kwargs.pop('platform', configuration['platform'])

        if isinstance(platform, NvidiaDevice):
            self.cflags.remove(f'-std={self.std}')
            # Add flags for OpenMP offloading
            if language in ['C', 'openmp']:
                cc = get_nvidia_cc()
                if cc:
                    self.cflags += ['-Xopenmp-target', f'-march=sm_{cc}']
                self.ldflags += ['-fopenmp', '-fopenmp-targets=nvptx64-nvidia-cuda']
        elif platform is AMDGPUX:
            self.cflags.remove(f'-std={self.std}')
            # Add flags for OpenMP offloading
            if language in ['C', 'openmp']:
                self.ldflags += ['-target', 'x86_64-pc-linux-gnu']
                self.ldflags += ['-fopenmp',
                                 '-fopenmp-targets=amdgcn-amd-amdhsa',
                                 '-Xopenmp-target=amdgcn-amd-amdhsa']
                self.ldflags += [f'-march={platform.march}']
        elif isinstance(platform, AppleArm):
            # NOTE:
            # Apple Mx supports OpenMP through Apple's LLVM compiler.
            # The compiler can be installed with Homebrew or can be built from scratch.
            # Check if installed and set compiler flags accordingly
            llvmm1 = get_m1_llvm_path(language)
            if llvmm1 and 'openmp' in language:
                mx = platform.march
                self.ldflags += [f'-mcpu=apple-{mx}',
                                 '-fopenmp', f'-L{llvmm1["libs"]}']
                self.cflags += ['-Xclang', f'-I{llvmm1["include"]}']
        else:
            if platform in [POWER8, POWER9]:
                # -march isn't supported on power architectures
                self.cflags += ['-mcpu=native']
            else:
                self.cflags += ['-march=native']
            if 'openmp' in language:
                self.ldflags += ['-fopenmp']

    def __lookup_cmds__(self):
        self.CC = 'clang'
        self.CXX = 'clang++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


class CrayCompiler(ClangCompiler):

    """HPE Cray's Clang compiler."""

    def __lookup_cmds__(self):
        self.CC = 'cc'
        self.CXX = 'CC'
        self.MPICC = 'cc'
        self.MPICXX = 'CC'


class AOMPCompiler(Compiler):

    """AMD's fork of Clang for OpenMP offloading on both AMD and NVidia cards."""

    def __init_finalize__(self, **kwargs):

        language = kwargs.pop('language', configuration['language'])
        platform = kwargs.pop('platform', configuration['platform'])

        self.cflags += ['-Wno-unused-result', '-Wno-unused-variable']
        if not configuration['safe-math']:
            self.cflags.append('-ffast-math')

        if 'openmp' in language:
            self.ldflags += ['-fopenmp']

        if isinstance(platform, NvidiaDevice):
            self.cflags.remove(f'-std={self.std}')
        elif platform is AMDGPUX:
            self.cflags.remove(f'-std={self.std}')
            # Add flags for OpenMP offloading
            if language in ['C', 'openmp']:
                self.ldflags += ['-target', 'x86_64-pc-linux-gnu']
                self.ldflags += [f'--offload-arch={platform.march}']
        elif platform in [POWER8, POWER9]:
            # It doesn't make much sense to use AOMP on Power, but it should work
            self.cflags.append('-mcpu=native')
        else:
            self.cflags.append('-march=native')

    def __lookup_cmds__(self):
        self.CC = 'amdclang'
        self.CXX = 'amdclang++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


class DPCPPCompiler(Compiler):

    def __init_finalize__(self, **kwargs):

        self.cflags += ['-qopenmp', '-fopenmp-targets=spir64']

    def __lookup_cmds__(self):
        # OneAPI Base Kit comes with dpcpp/icpx, both are clang++,
        # and icx, which is clang
        self.CC = 'icx'
        self.CXX = 'icpx'
        self.MPICC = 'mpic++'
        self.MPICXX = 'mpicxx'


class PGICompiler(Compiler):

    _default_cpp = True

    def __init_finalize__(self, **kwargs):

        self.cflags.remove('-O3')
        self.cflags.remove('-Wall')

        language = kwargs.pop('language', configuration['language'])
        platform = kwargs.pop('platform', configuration['platform'])

        if isinstance(platform, NvidiaDevice):
            if self.version >= Version("24.9"):
                self.cflags.append('-gpu=mem:separate:pinnedalloc')
            else:
                self.cflags.append('-gpu=pinned')

            if language == 'openacc':
                self.cflags.extend(['-mp', '-acc:gpu'])
            elif 'openmp' in language:
                self.cflags.extend(['-mp=gpu'])
        elif isinstance(platform, Cpu64):
            if 'openmp' in language:
                self.cflags.append('-mp')
            if isinstance(platform, NvidiaArm):
                self.cflags.append(f'-mcpu={platform.march}')

        if not configuration['safe-math']:
            self.cflags.append('-fast')
        # Default PGI compile for a target is GPU and single threaded host.
        # self.cflags += ['-ta=tesla,host']

    def __lookup_cmds__(self):
        # NOTE: using `pgc++` instead of `pgcc` because of issue #1219
        self.CC = 'pgc++'
        self.CXX = 'pgc++'
        self.MPICC = 'mpic++'
        self.MPICXX = 'mpicxx'


class NvidiaCompiler(PGICompiler):

    def __lookup_cmds__(self):
        self.CC = 'nvc++'
        self.CXX = 'nvc++'
        self.MPICC = 'mpic++'
        self.MPICXX = 'mpicxx'


class CudaCompiler(Compiler):

    _default_cpp = True
    linker_opt = "--linker-options="

    def __init_finalize__(self, **kwargs):

        self.cflags.remove('-Wall')
        self.cflags.remove('-fPIC')
        self.cflags.extend(['-Xcompiler', '-fPIC'])

        if configuration['mpi']:
            # We rather use `nvcc` to compile MPI, but for this we have to
            # explicitly pass the flags that an `mpicc` would implicitly use
            compile_flags, link_flags = sniff_mpi_flags('mpicxx')

            try:
                # No idea why `-pthread` would pop up among the `compile_flags`
                compile_flags.remove('-pthread')
            except ValueError:
                # Just in case they fix it, we wrap it up within a try-except
                pass
            self.cflags.extend(compile_flags)

            # Some arguments are for the host compiler
            proc_link_flags = []
            for i in link_flags:
                if i == '-pthread':
                    proc_link_flags.extend(['-Xcompiler', '-pthread'])
                elif i.startswith('-Wl'):
                    # E.g., `-Wl,-rpath` -> `-Xcompiler "-Wl\,-rpath"`
                    escaped_i = i.replace(",", r"\\,")
                    proc_link_flags.extend([
                        '-Xcompiler', f'"{escaped_i}"'
                    ])
                else:
                    proc_link_flags.append(i)
            self.ldflags.extend(proc_link_flags)

        cc = get_nvidia_cc()
        if cc:
            self.cflags.append(f'-arch=sm_{cc}')
        else:
            self.cflags.append('-arch=native')

        # Disable `warning #1650-D: result of call is not used`
        # See `https://gist.github.com/gavinb/f2320f9eaa0e0a7efca6877a34047a9d` about
        # disabling specific warnings with nvcc
        self.cflags.extend(['-Xcudafe', '--display_error_number',
                            '--diag-suppress', '1650'])
        # Same as above but for the host compiler
        self.cflags.extend(['-Xcompiler', '-Wno-unused-result'])

        if not configuration['safe-math']:
            self.cflags.append('--use_fast_math')

        if configuration['profiling'] == 'advanced2':
            # Optionally print out per-kernel shared memory and register usage
            self.cflags.append('--ptxas-options=-v')

            # Useful for Nsight Compute to associate the source code with
            # profiling information
            self.cflags.append('-lineinfo')

        self.src_ext = 'cu'

        # NOTE: not sure where we should place this. It definitely needs
        # to be executed once to warn the user in case there's a CUDA/driver
        # mismatch that would cause the program to run, but likely producing
        # garbage, since the CUDA kernel behaviour would be undefined
        check_cuda_runtime()

    def __lookup_cmds__(self):
        self.CC = 'nvcc'
        self.CXX = 'nvcc'
        self.MPICC = 'nvcc'
        self.MPICXX = 'nvcc'


class HipCompiler(Compiler):

    _default_cpp = True

    def __init_finalize__(self, **kwargs):

        if configuration['mpi']:
            # We rather use `hipcc` to compile MPI, but for this we have to
            # explicitly pass the flags that an `mpicc` would implicitly use
            compile_flags, link_flags = sniff_mpi_flags()
            self.cflags.extend(compile_flags)
            self.ldflags.extend(link_flags)
        else:
            if not configuration['safe-math']:
                self.cflags.append('-DHIP_FAST_MATH')

        self.src_ext = 'cpp'

    def __lookup_cmds__(self):
        self.CC = 'hipcc'
        self.CXX = 'hipcc'
        self.MPICC = 'hipcc'
        self.MPICXX = 'hipcc'


class IntelCompiler(Compiler):

    def __init_finalize__(self, **kwargs):
        platform = kwargs.pop('platform', configuration['platform'])
        language = kwargs.pop('language', configuration['language'])

        self.cflags.append("-xHost")

        if configuration['safe-math']:
            self.cflags.append("-fp-model=strict")
        else:
            self.cflags.append('-fp-model=fast')

        if platform.isa == 'avx512':
            # Systematically use 512-bit vectors if avx512 is available.
            self.cflags.append("-qopt-zmm-usage=high")

        if 'openmp' in language:
            self.ldflags.append('-qopenmp')

        if kwargs.get('mpi'):
            self.__init_intel_mpi__()
            self.__init_intel_mpi_flags__()

    def __init_intel_mpi__(self, **kwargs):
        # Make sure the MPI compiler uses an Intel compiler underneath,
        # whatever the MPI distro is
        mpi_distro = sniff_mpi_distro('mpiexec')
        if mpi_distro != 'IntelMPI':
            warning(f"Expected Intel MPI distribution with `{self.__class__.__name__}`,"
                    f"but found `{mpi_distro}`")

    def __init_intel_mpi_flags__(self, **kwargs):
        self.cflags.insert(0, f'-cc={self.CC}')

    def get_version(self):
        if configuration['mpi']:
            cmd = (self.cc, f"-cc={self.CC}", "--version")
        else:
            cmd = (self.cc, "--version")
        result, stdout, stderr = call_capture_output(cmd)
        if result != 0:
            raise RuntimeError(f"version query failed: {stderr}")
        return stdout

    def __lookup_cmds__(self):
        self.CC = 'icc'
        self.CXX = 'icpc'

        # On some systems, the Intel distribution of MPI may be available, in
        # which case the MPI compiler may be shipped either as `mpiicc` or `mpicc`.
        # On other systems, there may be no Intel distribution of MPI available,
        # thus the MPI compiler is expected to be the classic `mpicc`. Here,
        # we try to use `mpiicc` first, while `mpicc` is our fallback, which may
        # or may not be an Intel distribution
        try:
            check_output(["mpiicc", f"-cc={self.CC}", "--version"]).decode("utf-8")
            self.MPICC = 'mpiicc'
            self.MPICXX = 'mpicxx'
        except FileNotFoundError:
            self.MPICC = 'mpicc'
            self.MPICXX = 'mpicxx'


class IntelKNLCompiler(IntelCompiler):

    def __init_finalize__(self, **kwargs):
        IntelCompiler.__init_finalize__(self, **kwargs)

        language = kwargs.pop('language', configuration['language'])

        self.cflags.append('-xMIC-AVX512')

        if language != 'openmp':
            warning("Running on Intel KNL without OpenMP is highly discouraged")


class OneapiCompiler(IntelCompiler):

    def __init_finalize__(self, **kwargs):
        IntelCompiler.__init_finalize__(self, **kwargs)

        platform = kwargs.pop('platform', configuration['platform'])
        language = kwargs.pop('language', configuration['language'])

        if language == 'sycl':
            warning(f"Use SyclCompiler (`sycl`) to jit-compile sycl, not {self.name}")

        elif 'openmp' in language:
            # Earlier versions to OneAPI 2023.2.0 (clang17 underneath), have an
            # OpenMP bug concerning reductions, hence with them we're forced to
            # use the obsolete -fopenmp
            if self.version < Version('17.0.0'):
                self.ldflags.remove('-qopenmp')
                self.ldflags.append('-fopenmp')

            if isinstance(platform, NvidiaDevice):
                self.cflags.append('-fopenmp-targets=nvptx64-cuda')
            elif isinstance(platform, IntelDevice):
                self.cflags.append('-fiopenmp')
                self.cflags.append('-fopenmp-targets=spir64')
                self.cflags.append('-fopenmp-target-simd')

                self.cflags.remove('-g')  # -g disables some optimizations in IGC
                self.cflags.append('-gline-tables-only')
                self.cflags.append('-fdebug-info-for-profiling')

    def __init_intel_mpi__(self, **kwargs):
        IntelCompiler.__init_intel_mpi__(self, **kwargs)

        platform = kwargs.pop('platform', configuration['platform'])

        # The Intel toolchain requires the I_MPI_OFFLOAD env var to be set
        # to enable GPU-aware MPI (that is, passing device pointers to MPI calls)
        if isinstance(platform, IntelDevice):
            environ['I_MPI_OFFLOAD'] = '1'

    def __init_intel_mpi_flags__(self, **kwargs):
        pass

    get_version = Compiler.get_version

    def __lookup_cmds__(self):
        # OneAPI HPC ToolKit comes with icpx, which is clang++,
        # and icx, which is clang
        self.CC = 'icx'
        self.CXX = 'icpx'
        self.MPICC = 'mpiicx'
        self.MPICXX = 'mpiicpx'


class SyclCompiler(OneapiCompiler):

    _default_cpp = True

    def __init_finalize__(self, **kwargs):
        IntelCompiler.__init_finalize__(self, **kwargs)

        platform = kwargs.pop('platform', configuration['platform'])
        language = kwargs.pop('language', configuration['language'])

        if language != 'sycl':
            warning(f"Expected language sycl with SyclCompiler, not {language}")

        self.cflags.remove(f'-std={self.std}')
        self.cflags.append('-fsycl')

        self.cflags.remove('-g')  # -g disables some optimizations in IGC
        self.cflags.append('-gline-tables-only')
        self.cflags.append('-fdebug-info-for-profiling')

        if isinstance(platform, Cpu64):
            pass
        elif isinstance(platform, NvidiaDevice):
            self.cflags.append('-fsycl-targets=nvptx64-cuda')
        elif isinstance(platform, IntelDevice):
            self.cflags.append('-fsycl-targets=spir64')
        else:
            warning(f"Unsupported platform {platform}")


class CustomCompiler(Compiler):

    """
    Custom compiler based on standard environment flags.

    If no environment flags are found, defaults to the GNUCompiler.

    Notes
    -----
    Currently honours CC, CFLAGS and LDFLAGS, with defaults similar to the
    default GNU/gcc settings. If DEVITO_ARCH is enabled and the DEVITO_LANGUAGE
    is set to 'openmp', then the OpenMP linker flags are read from OMP_LDFLAGS
    or otherwise default to ``-fopenmp``.
    """

    def __new__(cls, *args, **kwargs):
        platform = kwargs.pop('platform', configuration['platform'])
        language = kwargs.pop('language', configuration['language'])

        if isinstance(platform, AppleArm):
            _base = ClangCompiler
        elif isinstance(platform, IntelDevice):
            _base = OneapiCompiler
        elif isinstance(platform, NvidiaDevice):
            if language == 'cuda':
                _base = CudaCompiler
            else:
                _base = NvidiaCompiler
        elif platform is AMDGPUX:
            if language == 'hip':
                _base = HipCompiler
            else:
                _base = AOMPCompiler
        else:
            _base = GNUCompiler

        obj = super().__new__(cls)
        # Keep base to initialize accordingly
        obj._base = kwargs.pop('base', _base)

        return obj

    def __init_finalize__(self, **kwargs):
        self._base.__init_finalize__(self, **kwargs)
        # Update cflags
        try:
            extrac = environ.get('CFLAGS').split(' ')
            self.cflags = self.cflags + extrac
        except AttributeError:
            pass
        # Update ldflags
        try:
            extrald = environ.get('LDFLAGS').split(' ')
            self.ldflags = self.ldflags + extrald
        except AttributeError:
            pass

    def __lookup_cmds__(self):
        self._base.__lookup_cmds__(self)
        # TODO: check for conflicts, for example using the nvhpc module file
        # will set CXX to nvc++ breaking  the cuda backend
        self.CC = environ.get('CC', self.CC)
        self.CXX = environ.get('CXX', self.CXX)
        self.MPICC = environ.get('MPICC', self.MPICC)
        self.MPICXX = environ.get('MPICXX', self.MPICXX)

    def __new_with__(self, **kwargs):
        return super().__new_with__(base=self._base, **kwargs)

    @property
    def _default_cpp(self):
        return self._base._default_cpp


class CompilerRegistry(dict):
    """
    Registry dict for deriving Compiler classes according to the environment variable
    DEVITO_ARCH. Developers should add new compiler classes here.
    """

    def __getitem__(self, key):
        if isinstance(key, Compiler):
            key = key.name

        if key.startswith('gcc-'):
            i = key.split('-')[1]
            return partial(GNUCompiler, suffix=i)

        return super().__getitem__(key)

    def __contains__(self, key):
        if isinstance(key, Compiler):
            key = key.name
        return key in self.keys() or key.startswith('gcc-')


_compiler_registry = {
    'custom': CustomCompiler,
    'gnu': GNUCompiler,
    'gcc': GNUCompiler,
    'arm': ArmCompiler,
    'clang': ClangCompiler,
    'cray': CrayCompiler,
    'aomp': AOMPCompiler,
    'amdclang': AOMPCompiler,
    'hip': HipCompiler,
    'pgcc': PGICompiler,
    'pgi': PGICompiler,
    'nvc': NvidiaCompiler,
    'nvc++': NvidiaCompiler,
    'nvidia': NvidiaCompiler,
    'cuda': CudaCompiler,
    'nvcc': CudaCompiler,
    'osx': ClangCompiler,
    'intel': OneapiCompiler,
    'icx': OneapiCompiler,
    'icpx': OneapiCompiler,
    'sycl': SyclCompiler,
    'icc': IntelCompiler,
    'icpc': IntelCompiler,
    'intel-knl': IntelKNLCompiler,
    'knl': IntelKNLCompiler,
    'dpcpp': DPCPPCompiler,
}


compiler_registry = CompilerRegistry(**_compiler_registry)
