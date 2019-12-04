import os
import subprocess
import warnings

from codepy.jit import compile_from_string
from devito.logger import warning
from devito.parameters import configuration
from devito.archinfo import NVIDIAX

__all__ = ['OPSCompiler']


class OPSMetaCompiler(configuration['compiler'].__class__):

    def __init__(self, *args, **kwargs):
        kwargs['cpp'] = True
        super(OPSMetaCompiler, self).__init__(*args, **kwargs)

        self._ops_install_path = os.environ.get('OPS_INSTALL_PATH')
        if not self._ops_install_path:
            raise ValueError("Couldn't find OPS_INSTALL_PATH\
                environment variable, please check your OPS installation")


class OPSCompiler(OPSMetaCompiler):

    def __init__(self, *args, **kwargs):
        super(OPSCompiler, self).__init__(*args, **kwargs)

    def jit_compile(self, soname, ccode, hcode):
        self._translate_ops(soname, ccode, hcode)
        self.ops_src = '%s/%s_ops.cpp' % (self.get_jit_dir(), soname)
        self.cache_dir = self.get_jit_dir()
        self.target = str(self.get_jit_dir().joinpath(soname))

        # Make a suite of cache directories based on the soname
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.ops_src, 'r') as f:
                self.code = f.read()
        except FileNotFoundError:
            warning("Couldn't find file: %s" % self.ops_src)

        if configuration['platform'] is NVIDIAX:
            self._compile(soname)
        elif configuration['platform'] == 'bdw':
            pass

    def _translate_ops(self, soname, ccode, hcode):
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
        translation = subprocess.run([translator, c_file.name], cwd=self.get_jit_dir())
        if translation.returncode == 1:
            raise ValueError("OPS Translation Error")

    def _compile(self, soname):
        # CUDA kernel compilation
        cuda_src = '%s/CUDA/%s_kernels.cu' % (self.get_jit_dir(), soname)
        cuda_target = '%s/%s_kernels_cu' % (self.get_jit_dir(), soname)

        try:
            with open(cuda_src, 'r') as f:
                cuda_code = f.read()
        except FileNotFoundError:
            raise ValueError("Couldn't find file: %s" % cuda_src)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Spinlock in case of MPI
            cuda_device_compiler = CUDADeviceCompiler()
            sleep_delay = 0 if configuration['mpi'] else 1
            _, _, cuda_o, _ = compile_from_string(
                cuda_device_compiler, cuda_target,
                cuda_code, cuda_src,
                cache_dir=self.cache_dir,
                debug=configuration['debug-compiler'],
                sleep_delay=sleep_delay,
                object=True
            )
            if not os.path.exists(cuda_o):
                raise ValueError("Error when compiling for device\
                                 Couldn't find file: %s" % cuda_o)

            cuda_host_compiler = CUDAHostCompiler()
            _, _, src_o, _ = compile_from_string(
                cuda_host_compiler, self.target,
                self.code, self.ops_src,
                cache_dir=self.cache_dir,
                debug=configuration['debug-compiler'],
                sleep_delay=sleep_delay,
                object=True
            )
            if not os.path.exists(src_o):
                raise ValueError("Error when compiling for host\
                                 Couldn't find file: %s" % src_o)

            try:
                cuda_host_compiler.link_extension(
                    '%s%s' % (self.target, cuda_host_compiler.so_ext),
                    [src_o, cuda_o],
                    debug=configuration['debug-compiler']
                )
            except:
                raise ValueError("Error at the linking stage")


class OPSCompilerCUDA(OPSMetaCompiler):
    def __init__(self, *args, **kwargs):
        super(OPSCompilerCUDA, self).__init__(*args, **kwargs)

        self._cuda_install_path = os.environ.get('CUDA_INSTALL_PATH')
        if not self._cuda_install_path:
            raise ValueError("Couldn't find CUDA_INSTALL_PATH a\
                environment variable, please check your CUDA installation")

    def _cmdline(self, files, object=False):
        if object:
            ld_options = ['-c']
            link = []
        else:
            ld_options = self.ldflags
            link = ["-L%s" % ldir for ldir in self.library_dirs]
            link.extend(["-l%s" % lib for lib in self.libraries])
        return (
            [self.cc]
            + self.cflags
            + ld_options
            + ["-D%s" % define for define in self.defines]
            + ["-U%s" % undefine for undefine in self.undefines]
            + ["-I%s" % idir for idir in self.include_dirs]
            + files
            + link
        )


class OPSCompilerSEQ(OPSMetaCompiler):

    def __init__(self, *args, **kwargs):
        super(OPSCompilerSEQ, self).__init__(*args, **kwargs)

    def _compile(self, soname):
        pass


class CUDADeviceCompiler(OPSCompilerCUDA):

    def __init__(self, *args, **kwargs):
        super(CUDADeviceCompiler, self).__init__(*args, **kwargs)
        self.o_ext = '.o'
        self.cflags = ['-Xcompiler="-fPIC"', '-O3', '-g', '-gencode']

        self._nv_arch = os.environ.get('NV_ARCH')
        if not self._nv_arch:
            raise ValueError("Select an NVIDIA device to compile in CUDA, e.g. \
                NV_ARCH=Kepler")

        if(self._nv_arch == 'Fermi'):
            self.cflags.append('arch=compute_20,code=sm_21')
        elif(self._nv_arch == 'Kepler'):
            self.cflags.append('arch=compute_35,code=sm_35')
        elif(self._nv_arch == 'Pascal'):
            self.cflags.append('arch=compute_60,code=sm_60')
        elif(self._nv_arch == 'Volta'):
            self.cflags.append('arch=compute_70,code=sm_70')
        else:
            raise ValueError("Unknown NVIDIA architecture, select: \
                Fermi, Kepler, Pascal or Volta")

        self.ldflags = []

        include_dirs = '%s %s/c/include' % (self.get_jit_dir(), self._ops_install_path)
        self.include_dirs = include_dirs.split(' ')
        self.include_dirs.append(os.path.join(self._cuda_install_path, 'include'))

    def __lookup_cmds__(self):
        self.CC = os.environ.get('CC', 'nvcc')
        self.CXX = os.environ.get('CXX', 'nvcc')


class CUDAHostCompiler(OPSCompilerCUDA):

    def __init__(self, *args, **kwargs):
        super(CUDAHostCompiler, self).__init__(*args, **kwargs)
        self.o_ext = '.o'
        self.cflags = ['-fopenmp', '-O3', '-fPIC', '-DUNIX', '-Wall', '-shared', '-g']

        self.include_dirs.append(os.path.join(self._cuda_install_path, 'include'))
        self.include_dirs.append(os.path.join(self._ops_install_path, 'c', 'include'))

        self.library_dirs.append(os.path.join(self._cuda_install_path, 'lib64'))
        self.library_dirs.append(os.path.join(self._ops_install_path, 'c', 'lib'))

        self.libraries.extend(['cudart', 'ops_cuda'])

    def __lookup_cmds__(self):
        self.CC = os.environ.get('CC', 'gcc')
        self.CXX = os.environ.get('CXX', 'g++')
        self.MPICC = os.environ.get('MPICC', 'mpicc')
        self.MPICXX = os.environ.get('MPICXX', 'mpicxx')
