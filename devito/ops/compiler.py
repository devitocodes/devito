import os
import subprocess
import warnings

from codepy.jit import compile_from_string
from devito.logger import warning
from devito.parameters import configuration


__all__ = ['CompilerOPS']


class CompilerOPS(configuration['compiler'].__class__):
    def __init__(self, *args, **kwargs):
        kwargs['cpp'] = True
        self._ops_install_path = os.environ.get('OPS_INSTALL_PATH')
        super(CompilerOPS, self).__init__(*args, **kwargs)

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

    def prepare_ops(self, soname, ccode, hcode):
        # Creating files
        file_name = str(self.get_jit_dir().joinpath(soname))
        h_file = open("%s.h" % (file_name), "w")
        c_file = open("%s.cpp" % (file_name), "w")

        c_file.write(ccode)
        h_file.write(hcode)

        c_file.close()
        h_file.close()
        if self._ops_install_path:
            # Calling OPS Translator
            translator = '%s/../ops_translator/c/ops.py' % (self._ops_install_path)
            subprocess.run([translator, c_file.name], cwd=self.get_jit_dir())
        else:
            warning("Couldn't find OPS_INSTALL_PATH \
                environment variable, please check your OPS installation")

    def cuda_compiler(self, soname):
        # CUDA kernel compilation
        cuda_src = '%s/CUDA/%s_kernels.cu' % (self.get_jit_dir(), soname)
        cuda_target = '%s/%s_kernels_cu' % (self.get_jit_dir(), soname)

        cuda_code = ""
        try:
            with open(cuda_src, 'r') as f:
                cuda_code = f.read()
        except FileNotFoundError:
            raise ValueError("The file %s isn't present" % cuda_src)

        cuda_device_compiler = CUDADeviceCompiler()
        cuda_host_compiler = CudaHostCompiler()

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            # Spinlock in case of MPI
            sleep_delay = 0 if configuration['mpi'] else 1
            _, _, cuda_o, _ = compile_from_string(
                cuda_device_compiler, cuda_target,
                cuda_code, cuda_src,
                cache_dir=self.cache_dir,
                debug=configuration['debug-compiler'],
                sleep_delay=sleep_delay,
                object=True
            )
            _, _, src_o, _ = compile_from_string(
                cuda_host_compiler, self.target,
                self.code, self.ops_src,
                cache_dir=self.cache_dir,
                debug=configuration['debug-compiler'],
                sleep_delay=sleep_delay,
                object=True
            )
            cuda_host_compiler.link_extension(
                '%s%s' % (self.target, cuda_host_compiler.so_ext),
                [src_o, cuda_o],
                debug=configuration['debug-compiler']
            )

    def jit_compile(self, soname):
        self.target = str(self.get_jit_dir().joinpath(soname))
        self.ops_src = '%s/%s_ops.cpp' % (self.get_jit_dir(), soname)
        self.cache_dir = self.get_codepy_dir().joinpath(soname[:7])

        # Typically we end up here
        # Make a suite of cache directories based on the soname
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.ops_src, 'r') as f:
                self.code = f.read()
        except FileNotFoundError:
            warning("The file %s isn't present" % self.ops_src)
        else:
            self.cuda_compiler(soname)


class CUDADeviceCompiler(CompilerOPS):

    def __init__(self, *args, **kwargs):
        super(CUDADeviceCompiler, self).__init__(*args, **kwargs)
        self.o_ext = '.o'
        self.cflags = ['-Xcompiler="-fPIC"', '-O3', '-g', '-gencode']
        nv_arch = os.environ.get('NV_ARCH')

        if(nv_arch is None):
            raise ValueError("select an NVIDIA device to compile in CUDA, e.g. \
                NV_ARCH=Kepler")
        elif(nv_arch == 'Fermi'):
            self.cflags.append('arch=compute_20,code=sm_21')
        elif(nv_arch == 'Kepler'):
            self.cflags.append('arch=compute_35,code=sm_35')
        elif(nv_arch == 'Pascal'):
            self.cflags.append('arch=compute_60,code=sm_60')
        elif(nv_arch == 'Volta'):
            self.cflags.append('arch=compute_70,code=sm_70')
        else:
            raise ValueError("Unknown NVIDIA architecture, select: \
                Fermi, Kepler, Pascal or Volta")

        self.ldflags = []

        include_dirs = '%s %s/c/include' % (self.get_jit_dir(), self._ops_install_path)
        self.include_dirs = include_dirs.split(' ')

    def __lookup_cmds__(self):
        self.CC = os.environ.get('CC', 'nvcc')
        self.CXX = os.environ.get('CXX', 'nvcc')


class CudaHostCompiler(CompilerOPS):

    def __init__(self, *args, **kwargs):
        super(CudaHostCompiler, self).__init__(*args, **kwargs)
        self.o_ext = '.o'
        self._cuda_install_path = os.environ.get('CUDA_INSTALL_PATH')
        if not self._cuda_install_path:
            raise ValueError("Couldn't find CUDA_INSTALL_PATH \
                environment variable, please check your CUDA installation")

        cflags = '-fopenmp -O3 -fPIC -DUNIX -Wall -shared -g'
        self.cflags = os.environ.get('CFLAGS', cflags).split(' ')

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
