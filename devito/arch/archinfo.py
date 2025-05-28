"""Collection of utilities to detect properties of the underlying architecture."""

from functools import cached_property
from subprocess import PIPE, Popen, DEVNULL, run
from pathlib import Path
import ctypes
import re
import os
import sys
import json

import cpuinfo
import numpy as np
import psutil

from devito.logger import warning
from devito.tools import as_tuple, all_equal, memoized_func

__all__ = ['platform_registry', 'get_cpu_info', 'get_gpu_info', 'get_nvidia_cc',
           'get_cuda_path', 'get_hip_path', 'check_cuda_runtime', 'get_m1_llvm_path',
           'get_advisor_path', 'Platform', 'Cpu64', 'Intel64', 'IntelSkylake', 'Amd',
           'Arm', 'Power', 'Device', 'NvidiaDevice', 'AmdDevice', 'IntelDevice',
           # Brand-agnostic
           'ANYCPU', 'ANYGPU',
           # Intel CPUs
           'INTEL64', 'SNB', 'IVB', 'HSW', 'BDW', 'KNL', 'KNL7210',
           'SKX', 'KLX', 'CLX', 'CLK', 'SPR',
           # AMD CPUs
           'AMD',
           # ARM CPUs
           'ARM', 'AppleArm', 'M1', 'M2', 'M3',
           'Graviton', 'GRAVITON2', 'GRAVITON3', 'GRAVITON4',
           'Cortex', 'NvidiaArm', 'GRACE',
           # Other legacy CPUs
           'POWER8', 'POWER9',
           # Generic GPUs
           'AMDGPUX', 'NVIDIAX', 'INTELGPUX',
           # Nvidia GPUs
           'VOLTA', 'AMPERE', 'HOPPER', 'BLACKWELL',
           # Intel GPUs
           'PVC', 'INTELGPUMAX', 'MAX1100', 'MAX1550']


@memoized_func
def get_cpu_info():
    """Attempt CPU info autodetection."""

    # Obtain textual cpu info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    cpu_info = {}

    # Extract CPU flags and branch
    if lines:
        # The /proc/cpuinfo format doesn't follow a standard, and on some
        # more or less exotic combinations of OS and platform it might not
        # contain the information we look for, hence the proliferation of
        # try-except below

        def get_cpu_flags():
            try:
                # ARM Thunder X2 is using 'Features' instead of 'flags'
                flags = [i for i in lines if (i.startswith('Features')
                                              or i.startswith('flags'))][0]
                return flags.split(':')[1].strip().split()
            except:
                return None

        def get_cpu_brand():
            try:
                # Xeons and i3/i5/... CPUs on Linux
                model_name = [i for i in lines if i.startswith('model name')][0]
                return model_name.split(':')[1].strip()
            except:
                pass

            try:
                # Power CPUs on Linux
                cpu = [i for i in lines if i.split(':')[0].strip() == 'cpu'][0]
                return cpu.split(':')[1].strip()
            except:
                pass

            try:
                # Certain ARM CPUs, e.g. Marvell Thunder X2
                return cpuinfo.get_cpu_info().get('arch').lower()
            except:
                return None

        cpu_info['flags'] = get_cpu_flags()
        cpu_info['brand'] = get_cpu_brand()

    if not cpu_info.get('flags'):
        try:
            cpu_info['flags'] = cpuinfo.get_cpu_info().get('flags')
        except:
            # We've rarely seen cpuinfo>=8 raising exceptions at this point,
            # while trying to fetch the cpu `flags`
            cpu_info['flags'] = None

    if not cpu_info.get('brand'):
        try:
            ret = cpuinfo.get_cpu_info()
            cpu_info['brand'] = ret.get('brand', ret.get('brand_raw'))
        except:
            cpu_info['brand'] = None

    # Detect number of logical cores
    logical = psutil.cpu_count(logical=True)
    if not logical:
        # Never bumped into a platform that make us end up here, yet
        # But we try to cover this case anyway, with `lscpu`
        try:
            logical = lscpu()['CPU(s)']
        except KeyError:
            warning("Logical core count autodetection failed")
            logical = 1
    cpu_info['logical'] = logical

    # Detect number of physical cores
    # Special case: in some ARM processors psutils fails to detect physical cores
    # correctly so we use lscpu()
    try:
        if 'arm' in cpu_info['brand']:
            cpu_info['physical'] = lscpu()['Core(s) per socket'] * lscpu()['Socket(s)']
            return cpu_info
    except:
        pass
    # TODO: on multi-socket systems + unix, can't use psutil due to
    # `https://github.com/giampaolo/psutil/issues/1558`
    mapper = {}
    if lines:
        # Copied and readapted from psutil
        current_info = {}
        for i in lines:
            line = i.strip().lower()
            if not line:
                # New section
                if ('physical id' in current_info and 'cpu cores' in current_info):
                    mapper[current_info['physical id']] = current_info['cpu cores']
                current_info = {}
            else:
                # Ongoing section
                if (line.startswith('physical id') or line.startswith('cpu cores')):
                    key, value = line.split('\t:', 1)
                    current_info[key] = int(value)
    physical = sum(mapper.values())
    if not physical:
        # Fallback 1: it should now be fine to use psutil
        physical = psutil.cpu_count(logical=False)
        if not physical:
            # Fallback 2: we might end up here on more exotic platforms such as Power8
            try:
                physical = lscpu()['Core(s) per socket'] * lscpu()['Socket(s)']
            except KeyError:
                warning("Physical core count autodetection failed")
                physical = 1
    cpu_info['physical'] = physical
    return cpu_info


@memoized_func
def get_gpu_info():
    """Attempt GPU info autodetection."""

    # Filter out virtual GPUs from a list of GPU dictionaries
    def filter_real_gpus(gpus):
        def is_real_gpu(gpu):
            return 'virtual' not in gpu['product'].lower()
        return list(filter(is_real_gpu, gpus))

    def homogenise_gpus(gpu_infos):
        """
        Run homogeneity checks on a list of GPUs, return GPU with count if
        homogeneous, otherwise None.
        """
        if gpu_infos == []:
            return {}

        # Check must ignore physical IDs as they may differ
        for gpu_info in gpu_infos:
            gpu_info.pop('physicalid', None)

        if all_equal(gpu_infos):
            gpu_infos[0]['ncards'] = len(gpu_infos)
            return gpu_infos[0]

        warning('Different models of graphics cards detected')

        return {'ncards': len(gpu_infos)}

    # Parse textual gpu info into a dict

    # *** First try: `nvidia-smi`, clearly only works with NVidia cards
    try:
        gpu_infos = []

        info_cmd = ['nvidia-smi', '-L']
        proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
        raw_info = str(proc.stdout.read())

        lines = raw_info.replace('\\n', '\n').replace('b\'', '')
        lines = lines.splitlines()

        for line in lines:
            gpu_info = {}
            if 'GPU' in line:
                gpu_info = {}
                match = re.match(r'GPU *[0-9]*\: ([\w]*) (.*) \(', line)
                if match:
                    if match.group(1) == 'Graphics':
                        gpu_info['architecture'] = 'unspecified'
                    else:
                        gpu_info['architecture'] = match.group(1)
                    if match.group(2) == 'Device':
                        gpu_info['product'] = 'unspecified'
                    else:
                        gpu_info['product'] = match.group(2)
                    gpu_info['vendor'] = 'NVIDIA'
                    gpu_infos.append(gpu_info)

        gpu_info = homogenise_gpus(gpu_infos)

        # Also attach callbacks to retrieve instantaneous memory info
        for i in ['total', 'free', 'used']:
            def make_cbk(i):
                def cbk(deviceid=0):
                    info_cmd = ['nvidia-smi', f'--query-gpu=memory.{i}', '--format=csv']
                    proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
                    raw_info = str(proc.stdout.read())

                    lines = raw_info.replace('\\n', '\n').replace('b\'', '')
                    lines = lines.splitlines()[1:-1]

                    try:
                        line = lines.pop(deviceid)
                        _, v, unit = re.split(r'([0-9]+)\s', line)
                        assert unit == 'MiB'
                        return int(v)*10**6
                    except:
                        # We shouldn't really end up here, unless nvidia-smi changes
                        # the output format (though we still have tests in place that
                        # will catch this)
                        return None

                    return lines

                return cbk

            gpu_info[f'mem.{i}'] = make_cbk(i)

        return gpu_info

    except OSError:
        pass

    # *** Second try: `rocm-smi`, clearly only works with AMD cards
    try:
        gpu_infos = {}

        # Base gpu info
        info_cmd = ['rocm-smi', '--showproductname']
        proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
        raw_info = str(proc.stdout.read())

        lines = raw_info.replace('\\n', '\n').replace('b\'', '').replace('\\t', '')
        lines = lines.splitlines()

        for line in lines:
            if 'GPU' in line:
                # Product
                pattern = r'GPU\[(\d+)\].*?Card [sS]eries:\s*(.*?)\s*$'
                match1 = re.match(pattern, line)

                if match1:
                    gid = match1.group(1)
                    gpu_infos.setdefault(gid, dict())
                    gpu_infos[gid]['physicalid'] = gid
                    gpu_infos[gid]['product'] = match1.group(2)

                # Model
                pattern = r'GPU\[(\d+)\].*?Card [mM]odel:\s*(.*?)\s*$'
                match2 = re.match(pattern, line)

                if match2:
                    gid = match2.group(1)
                    gpu_infos.setdefault(gid, dict())
                    gpu_infos[gid]['physicalid'] = match2.group(1)
                    gpu_infos[gid]['model'] = match2.group(2)

        gpu_info = homogenise_gpus(list(gpu_infos.values()))

        # Also attach callbacks to retrieve instantaneous memory info
        info_cmd = ['rocm-smi', '--showmeminfo', 'vram', '--json']
        proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
        raw_info = str(proc.stdout.read())
        lines = raw_info.replace('\\n', '').replace('b\'', '').replace('\'', '')
        info = json.loads(lines)

        for i in ['total', 'free', 'used']:
            def make_cbk(i):
                def cbk(deviceid=0):
                    try:
                        # Should only contain Used and total
                        assert len(info[f'card{deviceid}']) == 2
                        used = [int(v) for k, v in info[f'card{deviceid}'].items()
                                if 'Used' in k][0]
                        total = [int(v) for k, v in info[f'card{deviceid}'].items()
                                 if 'Used' not in k][0]
                        free = total - used
                        return {'total': total, 'free': free, 'used': used}[i]
                    except:
                        # We shouldn't really end up here, unless nvidia-smi changes
                        # the output format (though we still have tests in place that
                        # will catch this)
                        return None

                return cbk

            gpu_info[f'mem.{i}'] = make_cbk(i)

        gpu_info['architecture'] = 'unspecified'
        gpu_info['vendor'] = 'AMD'

        return gpu_info

    except (json.JSONDecodeError, OSError):
        pass

    # *** Third try: `sycl-ls`, clearly only works with Intel cards
    try:
        gpu_infos = {}

        # sycl-ls sometimes finds gpu twice with opencl and without so
        # we need to make sure we don't get duplicates
        selected_platform = None
        platform_block = ""

        proc = Popen(["sycl-ls", "--verbose"], stdout=PIPE, stderr=DEVNULL, text=True)
        sycl_output, _ = proc.communicate()

        # Extract platform blocks
        platforms = re.findall(r"Platform \[#(\d+)\]:([\s\S]*?)(?=Platform \[#\d+\]:|$)",
                               sycl_output)

        # Select Level-Zero if available, otherwise use OpenCL
        for platform_id, platform_content in platforms:
            if "Intel(R) Level-Zero" in platform_content:
                selected_platform = platform_id
                platform_block = platform_content
                break
            elif "Intel(R) OpenCL Graphics" in platform_content and \
                    selected_platform is None:
                selected_platform = platform_id
                platform_block = platform_content

        # Extract GPU devices from the selected platform
        devices = re.findall(r"Device \[#(\d+)\]:([\s\S]*?)(?=Device \[#\d+\]:|$)",
                             platform_block)

        for device_id, device_block in devices:
            if re.search(r"^\s*Type\s*:\s*gpu", device_block, re.MULTILINE):
                name_match = re.search(r"^\s*Name\s*:\s*(.+)", device_block, re.MULTILINE)

                if name_match:
                    name = name_match.group(1).strip()

                    # Store GPU info with correct physical ID
                    gpu_infos[device_id] = {
                        "physicalid": device_id,
                        "product": name
                    }

        gpu_info = homogenise_gpus(list(gpu_infos.values()))

        # Also attach callbacks to retrieve instantaneous memory info
        # Now this should be done using xpu-smi but for some reason
        # it throws a lot of weird errors in docker so skipping for now
        for i in ['total', 'free', 'used']:
            def make_cbk(i):
                def cbk(deviceid=0):
                    return None
                return cbk

            gpu_info['mem.%s' % i] = make_cbk(i)

        gpu_info['architecture'] = 'unspecified'
        gpu_info['vendor'] = 'INTEL'

        return gpu_info

    except OSError:
        pass

    # *** Fourth try: `lshw`
    try:
        info_cmd = ['lshw', '-C', 'video']
        proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
        raw_info = str(proc.stdout.read())

        def lshw_single_gpu_info(raw_info):
            # Separate the output into lines for processing
            lines = raw_info.replace('\\n', '\n')
            lines = lines.splitlines()

            # Define the processing functions
            if lines:
                def extract_gpu_info(keyword):
                    for line in lines:
                        if line.lstrip().startswith(keyword):
                            return line.split(':')[1].lstrip()

                def parse_product_arch():
                    for line in lines:
                        if line.lstrip().startswith('product') and '[' in line:
                            arch_match = re.search(r'\[([\w\s]+)\]', line)
                            if arch_match:
                                return arch_match.group(1)
                    return 'unspecified'

                # Populate the information
                gpu_info = {}
                gpu_info['product'] = extract_gpu_info('product')
                gpu_info['architecture'] = parse_product_arch()
                gpu_info['vendor'] = extract_gpu_info('vendor')
                gpu_info['physicalid'] = extract_gpu_info('physical id')

                return gpu_info

        # Parse the information for all the devices listed with lshw
        devices = raw_info.split('display')[1:]
        gpu_infos = [lshw_single_gpu_info(device) for device in devices]
        gpu_infos = filter_real_gpus(gpu_infos)

        return homogenise_gpus(gpu_infos)

    except OSError:
        pass

    # Fifth try: `lspci`, which is more readable but less detailed than `lshw`
    try:
        info_cmd = ['lspci']
        proc = Popen(info_cmd, stdout=PIPE, stderr=DEVNULL)
        raw_info = str(proc.stdout.read())

        # Note: due to the single line descriptive format of lspci, 'vendor'
        # and 'physicalid' elements cannot be reliably extracted so are left None

        # Separate the output into lines for processing
        lines = raw_info.replace('\\n', '\n')
        lines = lines.splitlines()

        gpu_infos = []
        for line in lines:
            # Graphics cards are listed as VGA or 3D controllers in lspci
            if any(i in line for i in ('VGA', '3D', 'Display')):
                gpu_info = {}
                # Lines produced by lspci command are of the form:
                #   xxxx:xx:xx.x Device Type: Name
                #   eg:
                #   0001:00:00.0 3D controller: NVIDIA Corp... [Tesla K80] (rev a1)
                name_match = re.match(
                    r'\d\d\d\d:\d\d:\d\d\.\d [\w\s]+: ([\w\s\(\)\[\]]*)', line
                )
                if name_match:
                    gpu_info['product'] = name_match.group(1)
                    arch_match = re.search(r'\[([\w\s]+)\]', line)
                    if arch_match:
                        gpu_info['architecture'] = arch_match.group(1)
                    else:
                        gpu_info['architecture'] = 'unspecified'
                else:
                    continue

                gpu_infos.append(gpu_info)

        gpu_infos = filter_real_gpus(gpu_infos)

        return homogenise_gpus(gpu_infos)

    except OSError:
        pass

    return None


@memoized_func
def get_nvidia_cc():
    libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        return None

    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()

    if cuda.cuInit(0) != 0:
        return None
    elif (cuda.cuDeviceComputeCapability(ctypes.byref(cc_major),
          ctypes.byref(cc_minor), 0) == 0):
        return 10*cc_major.value + cc_minor.value


@memoized_func
def get_cuda_path():
    # *** First try: via commonly used environment variables
    for i in ['CUDA_HOME', 'CUDA_ROOT']:
        cuda_home = os.environ.get(i)
        if cuda_home:
            return cuda_home

    # *** Second try: inspect the LD_LIBRARY_PATH
    llp = os.environ.get('LD_LIBRARY_PATH', '')
    for i in llp.split(':'):
        if re.match('.*/nvidia/hpc_sdk/.*/compilers/lib', i):
            cuda_home = os.path.join(os.path.dirname(os.path.dirname(i)), 'cuda')
            # Sanity check
            if os.path.exists(cuda_home):
                return cuda_home

    return None


@memoized_func
def get_advisor_path():
    """
    Detect if Intel Advisor is installed on the machine and return
    its location if it is.
    """
    path = None

    env_path = os.environ["PATH"]
    env_path_dirs = env_path.split(":")

    for env_path_dir in env_path_dirs:
        # intel/oneapi/advisor is the directory for Intel oneAPI
        if "intel/advisor" in env_path_dir or "intel/oneapi/advisor" in env_path_dir:
            path = Path(env_path_dir)
            if path.name.startswith('bin'):
                return path.parent

    return path


@memoized_func
def get_hip_path():
    # *** First try: via commonly used environment variables
    for i in ['HIP_HOME']:
        hip_home = os.environ.get(i)
        if hip_home:
            return hip_home

    return None


@memoized_func
def get_m1_llvm_path(language):
    # Check if Apple's llvm is installed (installable via Homebrew), which supports
    # OpenMP.
    # Check that we are on apple system
    if sys.platform != 'darwin':
        raise ValueError('Apple LLVM is only available on Mac OS X.')
    # *** First check if LLVM is installed
    ver = run(["clang", "--version"], stdout=PIPE, stderr=DEVNULL).stdout.decode("utf-8")
    # *** Second check if this clang version targets arm64 (M1)
    if "arm64-apple" in ver:
        # *** Third extract install path. clang version command contains the line:
        # InstalledDir: /path/to/llvm/bin
        # which points to the llvm root directory we need for libraries and includes
        prefix = [v.split(': ')[-1] for v in ver.split('\n')
                  if "InstalledDir" in v][0][:-4]
        # ** Fourth check if the libraries are installed
        if os.path.exists(os.path.join(prefix, 'lib', 'libomp.dylib')):
            libs = os.path.join(prefix, "lib")
            include = os.path.join(prefix, "include")
            return {'libs': libs, 'include': include}
        elif language == "openmp":
            warning("Apple's arm64 clang found but openmp libraries not found."
                    "Install Apple LLVM for OpenMP support, i.e. `brew install llvm`")
        else:
            pass
    else:
        if language == "openmp":
            warning("Apple's x86 clang found, OpenMP is not supported.")
    return None


@memoized_func
def check_cuda_runtime():
    libnames = ('libcudart.so', 'libcudart.dylib', 'cudart.dll')
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        warning("Unable to check compatibility of NVidia driver and runtime")
        return

    driver_version = ctypes.c_int()
    runtime_version = ctypes.c_int()

    if cuda.cudaDriverGetVersion(ctypes.byref(driver_version)) == 0 and \
       cuda.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) == 0:
        driver_version = driver_version.value
        runtime_version = runtime_version.value
        if driver_version < runtime_version:
            warning("The NVidia driver (v%d) on this system may not be compatible "
                    "with the CUDA runtime (v%d)" % (driver_version, runtime_version))
    else:
        warning("Unable to check compatibility of NVidia driver and runtime")


@memoized_func
def lscpu():
    try:
        p1 = Popen(['lscpu'], stdout=PIPE, stderr=PIPE)
    except OSError:
        return {}
    output, _ = p1.communicate()
    if output:
        lines = output.decode("utf-8").strip().split('\n')
        mapper = {}
        # Using split(':', 1) to avoid splitting lines where lscpu shows vulnerabilities
        # on some CPUs: https://askubuntu.com/questions/1248273/lscpu-vulnerabilities
        for k, v in [tuple(i.split(':', 1)) for i in lines]:
            try:
                mapper[k] = int(v)
            except ValueError:
                mapper[k] = v.strip()
        return mapper
    else:
        return {}


@memoized_func
def get_platform():
    """Attempt Platform autodetection."""

    try:
        cpu_info = get_cpu_info()
        brand = cpu_info['brand'].lower()
        if 'xeon' in brand:
            try:
                # Is it a Xeon?
                mapper = {
                    'v2': 'ivb',
                    'v3': 'hsw',
                    'v4': 'bdw',
                    'v5': 'skx',
                    'v6': 'klx',
                    'v7': 'clx'
                }
                return platform_registry[mapper[brand.split()[4]]]
            except:
                pass
            if 'phi' in brand:
                # Intel Xeon Phi?
                return platform_registry['knl']
            # Unknown Xeon ? May happen on some virtualized systems...
            return platform_registry['intel64']
        elif 'intel' in brand:
            # Most likely a desktop i3/i5/i7
            return platform_registry['intel64']
        elif 'power8' in brand:
            return platform_registry['power8']
        elif 'power9' in brand:
            return platform_registry['power8']
        elif 'arm' in brand:
            return platform_registry['arm']
        elif 'm1' in brand:
            return platform_registry['m1']
        elif 'amd' in brand:
            return platform_registry['amd']
    except:
        pass

    # Unable to detect platform. Stick to default...
    return ANYCPU


class Platform:

    registry = {}
    """
    The Platform registry.

    Each new Platform instance is automatically added to the registry.
    """

    max_mem_trans_nbytes = None
    """Maximum memory transaction size in bytes."""

    def __init__(self, name):
        self.name = name

        self.registry[name] = self

    def __eq__(self, other):
        return isinstance(other, Platform) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    @classmethod
    def _mro(cls):
        return [Platform]

    def __call__(self):
        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return "TargetPlatform[%s]" % self.name

    def _detect_isa(self):
        return 'unknown'

    @property
    def numa_domains(self):
        """
        Number of NUMA domains, or None if unknown.
        """
        return 1

    @property
    def threads_per_core(self):
        return self.cores_logical // self.cores_physical

    @property
    def cores_physical_per_numa_domain(self):
        return self.cores_physical // self.numa_domains

    @property
    def memtotal(self):
        """Physical memory size in bytes, or None if unknown."""
        return None

    def memavail(self, *args, **kwargs):
        """Available physical memory in bytes, or None if unknown."""
        return None

    def max_mem_trans_size(self, dtype):
        """
        Number of items of type `dtype` that can be transferred in a single
        memory transaction.
        """
        itemsize = np.dtype(dtype).itemsize

        # NOTE: This method conservatively uses the node's `max_mem_trans_size`,
        # instead of self's, so that we always pad by a compatible amount should
        # the user switch target platforms dynamically
        mmtb = node_max_mem_trans_nbytes(self)
        assert mmtb % itemsize == 0

        return int(mmtb / itemsize)

    def limits(self, compiler=None, language=None):
        """
        Return the architecture-specific limits for the given compiler and
        language.
        """
        return {
            'max-par-dims': sys.maxsize,
            'max-block-dims': sys.maxsize,
        }

    def supports(self, query, language=None):
        """
        Return True if the platform supports a given feature, False otherwise.
        """
        return False


class Cpu64(Platform):

    # The vast majority of CPUs have a 64-byte cache line size
    max_mem_trans_nbytes = 64

    # The known ISAs are to be provided by the subclasses
    known_isas = ()

    def __init__(self, name, cores_logical=None, cores_physical=None, isa=None):
        super().__init__(name)

        cpu_info = get_cpu_info()

        self.cores_logical = cores_logical or cpu_info['logical']
        self.cores_physical = cores_physical or cpu_info['physical']
        self.isa = isa or self._detect_isa()

    @classmethod
    def _mro(cls):
        # Retain only the CPU Platforms
        retval = []
        for i in cls.mro():
            if issubclass(i, Cpu64):
                retval.append(i)
            else:
                break
        return retval

    def _detect_isa(self):
        for i in reversed(self.known_isas):
            if any(j.startswith(i) for j in as_tuple(get_cpu_info()['flags'])):
                # Using `startswith`, rather than `==`, as a flag such as 'avx512'
                # appears as 'avx512f, avx512cd, ...'
                return i
        return 'cpp'

    @property
    def simd_reg_nbytes(self):
        """
        Size in bytes of a SIMD register.
        """
        return isa_registry.get(self.isa, 0)

    def simd_items_per_reg(self, dtype):
        """
        Number of items of type `dtype` that fit in a SIMD register.
        """
        assert self.simd_reg_nbytes % np.dtype(dtype).itemsize == 0
        return int(self.simd_reg_nbytes / np.dtype(dtype).itemsize)

    @cached_property
    def numa_domains(self):
        try:
            return int(lscpu()['NUMA node(s)'])
        except (ValueError, TypeError, KeyError):
            warning("NUMA domain count autodetection failed, assuming 1")
            return 1

    @cached_property
    def memtotal(self):
        return psutil.virtual_memory().total

    def memavail(self, *args, **kwargs):
        return psutil.virtual_memory().available


class Intel64(Cpu64):

    known_isas = ('cpp', 'sse', 'avx', 'avx2', 'avx512')


class IntelSkylake(Intel64):
    pass


class IntelGoldenCove(Intel64):
    pass


class Arm(Cpu64):

    known_isas = ('fp', 'asimd', 'asimdrdm')


class AppleArm(Arm):

    @cached_property
    def march(self):
        sysinfo = run(["sysctl", "-n", "machdep.cpu.brand_string"],
                      stdout=PIPE, stderr=DEVNULL).stdout.decode("utf-8")
        mx = sysinfo.split(' ')[1].lower()
        # Currently clang only supports up to m2
        return min(mx, 'm2')


class Graviton(Arm):

    @property
    def version(self):
        return int(self.name.split('graviton')[-1])

    @cached_property
    def march(self):
        if self.version >= 4:
            return 'neoverse-v2'
        elif self.version == 3:
            return 'neoverse-v1'
        else:
            return 'neoverse-n1'


class Cortex(Arm):

    @property
    def version(self):
        return int(self.name.split('cortexa')[-1])

    @cached_property
    def march(self):
        return 'armv8-a+crc+simd'

    @cached_property
    def mtune(self):
        return f'cortex-a{self.version}'


class NvidiaArm(Arm):

    @cached_property
    def march(self):
        if self.name == 'grace':
            return 'neoverse-v2'
        else:
            return 'native'


class Amd(Cpu64):

    known_isas = ('cpp', 'sse', 'avx', 'avx2')


class Power(Cpu64):

    def _detect_isa(self):
        return 'altivec'


class Device(Platform):

    def __init__(self, name, cores_logical=None, cores_physical=None, isa='cpp',
                 max_threads_per_block=1024, max_threads_dimx=1024,
                 max_threads_dimy=1024, max_threads_dimz=64,
                 max_thread_block_cluster_size=8):
        super().__init__(name)

        cpu_info = get_cpu_info()

        self.cores_logical = cores_logical or cpu_info['logical']
        self.cores_physical = cores_physical or cpu_info['physical']
        self.isa = isa

        self.max_threads_per_block = max_threads_per_block
        self.max_threads_dimx = max_threads_dimx
        self.max_threads_dimy = max_threads_dimy
        self.max_threads_dimz = max_threads_dimz
        self.max_thread_block_cluster_size = max_thread_block_cluster_size

    @classmethod
    def _mro(cls):
        # Retain only the Device Platforms
        retval = []
        for i in cls.mro():
            if issubclass(i, Device):
                retval.append(i)
            else:
                break
        return retval

    @property
    def march(self):
        return None

    @cached_property
    def numa_domains(self):
        info = get_gpu_info()
        try:
            return info['ncards']
        except KeyError:
            return None

    @cached_property
    def memtotal(self):
        info = get_gpu_info()
        try:
            return info['mem.total']()
        except (AttributeError, KeyError):
            return None

    def memavail(self, deviceid=0):
        """
        The amount of memory currently available on device.
        """
        info = get_gpu_info()
        try:
            return info['mem.free'](deviceid)
        except (AttributeError, KeyError):
            return None

    def limits(self, compiler=None, language=None):
        return {
            'max-par-dims': 3,
            'max-block-dims': 3,
        }


class IntelDevice(Device):

    max_mem_trans_nbytes = 64

    def __init__(self, *args, sub_group_size=32, **kwargs):
        super().__init__(*args, **kwargs)

        self.sub_group_size = sub_group_size

    @property
    def march(self):
        return ''


class NvidiaDevice(Device):

    max_mem_trans_nbytes = 128

    @cached_property
    def march(self):
        info = get_gpu_info()
        if info:
            architecture = info['architecture']
            if 'tesla' in architecture.lower():
                return 'tesla'
        return None

    def supports(self, query, language=None):
        if language != 'cuda':
            return False

        cc = get_nvidia_cc()
        if cc is None:
            # Something off with `get_nvidia_cc` on this system
            warning(f"Couldn't establish if `query={query}` is supported on this "
                    "system. Assuming it is not.")
            return False
        elif query == 'async-loads' and cc >= 80:
            # Asynchronous pipeline loads -- introduced in Ampere
            return True
        elif query in ('tma', 'thread-block-cluster') and cc >= 90:
            # Tensor Memory Accelerator -- introduced in Hopper
            return True
        else:
            return False


class Volta(NvidiaDevice):
    pass


class Ampere(Volta):

    def supports(self, query, language=None):
        if query == 'async-loads':
            return True
        else:
            return super().supports(query, language)


class Hopper(Ampere):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_thread_block_cluster_size', 16)
        super().__init__(*args, **kwargs)

    def supports(self, query, language=None):
        if query in ('tma', 'thread-block-cluster'):
            return True
        else:
            return super().supports(query, language)


class Blackwell(Hopper):
    pass


class AmdDevice(Device):

    max_mem_trans_nbytes = 256

    @cached_property
    def march(cls):
        # TODO: this corresponds to Vega, which acts as the fallback `march`
        # in case we don't manage to detect the actual `march`. Can we improve this?
        fallback = 'gfx900'

        # The AMD's AOMP compiler toolkit ships the `mygpu` program to (quoting
        # from the --help):
        #
        #     Print out the real gpu name for the current system
        #     or for the codename specified with -getgpuname option.
        #     mygpu will only print values accepted by cuda clang in
        #     the clang argument --cuda-gpu-arch.
        try:
            p1 = Popen(['offload-arch'], stdout=PIPE, stderr=PIPE)
        except OSError:
            try:
                p1 = Popen(['mygpu', '-d', fallback], stdout=PIPE, stderr=PIPE)
            except OSError:
                pass
            return fallback

        output, _ = p1.communicate()
        if output:
            return output.decode("utf-8").strip()
        else:
            return fallback


@memoized_func
def node_max_mem_trans_nbytes(platform):
    """
    Return the maximum memory transaction size in bytes for the underlying
    node, which, as such, takes into account all available platforms.
    """
    mmtb0 = platform.max_mem_trans_nbytes

    if isinstance(platform, Cpu64):
        gpu_info = get_gpu_info()
        if not gpu_info:
            # This node may simply not have a GPU
            return mmtb0

        mapper = {
            'NVIDIA': NvidiaDevice,
            'AMD': AmdDevice,
            'INTEL': IntelDevice,
        }
        try:
            mmtb1 = mapper[gpu_info['vendor']].max_mem_trans_nbytes
            return max(mmtb0, mmtb1)
        except KeyError:
            # Fallback -- act even more conservatively
            mmtb1 = max(p.max_mem_trans_nbytes for p in
                        [NvidiaDevice, AmdDevice, IntelDevice])
            mmtb = max(mmtb0, mmtb1)
            warning("Unable to determine GPU type, assuming a maximum memory "
                    f"transaction size of {mmtb} bytes")
            return mmtb
    elif isinstance(platform, Device):
        return max(Cpu64.max_mem_trans_nbytes, mmtb0)
    else:
        assert False, f"Unknown platform type: {type(platform)}"


# CPUs
ANYCPU = Cpu64('cpu64')
CPU64_DUMMY = Intel64('cpu64-dummy', cores_logical=2, cores_physical=1, isa='sse')

INTEL64 = Intel64('intel64')
SNB = Intel64('snb')  # Sandy Bridge
IVB = Intel64('ivb')  # Ivy Bridge
HSW = Intel64('hsw')  # Haswell
BDW = Intel64('bdw', isa='avx2')  # Broadwell
KNL = Intel64('knl')  # Knights Landing
KNL7210 = Intel64('knl7210', cores_logical=256, cores_physical=64, isa='avx512')
SKX = IntelSkylake('skx')  # Skylake
KLX = IntelSkylake('klx')  # Kaby Lake
CLX = IntelSkylake('clx')  # Coffee Lake
CLK = IntelSkylake('clk')  # Cascade Lake
SPR = IntelGoldenCove('spr')  # Sapphire Rapids

ARM = Arm('arm')
GRAVITON2 = Graviton('graviton2')
GRAVITON3 = Graviton('graviton3')
GRAVITON4 = Graviton('graviton4')
M1 = AppleArm('m1')
M2 = AppleArm('m2')
M3 = AppleArm('m3')
CORTEX = Cortex('cortex')
CORTEXA76 = Cortex('cortexa76')
GRACE = NvidiaArm('grace')

AMD = Amd('amd')

POWER8 = Power('power8')
POWER9 = Power('power9')

# Devices
ANYGPU = Cpu64('gpu')

NVIDIAX = NvidiaDevice('nvidiaX')
VOLTA = Volta('volta')
AMPERE = Ampere('ampere')
HOPPER = Hopper('hopper')
BLACKWELL = Blackwell('blackwell')

AMDGPUX = AmdDevice('amdgpuX')

INTELGPUX = IntelDevice('intelgpuX')
PVC = IntelDevice('pvc')  # Legacy codename for MAX GPUs
INTELGPUMAX = IntelDevice('intelgpuMAX')
MAX1100 = IntelDevice('max1100')
MAX1550 = IntelDevice('max1550')

platform_registry = Platform.registry
platform_registry['cpu64'] = get_platform  # Autodetection

isa_registry = {
    'cpp': 16,
    'sse': 16,
    'avx': 32,
    'avx2': 32,
    'avx512': 64,
    'altivec': 16,
    'fp': 8,
    'asimd': 16,
    'asimdrdm': 16
}
"""Size in bytes of a SIMD register in known ISAs."""
