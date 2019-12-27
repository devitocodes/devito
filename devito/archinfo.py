"""Collection of utilities to detect properties of the underlying architecture."""

from subprocess import PIPE, Popen

import numpy as np
import cpuinfo
import psutil

from devito.logger import warning
from devito.tools.memoization import memoized_func

__all__ = ['platform_registry',
           'INTEL64', 'SNB', 'IVB', 'HSW', 'BDW', 'SKX', 'KNL', 'KNL7210',
           'ARM',
           'POWER8', 'POWER9']


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
                flags = [i for i in lines if i.startswith('flags')][0]
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

            return None

        cpu_info['flags'] = get_cpu_flags()
        cpu_info['brand'] = get_cpu_brand()

    if not all(i in cpu_info for i in ('flags', 'brand')):
        # Fallback
        ci = cpuinfo.get_cpu_info()
        cpu_info['flags'] = ci.get('flags')
        cpu_info['brand'] = ci.get('brand')

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
            # Fallback 2: we might end up here on more exotic platforms such a Power8
            # Hopefully we can rely on `lscpu`
            try:
                physical = lscpu()['Core(s) per socket'] * lscpu()['Socket(s)']
            except KeyError:
                warning("Physical core count autodetection failed")
                physical = 1
    cpu_info['physical'] = physical

    return cpu_info


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
        for k, v in [tuple(i.split(':')) for i in lines]:
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
            # Unknown Xeon ? May happen on some virtualizes systems...
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
    except:
        pass

    # Unable to detect platform. Stick to default...
    return CPU64


class Platform(object):

    def __init__(self, name, **kwargs):
        self.name = name

        cpu_info = get_cpu_info()

        self.cores_logical = kwargs.get('cores_logical', cpu_info['logical'])
        self.cores_physical = kwargs.get('cores_physical', cpu_info['physical'])
        self.isa = kwargs.get('isa', self._detect_isa())

    def __call__(self):
        return self

    def __str__(self):
        return self.name

    def __repr__(self):
        return "TargetPlatform[%s]" % self.name

    def _detect_isa(self):
        return 'unknown'

    @property
    def threads_per_core(self):
        return self.cores_logical // self.cores_physical

    @property
    def simd_reg_size(self):
        """Size in bytes of a SIMD register."""
        return isa_registry.get(self.isa, 0)

    def simd_items_per_reg(self, dtype):
        """Number of items of type ``dtype`` that can fit in a SIMD register."""
        assert self.simd_reg_size % np.dtype(dtype).itemsize == 0
        return int(self.simd_reg_size / np.dtype(dtype).itemsize)


class Cpu64(Platform):

    # The known ISAs will be overwritten in the specialized classes
    known_isas = ()

    def _detect_isa(self):
        for i in reversed(self.known_isas):
            if any(j.startswith(i) for j in get_cpu_info()['flags']):
                # Using `startswith`, rather than `==`, as a flag such as 'avx512'
                # appears as 'avx512f, avx512cd, ...'
                return i
        return 'cpp'


class Intel64(Cpu64):

    known_isas = ('cpp', 'sse', 'avx', 'avx2', 'avx512')


class Arm(Cpu64):

    known_isas = ('fp', 'asimd', 'asimdrdm')


class Power(Cpu64):

    def _detect_isa(self):
        return 'altivec'


class Device(Platform):

    def __init__(self, name, cores_logical=1, cores_physical=1, isa='cpp'):
        self.name = name

        self.cores_logical = cores_logical
        self.cores_physical = cores_physical
        self.isa = isa


# CPUs
CPU64 = Cpu64('cpu64')
CPU64_DUMMY = Intel64('cpu64-dummy', cores_logical=2, cores_physical=1, isa='sse')
INTEL64 = Intel64('intel64')
SNB = Intel64('snb')
IVB = Intel64('ivb')
HSW = Intel64('hsw')
BDW = Intel64('bdw')
SKX = Intel64('skx')
KLX = Intel64('klx')
CLX = Intel64('clx')
KNL = Intel64('knl')
KNL7210 = Intel64('knl', cores_logical=256, cores_physical=64, isa='avx512')
ARM = Arm('arm')
POWER8 = Power('power8')
POWER9 = Power('power9')

# Devices
NVIDIAX = Device('nvidiax')


platform_registry = {
    'cpu64-dummy': CPU64_DUMMY,
    'intel64': INTEL64,
    'snb': SNB,  # Sandy Bridge
    'ivb': IVB,  # Ivy Bridge
    'hsw': HSW,  # Haswell
    'bdw': BDW,  # Broadwell
    'skx': SKX,  # Skylake
    'klx': KLX,  # Kaby Lake
    'clx': CLX,  # Coffee Lake
    'knl': KNL,
    'knl7210': KNL7210,
    'arm': ARM,
    'power8': POWER8,
    'power9': POWER9,
    'nvidiaX': NVIDIAX
}
"""
Registry dict for deriving Platform classes according to the environment variable
DEVITO_PLATFORM. Developers should add new platform classes here.
"""
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
