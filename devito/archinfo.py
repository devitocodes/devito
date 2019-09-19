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
    # Obtain textual cpu info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    cpu_info = {}

    # Extract CPU flags and branch
    if lines:
        try:
            get = lambda k: [i for i in lines if i.startswith(k)][0].split(':')[1].strip()
            cpu_info['flags'] = get('flags').split()
            cpu_info['brand'] = get('model name')
        except IndexError:
            # The /proc/cpuinfo format doesn't follow a standard, and on some
            # more or less exotic combinations of OS and platform it might not
            # be what we expect, hence ending up here
            pass
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

    # TODO: cannot autodetect the following platforms yet:
    # ['arm', 'power8', 'power9']

    try:
        # First, try leveraging `gcc`
        p1 = Popen(['gcc', '-march=native', '-Q', '--help=target'],
                   stdout=PIPE, stderr=PIPE)
        p2 = Popen(['grep', 'march'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        output, _ = p2.communicate()
        platform = output.decode("utf-8").split()[1]
        # Full list of possible `platform` values at this point at:
        # https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
        platform = {'sandybridge': 'snb', 'ivybridge': 'ivb', 'haswell': 'hsw',
                    'broadwell': 'bdw', 'skylake': 'skx', 'knl': 'knl'}[platform]
        return platform_registry[platform]
    except:
        pass

    # No luck so far; try instead from the brand name
    try:
        cpu_info = get_cpu_info()
        platform = cpu_info['brand'].split()[4]
        platform = {'v2': 'ivb', 'v3': 'hsw', 'v4': 'bdw', 'v5': 'skx'}[platform]
        return platform_registry[platform]
    except:
        pass

    # Stick to default
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

    def _detect_isa(self):
        return 'cpp'


class Intel64(Cpu64):

    def _detect_isa(self):
        known_isas = ['cpp', 'sse', 'avx', 'avx2', 'avx512']
        for i in reversed(known_isas):
            if any(j.startswith(i) for j in get_cpu_info()['flags']):
                # Using `startswith`, rather than `==`, as a flag such as 'avx512'
                # appears as 'avx512f, avx512cd, ...'
                return i
        return 'cpp'


class Arm(Cpu64):

    def _detect_isa(self):
        known_isas = ['fp', 'asimd', 'asimdrdm']
        for i in reversed(known_isas):
            if any(j.startswith(i) for j in get_cpu_info()['flags']):
                return i
        return 'fp'


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
INTEL64 = Intel64('intel64')
SNB = Intel64('snb')
IVB = Intel64('ivb')
HSW = Intel64('hsw')
BDW = Intel64('bdw')
SKX = Intel64('skx')
KNL = Intel64('knl')
KNL7210 = Intel64('knl', cores_logical=256, cores_physical=64, isa='avx512')
ARM = Arm('arm')
POWER8 = Power('power8')
POWER9 = Power('power9')

# Devices
NVIDIAX = Device('nvidiax')


platform_registry = {
    'intel64': INTEL64,
    'snb': SNB,
    'ivb': IVB,
    'hsw': HSW,
    'bdw': BDW,
    'skx': SKX,
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
