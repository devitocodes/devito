"""Collection of utilities to detect properties of the underlying architecture."""

from functools import partial
from subprocess import PIPE, Popen

import numpy as np
import cpuinfo
import psutil

from devito.logger import warning
from devito.tools.memoization import memoized_func

__all__ = ['platform_registry']


@memoized_func
def get_cpu_info():
    try:
        # On linux, the following should work and is super quick
        cpu_info = {}
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        get = lambda k: [i for i in lines if i.startswith(k)][0].split(':')[1].strip()
        cpu_info['flags'] = get('flags').split()
        cpu_info['brand'] = get('model name')
        # TODO: Other info omitted as currently unused
    except:
        # Fallback: rely on the slower `cpuinfo`
        cpu_info = cpuinfo.get_cpu_info()

    # Distinguish between *physical* and *logical* cores
    logical = psutil.cpu_count(logical=True)
    if logical:
        cpu_info['logical'] = logical
        cpu_info['physical'] = psutil.cpu_count(logical=False)
    else:
        # Fallback: we might end up here on more exotic platforms such a Power8
        # We attempt to use `lscpu` -- it's a bit sad because we have to spawn
        # a new process, or simply `lscpu` won't be available
        p1 = Popen(['lscpu'], stdout=PIPE, stderr=PIPE)
        output, _ = p1.communicate()
        if output:
            lines = output.decode("utf-8").split('\n')
            get = lambda k: [i for i in lines if i.startswith(k)][0].split(':')[1].strip()
            cpu_info['logical'] = int(get('CPU(s)'))
            cpu_info['physical'] = int(get('Core(s)')) * int(get('Socket(s)'))
        else:
            warning("Physical/logical core count autodetection failed")
            cpu_info['logical'] = 1
            cpu_info['physical'] = 1

    return cpu_info


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
        return platform_registry[platform]()
    except:
        pass

    # No luck so far; try instead from the brand name
    try:
        cpu_info = get_cpu_info()
        platform = cpu_info['brand'].split()[4]
        platform = {'v2': 'ivb', 'v3': 'hsw', 'v4': 'bdw', 'v5': 'skx'}[platform]
        return platform_registry[platform]()
    except:
        pass

    # Stick to default
    return CPU64('cpu64')


class Platform(object):

    def __init__(self, name):
        self.name = name

        self.cores_logical = 1
        self.cores_physical = 1

        self.isa = 'cpp'

    def __str__(self):
        return self.name

    def __repr__(self):
        return "DevitoTargetPlatform[%s]" % self.name

    @property
    def threads_per_core(self):
        return self.cores_logical // self.cores_physical


class CPU64(Platform):

    def __init__(self, name):
        self.name = name

        cpu_info = get_cpu_info()
        self.cores_logical = cpu_info['logical']
        self.cores_physical = cpu_info['physical']

        self.isa = self._detect_isa()

    def _detect_isa(self):
        return 'cpp'

    @property
    def simd_reg_size(self):
        """Size in bytes of a SIMD register."""
        return isa_registry[self.isa]

    def simd_items_per_reg(self, dtype):
        """Number of items of type ``dtype`` that can fit in a SIMD register."""
        assert self.simd_reg_size % np.dtype(dtype).itemsize == 0
        return int(self.simd_reg_size / np.dtype(dtype).itemsize)


class Intel64(CPU64):

    def _detect_isa(self):
        known_isas = ['cpp', 'sse', 'avx', 'avx2', 'avx512']
        for i in reversed(known_isas):
            if any(j.startswith(i) for j in get_cpu_info()['flags']):
                # Using `startswith`, rather than `==`, as a flag such as 'avx512'
                # appears as 'avx512f, avx512cd, ...'
                return i
        return 'cpp'


class KNL7210(Intel64):

    def __init__(self, name):
        self.name = name
        self.cores_logical = 256
        self.cores_physical = 64
        self.isa = 'avx512'


class Arm(CPU64):
    pass


class Power(CPU64):

    def _detect_isa(self):
        return 'altivec'


isa_registry = {
    'cpp': 16,
    'sse': 16,
    'avx': 32,
    'avx2': 32,
    'avx512': 64,
    'altivec': 16
}
"""Size in bytes of a SIMD register in known ISAs."""


platform_registry = {
    'cpu64': get_platform,  # Trigger autodetection
    'intel64': partial(Intel64, 'intel64'),
    'snb': partial(Intel64, 'snb'),
    'ivb': partial(Intel64, 'ivb'),
    'hsw': partial(Intel64, 'hsw'),
    'bdw': partial(Intel64, 'bdw'),
    'skx': partial(Intel64, 'skx'),
    'knl': partial(Intel64, 'knl'),
    'knl7210': partial(KNL7210, name='knl'),
    'arm': partial(Arm, name='arm'),
    'power8': partial(Power, name='power'),
    'power9': partial(Power, name='power')
}
"""
Registry dict for deriving Platform classes according to the environment variable
DEVITO_PLATFORM. Developers should add new platform classes here.
"""
