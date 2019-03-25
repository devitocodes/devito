"""Collection of utilities to detect properties of the underlying architecture."""

from subprocess import PIPE, Popen

import numpy as np
import cpuinfo

from devito.parameters import configuration
from devito.tools.memoization import memoized_func

__all__ = ['get_cpu_info', 'get_simd_isa', 'get_platform', 'simdinfo', 'get_simd_items']


@memoized_func
def get_cpu_info():
    try:
        # On linux, the following should work and is super quick
        cpu_info = {}
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()
        get = lambda key: [i for i in lines if i.startswith(key)][0].split(':')[1].strip()
        cpu_info['flags'] = get('flags').split()
        cpu_info['brand'] = get('model name')
        # TODO: Other info omitted as currently unused
        return cpu_info
    except:
        # Fallback: rely on the slower `cpuinfo`
        return cpuinfo.get_cpu_info()


@memoized_func
def get_simd_isa():
    """
    Retrieve the highest SIMD ISA on the current architecture.
    """
    cpu_info = get_cpu_info()
    isa = configuration._defaults['isa']
    for i in reversed(configuration._accepted['isa']):
        if any(j.startswith(i) for j in cpu_info['flags']):
            # Using `startswith`, rather than `==`, as a flag such as 'avx512'
            # appears as 'avx512f, avx512cd, ...'
            isa = i
            break
    return isa


@memoized_func
def get_platform():
    """
    Retrieve the architecture codename.
    """
    try:
        # First, try leveraging `gcc`
        p1 = Popen(['gcc', '-march=native', '-Q', '--help=target'], stdout=PIPE)
        p2 = Popen(['grep', 'march'], stdin=p1.stdout, stdout=PIPE)
        p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
        output, _ = p2.communicate()
        platform = output.decode("utf-8").split()[1]
        # Full list of possible /platform/ values at this point at:
        # https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html
        platform = {'sandybridge': 'snb', 'ivybridge': 'ivb', 'haswell': 'hsw',
                    'broadwell': 'bdw', 'skylake': 'skx', 'knl': 'knl'}[platform]
    except:
        # Then, try infer from the brand name, otherwise fallback to default
        try:
            cpu_info = get_cpu_info()
            platform = cpu_info['brand'].split()[4]
            platform = {'v2': 'ivb', 'v3': 'hsw', 'v4': 'bdw', 'v5': 'skx'}[platform]
        except:
            platform = None
    # Is it a known platform?
    if platform not in configuration._accepted['platform']:
        platform = configuration._defaults['platform']
    return platform


simdinfo = {
    # Sizes in bytes of a vector register
    'sse': 16,
    'avx': 32,
    'avx2': 32,
    'avx512': 64
}
"""
SIMD generic info
"""


@memoized_func
def get_simd_items(dtype):
    """
    Determine the number of items of type ``dtype`` that can fit in a SIMD
    register on the current architecture.
    """
    simd_size = simdinfo[get_simd_isa()]
    assert simd_size % np.dtype(dtype).itemsize == 0
    return int(simd_size / np.dtype(dtype).itemsize)
