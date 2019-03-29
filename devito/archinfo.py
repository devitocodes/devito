"""Collection of utilities to detect properties of the underlying architecture."""

from subprocess import PIPE, Popen

import numpy as np
import cpuinfo

from devito.tools.memoization import memoized_func

__all__ = ['known_isas', 'known_platforms', 'get_cpu_info', 'get_isa',
           'get_platform', 'get_simd_reg_size', 'get_simd_items_per_reg']


known_isas = ['cpp', 'sse', 'avx', 'avx2', 'avx512']
"""All known Instruction Set Architectures."""

known_platforms = ['intel64', 'snb', 'ivb', 'hsw', 'bdw', 'skx', 'knl']
"""All known platforms."""


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
def get_isa():
    """
    Retrieve the target architecture's highest SIMD instruction set architecture.
    """
    cpu_info = get_cpu_info()
    isa = 'cpp'
    for i in reversed(known_isas):
        if any(j.startswith(i) for j in cpu_info['flags']):
            # Using `startswith`, rather than `==`, as a flag such as 'avx512'
            # appears as 'avx512f, avx512cd, ...'
            isa = i
            break
    return isa


@memoized_func
def get_platform():
    """
    Retrieve the target architecture's codename.
    """
    try:
        # First, try leveraging `gcc`
        p1 = Popen(['gcc', '-march=native', '-Q', '--help=target'],
                   stdout=PIPE, stderr=PIPE)
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
    if platform not in known_platforms:
        platform = 'intel64'
    return platform


@memoized_func
def get_simd_reg_size(isa):
    """
    Retrieve the size in bytes of a SIMD register of the target architecture.
    """
    assert isa in known_isas
    return {'cpp': 16, 'sse': 16, 'avx': 32, 'avx2': 32, 'avx512': 64}[isa]


@memoized_func
def get_simd_items_per_reg(isa, dtype):
    """
    Retrieve the number of items of type ``dtype`` that can fit in a SIMD register
    of the target architecture.
    """
    simd_size = get_simd_reg_size(isa)
    assert simd_size % np.dtype(dtype).itemsize == 0
    return int(simd_size / np.dtype(dtype).itemsize)
