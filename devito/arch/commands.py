import json
import re
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from subprocess import run

from devito.warnings import warn


__all__ = [
    'lscpu',
    'lshw',
    'lspci',
    'lspci_gpu',
    'nvidia_smi',
    'nvidia_smi_memory',
    'proc_cpuinfo',
    'rocm_smi',
    'rocm_smi_memory',
    'sycl_ls',
    'xpu_smi_memory'
]


# Unified dataclass for describing RAM, SWAP and GPU RAM (maybe disk too?)
@dataclass
class Memory:
    total: int
    free: int
    used: int


def text2dict(text):
    return {
        line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
        for line in text.splitlines()
    }


def cast2numeric(adict):
    # Try and convert numeric values
    for k, v in adict.items():
        if not v:
            adict[k] = None
            continue
        for cast in [int, float]:
            with suppress(ValueError):
                adict[k] = cast(v)
                break
    return adict


def set2range(aset):
    if len(aset) == 1:
        r = aset.pop()
    else:
        r = aset
    return r


def proc_cpuinfo():
    """ Creates a `dict` containing the information in `/proc/cpuinfo`
    """
    # Obtain CPU info as text
    try:
        with open('/proc/cpuinfo') as f:
            lines = f.read()
        command = 'cat /proc/cpuinfo'
    except FileNotFoundError:
        lines = ''
        command = '`/proc/cpuinfo` not found'
        warn(f'File {command}')

    hwthreads = lines.strip().split('\n\n')
    logical = len(hwthreads)

    info = []
    for hwt in hwthreads:
        info.append(cast2numeric(text2dict(hwt)))

    # Nightmare
    variations = defaultdict(set)
    for hwt in info:
        for k, v in hwt.items():
            variations[k].add(v)

    final = {}
    for k, v in variations.items():
        final[k] = set2range(v)

    # `cpu MHz` is a "live" value so ignore it
    with suppress(KeyError):
        del final['cpu MHz']

    final['command'] = command

    return final


def lscpu():
    """ Creates a `dict` containing the information from `lscpu`
    """
    # Use `lscpu -J` if available (not available prior to v2.30, cerca 2017)
    try:
        ret = run(['lscpu', '-J'], capture_output=True, text=True)
        info = {
            x['field'].rstrip(':'): x['data']
            for x in json.loads(ret.stdout)['lscpu']
        }
        info['command'] = 'lscpu -J'
    except Exception:
        info = None

    # Use `lscpu` if `-J` argument is not available
    if info is None:
        try:
            ret = run(['lscpu'], capture_output=True, text=True)
            info = text2dict(ret.stdout)
            info['command'] = 'lscpu'
        except Exception:
            msg = '`lscpu` not found'
            warn(f'Command {msg}')
            info = {'command': msg}

    return cast2numeric(info)


def nvidia_smi():
    """ Creates a `list[dict]` containing the information from `nvidia-smi`
    """
    gpu_infos = []

    ret = run(['nvidia-smi', '-L'], capture_output=True, text=True)
    lines = ret.stdout.splitlines()

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

    return gpu_infos


def nvidia_smi_memory():
    ret = run(
        [
            'nvidia-smi',
            '--query-gpu=--query-gpu=memory.total,memory.free,memory.used',
            '--format=csv'
        ],
        capture_output=True,
        text=True
    )
    lines = ret.stdout.splitlines()[1:-1]
    mem = []
    for line in lines:
        with suppress(Exception):
            # This should not fail, unless `nvidia-smi` changes
            # the output format (though we still have tests in place that
            # will catch this)
            vals = []
            for value in line.split(', '):
                _, v, unit = re.split(r'([0-9]+)\s', line)
                assert unit == 'MiB'
                vals.append(int(v)*10**6)
            mem.append(Memory(*vals))
    return mem


def rocm_smi():
    gpu_infos = {}

    ret = run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
    lines = ret.stdout.splitlines()

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
                gpu_infos[gid]['vendor'] = 'AMD'
                gpu_infos[gid]['architecture'] = 'unspecified'

    return list(gpu_infos.values())


def rocm_smi_memory():
    ret = run(
        ['rocm-smi', '--showmeminfo', 'vram', '--json'],
        capture_output=True,
        text=True
    )
    json_data = json.loads(ret.stdout)

    mem = []
    for card in json_data:
        assert len(json_data[k]) == 2
        info = cast2numeric(json_data[k])
        vals = [0]*3
        for k, v in info:
            if 'Total' in k:
                vals[0] = v
            if 'Used' in k:
                vals[2] = v
        vals[1] = vals[0] - vals[2]
        mem.append(Memory(*vals))
    return mem


def sycl_ls():
    gpu_infos = {}

    # sycl-ls sometimes finds gpu twice with opencl and without so
    # we need to make sure we don't get duplicates
    selected_platform = None
    platform_block = ""

    ret = run(["sycl-ls", "--verbose"], capture_output=True, text=True)
    sycl_output = ret.stdout

    # Extract platform blocks
    platforms = re.findall(
        r"Platform \[#(\d+)\]:([\s\S]*?)(?=Platform \[#\d+\]:|$)",
        sycl_output
    )

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
    devices = re.findall(
        r"Device \[#(\d+)\]:([\s\S]*?)(?=Device \[#\d+\]:|$)",
        platform_block
    )

    for device_id, device_block in devices:
        if re.search(r"^\s*Type\s*:\s*gpu", device_block, re.MULTILINE):
            name_match = re.search(r"^\s*Name\s*:\s*(.+)", device_block, re.MULTILINE)

            if name_match:
                name = name_match.group(1).strip()

                # Store GPU info with correct physical ID
                gpu_infos[device_id] = {
                    "physicalid": device_id,
                    "product": name,
                    'vendor': 'INTEL',
                    'architecture': 'unspecified'
                }

    return list(gpu_infos.values())


def xpu_smi_memory():
    raise NotImplementedError


def filter_real_gpus(gpus):
    """ Filter out virtual GPUs from a list of GPU dictionaries
    """
    return list(filter(lambda g: 'virtual' not in g['product'].lower(), gpus))


def lshw():
    ret = run(['lshw', '-C', 'video'], capture_output=True, text=True)
    raw_info = ret.stdout

    # Parse the information for all the devices listed with lshw
    gpu_infos = []
    for block in raw_info.split('display')[1:]:
        # Separate the output block into lines for processing
        lines = block.splitlines()

        # Define the processing functions
        if lines:
            gpu_info = {}
            for line in lines:
                # Architecture
                if line.lstrip().startswith('product') and '[' in line:
                    arch_match = re.search(r'\[([\w\s]+)\]', line)
                    if arch_match:
                        gpu_info['architecture'] = arch_match.group(1)

                for keyword in ['product', 'vendor', 'physical id']:
                    if line.lstrip().startswith(keyword):
                        gpu_info[keyword] = line.split(':')[1].lstrip()

            gpu_info.set_default('architecture', 'unspecified')

            if 'physical id' in gpu_info:
                gpu_info['physicalid'] = gpu_info.pop('physical id')

            gpu_infos.append(gpu_info)

    return filter_real_gpus(gpu_infos)


def old_lspci():
    ret = run(['lspci'], capture_output=True, text=True)
    lines = ret.stdout.splitlines()

    # Note: due to the single line descriptive format of lspci, 'vendor'
    # and 'physicalid' elements cannot be reliably extracted so are left None

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

    return filter_real_gpus(gpu_infos)


def lspci():
    ret = run(['lspci', '-mm', '-v'], capture_output=True, text=True)
    blocks = ret.stdout.strip().split('\n\n')

    pci_info = {}
    for device in blocks:
        line = device.splitlines()
        slot = line[0].split(':', 1)[1].strip()
        info = {
            (part:=l.split(':', 1))[0]: part[1].strip()
            for l in line[1:]
        }
        pci_info[slot] = info
    return pci_info


def lspci_gpu():
    devices = lspci()

    gpu_info = []
    for dev in devices.values():
        # Graphics cards are listed as VGA, 3D controllers or Displays in lspci
        if any(sub in dev['Class'] for sub in ('VGA', '3D', 'Display')):
            info = {}
            info['class'] = dev['Class']
            info['vendor'] = dev['Vendor']
            info['device'] = dev['Device']
            info['product'] = dev['Device']
            info['architecture'] = 'unspecified'
            info['physicalid'] = 'unknown'
            gpu_info.append(info)

    return filter_real_gpus(gpu_info)
