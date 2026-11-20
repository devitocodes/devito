import ctypes
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from subprocess import check_call
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from devito import configuration, switchconfig
from devito.arch import BLACKWELL, NVIDIAX, archinfo, platform_registry
from devito.arch.archinfo import AppleArm, get_apple_platform, get_m1_llvm_path
from devito.arch.compiler import (
    Compiler, CudaCompiler, CustomCompiler, GNUCompiler, NvidiaCompiler,
    compiler_registry, sniff_compiler_version
)
from devito.tools import os_helper


@pytest.mark.parametrize("cc", [
    "doesn'texist",
    "/root/doesn'texist",
])
def test_sniff_compiler_version(cc):
    with pytest.raises(RuntimeError, match=cc):
        sniff_compiler_version(cc)


@pytest.mark.skipif(not hasattr(os, 'RTLD_DEEPBIND') or shutil.which('gcc') is None,
                    reason='Requires ELF deep binding and GCC')
@pytest.mark.parametrize('custom', [False, True])
def test_compiler_runtime_symbols(tmp_path, monkeypatch, custom):
    """A globally loaded runtime must not interpose on NVIDIA kernel dependencies."""
    monkeypatch.setattr('devito.arch.compiler.sniff_compiler_version',
                        lambda cc, **kwargs: Version('16.0'))
    symbol = f'devito_test_runtime_{str(custom).lower()}'
    vendor = f'vendor_{custom}'
    sources = {
        'global': f'int {symbol}(void) {{ return 17; }}',
        vendor: f'int {symbol}(void) {{ return 42; }}',
        'kernel': f'extern int {symbol}(void); int check(void) {{ return {symbol}(); }}'
    }
    for name, source in sources.items():
        src = tmp_path / f'{name}.c'
        src.write_text(source)
        args = ['gcc', '-shared', '-fPIC', str(src), '-o',
                str(tmp_path / f'lib{name}.so')]
        if name == 'kernel':
            args += [f'-L{tmp_path}', f'-l{vendor}', f'-Wl,-rpath,{tmp_path}']
        check_call(args)

    global_runtime = ctypes.CDLL(str(tmp_path / 'libglobal.so'), mode=ctypes.RTLD_GLOBAL)
    assert getattr(global_runtime, symbol)() == 17
    platform = platform_registry['cpu64-dummy']()
    compiler = (CustomCompiler(base=NvidiaCompiler, platform=platform) if custom else
                NvidiaCompiler(platform=platform))
    monkeypatch.setattr(compiler, 'get_jit_dir', lambda: tmp_path)
    assert compiler.load('libkernel').check() == 42


@pytest.mark.parametrize("cc", ['gcc-4.9', 'gcc-11', 'gcc', 'gcc-14', 'gcc-123'])
def test_gcc(cc):
    assert cc in compiler_registry


@pytest.mark.parametrize("brand, expected", [
    ("Apple M1", 'm1'),
    ("Apple M1 (Virtual)", 'm1'),
    ("Apple M2 Pro", 'm2'),
    ("Apple M3 Max", 'm3'),
    ("Apple M4 Max", 'm3'),
    ("Apple M5", 'm3'),
])
def test_apple_platform(brand, expected):
    platform = get_apple_platform(brand)
    assert isinstance(platform, AppleArm)
    assert platform.name == expected


@pytest.mark.parametrize('loaded', [
    [], ['/numerical/lib/libomp.dylib'], ['/usr/lib/libSystem.B.dylib']
])
def test_apple_openmp_runtime(monkeypatch, loaded):
    monkeypatch.setattr('devito.arch.archinfo.sys.platform', 'darwin')
    monkeypatch.setattr('devito.arch.archinfo.run', lambda *a, **kw: Mock(
        stdout=b'Target: arm64-apple-darwin\nInstalledDir: /llvm/bin\n'))
    monkeypatch.setattr('devito.arch.archinfo.os.path.exists', lambda p: True)
    dyld = Mock()
    dyld._dyld_image_count.return_value = len(loaded)
    dyld._dyld_get_image_name.side_effect = [p.encode() for p in loaded]
    monkeypatch.setattr('devito.arch.archinfo.ctypes.CDLL', lambda name: dyld)

    paths = get_m1_llvm_path.func('openmp')
    assert paths['include'] == '/llvm/include'
    assert paths['libs'] == ('/numerical/lib' if '/numerical/lib/libomp.dylib'
                             in loaded else '/llvm/lib')


def test_switcharch():
    old_compiler = configuration['compiler']
    with switchconfig(compiler='gcc-4.9'):
        tmp_comp = configuration['compiler']
        assert isinstance(tmp_comp, GNUCompiler)
        assert tmp_comp.suffix == '4.9'

    tmp_comp = configuration['compiler']
    assert isinstance(tmp_comp, old_compiler.__class__)
    assert old_compiler.suffix == tmp_comp.suffix
    assert old_compiler.name == tmp_comp.name


def test_save_header_concurrent(tmp_path, monkeypatch):
    """
    The JIT dir is shared across processes, so a header being rewritten must
    never be seen truncated or half-written by a concurrent reader.
    """
    compiler = configuration['compiler']
    monkeypatch.setattr(compiler, 'get_jit_dir', lambda: tmp_path)

    hfile = tmp_path.joinpath('header.h')
    codes = ['a'*2**20, 'b'*2**20]

    def write(code):
        for _ in range(20):
            compiler.save_header('header', code)

    def read(writers):
        seen = set()
        while not all(w.done() for w in writers):
            with suppress(FileNotFoundError):
                seen.add(hfile.read_text())
        return seen

    with ThreadPoolExecutor(max_workers=8) as tpe:
        writers = [tpe.submit(write, codes[i % 2]) for i in range(4)]
        readers = [tpe.submit(read, writers) for _ in range(4)]
        for w in writers:
            w.result()
        seen = set.union(*[r.result() for r in readers])

    # Only complete headers are ever read, and no temporary file is left behind
    assert seen and seen <= set(codes)
    assert list(tmp_path.iterdir()) == [hfile]


@pytest.mark.parametrize('platform,cc,target', [
    (NVIDIAX, None, '-arch=native'),
    (NVIDIAX, 80, '-arch=sm_80'),
    (NVIDIAX, 90, '-arch=sm_90'),
    (NVIDIAX, 100, '-gencode=arch=compute_100a,code=sm_100a'),
    (NVIDIAX, 103, '-arch=sm_103'),
    (NVIDIAX, 120, '-arch=sm_120'),
    (BLACKWELL, None, '-gencode=arch=compute_100a,code=sm_100a'),
    (BLACKWELL, 100, '-gencode=arch=compute_100a,code=sm_100a'),
    (BLACKWELL, 103, '-arch=sm_103'),
    (BLACKWELL, 120, '-arch=sm_120')
])
def test_cuda_target(monkeypatch, platform, cc, target):
    with switchconfig(mpi=False), monkeypatch.context() as m:
        m.setattr('devito.arch.compiler.get_nvidia_cc', lambda: cc)
        m.setattr('devito.arch.compiler.get_cuda_version', lambda: Version('13.0'))
        m.setattr('devito.arch.compiler.check_cuda_runtime', lambda: None)
        compiler = CudaCompiler(platform=platform)

    flags = [i for i in compiler.cflags if i.startswith(('-arch=', '-gencode='))]
    assert flags == [target]


@pytest.mark.parametrize('language', [None, 'cuda', 'openacc', 'hip', 'sycl'])
@pytest.mark.parametrize('platform,cc,expected', [
    ('nvidiaX', None, False),
    ('nvidiaX', 90, False),
    ('nvidiaX', 100, True),
    ('nvidiaX', 120, True),
    ('cpu64', 100, False),
    ('volta', 100, True),
    ('ampere', 100, True),
    ('hopper', 100, True),
    ('blackwell', None, True),
    ('blackwell', 90, True)
])
def test_fp32x2_support(platform, cc, expected, language, monkeypatch):
    monkeypatch.setattr('devito.arch.archinfo.get_nvidia_cc', lambda: cc)
    platform = platform_registry[platform]()
    assert platform.supports('fp32x2', language) is (expected and language == 'cuda')


@pytest.mark.parametrize('memory', ['total', 'free', 'used'])
def test_nvidia_memory_units(monkeypatch, memory):
    responses = {
        ('nvidia-smi', '-L'): (
            b'GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-0)\n'
            b'GPU 1: NVIDIA A100-SXM4-80GB (UUID: GPU-1)\n'
        ),
        ('nvidia-smi', f'--query-gpu=memory.{memory}', '--format=csv'): (
            f'memory.{memory} [MiB]\n1024 MiB\n81920 MiB\n'.encode()
        )
    }

    def popen(command, **kwargs):
        return SimpleNamespace(stdout=BytesIO(responses[tuple(command)]))

    monkeypatch.setattr(archinfo, 'Popen', popen)
    info = archinfo.get_gpu_info.func()

    assert info[f'mem.{memory}'](0) == 1024**3
    assert info[f'mem.{memory}'](1) == 80*1024**3


class TestLibraryPublication:
    """Concurrent Operator restoration must never expose a partial library."""

    def test_complete_binary(self, tmp_path, monkeypatch):
        compiler = SimpleNamespace(get_jit_dir=lambda: tmp_path, so_ext='.so')
        target = tmp_path/'shared.so'
        binary = b'ELF-binary-content'*1024
        original = os_helper.os.link
        publications = []

        def observe(source, destination):
            assert not target.exists()
            assert Path(source).read_bytes() == binary
            publications.append(source)
            original(source, destination)

        monkeypatch.setattr(os_helper.os, 'link', observe)
        Compiler.save(compiler, 'shared', binary)
        assert publications
        assert target.read_bytes() == binary
        assert list(tmp_path.iterdir()) == [target]
        Compiler.save(compiler, 'shared', b'different')
        assert target.read_bytes() == binary

    def test_concurrent_binary(self, tmp_path):
        compiler = SimpleNamespace(get_jit_dir=lambda: tmp_path, so_ext='.so')
        target = tmp_path/'shared.so'
        binary = b'ELF-binary-content'*(1 << 18)

        def restore(_):
            Compiler.save(compiler, 'shared', binary)
            assert target.read_bytes() == binary

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(restore, range(32)))
        assert list(tmp_path.iterdir()) == [target]
