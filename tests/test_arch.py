import pytest
from packaging.version import Version

from devito import configuration, switchconfig
from devito.arch import BLACKWELL, NVIDIAX
from devito.arch.compiler import (
    CudaCompiler, GNUCompiler, compiler_registry, sniff_compiler_version
)


@pytest.mark.parametrize("cc", [
    "doesn'texist",
    "/root/doesn'texist",
])
def test_sniff_compiler_version(cc):
    with pytest.raises(RuntimeError, match=cc):
        sniff_compiler_version(cc)


@pytest.mark.parametrize("cc", ['gcc-4.9', 'gcc-11', 'gcc', 'gcc-14', 'gcc-123'])
def test_gcc(cc):
    assert cc in compiler_registry


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
