import pytest

from devito import configuration, switchconfig
from devito.arch import archinfo
from devito.arch.compiler import GNUCompiler, compiler_registry, sniff_compiler_version


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


@pytest.mark.parametrize('brand,expected', [
    ('Apple M1', 'm1'),
    ('Apple M2 Pro', 'm2'),
    ('Apple M3 Max', 'm3'),
    ('Apple M4 Max', 'm3'),
])
def test_detect_apple_silicon(monkeypatch, brand, expected):
    monkeypatch.setattr(archinfo, 'get_cpu_info', lambda: {'brand': brand})
    assert archinfo.get_platform.func() is archinfo.platform_registry[expected]


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
