import pytest

from devito.arch.compiler import sniff_compiler_version, compiler_registry


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
