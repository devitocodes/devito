from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import pytest

from devito import configuration, switchconfig
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
