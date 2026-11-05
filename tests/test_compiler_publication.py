"""Concurrent Operator restoration must never expose a partial library."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from devito.arch.compiler import Compiler
from devito.tools import os_helper


def test_complete_binary_publication(tmp_path, monkeypatch):
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


def test_concurrent_binary_publication(tmp_path):
    compiler = SimpleNamespace(get_jit_dir=lambda: tmp_path, so_ext='.so')
    target = tmp_path/'shared.so'
    binary = b'ELF-binary-content'*(1 << 18)

    def restore(_):
        Compiler.save(compiler, 'shared', binary)
        assert target.read_bytes() == binary

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore, range(32)))
    assert list(tmp_path.iterdir()) == [target]
