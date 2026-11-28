"""Default allocator selection through Devito's configuration and factory."""

import os
import subprocess
import sys

import numpy as np
import pytest

from devito import ALLOC_ALIGNED, ALLOC_GUARD, Function, Grid, configuration, switchconfig
from devito.data.allocators import default_allocator, register_allocator


class TestDefaultAllocator:

    def test_restore_automatic_selection(self):
        previous = configuration['allocator']
        with switchconfig(allocator='fallback'):
            assert configuration['allocator'] == 'fallback'
            with switchconfig(allocator=None, develop_mode=True):
                assert configuration['allocator'] == 'auto'
                assert default_allocator() is ALLOC_GUARD
            assert configuration['allocator'] == 'fallback'
        assert configuration['allocator'] == previous

    def test_automatic(self):
        with switchconfig(allocator=None, develop_mode=True):
            assert default_allocator() is ALLOC_GUARD

    def test_registered(self):
        register_allocator('test-configured', ALLOC_ALIGNED)
        with switchconfig(allocator='test-configured', develop_mode=True):
            assert default_allocator() is ALLOC_ALIGNED
            f = Function(name='configured', grid=Grid(shape=(4, 4)))
            assert f._allocator is ALLOC_ALIGNED
            assert isinstance(f.data, np.ndarray)
            explicit = Function(name='explicit', grid=f.grid, allocator=ALLOC_GUARD)
            assert explicit._allocator is ALLOC_GUARD
            assert default_allocator('fallback') is ALLOC_ALIGNED

    def test_unknown(self):
        with switchconfig(allocator='unknown-allocator'), \
                pytest.raises(ValueError, match='Unknown allocator'):
            default_allocator()

    def test_environment(self):
        environment = dict(os.environ, DEVITO_ALLOCATOR='fallback')
        result = subprocess.run([sys.executable, '-c', '''
from devito import ALLOC_ALIGNED, configuration
from devito.data.allocators import default_allocator
assert configuration['allocator'] == 'fallback'
assert default_allocator() is ALLOC_ALIGNED
'''], env=environment, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
