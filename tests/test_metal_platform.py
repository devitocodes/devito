"""
Tests for Apple GPU platform detection and Metal compiler infrastructure.
"""

import platform

import pytest

from devito.arch.archinfo import AppleGPU, Device, APPLEGPUX
from devito.arch.compiler import MetalCompiler, compiler_registry
from devito.operator.registry import OperatorRegistry


class TestAppleGPUPlatform:
    """Tests for the AppleGPU platform class."""

    def test_apple_gpu_inherits_device(self):
        assert issubclass(AppleGPU, Device)

    def test_apple_gpu_thread_group_size(self):
        assert AppleGPU.thread_group_size == 32

    def test_apple_gpu_max_mem_trans_nbytes(self):
        assert AppleGPU.max_mem_trans_nbytes == 128

    def test_applegpux_instance(self):
        assert APPLEGPUX.name == 'applegpuX'
        assert isinstance(APPLEGPUX, AppleGPU)
        assert isinstance(APPLEGPUX, Device)

    def test_apple_gpu_defaults(self):
        gpu = AppleGPU('test-gpu')
        assert gpu.max_threads_per_block == 1024
        assert gpu.max_threads_dimx == 1024
        assert gpu.max_threads_dimy == 1024
        assert gpu.max_threads_dimz == 1024

    def test_apple_gpu_custom_kwargs(self):
        gpu = AppleGPU('test-gpu', max_threads_per_block=512)
        assert gpu.max_threads_per_block == 512

    @pytest.mark.skipif(platform.system() != 'Darwin' or
                        platform.machine() != 'arm64',
                        reason="Apple Silicon only")
    def test_apple_gpu_march_detection(self):
        gpu = AppleGPU('test-gpu')
        march = gpu.march
        assert march is not None
        assert march.startswith('m')

    def test_apple_gpu_mro(self):
        mro = AppleGPU._mro()
        assert AppleGPU in mro
        assert Device in mro

    def test_apple_gpu_in_platform_registry(self):
        from devito.arch.archinfo import platform_registry
        assert 'applegpuX' in platform_registry


class TestMetalCompiler:
    """Tests for the MetalCompiler class."""

    def test_metal_in_compiler_registry(self):
        assert 'metal' in compiler_registry

    def test_metal_compiler_class(self):
        assert compiler_registry['metal'] is MetalCompiler

    def test_metal_compiler_is_cpp(self):
        assert MetalCompiler._default_cpp is True

    def test_metal_compiler_std(self):
        comp = MetalCompiler.__new__(MetalCompiler)
        assert comp.std == 'c++17'

    @pytest.mark.skipif(platform.system() != 'Darwin',
                        reason="macOS only")
    def test_metal_compiler_creation(self):
        comp = MetalCompiler(platform=APPLEGPUX)
        assert comp.cc == 'clang++'
        assert comp.src_ext == 'cpp'
        assert '-std=c++17' in comp.cflags
        assert '-framework' in comp.ldflags
        assert 'Metal' in comp.ldflags
        assert 'Foundation' in comp.ldflags
        assert '-lobjc' in comp.ldflags

    @pytest.mark.skipif(platform.system() != 'Darwin',
                        reason="macOS only")
    def test_metal_compiler_no_c_std(self):
        """Ensure C99 std is not present, only C++17."""
        comp = MetalCompiler(platform=APPLEGPUX)
        assert '-std=c99' not in comp.cflags
        assert '-std=c++14' not in comp.cflags


class TestMetalLanguageConfig:
    """Tests for metal language registration."""

    def test_metal_in_languages(self):
        assert 'metal' in OperatorRegistry._languages
