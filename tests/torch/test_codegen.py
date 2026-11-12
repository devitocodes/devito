"""Torch language selection, optimized C++ printing and native JIT execution."""

import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest
from packaging.version import Version

torch = pytest.importorskip('torch')

from conftest import _R  # noqa: E402
from devito import (  # noqa: E402
    Constant, Dimension, Eq, Function, Grid, Operator, TimeFunction, TorchAllocator,
    configuration, switchconfig
)
from devito.arch import M1, Cpu64  # noqa: E402
from devito.arch.compiler import (  # noqa: E402
    ClangCompiler, CustomCompiler, GNUCompiler, IntelCompiler, NvidiaCompiler,
    OneapiCompiler
)
from devito.core.torch import Cpu64AdvTorchOperator  # noqa: E402
from devito.ir import Dereference, FindNodes  # noqa: E402
from devito.ir.iet.torch import TorchCGen  # noqa: E402
from devito.operator.operator import parse_kwargs  # noqa: E402


@pytest.mark.parametrize('first', ['devito', 'torch'])
def test_openmp_runtime_interoperability(first):
    # Keep a runtime abort in the child so its diagnostic is reported by pytest
    code = f'import {first}\n' + dedent('''
        import torch
        from devito import Eq, Function, Grid, Operator, TorchAllocator

        for platform in (None, 'cpu64-dummy'):
            for allocator in (None, TorchAllocator()):
                f = Function(name='f', grid=Grid(shape=(128, 128)), space_order=0,
                             allocator=allocator)
                language = 'torch' if allocator else 'openmp'
                op = Operator(Eq(f, f+1), language=language, platform=platform,
                              opt=('blocking', {'openmp': True}))
                assert '#pragma omp parallel' in str(op.ccode)
                op.apply()
                assert (f.data == 1).all()
                assert torch.ones((512, 512)).sum().item() == 512**2
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('opt', ['noop', 'advanced', 'advanced-fsg'])
def test_standard_operator(opt):
    u = TimeFunction(name='u', grid=Grid(shape=(8, 8)), allocator=TorchAllocator())
    op = Operator(Eq(u.forward, u+1), language='torch', opt=opt)
    assert isinstance(op, Operator)
    assert op._ccode_handler is TorchCGen
    if opt == 'advanced':
        assert type(op) is Cpu64AdvTorchOperator
    assert op._compiler.cc == op._compiler.CXX + (
        f'-{op._compiler.suffix}' if op._compiler.suffix else '')
    assert op._compiler.src_ext == 'cpp'
    assert 'at::Tensor' in str(op.ccode)
    assert '.data_ptr<float>()' in str(op.ccode)
    assert 'for (int time' in str(op.ccode)
    assert 'pybind11' not in str(op.ccode)
    ptr, version = u._data_allocated.data_ptr(), u._data_allocated._version
    op.apply(time_M=3)
    torch.testing.assert_close(u.data[0], torch.full((8, 8), 4.))
    assert u._data_allocated.data_ptr() == ptr
    assert u._data_allocated._version > version
    # Standard ctypes loader and cache, not torch's extension mechanism.
    assert op.cfunction is op._cfunction
    assert op._lib._name.endswith(op._compiler.so_ext)
    assert str(op._compiler.get_jit_dir()) in op._lib._name


@pytest.mark.parametrize('parallel,dtype', [(False, np.float32), (True, np.float64)])
def test_cpp_stencil(parallel, dtype):
    grid = Grid(shape=(12, 12, 12), extent=(11., 11., 11.))
    u = TimeFunction(name='u', grid=grid, dtype=dtype, space_order=2,
                     allocator=TorchAllocator())
    ref = TimeFunction(name='ref', grid=grid, dtype=dtype, space_order=2)
    u.data[0, 6, 6, 6] = 1
    ref.data[0, 6, 6, 6] = 1
    scale = Constant(name='scale', value=0.01, dtype=dtype)
    opt = ('advanced', {'blockinner': True, 'openmp': parallel})
    op = Operator(Eq(u.forward, u + scale*u.laplace), language='torch',
                  opt=opt)
    reference = Operator(Eq(ref.forward, ref + scale*ref.laplace),
                         opt=opt)
    op.apply(time_M=2)
    reference.apply(time_M=2)
    np.testing.assert_allclose(u.data.numpy(), ref.data, rtol=1e-5, atol=1e-6)
    assert any(d.is_Block for d in op.dimensions)
    assert '#pragma omp simd' in str(op.ccode)
    assert ('#pragma omp parallel' in str(op.ccode)) == parallel


def test_allocator_with_c_language():
    # Storage choice is independent of language, as with other allocators.
    f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator())
    Operator(Eq(f, f+1), language='C').apply()
    assert torch.all(f.data == 1)


@pytest.mark.parametrize('dtype', [np.int32, np.int64])
def test_integer_codegen(dtype):
    f = Function(name='integer', grid=Grid(shape=(8, 8)), space_order=0,
                 dtype=dtype, allocator=TorchAllocator())
    op = Operator(Eq(f, f+3), language='torch')
    assert f'data_ptr<{np.dtype(dtype).name}_t>()' in str(op.ccode)
    op.apply()
    assert torch.all(f.data == 3)


def test_custom_compiler(monkeypatch):
    monkeypatch.setenv('CXX', configuration['compiler'].CXX)
    monkeypatch.setenv('CXXFLAGS', '-O2 -fPIC')
    f = Function(name='custom', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator())
    op = Operator(Eq(f, f+1), language='torch', compiler='custom')
    assert '-std=c++20' in op._compiler.cflags
    assert '-std=c99' not in op._compiler.cflags
    op.apply()
    assert torch.all(f.data == 1)


def test_torch_language_with_host_allocator():
    f = Function(name='host', grid=Grid(shape=(8, 8)), space_order=0)
    previous_ext = configuration['compiler'].src_ext
    with switchconfig(language='torch'):
        assert configuration['compiler'].src_ext == previous_ext
        op = Operator(Eq(f, f+1))
        op.apply()
    assert configuration['compiler'].src_ext == previous_ext
    assert np.all(f.data == 1)


def test_preemptive_autotuning_preserves_tensor_state(monkeypatch):
    u = TimeFunction(name='tuned', grid=Grid(shape=(12, 12, 12)),
                     allocator=TorchAllocator())
    op = Operator(Eq(u.forward, u.dx+1), language='torch',
                  opt=('blocking', {'openmp': False, 'blockinner': True}))

    def no_numpy(*args, **kwargs):
        raise AssertionError('Autotuning must preserve tensor-owned storage')

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'numpy', no_numpy)
        op.apply(time_M=0, autotune=('basic', 'preemptive'))

    assert op._state['autotuning'][-1]['runs'] > 0
    torch.testing.assert_close(u.data[0], torch.zeros((12, 12, 12)))
    torch.testing.assert_close(u.data[1], torch.ones((12, 12, 12)))


def test_function_names_do_not_collide_with_tensor_views():
    grid = Grid(shape=(8, 8))
    f = Function(name='field', grid=grid, space_order=0, allocator=TorchAllocator())
    g = Function(name='field_tensor', grid=grid, space_order=0,
                 allocator=TorchAllocator())
    f.data[:] = 1
    g.data[:] = 2
    op = Operator(Eq(f, f+g), language='torch')
    op.apply()
    torch.testing.assert_close(f.data, torch.full((8, 8), 3.))


@pytest.mark.parametrize('compiler_type,flag', [
    (ClangCompiler, '-fopenmp'), (IntelCompiler, '-qopenmp'),
    (OneapiCompiler, '-fopenmp')
])
def test_cpu_compiler_enables_openmp(monkeypatch, compiler_type, flag):
    # Inspect toolchain flags without requiring every vendor compiler locally.
    monkeypatch.setattr('devito.arch.compiler.sniff_compiler_version',
                        lambda cc, **kwargs: Version('16.0'))
    compiler = compiler_type(cpp=True, openmp=True,
                             platform=Cpu64('test', isa='cpp'))
    assert flag in compiler.ldflags
    assert compiler.src_ext == 'cpp'


@pytest.mark.parametrize('openmp', [False, True])
@pytest.mark.parametrize('platform', [M1, Cpu64('test', isa='cpp')])
def test_apple_compiler_honors_openmp_option(monkeypatch, openmp, platform):
    monkeypatch.setattr('devito.arch.compiler.sys.platform', 'darwin')
    monkeypatch.setattr('devito.arch.compiler.sniff_compiler_version',
                        lambda cc, **kwargs: Version('16.0'))
    monkeypatch.setattr(type(M1), 'march', 'm1')
    paths = {'libs': '/llvm/lib', 'include': '/llvm/include'}
    monkeypatch.setattr('devito.arch.compiler.get_m1_llvm_path', lambda language: paths)
    compiler = ClangCompiler(cpp=True, openmp=openmp, language='torch', platform=platform)
    assert ('-fopenmp' in compiler.ldflags) == openmp
    assert ('-Wl,-rpath,/llvm/lib' in compiler.ldflags) == openmp


def test_custom_compiler_honors_cxxflags(monkeypatch):
    for name in ('CC', 'CXX', 'CFLAGS', 'LDFLAGS'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CXXFLAGS', '-O2 -fPIC')
    compiler = CustomCompiler(cpp=True)
    assert '-O2' in compiler.cflags
    assert compiler.cflags.index('-O2') > compiler.cflags.index('-O3')


@pytest.mark.parametrize('rotate', [False, True])
def test_thread_private_temporaries(rotate):
    grid = Grid(shape=(8, 8, 8))
    x, y, z = grid.dimensions
    t = grid.stepping_dim
    d = Dimension(name='d')
    allocator = TorchAllocator()
    c = Function(name='coefficient', grid=grid, shape=(2, 8), dimensions=(d, z),
                 allocator=allocator)
    u = TimeFunction(name='temporary_state', grid=grid, space_order=3,
                     allocator=allocator)
    ref = TimeFunction(name='temporary_reference', grid=grid, space_order=3,
                       allocator=allocator)
    c.data_with_halo[:] = 1
    u.data_with_halo[:] = 1.5
    ref.data_with_halo[:] = 1.5
    expression = Eq(u.forward,
                    _R(_R(c[0, z]*u[t, x+1, y+1, z] +
                          c[1, z+1]*u[t, x+1, y+1, z+1]) +
                       _R(c[0, z]*u[t, x+2, y+2, z] +
                          c[1, z+1]*u[t, x+2, y+2, z+1]) +
                       _R(u[t, x, y+1, z+1] + 3*u[t, x+1, y+1, z+1]) +
                       _R(u[t, x, y+3, z+1] + 3*u[t, x+1, y+3, z+1])))
    reference = Operator(expression, language='C', opt='noop')
    op = Operator(expression, language='torch',
                  opt=('advanced', {'openmp': True, 'min-storage': True,
                                    'cire-mingain': 0, 'cire-rotate': rotate}))
    assert FindNodes(Dereference).visit(op)
    reference.apply(temporary_state=ref, time_M=1)
    op.apply(time_M=1)
    torch.testing.assert_close(u.data, ref.data)


def test_torch_compiler_requirements_are_operator_local():
    template = configuration['compiler']
    previous = (template.cc, template.src_ext, list(template.cflags),
                list(template.include_dirs), list(template.libraries))
    f = Function(name='local_toolchain', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator())
    op = Operator(Eq(f, f+1), language='torch')
    assert op._compiler.src_ext == 'cpp'
    assert '-std=c++20' in op._compiler.cflags
    assert 'torch_cpu' in op._compiler.libraries
    assert previous == (template.cc, template.src_ext, template.cflags,
                        template.include_dirs, template.libraries)
    op.apply()
    torch.testing.assert_close(f.data, torch.ones((8, 8)))


def test_cpp_only_compiler_accepts_backend_options(monkeypatch):
    monkeypatch.setattr('devito.arch.compiler.sniff_compiler_version',
                        lambda cc, **kwargs: Version('16.0'))
    compiler = NvidiaCompiler(cpp=True, openmp=True,
                              platform=Cpu64('test', isa='cpp'))
    assert compiler.src_ext == 'cpp'
    assert '-mp' in compiler.cflags


def test_normalize_kwargs_leaves_compiler_unchanged():
    kwargs = parse_kwargs(language='torch')
    compiler = kwargs['compiler']
    attrs = ('cflags', 'defines', 'include_dirs', 'libraries', 'library_dirs', 'ldflags')
    previous = {name: list(getattr(compiler, name)) for name in attrs}
    normalized = Cpu64AdvTorchOperator._normalize_kwargs(**kwargs)
    assert normalized['compiler'] is compiler
    assert previous == {name: getattr(compiler, name) for name in attrs}


@pytest.mark.parametrize('cpp,expected', [(False, 'c99'), (True, 'c++14')])
def test_compiler_standard_property(cpp, expected):
    compiler = GNUCompiler(cpp=cpp)
    assert compiler.std == expected
    assert f'-std={compiler.std}' in compiler.cflags


def test_compiler_standard_class_override_and_clone():
    class Cxx20Compiler(GNUCompiler):
        _cxxstd = 'c++20'

    compiler = Cxx20Compiler(cpp=True)
    clone = compiler.__new_with__()
    assert clone is not compiler
    assert compiler.std == clone.std == 'c++20'
    assert '-std=c++20' in clone.cflags
    assert clone.cc == compiler.cc
