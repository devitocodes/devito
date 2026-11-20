"""Tensor allocation, external ownership and native argument validation."""

import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from devito import (  # noqa: E402
    ALLOC_ALIGNED, Eq, Function, Grid, Operator, TimeFunction, TorchAllocator,
    configuration, switchconfig
)
from devito.data.allocators import default_allocator  # noqa: E402
from devito.exceptions import InvalidArgument  # noqa: E402


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32, np.int64])
def test_allocator(dtype):
    f = Function(name='f', grid=Grid(shape=(8, 8)), dtype=dtype,
                 allocator=TorchAllocator())
    assert type(f) is Function
    assert isinstance(f.data, torch.Tensor)
    f.data.fill_(3)
    assert torch.all(f.data == 3)
    data = f._data_allocated
    obj = f._C_make_dataobj(**{f.name: data})
    assert obj._obj.data.value == data.data_ptr()
    assert obj._obj.underlying_array.data_ptr() == data.data_ptr()
    assert f._C_as_ndarray(obj).data_ptr() == data.data_ptr()
    snapshot = f.data_ro_domain
    snapshot.zero_()
    assert torch.all(f.data == 3)


def test_external_tensor(monkeypatch):
    tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator(tensor=tensor), initializer=lambda _: None)
    assert f.data.data_ptr() == tensor.data_ptr()
    op = Operator(Eq(f, 2*f+1), language='torch')
    expected = 2*tensor + 1

    def no_numpy(*args, **kwargs):
        raise AssertionError('Tensor data must not pass through NumPy')

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'numpy', no_numpy)
        op.apply()
    torch.testing.assert_close(tensor, expected)
    replacement = torch.zeros_like(tensor)
    op.apply(f=replacement)
    torch.testing.assert_close(replacement, torch.ones_like(tensor))
    torch.testing.assert_close(tensor, expected)


@pytest.mark.parametrize('initializer', [
    torch.ones((8, 8)), np.ones((8, 8)), [[1]*8]*8
])
def test_initializers(initializer):
    f = Function(name='initialized', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator(), initializer=initializer)
    assert torch.all(f.data == 1)


@pytest.mark.parametrize('kind', [
    'dtype', 'strides', 'grad', 'numpy', 'alignment', 'sparse'
])
def test_invalid_override(kind):
    f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator())
    op = Operator(Eq(f, f+1), language='torch')
    if kind == 'dtype':
        value = torch.zeros((8, 8), dtype=torch.float64)
    elif kind == 'strides':
        value = torch.zeros((8, 8)).T
    elif kind == 'sparse':
        value = torch.zeros((8, 8)).to_sparse()
    elif kind == 'grad':
        value = torch.zeros((8, 8), requires_grad=True)
    elif kind == 'alignment':
        value = torch.zeros(65)[1:].reshape(8, 8)
    else:
        value = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(InvalidArgument):
        op.arguments(f=value)


def test_default_cuda_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 2)
    allocator = TorchAllocator(device='cuda')
    assert allocator.device == torch.device('cuda:2')
    # A supplied tensor determines the device unless explicitly overridden.
    tensor = torch.empty((8, 8))
    assert TorchAllocator(tensor=tensor).device == tensor.device


def test_indexed_cpu_device():
    f = Function(name='cpu_index', grid=Grid(shape=(4, 4)),
                 allocator=TorchAllocator(device='cpu:0'))
    f.data[:] = 2
    torch.testing.assert_close(f.data, torch.full((4, 4), 2.))


class TestDefaultTorchAllocator:

    def test_configuration_and_overrides(self):
        with switchconfig(allocator='torch'):
            assert configuration['allocator'] == 'torch'
            assert isinstance(default_allocator(), TorchAllocator)
            assert isinstance(default_allocator('torch'), TorchAllocator)
            assert default_allocator('fallback') is ALLOC_ALIGNED
            f = Function(name='torch_default', grid=Grid(shape=(9, 13)))
            assert isinstance(f.data, torch.Tensor)
            explicit = Function(name='numpy_override', grid=f.grid,
                                allocator=ALLOC_ALIGNED)
            assert isinstance(explicit.data, np.ndarray)
            op = Operator(Eq(f, f+1), language='torch')
            op.apply()
            torch.testing.assert_close(f.data, torch.ones_like(f.data))

    def test_environment(self):
        result = subprocess.run([sys.executable, '-c', '''
import torch
from devito import Function, Grid, configuration
assert configuration['allocator'] == 'torch'
f = Function(name='f', grid=Grid(shape=(4, 4)))
assert isinstance(f.data, torch.Tensor)
'''], env=dict(os.environ, DEVITO_ALLOCATOR='torch'), capture_output=True,
            text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


class TestTensorLayout:

    @pytest.mark.parametrize('autopadding', [False, True])
    @pytest.mark.parametrize('dtype', [np.float32, np.float64])
    @pytest.mark.parametrize('external', [False, True])
    def test_halo_padding_and_compiled_execution(self, autopadding, dtype, external):
        with switchconfig(autopadding=autopadding):
            grid = Grid(shape=(9, 13), extent=(8., 12.))
            reference = TimeFunction(name='reference', grid=grid, space_order=2,
                                     dtype=dtype, allocator=ALLOC_ALIGNED)
            tensor = torch.zeros(tuple(reference.shape_allocated),
                                 dtype=torch.float32 if dtype == np.float32
                                 else torch.float64)
            allocator = TorchAllocator(tensor=tensor) if external else TorchAllocator()
            actual = TimeFunction(name='actual', grid=grid, space_order=2,
                                  dtype=dtype, allocator=allocator,
                                  initializer=lambda _: None)
            assert actual.shape_allocated == reference.shape_allocated
            assert actual._halo == reference._halo
            assert actual._padding == reference._padding
            if autopadding:
                assert actual.shape_allocated[-1] > grid.shape[-1] + 4
            raw = actual._data_allocated
            ref = reference._data_allocated
            assert tuple(i*raw.element_size() for i in raw.stride()) == ref.strides
            assert raw.data_ptr() % actual._data_alignment == 0
            assert ref.ctypes.data % reference._data_alignment == 0
            if external:
                assert raw.data_ptr() == tensor.data_ptr()
            raw.fill_(-123)
            ref.fill(-123)
            actual.data_with_halo[:] = reference.data_with_halo[:] = 0
            actual.data[0, 4, 6] = reference.data[0, 4, 6] = 1
            for function, language in [(reference, 'C'), (actual, 'torch')]:
                op = Operator(Eq(function.forward, function + .1*function.laplace),
                              language=language)
                op.apply(time_M=2)
            np.testing.assert_allclose(actual.data.numpy(), reference.data,
                                       rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(actual.data_with_halo.numpy(),
                                       reference.data_with_halo, rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(raw.numpy(), ref, rtol=1e-6, atol=1e-7)
            strides = tuple(i//ref.itemsize for i in reference.data.strides)
            assert actual.data.stride() == strides
            replacement = torch.zeros_like(raw)
            Operator(Eq(actual.forward, actual+1), language='torch').apply(
                actual=replacement, time_m=0, time_M=0)
            torch.testing.assert_close(replacement[actual._mask_domain][1],
                                       torch.ones(grid.shape, dtype=raw.dtype))
