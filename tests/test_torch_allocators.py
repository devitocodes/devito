"""Tensor allocation, external ownership and native argument validation."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from devito import Eq, Function, Grid, Operator, TorchAllocator  # noqa: E402
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
