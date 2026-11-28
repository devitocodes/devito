"""Compiled OpenACC host tests and CUDA tensor interoperability tests.

GCC host execution checks generated OpenACC kernels without requiring a GPU.
CUDA cases require CUDA-enabled PyTorch, an NVIDIA GPU, and the HPC SDK.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from devito import (  # noqa: E402
    Eq, Function, Grid, Operator, TimeFunction, TorchAllocator, configuration
)
from devito.arch import NVIDIAX  # noqa: E402
from devito.arch.compiler import GNUCompiler, compiler_registry  # noqa: E402
from devito.exceptions import InvalidArgument, InvalidOperator  # noqa: E402
from devito.passes.iet.languages.openacc import AccBB  # noqa: E402
from devito.symbolics import Macro  # noqa: E402


class HostOpenACCCompiler(GNUCompiler):
    """Use GCC's C++ OpenACC frontend for host-only execution checks."""

    _default_cpp = True

    def __init_finalize__(self, **kwargs):
        super().__init_finalize__(**kwargs)
        self.cflags.extend(['-fopenacc', '-foffload=disable'])
        self.ldflags.extend(['-fopenacc', '-foffload=disable'])


def _reference(value, steps):
    weight = value.new_tensor([[0, .1, 0], [.1, .6, .1], [0, .1, 0]])[None, None]
    # Compare the discretization at float32 precision, not cuDNN's optional TF32
    with torch.backends.cudnn.flags(allow_tf32=False):
        for _ in range(steps):
            value = torch.nn.functional.conv2d(value[None, None], weight, padding=1)[0, 0]
    return value


def _host_case(case, precision):
    """Execute an actual GCC OpenACC kernel against tensor-owned host storage."""
    AccBB.mapper[NVIDIAX] = Macro('acc_device_host')
    compiler_registry['openacc-host'] = HostOpenACCCompiler
    dtype = getattr(np, precision)
    grid = Grid(shape=(12, 12), extent=(11., 11.))
    allocator = TorchAllocator()
    u = TimeFunction(name='acc_u', grid=grid, dtype=dtype, space_order=2,
                     allocator=allocator)
    if case == 'mixed':
        coefficient = Function(name='coefficient', grid=grid, dtype=dtype)
        coefficient.data[:] = 2
        expression = Eq(u.forward, u+coefficient)
    else:
        expression = Eq(u.forward, u+.1*u.laplace) if case == 'stencil' else Eq(
            u.forward, u+1)
    op = Operator(expression, platform='nvidiaX', language='openacc',
                  compiler='openacc-host')
    if case == 'override':
        data = torch.zeros(tuple(u.shape_allocated), dtype=getattr(torch, precision))
        ptr = data.data_ptr()
        op.apply(acc_u=data, time_m=0, time_M=4)
        assert data.data_ptr() == ptr
        assert u._data is None
        torch.testing.assert_close(data[u._mask_domain][1], torch.full(
            grid.shape, 5., dtype=data.dtype))
    else:
        torch.manual_seed(3)
        initial = torch.randn(grid.shape, dtype=getattr(torch, precision))
        u.data[0].copy_(initial)
        ptr = u._data_allocated.data_ptr()
        op.apply(time_M=4)
        expected = _reference(initial, 5) if case == 'stencil' else initial + (
            10 if case == 'mixed' else 5)
        torch.testing.assert_close(u.data[5], expected)
        assert u._data_allocated.data_ptr() == ptr
    assert 'acc parallel loop' in str(op.ccode)
    assert op.cfunction is not None


@pytest.mark.skipif(shutil.which('gcc') is None or sys.platform != 'linux',
                    reason='Requires GCC OpenACC on Linux')
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('case', ['update', 'stencil', 'override', 'mixed'])
def test_openacc_host_execution(case, precision):
    # PyTorch 2.2 bundles an older libgomp without current GOACC symbols. Load
    # the system compiler's runtime first in a fresh interpreter, preserving
    # inherited preloads. This is host OpenACC execution, not CUDA validation.
    runtime = subprocess.check_output(['gcc', '-print-file-name=libgomp.so.1'],
                                      text=True).strip()
    environment = os.environ.copy()
    environment['LD_PRELOAD'] = ':'.join(filter(None, [
        runtime, environment.get('LD_PRELOAD')
    ]))
    environment['DEVITO_LOGGING'] = 'ERROR'
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             '--host', case, precision], env=environment,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


cuda_available = torch.cuda.is_available() and shutil.which('nvc++') is not None
cuda_required = pytest.mark.skipif(not cuda_available,
                                   reason='Requires CUDA PyTorch, NVIDIA GPU and nvc++')


@pytest.mark.skipif(os.environ.get('CI') != 'true' or
                    configuration['platform'] is not NVIDIAX,
                    reason='Requires the NVIDIA CI runtime')
def test_openacc_cuda_runtime():
    assert torch.cuda.is_available(), 'NVIDIA CI requires driver-compatible CUDA PyTorch'
    assert shutil.which('nvc++') is not None


def test_openacc_deviceptr_codegen():
    u = TimeFunction(name='u', grid=Grid(shape=(8, 8)),
                     allocator=TorchAllocator(device='cuda:0'))
    op = Operator(Eq(u.forward, u+1), platform='nvidiaX',
                  language='openacc', compiler='gcc')
    code = str(op.ccode)
    assert '#pragma acc parallel loop' in code
    assert 'deviceptr(u)' in code
    assert 'copyin(' not in code
    assert 'copyout(' not in code
    assert 'acc enter data' not in code
    assert 'acc exit data' not in code
    assert u._data is None


def test_openacc_host_access_rejected():
    u = TimeFunction(name='u', grid=Grid(shape=(8, 8)),
                     allocator=TorchAllocator(device='cuda:0'))
    with pytest.raises(InvalidOperator, match='host code'):
        Operator(Eq(u[0, 0, 0], 1), platform='nvidiaX',
                 language='openacc', compiler='gcc')


def test_openacc_mixed_storage_codegen():
    grid = Grid(shape=(8, 8))
    u = TimeFunction(name='u', grid=grid, allocator=TorchAllocator(device='cuda:0'))
    a = Function(name='a', grid=grid)
    op = Operator(Eq(u.forward, u+a), platform='nvidiaX',
                  language='openacc', compiler='gcc')
    code = str(op.ccode)
    assert 'deviceptr(u)' in code
    assert 'present(a)' in code
    assert 'copyin(a[' in code
    assert 'copyin(u[' not in code


@cuda_required
@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('case', ['owned', 'external', 'override', 'stencil', 'mixed'])
def test_openacc_cuda_storage(case, precision):
    grid = Grid(shape=(12, 12), extent=(11., 11.))
    dtype = getattr(np, precision)
    tensor_dtype = getattr(torch, precision)
    if case == 'external':
        # space_order=0 makes the full external allocation equal to the domain.
        data = torch.ones(grid.shape, dtype=tensor_dtype, device='cuda:0')
        f = Function(name='f', grid=grid, space_order=0, dtype=dtype,
                     allocator=TorchAllocator(tensor=data), initializer=lambda _: None)
        op = Operator(Eq(f, 2*f+1), platform='nvidiaX', language='openacc',
                      compiler='nvc')
        ptr = data.data_ptr()
        op.apply()
        torch.testing.assert_close(data, torch.full_like(data, 3))
        assert data.data_ptr() == ptr
        return
    u = TimeFunction(name='cuda_u', grid=grid, dtype=dtype, space_order=2,
                     allocator=TorchAllocator(device='cuda:0'))
    expression = Eq(u.forward, u+.1*u.laplace) if case == 'stencil' else Eq(
        u.forward, u+1)
    if case == 'mixed':
        coefficient = Function(name='coefficient', grid=grid, dtype=dtype)
        coefficient.data[:] = 2
        expression = Eq(u.forward, u+coefficient)
    op = Operator(expression, platform='nvidiaX', language='openacc', compiler='nvc')
    if case == 'override':
        data = torch.zeros(tuple(u.shape_allocated), dtype=tensor_dtype, device='cuda:0')
        # Producer work on a nondefault stream must finish before OpenACC runs.
        producer = torch.cuda.Stream()
        producer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(producer):
            data.fill_(2)
        ptr = data.data_ptr()
        op.apply(cuda_u=data, time_m=0, time_M=4)
        torch.testing.assert_close(data[u._mask_domain][1], torch.full(
            grid.shape, 7., dtype=tensor_dtype, device='cuda:0'))
        assert data.data_ptr() == ptr
    else:
        initial = torch.randn(grid.shape, dtype=tensor_dtype, device='cuda:0')
        u.data[0].copy_(initial)
        ptr = u._data_allocated.data_ptr()
        op.apply(time_M=4)
        expected = _reference(initial, 5) if case == 'stencil' else initial + (
            10 if case == 'mixed' else 5)
        torch.testing.assert_close(u.data[5], expected)
        assert u._data_allocated.data_ptr() == ptr
    code = str(op.ccode)
    assert 'deviceptr(cuda_u)' in code
    assert 'copyin(cuda_u[' not in code
    assert 'copyout(cuda_u[' not in code


@cuda_required
def test_openacc_cuda_rejects_incorrect_device():
    f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator(device='cuda:0'))
    op = Operator(Eq(f, f+1), platform='nvidiaX', language='openacc', compiler='nvc')
    with pytest.raises(InvalidArgument, match='deviceid'):
        op.apply(deviceid=1)


@cuda_required
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='Requires two CUDA devices')
def test_openacc_cuda_second_device():
    data = torch.ones((8, 8), device='cuda:1')
    f = Function(name='f', grid=Grid(shape=(8, 8)), space_order=0,
                 allocator=TorchAllocator(tensor=data), initializer=lambda _: None)
    op = Operator(Eq(f, f+1), platform='nvidiaX', language='openacc', compiler='nvc')
    op.apply()
    torch.testing.assert_close(data, torch.full_like(data, 2))


@pytest.mark.skipif(not torch.cuda.is_available() or shutil.which('nvc++') is None,
                    reason='Requires CUDA PyTorch, an NVIDIA GPU and nvc++')
def test_openacc_cuda_execution():
    grid = Grid(shape=(16, 16))
    u = TimeFunction(name='u', grid=grid, allocator=TorchAllocator(device='cuda:0'))
    op = Operator(Eq(u.forward, u+1), platform='nvidiaX',
                  language='openacc', compiler='nvc')
    tensor = torch.zeros(tuple(u.shape_allocated), device='cuda:0')
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        tensor.fill_(2)
    ptr = tensor.data_ptr()
    op.apply(u=tensor, time_m=0, time_M=3)
    assert tensor.data_ptr() == ptr
    torch.testing.assert_close(tensor[u._mask_domain][0],
                               torch.full((16, 16), 6., device='cuda:0'))


if __name__ == '__main__':
    assert sys.argv[1] == '--host'
    _host_case(sys.argv[2], sys.argv[3])
