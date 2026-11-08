# PyTorch allocation and code generation

Use the standard Devito API with `TorchAllocator` and `language='torch'`:

```python
from devito import Eq, Grid, Operator, TimeFunction, TorchAllocator

grid = Grid(shape=(64, 64))
u = TimeFunction(name='u', grid=grid, space_order=2, allocator=TorchAllocator())
u.data.fill_(1)
op = Operator(Eq(u.forward, u + 1), language='torch', compiler='gcc')
op.apply(time_M=9)
```

`u` is an ordinary Devito `TimeFunction`, and `op` is selected through the usual
operator registry. The allocator chooses PyTorch-owned storage; `.data` and
`.data_with_halo` expose tensor views inheriting Devito's `Data` behavior.
NumPy remains a dependency for
compiler metadata, not numerical tensor storage.

The torch language uses the existing CPU optimization pipeline and a `CGen`
subclass. It emits non-owning `at::Tensor` views over the argument storage and
obtains typed pointers with `data_ptr<T>()`. Derivative expansion, blocking,
SIMD, and optimized stencil loops remain Devito's. Set
`opt=('advanced', {'openmp': True})` for OpenMP parallel loops.

Compilation and execution use Devito's standard C++ compiler configuration,
ctypes ABI, JIT cache, argument checks, and profiler. There is no Python operator
wrapper, PyBind module, or separate PyTorch extension build. The printer uses
flat indexing because C variable-length arrays are not part of C++.

## Installation

Install PyTorch from the appropriate official CPU or CUDA wheel index, then:

```sh
python -m pip install -e '.[torch,tests]'
```

The optional extra uses the current main branch's dependency ranges. CPU code
generation requires a C++20 compiler. The compiler standard is selected at construction through its
`std` property. The Torch target declares headers, libraries and ABI settings
as IET pass metadata. The existing graph machinery applies those
requirements to the operator-local compiler. Keyword normalization does not
configure the compiler, and the configured template is unchanged. Ninja is not
needed.

## Existing tensors

Use the allocator in the same way as `ExternalAllocator`. A supplied tensor must
include the Function's halo and padding. A no-op initializer preserves its values:

```python
import torch
from devito import Function

tensor = torch.ones((64, 64))
f = Function(name='f', grid=grid, space_order=0,
             allocator=TorchAllocator(tensor=tensor), initializer=lambda _: None)
Operator(Eq(f, f + 1), language='torch', compiler='gcc').apply()
# tensor now contains 2, with the same allocation.
```

A contiguous tensor can also be supplied through the normal runtime override
`op.apply(f=tensor)`. The usual Devito rules for overrides, dimension bounds,
halo layout, and grid spacing apply. Storage must match the Function's dtype,
device, and alignment. No implicit contiguous conversion or device copy occurs.
Supported dtypes are float32, float64, int32, and int64, specified through the
normal Devito `dtype` argument. Tensor initializers are also accepted.

Storage and code generation remain independent: CPU TorchAllocator allocations
can be used with the existing C language, and the torch printer can operate on
ordinary host allocations.

## OpenACC and CUDA tensors

Select the existing OpenACC language and a CUDA allocator:

```python
u = TimeFunction(name='u', grid=grid, space_order=2,
                 allocator=TorchAllocator(device='cuda:0'))
op = Operator(Eq(u.forward, u + 1), platform='nvidiaX',
              language='openacc', compiler='nvc')
op.apply(time_M=9)
```

The allocator marks storage as device-resident. The existing OpenACC passes
emit `deviceptr(...)` clauses and omit host/device mapping and transfer
operations for those Functions. Tensor pointers pass through the normal Devito
data-object ABI. PyTorch owns the allocations throughout their lifetime.

The allocator checks the target and device, supplies a matching default
`deviceid`, and synchronizes CUDA before and after execution so PyTorch and
OpenACC streams can safely consume each other's results. Written tensors have
their PyTorch version counters updated. CUDA Functions cannot be accessed by
host fallback code.

Runtime validation requires CUDA-enabled PyTorch, an NVIDIA GPU, and the NVIDIA
HPC SDK OpenACC compiler. GPU code-generation tests can run without these;
the GPU execution test is skipped when they are unavailable.

## Current limits and tests

`Data` is the backend-independent base for `NumpyData(Data, np.ndarray)` and
`TorchData(Data, torch.Tensor)`. It owns modulo indexing, decomposition metadata,
global/local index translation, distributed assignment, and gather. Constructing
`Data` directly still selects NumPy by default, preserving the existing API.
Torch views and arithmetic results retain this metadata, so buffered time indices
wrap and distributed views retain Devito's indexing semantics. Private allocation
and inhalo accessors return native arrays, as in the NumPy API, without Devito's
modulo or global indexing.

Both backends use the same MPI indexing and gather implementation. Data owns
native storage access and communication views; the allocator owns allocations.
CPU tensors expose zero-copy NumPy views to MPI through PyTorch’s standard
`.numpy()` interoperability. CUDA communication uses explicit host staging and
synchronization;
CUDA-aware MPI is not assumed. OpenACC kernels continue to consume GPU tensor
pointers directly, without staging. Install `.[mpi]` and an MPI runtime to use
distributed execution.

Autograd tensors are not supported. PyTorch has no read-only views, so
`data_ro_*` returns independent snapshots. Negative-stride tensor reads produce
copies because PyTorch cannot represent negative-stride views; direct assignment
through negative slices writes back to the original allocation. Combining
negative strides with advanced tensor indexing is not currently supported.

For an explicit differentiable integration, see
[the diffusion layer example](../examples/pytorch/devito_layer.py).
That CPU-only `nn.Module`
supplies a verified first-order input adjoint for a fixed linear diffusion
operator. Its native kernels receive detached workspaces; the autograd bridge
does not make arbitrary Operators or physical coefficients differentiable.

```sh
OMP_NUM_THREADS=2 python -m pytest -q \
  tests/test_torch_allocators.py tests/test_torch_codegen.py
OMP_NUM_THREADS=2 python -m pytest -q tests/test_data_backends.py
```

Tests cover the standard allocator and operator API, native JIT execution,
numerical parity for blocked 3D stencils, OpenMP, external tensors, rejected
runtime layouts, shared indexing/decomposition semantics, MPI halo exchange and
gather, and OpenACC device-pointer generation. The shared-backend MPI tests use
the repository's existing multi-process pytest runner. In managed environments
without the CMA syscall, MPICH can use `MPIR_CVAR_CH4_CMA_ENABLE=0 UCX_TLS=self,tcp`.
A skipped CUDA execution test does not establish GPU runtime compatibility.

## Internal organization

The backend follows the existing module and argument-provider structure:

| Module | Torch support |
| --- | --- |
| `devito/data/allocators.py` | `TorchAllocator`, alongside the other allocators; allocation and external ownership |
| `devito/data/data.py` | `TorchData`, alongside `Data` and `NumpyData`; native views and the shared indexing/MPI implementation |
| `devito/tools/dtypes_lowering.py` | Scalar dtype conversion |
| `devito/types/dense.py`, `devito/types/sparse.py` | The existing `_arg_check`, `_arg_finalize` and `_arg_apply` lifecycle |
| `devito/core/torch.py`, `devito/ir/iet/torch.py` | Language selection and C++ printing through the existing optimization pipeline |
| `devito/passes/iet/languages/torch.py` | Runtime headers and JIT requirements declared as IET metadata |

Allocators return their native array type. `Data` creates the corresponding view
through its shared construction path. Function argument validation wraps native
overrides in non-owning Data views, and the usual data-object ABI retains those
views for the duration of the call. Tensor write synchronization and version
updates run through `_arg_apply`.

There is no separate storage runtime or backend package. The public API remains
`from devito import TorchAllocator`, tensor `.data`, and
`Operator(..., language='torch')`.

Tests cover allocation in `test_torch_allocators.py`, C++ generation in
`test_torch_codegen.py`, shared indexing/MPI in `test_data_backends.py`, OpenACC
in `test_torch_openacc.py`, and the example layer in `test_torch_layer.py`.

## Contribution validation

Torch tests run in the existing CPU and MPI workflows, which collect `tests/`,
and in the NVIDIA OpenACC job of the existing GPU workflow. PyTorch is installed
through the standard test dependencies, including in the shared Docker image;
there is no separate Torch workflow or workflow-specific dependency installation.
The tests use the same dependency requirements as the main branch.

```sh
# All CPU tests, including the new torch and OpenACC-host cases
OMP_NUM_THREADS=2 pytest -q -m 'not parallel' --maxfail=0 tests/

# All MPI tests; transport flags are needed only where CMA is unavailable
MPIR_CVAR_CH4_CMA_ENABLE=0 UCX_TLS=self,tcp MPIEXEC_TIMEOUT=900 \
  OMP_NUM_THREADS=1 pytest -q -m parallel --maxfail=0 tests/

# Actual GCC OpenACC compilation and host execution, without a GPU
OMP_NUM_THREADS=2 pytest -q tests/test_torch_openacc.py -k host

# NVIDIA OpenACC interoperability; requires CUDA PyTorch, GPU and nvc++
python -c 'import torch; assert torch.cuda.is_available()'
nvc++ --version
pytest -q tests/test_torch_openacc.py -k cuda
```

The OpenACC host cases execute generated code on GCC's `acc_device_host` target
and compare stencil results with PyTorch. They launch a fresh interpreter with
the compiler's system `libgomp` loaded before PyTorch's bundled runtime, which
may lack newer OpenACC symbols. This is a test configuration, not a CUDA-device
emulation or a substitute for the NVIDIA cases.

CUDA coverage includes float32/float64 stencils, allocator-owned and external
storage, runtime overrides, nondefault producer streams, mixed NumPy/tensor
inputs, and device selection. A second-device case requires two GPUs. The
workflow's CUDA job is opt-in through `workflow_dispatch` and requires a
configured self-hosted runner labelled `nvidiagpu`; it fails its preflight if
CUDA or the NVIDIA compiler is unavailable, rather than reporting an all-skipped
CUDA run as successful.
