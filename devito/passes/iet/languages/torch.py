"""JIT requirements for code using the PyTorch C++ runtime."""

try:
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
except ImportError:
    torch = None

from devito.passes.iet.engine import iet_pass
from devito.passes.iet.languages.CXX import CXXPrinter

__all__ = ['TorchPrinter', 'initialize']


@iet_pass
def initialize(iet, **kwargs):
    """Declare the headers and toolchain requirements of the Torch language."""
    if torch is None:
        raise ImportError("PyTorch C++ support is required for language='torch'")

    paths = library_paths()
    return iet, {
        'includes': ['ATen/ops/from_blob.h'],
        'include_dirs': include_paths(),
        'lib_dirs': paths,
        'libs': ['torch_cpu', 'torch', 'c10'],
        'defines': [f'_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'],
        'rpath': True
    }


class TorchPrinter(CXXPrinter):
    pass
