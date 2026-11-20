"""C++ printer for optimized loops over PyTorch tensor storage."""

import cgen as c
import numpy as np

from devito.data import FULL
from devito.ir.iet.visitors import CGen


class TorchCGen(CGen):
    """Print typed ATen storage views followed by Devito's flattened loops."""

    def visit_PointerCast(self, o):
        cast = super().visit_PointerCast(o)
        f = o.function
        if not f.is_DiscreteFunction:
            return cast
        shape = ', '.join(
            f'static_cast<int64_t>({self.ccode(f._C_get_field(FULL, d).size)})'
            for d in f.dimensions
        )
        dtype = self.ccode(f.indexed._C_typedata)
        # ATen specializes fixed-width integer types. NumPy's C spelling (e.g.
        # long) need not name the same C++ type as int64_t on every platform
        scalar_dtype = (f'{np.dtype(f.dtype).name}_t'
                        if np.issubdtype(f.dtype, np.integer) else dtype)
        tensor = (f'at::from_blob({cast.data}, {{{shape}}}, '
                  'at::TensorOptions().dtype('
                  f'c10::CppTypeToScalarType<{scalar_dtype}>::value)'
                  '.device(at::kCPU))')
        pointer = f'{tensor}.data_ptr<{scalar_dtype}>()'
        if scalar_dtype != dtype:
            pointer = f'reinterpret_cast<{dtype} *>({pointer})'
        return c.Initializer(cast.vdecl, pointer)
