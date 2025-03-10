import os
import sys
from ctypes import POINTER, cast, c_char
import atexit

from devito import Operator
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize

global _petsc_initialized
_petsc_initialized = False


def PetscInitialize():
    global _petsc_initialized
    if not _petsc_initialized:
        dummy = Symbol(name='d')
        # TODO: Potentially just use cgen + the compiler machinery in Devito
        # to generate these "dummy_ops" instead of using the Operator class.
        # This would prevent circular imports when initializing during import
        # from the PETSc module.
        op_init = Operator(
            [PetscEq(dummy, Initialize(dummy))],
            name='kernel_init', opt='noop'
        )
        op_finalize = Operator(
            [PetscEq(dummy, Finalize(dummy))],
            name='kernel_finalize', opt='noop'
        )

        # `argv_bytes` must be a list so the memory address persists
        # `os.fsencode` should be preferred over `string().encode('utf-8')`
        # in case there is some system specific encoding in use
        argv_bytes = list(map(os.fsencode, sys.argv))
        argv_pointer = (POINTER(c_char)*len(sys.argv))(
            *map(lambda s: cast(s, POINTER(c_char)), argv_bytes)
        )
        op_init.apply(argc=len(sys.argv), argv=argv_pointer)

        atexit.register(op_finalize.apply)
        _petsc_initialized = True
