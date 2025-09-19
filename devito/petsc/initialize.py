import os
import sys
from ctypes import POINTER, cast, c_char
import atexit

from devito import Operator, switchconfig
from devito.types import Symbol
from devito.types.equation import PetscEq
from devito.petsc.types import Initialize, Finalize

global _petsc_initialized
_petsc_initialized = False


global _petsc_clargs


def PetscInitialize(clargs=sys.argv):
    global _petsc_initialized
    global _petsc_clargs

    if not _petsc_initialized:
        dummy = Symbol(name='d')

        if clargs is not sys.argv:
            clargs = (sys.argv[0], *clargs)

        _petsc_clargs = clargs

        # TODO: Potentially just use cgen + the compiler machinery in Devito
        # to generate these "dummy_ops" instead of using the Operator class.
        # This would prevent circular imports when initializing during import
        # from the PETSc module.
        with switchconfig(language='petsc'):
            op_init = Operator(
                [PetscEq(dummy, Initialize(dummy))],
                name='kernel_init', opt='noop'
            )
            op_finalize = Operator(
                [PetscEq(dummy, Finalize(dummy))],
                name='kernel_finalize', opt='noop'
            )

        # Convert each string to a bytes object (e.g: '-ksp_type' -> b'-ksp_type')
        # `argv_bytes` must be a list so the memory address persists
        # `os.fsencode` should be preferred over `string().encode('utf-8')`
        # in case there is some system specific encoding in use
        argv_bytes = list(map(os.fsencode, clargs))

        # POINTER(c_char) is equivalent to char * in C
        # (POINTER(c_char) * len(clargs)) creates a C array type: char *[len(clargs)]
        # Instantiating it with (*map(...)) casts each bytes object to a char * and
        # fills the array. The result is a char *argv[]
        argv_pointer = (POINTER(c_char)*len(clargs))(
            *map(lambda s: cast(s, POINTER(c_char)), argv_bytes)
        )
        op_init.apply(argc=len(clargs), argv=argv_pointer)
        atexit.register(op_finalize.apply)
        _petsc_initialized = True
