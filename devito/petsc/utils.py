import os
import ctypes

from pathlib import Path
from devito.tools import memoized_func

from petsctools import get_petscvariables


class PetscOSError(OSError):
    pass


solver_mapper = {
    'gmres': 'KSPGMRES',
    'jacobi': 'PCJACOBI',
    None: 'PCNONE'
}


@memoized_func
def get_petsc_dir():
    petsc_dir = os.environ.get('PETSC_DIR')
    if petsc_dir is None:
        raise PetscOSError("PETSC_DIR environment variable not set")
    else:
        petsc_dir = (Path(petsc_dir),)

    petsc_arch = os.environ.get('PETSC_ARCH')
    if petsc_arch is not None:
        petsc_dir += (petsc_dir[0] / petsc_arch,)

    petsc_installed = petsc_dir[-1] / 'include' / 'petscconf.h'
    if not petsc_installed.is_file():
        raise PetscOSError("PETSc is not installed")

    return petsc_dir


@memoized_func
def core_metadata():
    petsc_dir = get_petsc_dir()

    petsc_include = tuple([arch / 'include' for arch in petsc_dir])
    petsc_lib = tuple([arch / 'lib' for arch in petsc_dir])

    return {
        'includes': ('petscsnes.h', 'petscdmda.h'),
        'include_dirs': petsc_include,
        'libs': ('petsc'),
        'lib_dirs': petsc_lib,
        'ldflags': tuple([f"-Wl,-rpath,{lib}" for lib in petsc_lib])
    }


petsc_variables = get_petscvariables()


def get_petsc_type_mappings():
    try:
        petsc_precision = petsc_variables['PETSC_PRECISION']
    except KeyError:
        mapper = {}
    else:
        petsc_scalar = 'PetscScalar'
        # TODO: Check to see whether Petsc is compiled with
        # 32-bit or 64-bit integers
        mapper = {ctypes.c_int: 'PetscInt'}

        if petsc_precision == 'single':
            mapper[ctypes.c_float] = petsc_scalar
        elif petsc_precision == 'double':
            mapper[ctypes.c_double] = petsc_scalar
    return mapper


petsc_type_mappings = get_petsc_type_mappings()

petsc_languages = ['petsc']
