import os
import ctypes
from pathlib import Path

from devito.tools import memoized_func, filter_ordered, as_tuple
from devito.types import Symbol, SteppingDimension, TimeDimension
from devito.operations.solve import eval_time_derivatives
from devito.symbolics import retrieve_functions, retrieve_dimensions


class PetscOSError(OSError):
    pass


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


@memoized_func
def get_petsc_variables():
    """
    Taken from https://www.firedrakeproject.org/_modules/firedrake/petsc.html
    Get a dict of PETSc environment variables from the file:
    $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables
    """
    try:
        petsc_dir = get_petsc_dir()
    except PetscOSError:
        petsc_variables = {}
    else:
        path = [petsc_dir[-1], 'lib', 'petsc', 'conf', 'petscvariables']
        variables_path = Path(*path)

        with open(variables_path) as fh:
            # Split lines on first '=' (assignment)
            splitlines = (line.split("=", maxsplit=1) for line in fh.readlines())
        petsc_variables = {k.strip(): v.strip() for k, v in splitlines}

    return petsc_variables


petsc_variables = get_petsc_variables()
# TODO: Use petsctools get_petscvariables() instead?


def get_petsc_type_mappings():
    try:
        petsc_precision = petsc_variables['PETSC_PRECISION']
    except KeyError:
        printer_mapper = {}
        petsc_type_to_ctype = {}
    else:
        petsc_scalar = 'PetscScalar'
        # TODO: Check to see whether Petsc is compiled with
        # 32-bit or 64-bit integers
        printer_mapper = {ctypes.c_int: 'PetscInt'}

        if petsc_precision == 'single':
            printer_mapper[ctypes.c_float] = petsc_scalar
        elif petsc_precision == 'double':
            printer_mapper[ctypes.c_double] = petsc_scalar

        # Used to construct ctypes.Structures that wrap PETSc objects
        petsc_type_to_ctype = {v: k for k, v in printer_mapper.items()}
        # Add other PETSc types
        petsc_type_to_ctype.update({
            'KSPType': ctypes.c_char_p,
            'KSPConvergedReason': petsc_type_to_ctype['PetscInt'],
            'KSPNormType': petsc_type_to_ctype['PetscInt'],
        })
    return printer_mapper, petsc_type_to_ctype


petsc_type_mappings, petsc_type_to_ctype = get_petsc_type_mappings()


petsc_languages = ['petsc']


def get_funcs(exprs):
    funcs = [
        f for e in exprs
        for f in retrieve_functions(eval_time_derivatives(e.lhs - e.rhs))
    ]
    return as_tuple(filter_ordered(funcs))


def generate_time_mapper(exprs):
    """
    Replace time indices with `Symbols` in expressions used within
    PETSc callback functions. These symbols are Uxreplaced at the IET
    level to align with the `TimeDimension` and `ModuloDimension` objects
    present in the initial lowering.
    NOTE: All functions used in PETSc callback functions are attached to
    the `SolverMetaData` object, which is passed through the initial lowering
    (and subsequently dropped and replaced with calls to run the solver).
    Therefore, the appropriate time loop will always be correctly generated inside
    the main kernel.
    Examples
    --------
    >>> exprs = (Eq(f1(t + dt, x, y), g1(t + dt, x, y) + g2(t, x, y)*f1(t, x, y)),)
    >>> generate_time_mapper(exprs)
    {t + dt: tau0, t: tau1}
    """
    # First, map any actual TimeDimensions
    time_indices = [d for d in retrieve_dimensions(exprs) if isinstance(d, TimeDimension)]

    funcs = get_funcs(exprs)

    time_indices.extend(list({
        i if isinstance(d, SteppingDimension) else d
        for f in funcs
        for i, d in zip(f.indices, f.dimensions)
        if d.is_Time
    }))
    tau_symbs = [Symbol('tau%d' % i) for i in range(len(time_indices))]
    return dict(zip(time_indices, tau_symbs))
