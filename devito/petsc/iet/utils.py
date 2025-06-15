from devito.petsc.iet.nodes import PetscMetaData, PETScCall
from devito.ir.equations import OpPetsc
from devito.ir.iet import Dereference, FindSymbols, Uxreplace
from devito.types.basic import AbstractFunction


def petsc_call(specific_call, call_args):
    return PETScCall('PetscCall', [PETScCall(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    return PETScCall('PetscCallMPI', [PETScCall(specific_call, arguments=call_args)])


def petsc_struct(name, fields, pname, liveness='lazy', modifier=None):
    # TODO: Fix this circular import
    from devito.petsc.types.object import PETScStruct
    return PETScStruct(name=name, pname=pname,
                       fields=fields, liveness=liveness,
                       modifier=modifier)


def zero_vector(vec):
    """
    Set all entries of a PETSc vector to zero.
    """
    return petsc_call('VecSet', [vec, 0.0])


def dereference_funcs(struct, fields):
    """
    Dereference AbstractFunctions from a struct.
    """
    return tuple(
        [Dereference(i, struct) for i in
         fields if isinstance(i.function, AbstractFunction)]
    )


def residual_bundle(body, bundles):
    """
    Replaces PetscArrays in `body` with PetscBundle struct field accesses
    (e.g., f_v[ix][iy] -> f_bundle[ix][iy].v).

    Example:
        f_v[ix][iy] = x_v[ix][iy];
        f_u[ix][iy] = x_u[ix][iy];
    becomes:
        f_bundle[ix][iy].v = x_bundle[ix][iy].v;
        f_bundle[ix][iy].u = x_bundle[ix][iy].u;

    NOTE: This is used because the data is interleaved for
    multi-component DMDAs in PETSc.
    """
    mapper = bundles['bundle_mapper']
    indexeds = FindSymbols('indexeds').visit(body)
    subs = {}

    for i in indexeds:
        if i.base in mapper:
            bundle = mapper[i.base]
            index = bundles['target_indices'][i.function.target]
            index = (index,) + i.indices
            subs[i] = bundle.__getitem__(index)

    body = Uxreplace(subs).visit(body)
    return body


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpPetsc: PetscMetaData}
