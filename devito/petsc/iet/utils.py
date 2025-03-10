from devito.petsc.iet.nodes import PetscMetaData, PETScCall
from devito.ir.equations import OpPetsc


def petsc_call(specific_call, call_args):
    return PETScCall('PetscCall', [PETScCall(specific_call, arguments=call_args)])


def petsc_call_mpi(specific_call, call_args):
    return PETScCall('PetscCallMPI', [PETScCall(specific_call, arguments=call_args)])


def petsc_struct(name, fields, pname, liveness='lazy'):
    # TODO: Fix this circular import
    from devito.petsc.types.object import PETScStruct
    return PETScStruct(name=name, pname=pname,
                       fields=fields, liveness=liveness)


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpPetsc: PetscMetaData}
