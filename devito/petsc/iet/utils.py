from devito.ir.equations import OpPetsc
from devito.ir.iet import Dereference, FindSymbols, Uxreplace
from devito.types.basic import AbstractFunction

from devito.petsc.iet.nodes import PetscMetaData, PETScCall


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


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of Eq utilised within PETScSolve.
petsc_iet_mapper = {OpPetsc: PetscMetaData}


void = 'void'
insert_vals = 'INSERT_VALUES'
add_vals = 'ADD_VALUES'
sreverse = 'SCATTER_REVERSE'
sforward = 'SCATTER_FORWARD'
