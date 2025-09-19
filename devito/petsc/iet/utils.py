from devito.ir.iet import FindSymbols
from devito.types import Temp, TempArray

from devito.petsc.iet.nodes import PETScCall


def petsc_call(specific_call, call_args):
    return PETScCall('PetscCall', [PETScCall(specific_call, arguments=call_args)])


def get_user_struct_fields(iet):
    fields = [f.function for f in FindSymbols('basics').visit(iet)]
    from devito.types.basic import LocalType
    avoid = (Temp, TempArray, LocalType)
    fields = [f for f in fields if not isinstance(f.function, avoid)]
    fields = [
        f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
    ]
    return fields
