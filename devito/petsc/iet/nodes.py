from devito.ir.equations import OpPetsc
from devito.ir.iet import Call, Callback, Expression, FixedArgsCallable


class PetscMetaData(Expression):
    """
    Base class for general expressions required to run a PETSc solver.
    """
    def __init__(self, expr, pragmas=None, operation=OpPetsc):
        super().__init__(expr, pragmas=pragmas, operation=operation)


class PETScCallable(FixedArgsCallable):
    pass


class MatShellSetOp(Callback):
    @property
    def callback_form(self):
        param_types_str = ', '.join([str(t) for t in self.param_types])
        return f"({self.retval} (*)({param_types_str})){self.name}"


class PetscCallback(Callback):
    @property
    def callback_form(self):
        return f'{self.name}'


class PETScCall(Call):
    pass


class MgPopulateCall(PETScCall):
    """
    A call to PopulateUserContext that carries its MG level so the
    fix_mg_populate_calls pass can identify and scale arguments without
    parsing the call's argument list.
    """
    def __init__(self, name, arguments=None, retobj=None, is_indirect=False,
                 cast=False, writes=None, templates=None, level=0, hierarchy=None):
        super().__init__(name, arguments=arguments, retobj=retobj,
                         is_indirect=is_indirect, cast=cast,
                         writes=writes, templates=templates)
        self.level = level
        self.hierarchy = hierarchy


def petsc_call(specific_call, call_args):
    return PETScCall('PetscCall', [PETScCall(specific_call, arguments=call_args)])


# Mapping special Eq operations to their corresponding IET Expression subclass types.
# These operations correspond to subclasses of `Eq`` utilised within `petscsolve``.
petsc_iet_mapper = {OpPetsc: PetscMetaData}
