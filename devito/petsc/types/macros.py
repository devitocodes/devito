import cgen as c
from devito.symbolics import Macro


Null = Macro('NULL')

# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
