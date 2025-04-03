from ctypes import POINTER, c_char
from devito.tools import CustomDtype, dtype_to_ctype, as_tuple, CustomIntType
from devito.types import (LocalObject, LocalCompositeObject, ModuloDimension,
                          TimeDimension, ArrayObject, CustomDimension)
from devito.symbolics import Byref, cast
from devito.types.basic import DataSymbol
from devito.petsc.iet.utils import petsc_call


class CallbackDM(LocalObject):
    """
    PETSc Data Management object (DM). This is the DM instance
    accessed within the callback functions via `SNESGetDM` and
    is not destroyed during callback execution.
    """
    dtype = CustomDtype('DM')


class DM(LocalObject):
    """
    PETSc Data Management object (DM). This is the primary DM instance
    created within the main kernel and linked to the SNES
    solver using `SNESSetDM`.
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, dofs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dofs = dofs

    @property
    def dofs(self):
        return self._dofs

    @property
    def _C_free(self):
        return petsc_call('DMDestroy', [Byref(self.function)])

    # TODO: Switch to an enumeration?
    @property
    def _C_free_priority(self):
        return 4


DMCast = cast('DM')


class CallbackMat(LocalObject):
    """
    PETSc Matrix object (Mat) used within callback functions.
    These instances are not destroyed during callback execution;
    instead, they are managed and destroyed in the main kernel.
    """
    dtype = CustomDtype('Mat')


class Mat(LocalObject):
    dtype = CustomDtype('Mat')

    @property
    def _C_free(self):
        return petsc_call('MatDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 2


class CallbackVec(LocalObject):
    """
    PETSc vector object (Vec).
    """
    dtype = CustomDtype('Vec')


class Vec(CallbackVec):
    @property
    def _C_free(self):
        return petsc_call('VecDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 1


class PetscMPIInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class PetscInt(LocalObject):
    """
    PETSc datatype used to represent `int` parameters
    to PETSc functions.
    """
    dtype = CustomIntType('PetscInt')


class KSP(LocalObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


class CallbackSNES(LocalObject):
    """
    PETSc SNES : Non-Linear Systems Solvers.
    """
    dtype = CustomDtype('SNES')


class SNES(CallbackSNES):
    @property
    def _C_free(self):
        return petsc_call('SNESDestroy', [Byref(self.function)])

    @property
    def _C_free_priority(self):
        return 3


class PC(LocalObject):
    """
    PETSc object that manages all preconditioners (PC).
    """
    dtype = CustomDtype('PC')


class KSPConvergedReason(LocalObject):
    """
    PETSc object - reason a Krylov method was determined
    to have converged or diverged.
    """
    dtype = CustomDtype('KSPConvergedReason')


class DMDALocalInfo(LocalObject):
    """
    PETSc object - C struct containing information
    about the local grid.
    """
    dtype = CustomDtype('DMDALocalInfo')


class PetscErrorCode(LocalObject):
    """
    PETSc datatype used to return PETSc error codes.
    https://petsc.org/release/manualpages/Sys/PetscErrorCode/
    """
    dtype = CustomDtype('PetscErrorCode')


class DummyArg(LocalObject):
    """
    A void pointer used to satisfy the function
    signature of the `FormFunction` callback.
    """
    dtype = CustomDtype('void', modifier='*')


class MatReuse(LocalObject):
    dtype = CustomDtype('MatReuse')


class VecScatter(LocalObject):
    dtype = CustomDtype('VecScatter')


class StartPtr(LocalObject):
    def __init__(self, name, dtype):
        super().__init__(name=name)
        self.dtype = POINTER(dtype_to_ctype(dtype))


class SingleIS(LocalObject):
    dtype = CustomDtype('IS')


class PETScStruct(LocalCompositeObject):

    @property
    def time_dim_fields(self):
        """
        Fields within the struct that are updated during the time loop.
        These are not set in the `PopulateMatContext` callback.
        """
        return [f for f in self.fields
                if isinstance(f, (ModuloDimension, TimeDimension))]

    @property
    def callback_fields(self):
        """
        Fields within the struct that are initialized in the `PopulateMatContext`
        callback. These fields are not updated in the time loop.
        """
        return [f for f in self.fields if f not in self.time_dim_fields]

    _C_modifier = ' *'


class JacobianStruct(PETScStruct):
    def __init__(self, name='jctx', pname='JacobianCtx', fields=None,
                 modifier='', liveness='lazy'):
        super().__init__(name, pname, fields, modifier, liveness)
    _C_modifier = None


class SubMatrixStruct(PETScStruct):
    def __init__(self, name='subctx', pname='SubMatrixCtx', fields=None,
                 modifier=' *', liveness='lazy'):
        super().__init__(name, pname, fields, modifier, liveness)
    _C_modifier = None


JacobianStructCast = cast('struct JacobianCtx *')


class PETScArrayObject(ArrayObject):
    _data_alignment = False

    def __init_finalize__(self, *args, **kwargs):
        self._nindices = kwargs.pop('nindices', 1)
        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def __indices_setup__(cls, **kwargs):
        try:
            return as_tuple(kwargs['dimensions']), as_tuple(kwargs['dimensions'])
        except KeyError:
            nindices = kwargs.get('nindices', 1)
            dim = CustomDimension(name='d', symbolic_size=nindices)
            return (dim,), (dim,)

    @property
    def dim(self):
        assert len(self.dimensions) == 1
        return self.dimensions[0]

    @property
    def nindices(self):
        return self._nindices

    @property
    def _C_name(self):
        return self.name

    @property
    def _mem_stack(self):
        return False

    @property
    def _C_free_priority(self):
        return 0


class CallbackPointerIS(PETScArrayObject):
    """
    Index set object used for efficient indexing into vectors and matrices.
    https://petsc.org/release/manualpages/IS/IS/
    """
    @property
    def dtype(self):
        return CustomDtype('IS', modifier=' *')


class PointerIS(CallbackPointerIS):
    @property
    def _C_free(self):
        destroy_calls = [
            petsc_call('ISDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls


class CallbackPointerDM(PETScArrayObject):
    @property
    def dtype(self):
        return CustomDtype('DM', modifier=' *')


class PointerDM(CallbackPointerDM):
    @property
    def _C_free(self):
        destroy_calls = [
            petsc_call('DMDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls


class PointerMat(PETScArrayObject):
    _C_modifier = ' *'

    @property
    def dtype(self):
        return CustomDtype('Mat', modifier=' *')


class ArgvSymbol(DataSymbol):
    @property
    def _C_ctype(self):
        return POINTER(POINTER(c_char))
