from ctypes import POINTER, c_char

from devito.tools import CustomDtype, dtype_to_ctype, as_tuple, CustomIntType
from devito.types import (
    LocalObject, LocalCompositeObject, ModuloDimension, TimeDimension, ArrayObject,
    CustomDimension, Scalar
)
from devito.symbolics import Byref, cast, FieldFromComposite
from devito.types.basic import DataSymbol, LocalType

from devito.petsc.iet.nodes import petsc_call


# TODO: unnecessary use of "CALLBACK" types - just create a simple
# way of destroying or not destroying a certain type


class PetscMixin:
    def __init__(self, *args, destroy=True, **kwargs):
        super().__init__(*args, **kwargs)
        self._destroy = destroy

    @property
    def _C_free(self):
        if not self._destroy:
            return None
        return self._C_free_impl()

    def _C_free_impl(self):
        """
        Implement the PETSc destruction logic for this object.

        Subclasses should override this method to emit the appropriate
        PETSc destroy call(s) (e.g., `DMDestroy`, `VecDestroy`, etc.).

        Objects obtained inside callback functions via PETSc "Get"
        routines (e.g. `SNESGetDM`, `KSPGetPC`) are managed elsewhere
        and must not be destroyed here. In such cases, this method
        should return ``None``.
        """
        return None

    @property
    def _C_free_priority(self):
        for cls, prio in FREE_PRIORITY.items():
            if isinstance(self, cls):
                return prio
        return super()._C_free_priority


class PetscObject(PetscMixin, LocalObject):
    pass


class DM(PetscObject):
    """
    PETSc Data Management object (DM).
    """
    dtype = CustomDtype('DM')

    def __init__(self, *args, dofs=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._dofs = dofs

    @property
    def dofs(self):
        return self._dofs

    def _C_free_impl(self):
        return petsc_call('DMDestroy', [Byref(self.function)])


DMCast = cast('DM')
PetscObjectCast = cast('PetscObject')


class Mat(PetscObject):
    """
    PETSc Matrix object (Mat).
    """
    dtype = CustomDtype('Mat')

    def _C_free_impl(self):
        return petsc_call('MatDestroy', [Byref(self.function)])


class Vec(PetscObject):
    """
    PETSc vector object (Vec).
    """
    dtype = CustomDtype('Vec')

    def _C_free_impl(self):
        return petsc_call('VecDestroy', [Byref(self.function)])


class PetscMPIInt(PetscObject):
    """
    PETSc datatype used to represent `int` parameters
    to MPI functions.
    """
    dtype = CustomDtype('PetscMPIInt')


class PetscInt(PetscObject):
    """
    PETSc datatype used to represent `int` parameters
    to PETSc functions.
    """
    dtype = CustomIntType('PetscInt')


class CallbackPetscInt(PetscObject):
    """
    """
    dtype = CustomIntType('PetscInt', modifier=' *')


class PetscIntPtr(PetscObject):
    """
    """
    dtype = CustomIntType('PetscInt')

    _C_modifier = ' *'


class PetscScalar(PetscObject):
    dtype = CustomIntType('PetscScalar')


class PetscBool(PetscObject):
    dtype = CustomDtype('PetscBool')


class KSP(PetscObject):
    """
    PETSc KSP : Linear Systems Solvers.
    Manages Krylov Methods.
    """
    dtype = CustomDtype('KSP')


class KSPType(PetscObject):
    dtype = CustomDtype('KSPType')


class KSPNormType(PetscObject):
    dtype = CustomDtype('KSPNormType')


class SNES(PetscObject):
    """
    PETSc SNES : Non-Linear Systems Solvers.
    """
    dtype = CustomDtype('SNES')

    def _C_free_impl(self):
        return petsc_call('SNESDestroy', [Byref(self.function)])


class PC(PetscObject):
    """
    PETSc object that manages all preconditioners (PC).
    """
    dtype = CustomDtype('PC')


class KSPConvergedReason(PetscObject):
    """
    PETSc object - reason a Krylov method was determined
    to have converged or diverged.
    """
    dtype = CustomDtype('KSPConvergedReason')


class DMDALocalInfo(PetscObject):
    """
    PETSc object - C struct containing information
    about the local grid.
    """
    dtype = CustomDtype('DMDALocalInfo')


class PetscErrorCode(PetscObject):
    """
    PETSc datatype used to return PETSc error codes.
    https://petsc.org/release/manualpages/Sys/PetscErrorCode/
    """
    dtype = CustomDtype('PetscErrorCode')


class DummyArg(PetscObject):
    """
    A void pointer used to satisfy the function
    signature of the `FormFunction` callback.
    """
    dtype = CustomDtype('void', modifier='*')


class MatReuse(PetscObject):
    dtype = CustomDtype('MatReuse')


class VecScatter(PetscObject):
    dtype = CustomDtype('VecScatter')

    def _C_free_impl(self):
        return petsc_call('VecScatterDestroy', [Byref(self.function)])


class StartPtr(PetscObject):
    def __init__(self, name, dtype):
        super().__init__(name=name)
        self.dtype = POINTER(dtype_to_ctype(dtype))


class SingleIS(PetscObject):
    dtype = CustomDtype('IS')


class PetscSectionGlobal(PetscObject):
    dtype = CustomDtype('PetscSection')

    def _C_free_impl(self):
        return petsc_call('PetscSectionDestroy', [Byref(self.function)])


class PetscSectionLocal(PetscObject):
    dtype = CustomDtype('PetscSection')


class PetscSF(PetscObject):
    dtype = CustomDtype('PetscSF')


class PETScStruct(PetscMixin, LocalCompositeObject):

    @property
    def time_dim_fields(self):
        """
        Fields within the struct that are updated during the time loop.
        These are not set in the `PopulateUserContext` callback.
        """
        return [f for f in self.fields
                if isinstance(f, (ModuloDimension, TimeDimension))]

    @property
    def callback_fields(self):
        """
        Fields within the struct that are initialized in the `PopulateUserContext`
        callback. These fields are not updated in the time loop.
        """
        return [f for f in self.fields if f not in self.time_dim_fields]

    _C_modifier = ' *'


class MainUserStruct(PETScStruct):
    pass


class CallbackUserStruct(PETScStruct):
    __rkwargs__ = PETScStruct.__rkwargs__ + ('parent',)

    def __init__(self, *args, parent=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._parent = parent

    @property
    def parent(self):
        return self._parent


class JacobianStruct(PETScStruct):
    def __init__(self, name='jctx', pname='JacobianCtx', fields=None,
                 modifier='', liveness='lazy', no_of_submats=0, **kwargs):
        super().__init__(name, pname, fields, modifier, liveness, **kwargs)
        self._no_of_submats = no_of_submats
    _C_modifier = None

    def _C_free_impl(self):
        pointer_mat = [i for i in self.fields if isinstance(i, PointerMat)]
        assert len(pointer_mat) == 1
        pointer_mat = pointer_mat[0]

        # Destroy each sub-matrix
        destroy_calls = [
            petsc_call(
                'MatDestroy',
                [Byref(FieldFromComposite(pointer_mat.indexed[i],
                                          self.function))]
            )
            for i in range(self._no_of_submats)
        ]
        # Free the allocated matrix pointer array
        destroy_calls.append(
            petsc_call(
                'PetscFree',
                [FieldFromComposite(pointer_mat.base, self.function)]
            )
        )
        return destroy_calls


class SubMatrixStruct(PETScStruct):
    def __init__(self, name='subctx', pname='SubMatrixCtx', fields=None,
                 modifier=' *', liveness='lazy'):
        super().__init__(name, pname, fields, modifier, liveness)
    _C_modifier = None


class PETScArrayObject(PetscMixin, ArrayObject, LocalType):
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


class PointerIS(PETScArrayObject):
    """
    Index set object used for efficient indexing into vectors and matrices.
    https://petsc.org/release/manualpages/IS/IS/
    """
    @property
    def dtype(self):
        return CustomDtype('IS', modifier=' *')

    def _C_free_impl(self):
        destroy_calls = [
            petsc_call('ISDestroy', [Byref(self.indexify().subs({self.dim: i}))])
            for i in range(self._nindices)
        ]
        destroy_calls.append(petsc_call('PetscFree', [self.function]))
        return destroy_calls


class PointerPetscInt(PETScArrayObject):
    """
    TODO: figure out names for this class vs PetscIntPtr class
    """
    @property
    def dtype(self):
        return CustomDtype('PetscInt', modifier=' *')


class PointerDM(PETScArrayObject):
    @property
    def dtype(self):
        return CustomDtype('DM', modifier=' *')

    def _C_free_impl(self):
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


class NofSubMats(Scalar, LocalType):
    pass


# Can this be attached to the consrain bc object in metadata maybe? probs
# shoulnd't be here
Counter = PetscInt(name='count')


FREE_PRIORITY = {
    JacobianStruct: 0,
    PETScArrayObject: 1,
    VecScatter: 2,
    Vec: 3,
    Mat: 4,
    SNES: 5,
    PetscSectionGlobal: 6,
    DM: 7,
}
