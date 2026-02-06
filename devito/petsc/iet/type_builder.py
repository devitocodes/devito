import numpy as np

from devito.symbolics import String
from devito.types import Symbol
from devito.types.misc import PostIncrementIndex
from devito.tools import frozendict

from devito.petsc.types import (
    PetscBundle, DM, Mat, CallbackVec, Vec, KSP, PC, SNES, PetscInt, StartPtr,
    PointerIS, PointerDM, VecScatter, JacobianStruct, SubMatrixStruct, CallbackDM,
    PetscMPIInt, PetscErrorCode, PointerMat, MatReuse, CallbackPointerDM,
    CallbackPointerIS, CallbackMat, DummyArg, NofSubMats, PetscSectionGlobal,
    PetscSectionLocal, PetscSF, CallbackPetscInt, CallbackPointerPetscInt, SingleIS
)


class BaseTypeBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """
    def __init__(self, **kwargs):
        self.inject_solve = kwargs.get('inject_solve')
        self.objs = kwargs.get('objs')
        self.sregistry = kwargs.get('sregistry')
        self.comm = kwargs.get('comm')
        self.field_data = self.inject_solve.expr.rhs.field_data
        self.solver_objs = self._build()

    def _build(self):
        """
        # TODO: update docs
        Constructs the core dictionary of solver objects and allows
        subclasses to extend or modify it via `_extend_build`.
        Returns:
            dict: A dictionary containing the following objects:
                - 'Jac' (Mat): A matrix representing the jacobian.
                - 'xglobal' (GlobalVec): The global solution vector.
                - 'xlocal' (LocalVec): The local solution vector.
                - 'bglobal': (GlobalVec) Global RHS vector `b`, where `F(x) = b`.
                - 'blocal': (LocalVec) Local RHS vector `b`, where `F(x) = b`.
                - 'ksp': (KSP) Krylov solver object that manages the linear solver.
                - 'pc': (PC) Preconditioner object.
                - 'snes': (SNES) Nonlinear solver object.
                - 'localsize' (PetscInt): The local length of the solution vector.
                - 'dmda' (DM): The DMDA object associated with this solve, linked to
                   the SNES object via `SNESSetDM`.
                - 'callbackdm' (CallbackDM): The DM object accessed within callback
                   functions via `SNESGetDM`.
        """
        sreg = self.sregistry
        targets = self.field_data.targets

        snes_name = sreg.make_name(prefix='snes')
        formatted_prefix = self.inject_solve.expr.rhs.formatted_prefix

        base_dict = {
            'Jac': Mat(sreg.make_name(prefix='J')),
            'xglobal': Vec(sreg.make_name(prefix='xglobal')),
            'xlocal': Vec(sreg.make_name(prefix='xlocal')),
            'bglobal': Vec(sreg.make_name(prefix='bglobal')),
            'blocal': CallbackVec(sreg.make_name(prefix='blocal')),
            'ksp': KSP(sreg.make_name(prefix='ksp')),
            'pc': PC(sreg.make_name(prefix='pc')),
            'snes': SNES(snes_name),
            'localsize': PetscInt(sreg.make_name(prefix='localsize')),
            'dmda': DM(sreg.make_name(prefix='da'), dofs=len(targets)),
            'callbackdm': CallbackDM(sreg.make_name(prefix='dm')),
            'snes_prefix': String(formatted_prefix),
        }

        base_dict['comm'] = self.comm
        self._target_dependent(base_dict)
        return self._extend_build(base_dict)

    def _target_dependent(self, base_dict):
        """
        '_ptr' (StartPtr): A pointer to the beginning of the solution array
        that will be updated at each time step.
        """
        sreg = self.sregistry
        target = self.field_data.target
        base_dict[f'{target.name}_ptr'] = StartPtr(
            sreg.make_name(prefix=f'{target.name}_ptr'), target.dtype
        )

    def _extend_build(self, base_dict):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of solver objects.
        """
        return base_dict


class CoupledTypeBuilder(BaseTypeBuilder):
    def _extend_build(self, base_dict):
        sreg = self.sregistry
        objs = self.objs
        targets = self.field_data.targets
        arrays = self.field_data.arrays

        base_dict['fields'] = PointerIS(
            name=sreg.make_name(prefix='fields'), nindices=len(targets)
        )
        base_dict['subdms'] = PointerDM(
            name=sreg.make_name(prefix='subdms'), nindices=len(targets)
        )
        base_dict['nfields'] = PetscInt(sreg.make_name(prefix='nfields'))

        space_dims = len(self.field_data.grid.dimensions)

        dim_labels = ["M", "N", "P"]
        base_dict.update({
            dim_labels[i]: PetscInt(dim_labels[i]) for i in range(space_dims)
        })

        submatrices = self.field_data.jacobian.nonzero_submatrices

        base_dict['jacctx'] = JacobianStruct(
            name=sreg.make_name(prefix=objs['ljacctx'].name),
            fields=objs['ljacctx'].fields,
        )

        for sm in submatrices:
            name = sm.name
            base_dict[name] = Mat(name=name)
            base_dict[f'{name}ctx'] = SubMatrixStruct(
                name=f'{name}ctx',
                fields=objs['subctx'].fields,
            )
            base_dict[f'{name}X'] = CallbackVec(f'{name}X')
            base_dict[f'{name}Y'] = CallbackVec(f'{name}Y')
            base_dict[f'{name}F'] = CallbackVec(f'{name}F')

        # Bundle objects/metadata required by the coupled residual callback
        f_components, x_components = [], []
        bundle_mapper = {}
        pname = sreg.make_name(prefix='Field')

        target_indices = {t: i for i, t in enumerate(targets)}

        for t in targets:
            f_arr = arrays[t]['f']
            x_arr = arrays[t]['x']
            f_components.append(f_arr)
            x_components.append(x_arr)

        fbundle = PetscBundle(
            name='f_bundle', components=f_components, pname=pname
        )
        xbundle = PetscBundle(
            name='x_bundle', components=x_components, pname=pname
        )

        # Build the bundle mapper
        for f_arr, x_arr in zip(f_components, x_components):
            bundle_mapper[f_arr.base] = fbundle
            bundle_mapper[x_arr.base] = xbundle

        base_dict['bundles'] = {
            'f': fbundle,
            'x': xbundle,
            'bundle_mapper': bundle_mapper,
            'target_indices': target_indices
        }

        return base_dict

    def _target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.field_data.targets
        for t in targets:
            name = t.name
            base_dict[f'{name}_ptr'] = StartPtr(
                sreg.make_name(prefix=f'{name}_ptr'), t.dtype
            )
            base_dict[f'xlocal{name}'] = CallbackVec(
                sreg.make_name(prefix=f'xlocal{name}'), liveness='eager'
            )
            base_dict[f'Fglobal{name}'] = CallbackVec(
                sreg.make_name(prefix=f'Fglobal{name}'), liveness='eager'
            )
            base_dict[f'Xglobal{name}'] = CallbackVec(
                sreg.make_name(prefix=f'Xglobal{name}')
            )
            base_dict[f'xglobal{name}'] = Vec(
                sreg.make_name(prefix=f'xglobal{name}')
            )
            base_dict[f'blocal{name}'] = CallbackVec(
                sreg.make_name(prefix=f'blocal{name}'), liveness='eager'
            )
            base_dict[f'bglobal{name}'] = Vec(
                sreg.make_name(prefix=f'bglobal{name}')
            )
            base_dict[f'da{name}'] = DM(
                sreg.make_name(prefix=f'da{name}'), liveness='eager'
            )
            base_dict[f'scatter{name}'] = VecScatter(
                sreg.make_name(prefix=f'scatter{name}')
            )


class ConstrainedBCTypeBuilder(BaseTypeBuilder):
    def _extend_build(self, base_dict):
        sreg = self.sregistry
        base_dict['lsection'] = PetscSectionLocal(
            name=sreg.make_name(prefix='lsection')
        )
        base_dict['gsection'] = PetscSectionGlobal(
            name=sreg.make_name(prefix='gsection')
        )
        base_dict['sf'] = PetscSF(
            name=sreg.make_name(prefix='sf')
        )
        name = sreg.make_name(prefix='numBC')
        base_dict['numBC'] = PetscInt(
            name=name, initvalue=0
        )
        base_dict['numBCPtr'] = CallbackPetscInt(
            name=sreg.make_name(prefix='numBCPtr'), initvalue=0
        )
        base_dict['bcPointsArr'] = CallbackPointerPetscInt(
            name=sreg.make_name(prefix='bcPointsArr')
        )
        base_dict['k_iter'] = PostIncrementIndex(
            name='k_iter', initvalue=0
        )
        # change names etc..
        base_dict['bcPointsIS'] = SingleIS(name='bcPointsIS')
        base_dict['bcPoints'] = PointerIS(name='bcPoints')
        return base_dict


subdms = PointerDM(name='subdms')
fields = PointerIS(name='fields')
submats = PointerMat(name='submats')
rows = PointerIS(name='rows')
cols = PointerIS(name='cols')


# A static dict containing shared symbols and objects that are not
# unique to each `petscsolve` call.
# Many of these objects are used as arguments in callback functions to make
# the C code cleaner and more modular.
objs = frozendict({
    'size': PetscMPIInt(name='size'),
    'err': PetscErrorCode(name='err'),
    'block': CallbackMat('block'),
    'submat_arr': PointerMat(name='submat_arr'),
    'subblockrows': PetscInt('subblockrows'),
    'subblockcols': PetscInt('subblockcols'),
    'rowidx': PetscInt('rowidx'),
    'colidx': PetscInt('colidx'),
    'J': Mat('J'),
    'X': Vec('X'),
    'xloc': CallbackVec('xloc'),
    'Y': Vec('Y'),
    'yloc': CallbackVec('yloc'),
    'F': Vec('F'),
    'floc': CallbackVec('floc'),
    'B': Vec('B'),
    'nfields': PetscInt('nfields'),
    'irow': PointerIS(name='irow'),
    'icol': PointerIS(name='icol'),
    'nsubmats': NofSubMats('nsubmats', dtype=np.int32),
    # 'nsubmats': PetscInt('nsubmats'),
    'matreuse': MatReuse('scall'),
    'snes': SNES('snes'),
    'rows': rows,
    'cols': cols,
    'Subdms': subdms,
    'LocalSubdms': CallbackPointerDM(name='subdms'),
    'Fields': fields,
    'LocalFields': CallbackPointerIS(name='fields'),
    'Submats': submats,
    'ljacctx': JacobianStruct(
        fields=[subdms, fields, submats], modifier=' *'
    ),
    'subctx': SubMatrixStruct(fields=[rows, cols]),
    'dummyctx': Symbol('lctx'),
    'dummyptr': DummyArg('dummy'),
    'dummyefunc': Symbol('dummyefunc'),
    'dof': PetscInt('dof'),
})
