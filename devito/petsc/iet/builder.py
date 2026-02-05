import math
from functools import cached_property

from devito.ir.iet import DummyExpr, BlankLine
from devito.symbolics import (Byref, FieldFromPointer, VOID,
                              FieldFromComposite, Null, String)

from devito.petsc.iet.nodes import (
    FormFunctionCallback, MatShellSetOp, PETScCall, petsc_call
)


def make_core_petsc_calls(objs, comm):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [comm, Byref(objs['size'])])
    return call_mpi, BlankLine


class BuilderBase:
    def __init__(self, **kwargs):
        self.inject_solve = kwargs.get('inject_solve')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.callback_builder = kwargs.get('callback_builder')
        self.field_data = self.inject_solve.expr.rhs.field_data
        self.formatted_prefix = self.inject_solve.expr.rhs.formatted_prefix

    @cached_property
    def calls(self):
        return self._setup()

    @property
    def snes_ctx(self):
        """
        The [optional] context for private data for the function evaluation routine.
        https://petsc.org/main/manualpages/SNES/SNESSetFunction/
        """
        return VOID(self.solver_objs['dmda'], stars='*')

    def _setup(self):
        sobjs = self.solver_objs
        dmda = sobjs['dmda']

        snes_create = petsc_call('SNESCreate', [sobjs['comm'], Byref(sobjs['snes'])])

        snes_options_prefix = petsc_call(
            'SNESSetOptionsPrefix', [sobjs['snes'], sobjs['snes_prefix']]
        ) if self.formatted_prefix else None

        set_options = petsc_call(
            self.callback_builder._set_options_efunc.name, []
        )

        snes_set_dm = petsc_call('SNESSetDM', [sobjs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(sobjs['Jac'])])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [sobjs['snes'], sobjs['Jac'],
                                sobjs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['xglobal'])])

        target = self.field_data.target
        field_from_ptr = FieldFromPointer(
            target.function._C_field_data, target.function._C_symbol
        )

        local_size = math.prod(
            v for v, dim in zip(target.shape_allocated, target.dimensions) if dim.is_Space
        )
        # TODO: Check - VecCreateSeqWithArray
        local_x = petsc_call('VecCreateMPIWithArray',
                             [sobjs['comm'], 1, local_size, 'PETSC_DECIDE',
                              field_from_ptr, Byref(sobjs['xlocal'])])

        # TODO: potentially also need to set the DM and local/global map to xlocal

        get_local_size = petsc_call('VecGetSize',
                                    [sobjs['xlocal'], Byref(sobjs['localsize'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['bglobal'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [sobjs['snes'], Byref(sobjs['ksp'])])

        matvec = self.callback_builder.main_matvec_efunc
        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_MULT', MatShellSetOp(matvec.name, void, void)]
        )
        formfunc = self.callback_builder._F_efunc
        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [sobjs['snes'], Null, FormFunctionCallback(formfunc.name, void, void),
             self.snes_ctx]
        )

        snes_set_options = petsc_call(
            'SNESSetFromOptions', [sobjs['snes']]
        )

        dmda_calls = self._create_dmda_calls(dmda)

        # TODO: maybe don't need to explictly set this
        mat_set_dm = petsc_call('MatSetDM', [sobjs['Jac'], dmda])

        base_setup = dmda_calls + (
            snes_create,
            snes_options_prefix,
            set_options,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            global_x,
            local_x,
            get_local_size,
            global_b,
            snes_get_ksp,
            matvec_operation,
            formfunc_operation,
            snes_set_options,
            mat_set_dm,
            BlankLine
        )
        extended_setup = self._extend_setup()
        return base_setup + extended_setup

    def _extend_setup(self):
        """
        Hook for subclasses to add additional setup calls.
        """
        return ()

    def _create_dmda_calls(self, dmda):
        dmda_create = self._create_dmda(dmda)
        # TODO: probs need to set the dm options prefix the same as snes?
        dm_set_from_opts = petsc_call('DMSetFromOptions', [dmda])
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        mainctx = self.solver_objs['userctx']

        call_struct_callback = petsc_call(
            self.callback_builder.user_struct_efunc.name, [Byref(mainctx)]
        )

        calls_set_app_ctx = petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])

        return (
            dmda_create,
            dm_set_from_opts,
            dm_setup,
            dm_mat_type,
            call_struct_callback,
            calls_set_app_ctx
        )

    def _create_dmda(self, dmda):
        sobjs = self.solver_objs
        grid = self.field_data.grid
        nspace_dims = len(grid.dimensions)

        # MPI communicator
        args = [sobjs['comm']]

        # Type of ghost nodes
        args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(nspace_dims)])

        # Stencil type
        if nspace_dims > 1:
            args.append('DMDA_STENCIL_BOX')

        # Global dimensions
        args.extend(list(grid.shape)[::-1])
        # No.of processors in each dimension
        if nspace_dims > 1:
            args.extend(list(grid.distributor.topology)[::-1])

        # Number of degrees of freedom per node
        args.append(dmda.dofs)
        # "Stencil width" -> size of overlap
        # TODO: Instead, this probably should be
        # extracted from field_data.target._size_outhalo?
        stencil_width = self.field_data.space_order

        args.append(stencil_width)
        args.extend([Null]*nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call(f'DMDACreate{nspace_dims}d', args)

        return dmda


class CoupledBuilder(BuilderBase):
    def _setup(self):
        # TODO: minimise code duplication with superclass
        objs = self.objs
        sobjs = self.solver_objs
        dmda = sobjs['dmda']

        snes_create = petsc_call('SNESCreate', [sobjs['comm'], Byref(sobjs['snes'])])

        snes_options_prefix = petsc_call(
            'SNESSetOptionsPrefix', [sobjs['snes'], sobjs['snes_prefix']]
        ) if self.formatted_prefix else None

        set_options = petsc_call(
            self.callback_builder._set_options_efunc.name, []
        )

        snes_set_dm = petsc_call('SNESSetDM', [sobjs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(sobjs['Jac'])])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [sobjs['snes'], sobjs['Jac'],
                                sobjs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['xglobal'])])

        local_x = petsc_call('DMCreateLocalVector', [dmda, Byref(sobjs['xlocal'])])

        get_local_size = petsc_call('VecGetSize',
                                    [sobjs['xlocal'], Byref(sobjs['localsize'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [sobjs['snes'], Byref(sobjs['ksp'])])

        matvec = self.callback_builder.main_matvec_efunc
        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_MULT', MatShellSetOp(matvec.name, void, void)]
        )
        formfunc = self.callback_builder._F_efunc
        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [sobjs['snes'], Null, FormFunctionCallback(formfunc.name, void, void),
             self.snes_ctx]
        )

        snes_set_options = petsc_call(
            'SNESSetFromOptions', [sobjs['snes']]
        )

        dmda_calls = self._create_dmda_calls(dmda)

        mainctx = sobjs['userctx']

        call_struct_callback = petsc_call(
            self.callback_builder.user_struct_efunc.name, [Byref(mainctx)]
        )

        # TODO: maybe don't need to explictly set this
        mat_set_dm = petsc_call('MatSetDM', [sobjs['Jac'], dmda])

        create_field_decomp = petsc_call(
            'DMCreateFieldDecomposition',
            [dmda, Byref(sobjs['nfields']), Null, Byref(sobjs['fields']),
             Byref(sobjs['subdms'])]
        )
        submat_cb = self.callback_builder.submatrices_callback
        matop_create_submats_op = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_CREATE_SUBMATRICES',
             MatShellSetOp(submat_cb.name, void, void)]
        )

        call_coupled_struct_callback = petsc_call(
            'PopulateMatContext',
            [Byref(sobjs['jacctx']), sobjs['subdms'], sobjs['fields']]
        )

        shell_set_ctx = petsc_call(
            'MatShellSetContext', [sobjs['Jac'], Byref(sobjs['jacctx']._C_symbol)]
        )

        create_submats = petsc_call(
            'MatCreateSubMatrices',
            [sobjs['Jac'], sobjs['nfields'], sobjs['fields'],
             sobjs['fields'], 'MAT_INITIAL_MATRIX',
             Byref(FieldFromComposite(objs['Submats'].base, sobjs['jacctx']))]
        )

        targets = self.field_data.targets

        deref_dms = [
            DummyExpr(sobjs[f'da{t.name}'], sobjs['subdms'].indexed[i])
            for i, t in enumerate(targets)
        ]

        xglobals = [petsc_call(
            'DMCreateGlobalVector',
            [sobjs[f'da{t.name}'], Byref(sobjs[f'xglobal{t.name}'])]
        ) for t in targets]

        xlocals = []
        for t in targets:
            target_xloc = sobjs[f'xlocal{t.name}']
            local_size = math.prod(
                v for v, dim in zip(t.shape_allocated, t.dimensions) if dim.is_Space
            )
            field_from_ptr = FieldFromPointer(
                t.function._C_field_data, t.function._C_symbol
            )
            # TODO: Check - VecCreateSeqWithArray?
            xlocals.append(petsc_call(
                'VecCreateMPIWithArray',
                [sobjs['comm'], 1, local_size, 'PETSC_DECIDE',
                 field_from_ptr, Byref(target_xloc)]
            ))

        coupled_setup = dmda_calls + (
            snes_create,
            snes_options_prefix,
            set_options,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            global_x,
            local_x,
            get_local_size,
            snes_get_ksp,
            matvec_operation,
            formfunc_operation,
            snes_set_options,
            call_struct_callback,
            mat_set_dm,
            create_field_decomp,
            matop_create_submats_op,
            call_coupled_struct_callback,
            shell_set_ctx,
            create_submats) + \
            tuple(deref_dms) + tuple(xglobals) + tuple(xlocals) + (BlankLine,)
        return coupled_setup


class ConstrainedBCMixin:
    """
    """
    def _create_dmda_calls(self, dmda):
        sobjs = self.solver_objs
        mainctx = sobjs['userctx']

        dmda_create = self._create_dmda(dmda)

        # TODO: likely need to set the dm options prefix the same as snes?
        # likely shouldn't hardcode this option like this.. (should be set in the options
        # callback)
        da_create_section = petsc_call(
            'PetscOptionsSetValue', [Null, String("-da_use_section"), Null]
        )
        dm_set_from_opts = petsc_call('DMSetFromOptions', [dmda])
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])

        count_bcs = petsc_call(
            self.callback_builder._count_bc_efunc.name, [dmda, Byref(sobjs['numBC'])]
        )

        set_point_bcs = petsc_call(
            self.callback_builder._point_bc_efunc.name, [dmda, sobjs['numBC']]
        )

        get_local_section = petsc_call(
            'DMGetLocalSection', [dmda, Byref(sobjs['lsection'])]
        )

        get_point_sf = petsc_call('DMGetPointSF', [dmda, Byref(sobjs['sf'])])

        create_global_section = petsc_call(
            'PetscSectionCreateGlobalSection',
            [sobjs['lsection'], sobjs['sf'], 'PETSC_TRUE', 'PETSC_FALSE', 'PETSC_FALSE',
             Byref(sobjs['gsection'])]
        )

        dm_set_global_section = petsc_call(
            'DMSetGlobalSection', [dmda, sobjs['gsection']]
        )

        dm_create_section_sf = petsc_call(
            'DMCreateSectionSF', [dmda, sobjs['lsection'], sobjs['gsection']]
        )

        call_struct_callback = petsc_call(
            self.callback_builder.user_struct_efunc.name, [Byref(mainctx)]
        )

        calls_set_app_ctx = petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])

        return (
            dmda_create,
            da_create_section,
            dm_set_from_opts,
            dm_setup,
            dm_mat_type,
            call_struct_callback,
            calls_set_app_ctx,
            count_bcs,
            set_point_bcs,
            get_local_section,
            get_point_sf,
            create_global_section,
            dm_set_global_section,
            dm_create_section_sf
        )


class ConstrainedBCBuilder(ConstrainedBCMixin, BuilderBase):
    pass


# TODO: Implement this class properly
class CoupledConstrainedBCBuilder(ConstrainedBCMixin, CoupledBuilder):
    pass


def petsc_call_mpi(specific_call, call_args):
    return PETScCall('PetscCallMPI', [PETScCall(specific_call, arguments=call_args)])


void = VOID._dtype
