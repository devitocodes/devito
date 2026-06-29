import math
from functools import cached_property

from devito.ir.iet import BlankLine, DummyExpr
from devito.petsc.iet.nodes import (
    MgPopulateCall, PetscCallback, MatShellSetOp, PETScCall, petsc_call
)
from devito.symbolics import (
    VOID, Byref, FieldFromComposite, FieldFromPointer, IndexedPointer,
    Null, String
)


def make_core_petsc_calls(objs, comm):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [comm, Byref(objs['size'])])
    return call_mpi, BlankLine


class BuilderBase:
    """
    Generates the PETSc solver setup calls emitted at the top of the Kernel.
    The set of calls are accessed via the `calls` property. Extend via mixins
    that override the relevant hooks to add/modify certain calls.
    """
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
    def snes_set_function_context(self):
        """
        The context for private data passed to the `SNESSetFunction` routine.
        https://petsc.org/main/manualpages/SNES/SNESSetFunction/
        """
        return VOID(self.solver_objs['dmda'], stars='*')

    def _solver_dm(self):
        """
        DM passed to `SNESSetDM`.
        """
        return self.solver_objs['dmda']

    def _create_local_x_vector(self):
        """
        Create the local solution vector x where F(x) = b. Array space is provided
        to store the vector values instead of allocating new memory on the PETSc side.
        """
        sobjs = self.solver_objs
        target = self.field_data.target
        field_from_ptr = FieldFromPointer(
            target.function._C_field_data, target.function._C_symbol
        )
        local_size = math.prod(
            v for v, dim in zip(target.shape_allocated, target.dimensions, strict=True)
            if dim.is_Space
        )
        # TODO: Check - VecCreateSeqWithArray vs VecCreateMPIWithArray
        return petsc_call('VecCreateSeqWithArray',
                          ['PETSC_COMM_SELF', 1, local_size,
                           field_from_ptr, Byref(sobjs['xlocal'])])

    def _create_global_b_vector(self):
        """
        Create the global vector b where F(x) = b.
        """
        sobjs = self.solver_objs
        return petsc_call('DMCreateGlobalVector',
                          [sobjs['dmda'], Byref(sobjs['bglobal'])])
    
    def _mat_set_dm(self):
        """
        Set the DM for the Jacobian matrix.
        """
        sobjs = self.solver_objs
        # TODO: maybe don't need to explicitly set this?
        return petsc_call('MatSetDM', [sobjs['Jac'], sobjs['dmda']])
    
    def _mat_shell_set_matop_mult(self):
        """
        Set the MATOP_MULT operation for the Jacobian.
        """
        sobjs = self.solver_objs
        matvec = self.callback_builder.main_matvec_efunc
        return petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_MULT', MatShellSetOp(matvec.name, void, void)]
        )

    def _mat_shell_set_matop_get_diagonal(self):
        """
        Register MATOP_GET_DIAGONAL on the Jacobian
        """
        efunc = getattr(self.callback_builder, 'get_diagonal_efunc', None)
        if efunc is None:
            return None
        sobjs = self.solver_objs
        return petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_GET_DIAGONAL', MatShellSetOp(efunc.name, void, void)]
        )

    def _setup(self):
        sobjs = self.solver_objs
        dmda = sobjs['dmda']
        solver_dm = self._solver_dm()

        dmda_calls = self._create_dmda_calls(dmda)

        snes_create = petsc_call('SNESCreate', [sobjs['comm'], Byref(sobjs['snes'])])

        snes_options_prefix = petsc_call(
            'SNESSetOptionsPrefix', [sobjs['snes'], sobjs['snes_prefix']]
        ) if self.formatted_prefix else None

        set_options = petsc_call(
            self.callback_builder._set_options_efunc.name, []
        )

        snes_set_dm = petsc_call('SNESSetDM', [sobjs['snes'], solver_dm])

        create_matrix = petsc_call('DMCreateMatrix', [solver_dm, Byref(sobjs['Jac'])])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [sobjs['snes'], sobjs['Jac'],
                                sobjs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['xglobal'])])

        # TODO: potentially also need to set the DM and local/global map to xlocal

        get_local_size = petsc_call('VecGetLocalSize',
                                     [sobjs['xlocal'], Byref(sobjs['localsize'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [sobjs['snes'], Byref(sobjs['ksp'])])

        formfunc = self.callback_builder._F_efunc
        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [sobjs['snes'], Null, PetscCallback(formfunc.name, void, void),
             self.snes_set_function_context]
        )

        snes_set_options = petsc_call('SNESSetFromOptions', [sobjs['snes']])

        base_setup = dmda_calls + (
            snes_create,
            snes_options_prefix,
            set_options,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            global_x,
            self._create_local_x_vector(),
            get_local_size,
            self._create_global_b_vector(),
            snes_get_ksp,
            self._mat_shell_set_matop_mult(),
            self._mat_shell_set_matop_get_diagonal(),
            formfunc_operation,
            snes_set_options,
            self._mat_set_dm()
        )
        extended_setup = self._extend_setup()
        return base_setup + extended_setup

    def _extend_setup(self):
        """
        Hook for subclasses to add additional setup calls.
        """
        return ()

    def _create_dmda_calls(self, dmda):
        mainctx = self.solver_objs['userctx']

        call_struct_callback = petsc_call(
            self.callback_builder.user_struct_efunc.name, [Byref(mainctx)]
        )
        return (
            self._create_dmda(dmda, lc_arrays=self.solver_objs['lc'][0]),
            # TODO: probably need to set the dm options prefix the same as snes?
            petsc_call('DMSetFromOptions', [dmda]),
            petsc_call('DMSetUp', [dmda]),
            petsc_call('DMSetMatType', [dmda, 'MATSHELL']),
            call_struct_callback,
            petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])
        )

    def _create_dmda(self, dmda, shape=None, lc_arrays=None):
        sobjs = self.solver_objs
        grid = self.field_data.grid
        nspace_dims = len(grid.dimensions)
        shape = shape or grid.shape

        # MPI communicator
        args = [sobjs['comm']]

        # Type of ghost nodes
        args.extend(['DM_BOUNDARY_GHOSTED' for _ in range(nspace_dims)])

        # Stencil type
        if nspace_dims > 1:
            args.append('DMDA_STENCIL_BOX')

        # Global dimensions
        args.extend(list(shape)[::-1])
        # No. of processors in each dimension
        if nspace_dims > 1:
            args.extend(list(grid.distributor.topology)[::-1])

        # Number of degrees of freedom per node
        args.append(dmda.dofs if hasattr(dmda, 'dofs') else 1)
        # "Stencil width" -> size of overlap
        # TODO: Instead, this probably should be
        # extracted from field_data.target._size_outhalo?
        stencil_width = self.field_data.space_order
        args.append(stencil_width)

        # Per-rank node counts
        if lc_arrays is not None:
            args.extend(lc.indexed for lc in lc_arrays)
        else:
            args.extend([Null] * nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        return petsc_call(f'DMDACreate{nspace_dims}d', args)


class CoupledBuilderMixin(BuilderBase):
    """
    Mixin for multi-field (coupled) solver setup calls emitted at the top
    of the Kernel.
    """
    def _create_local_x_vector(self):
        sobjs = self.solver_objs
        return petsc_call(
            'DMCreateLocalVector', [sobjs['dmda'], Byref(sobjs['xlocal'])]
        )
    
    def _create_global_b_vector(self):
        return None
    
    def _extend_setup(self):
        base = super()._extend_setup()
        sobjs = self.solver_objs
        objs = self.objs
        targets = self.field_data.targets

        create_field_decomp = petsc_call(
            'DMCreateFieldDecomposition',
            [sobjs['dmda'], Byref(sobjs['nfields']), Null, Byref(sobjs['fields']),
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
                v for v, dim in zip(t.shape_allocated, t.dimensions, strict=True)
                if dim.is_Space
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

        return base + (
            create_field_decomp,
            matop_create_submats_op,
            call_coupled_struct_callback,
            shell_set_ctx,
            create_submats,
            ) + tuple(deref_dms) + tuple(xglobals) + tuple(xlocals)


class ConstrainedBCMixin:
    """
    Mixin that overrides the DMDA setup calls for solvers with constrained
    boundary nodes.
    """
    def _create_dmda_calls(self, dmda):
        sobjs = self.solver_objs
        mainctx = sobjs['userctx']

        dmda_create = self._create_dmda(dmda, lc_arrays=self.solver_objs['lc'][0])

        # TODO: likely need to set the dm options prefix the same as snes?
        # Probably shouldn't hardcode this option.. (should be set in the options
        # callback)
        da_create_section = petsc_call(
            'PetscOptionsSetValue', [Null, String("-da_use_section"), Null]
        )
        dm_set_from_opts = petsc_call('DMSetFromOptions', [dmda])
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])

        targets = self.field_data.targets
        count_bcs = petsc_call(
            self.callback_builder._count_bc_efunc.name,
            [dmda] + [Byref(sobjs[f'numBC_{t.name}']) for t in targets]
        )

        set_point_bcs = petsc_call(
            self.callback_builder._set_point_bc_efunc.name,
            [dmda] + [sobjs[f'numBC_{t.name}'] for t in targets]
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
    

class MultigridBuilderMixin:
    """
    Mixin for geometric multigrid solver setup calls emitted at the top
    of the Kernel.
    """
    def _solver_dm(self):
        return self.solver_objs['shell']
    
    @property
    def snes_set_function_context(self):
        # Context is retrieved from the DM at runtime via DMShellGetContext.
        return Null

    def _mat_set_dm(self):
        # Set inside CreateMatrix DMShell callback instead
        return None

    def _mat_shell_set_matop_mult(self):
        # Set inside CreateMatrix DMShell callback instead
        return None

    def _mat_shell_set_matop_get_diagonal(self):
        # Set inside CreateMatrix DMShell callback instead
        return None

    def _create_dmda_calls(self, dmda):
        multigrid_metadata = self.inject_solve.expr.rhs.multigrid_metadata
        sobjs = self.solver_objs
        # Array of structs that carrys grid info (separate context) on each level
        all_ctx = sobjs['all_ctx']
        shell_context = sobjs['sctx']
        sctxnew = sobjs['sctxnew']
        shell_dm = sobjs['shell']
        refine_levels = sobjs['refine_levels']
        all_da = sobjs['all_da']
        all_shells = sobjs['all_shells']
        hierarchy = multigrid_metadata.hierarchy
        n_levels = hierarchy.nlevels
        make_shell_ctx_name = self.callback_builder.make_shell_ctx_efunc.name
        make_dm_shell_name = self.callback_builder.make_dm_shell_efunc.name

        # Allocate an array of n_levels UserCtx structs, one per unique MG level
        malloc_all_ctx = petsc_call('PetscMalloc1', [n_levels, Byref(all_ctx)])

        # Populate each UserCtx struct — arguments are finalised later in
        # fix_mg_populate_calls once lower_petsc_symbols has built the full
        # parameter list.
        # NOTE/TODO: data Functions (f_vec, bc_vec) are shared across all levels — coarse
        # levels receive the same fine-level array pointer. This is correct for
        # standard PCMG (FormRHS/FormFunction are only called at the fine level).
        # For FMG, coarse levels need rediscretised data: requires Function support
        # on SubGrid so Devito can automatically inject fine->coarse at op.apply()
        # time. See SubGrid TODO comment in devito/types/grid.py.
        populate_name = self.callback_builder.user_struct_efunc.name
        populate_all_level_contexts = tuple(
            PETScCall('PetscCall', [MgPopulateCall(
                populate_name,
                arguments=[Byref(IndexedPointer(all_ctx, i))],
                level=i,
                hierarchy=hierarchy
            )])
            for i in range(n_levels)
        )

        # Allocate the pre-built DMDA and shell arrays
        malloc_all_da = petsc_call('PetscMalloc1', [n_levels, Byref(all_da)])
        malloc_all_shells = petsc_call('PetscMalloc1', [n_levels, Byref(all_shells)])

        # Pre-build all DMDAs: fine at index 0, coarse levels at 1..n_levels-1
        lc = self.solver_objs['lc']
        dmda_creates = [
            self._create_dmda(IndexedPointer(all_da, 0), lc_arrays=lc[0]),
            petsc_call('DMSetFromOptions', [IndexedPointer(all_da, 0)]),
            petsc_call('DMSetUp', [IndexedPointer(all_da, 0)]),
        ]
        for i, sublevel in enumerate(hierarchy.coarse_levels, start=1):
            dmda_creates += [
                self._create_dmda(IndexedPointer(all_da, i), shape=sublevel.shape,
                                  lc_arrays=lc[i]),
                petsc_call('DMSetFromOptions', [IndexedPointer(all_da, i)]),
                petsc_call('DMSetUp', [IndexedPointer(all_da, i)]),
            ]

        # Pre-build all shells: for each level, malloc sctx, call MakeShellCtx,
        # call UserDMShellCreate. Reuse sctxnew for coarse levels (each PetscMalloc1
        # gives a fresh allocation; the shell stores the pointer).
        shell_creates = [
            petsc_call('PetscMalloc1', [1, Byref(shell_context)]),
            petsc_call(make_shell_ctx_name,
                       [IndexedPointer(all_da, 0), 0, all_ctx,
                        all_da, all_shells, shell_context]),
            petsc_call(make_dm_shell_name,
                       [shell_context, Byref(IndexedPointer(all_shells, 0))]),
        ]
        for i in range(1, n_levels):
            shell_creates += [
                petsc_call('PetscMalloc1', [1, Byref(sctxnew)]),
                petsc_call(make_shell_ctx_name,
                           [IndexedPointer(all_da, i), i, all_ctx,
                            all_da, all_shells, sctxnew]),
                petsc_call(make_dm_shell_name,
                           [sctxnew, Byref(IndexedPointer(all_shells, i))]),
            ]
        from devito.petsc.types.object import DM as DMType
        sobjs['dmda'] = DMType(dmda.name, dofs=sobjs['dmda'].dofs, destroy=False)
        sobjs['shell'] = DMType(shell_dm.name, destroy=False)
        assign_dmda = DummyExpr(sobjs['dmda'], IndexedPointer(all_da, 0))
        assign_shell = DummyExpr(sobjs['shell'], IndexedPointer(all_shells, 0))

        get_refine = petsc_call(
            'DMGetRefineLevel', [IndexedPointer(all_da, 0), Byref(refine_levels)]
        )
        set_refine = petsc_call(
            'DMSetRefineLevel', [IndexedPointer(all_shells, 0), refine_levels]
        )

        return (
            malloc_all_ctx,
            *populate_all_level_contexts,
            malloc_all_da,
            malloc_all_shells,
            *dmda_creates,
            *shell_creates,
            assign_dmda,
            assign_shell,
            get_refine,
            set_refine,
        )

def make_builder_cls(is_coupled, is_multigrid, is_constrained_bc):
    """
    Construct a Builder class by composing the appropriate mixins
    for the given solver properties.

    Parameters
    ----------
    is_coupled : bool
    is_multigrid : bool
    is_constrained_bc : bool
    """
    if is_multigrid and is_coupled:
        raise NotImplementedError(
            "Multigrid not yet supported for multi-field (coupled) solvers."
        )
    # TODO: implement this in this PR
    if is_multigrid and is_constrained_bc:
        raise NotImplementedError(
            "Multigrid not yet supported for solvers with constrained boundary nodes."
        )
    
    mixins = []
    if is_multigrid:
        mixins.append(MultigridBuilderMixin)
    if is_coupled:
        mixins.append(CoupledBuilderMixin)
    if is_constrained_bc:
        mixins.append(ConstrainedBCMixin)
    mixins.append(BuilderBase)

    if len(mixins) == 1:
        return BuilderBase

    return type('Builder', tuple(mixins), {})

def petsc_call_mpi(specific_call, call_args):
    return PETScCall('PetscCallMPI', [PETScCall(specific_call, arguments=call_args)])


void = VOID._dtype
