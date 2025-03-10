from collections import OrderedDict

import cgen as c

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, FindNodes,
                           retrieve_iteration_tree, filter_iterations)
from devito.symbolics import (Byref, FieldFromPointer, Macro, cast_mapper,
                              FieldFromComposite)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import Temp, Symbol
from devito.tools import filter_ordered

from devito.petsc.types import PETScArray
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatVecCallback, PetscMetaData)
from devito.petsc.iet.utils import petsc_call, petsc_struct
from devito.petsc.utils import solver_mapper
from devito.petsc.types import (DM, CallbackDM, Mat, LocalVec, GlobalVec, KSP, PC,
                                SNES, DummyArg, PetscInt, StartPtr)


class CallbackBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __init__(self, injectsolve, objs, solver_objs,
                 rcompile=None, sregistry=None, timedep=None, **kwargs):

        self.rcompile = rcompile
        self.sregistry = sregistry
        self.timedep = timedep
        self.solver_objs = solver_objs

        self._efuncs = OrderedDict()
        self._struct_params = []

        self._matvec_callback = None
        self._formfunc_callback = None
        self._formrhs_callback = None
        self._struct_callback = None

        self._make_core(injectsolve, objs, solver_objs)
        self._main_struct(solver_objs)
        self._make_struct_callback(solver_objs, objs)
        self._local_struct(solver_objs)
        self._efuncs = self._uxreplace_efuncs()

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def struct_params(self):
        return self._struct_params

    @property
    def filtered_struct_params(self):
        return filter_ordered(self.struct_params)

    @property
    def matvec_callback(self):
        return self._matvec_callback

    @property
    def formfunc_callback(self):
        return self._formfunc_callback

    @property
    def formrhs_callback(self):
        return self._formrhs_callback

    @property
    def struct_callback(self):
        return self._struct_callback

    def _make_core(self, injectsolve, objs, solver_objs):
        self._make_matvec(injectsolve, objs, solver_objs)
        self._make_formfunc(injectsolve, objs, solver_objs)
        self._make_formrhs(injectsolve, objs, solver_objs)

    def _make_matvec(self, injectsolve, objs, solver_objs):
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_matvec, _ = self.rcompile(injectsolve.expr.rhs.matvecs,
                                      options={'mpi': False}, sregistry=self.sregistry)
        body_matvec = self._create_matvec_body(injectsolve,
                                               List(body=irs_matvec.uiet.body),
                                               solver_objs, objs)

        matvec_callback = PETScCallable(
            self.sregistry.make_name(prefix='MyMatShellMult_'), body_matvec,
            retval=objs['err'],
            parameters=(
                solver_objs['Jac'], solver_objs['X_global'], solver_objs['Y_global']
            )
        )
        self._matvec_callback = matvec_callback
        self._efuncs[matvec_callback.name] = matvec_callback

    def _create_matvec_body(self, injectsolve, body, solver_objs, objs):
        linsolve_expr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body, solver_objs)

        y_matvec = linsolve_expr.arrays['y_matvec']
        x_matvec = linsolve_expr.arrays['x_matvec']

        mat_get_dm = petsc_call('MatGetDM', [solver_objs['Jac'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                     'INSERT_VALUES', solver_objs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['Y_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['Y_local'], Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local'], Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, solver_objs['Y_local'], 'INSERT_VALUES', solver_objs['Y_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['Y_local'])]
        )

        # TODO: Some of the calls are placed in the `stacks` argument of the
        # `CallableBody` to ensure that they precede the `cast` statements. The
        # 'casts' depend on the calls, so this order is necessary. By doing this,
        # you avoid having to manually construct the `casts` and can allow
        # Devito to handle their construction. This is a temporary solution and
        # should be revisited

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            mat_get_dm,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self._struct_params.extend(fields)

        return matvec_body

    def _make_formfunc(self, injectsolve, objs, solver_objs):
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            injectsolve.expr.rhs.formfuncs,
            options={'mpi': False}, sregistry=self.sregistry
        )
        body_formfunc = self._create_formfunc_body(injectsolve,
                                                   List(body=irs_formfunc.uiet.body),
                                                   solver_objs, objs)

        formfunc_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction_'), body_formfunc,
            retval=objs['err'],
            parameters=(solver_objs['snes'], solver_objs['X_global'],
                        solver_objs['F_global'], dummyptr)
        )
        self._formfunc_callback = formfunc_callback
        self._efuncs[formfunc_callback.name] = formfunc_callback

    def _create_formfunc_body(self, injectsolve, body, solver_objs, objs):
        linsolve_expr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body, solver_objs)

        f_formfunc = linsolve_expr.arrays['f_formfunc']
        x_formfunc = linsolve_expr.arrays['x_formfunc']

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, solver_objs['X_global'],
                                     'INSERT_VALUES', solver_objs['X_local']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, solver_objs['X_global'], 'INSERT_VALUES', solver_objs['X_local']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(solver_objs['F_local'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [solver_objs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [solver_objs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [solver_objs['F_local'], Byref(f_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [solver_objs['X_local'], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, solver_objs['F_local'], 'INSERT_VALUES', solver_objs['F_global']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, solver_objs['F_local'], 'INSERT_VALUES', solver_objs['F_global']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['X_local'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(solver_objs['F_local'])]
        )

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_y,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for i in fields}
        formfunc_body = Uxreplace(subs).visit(formfunc_body)

        self._struct_params.extend(fields)

        return formfunc_body

    def _make_formrhs(self, injectsolve, objs, solver_objs):
        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(injectsolve.expr.rhs.formrhs,
                                       options={'mpi': False}, sregistry=self.sregistry)
        body_formrhs = self._create_form_rhs_body(injectsolve,
                                                  List(body=irs_formrhs.uiet.body),
                                                  solver_objs, objs)

        formrhs_callback = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS_'), body_formrhs, retval=objs['err'],
            parameters=(
                solver_objs['snes'], solver_objs['b_local']
            )
        )
        self._formrhs_callback = formrhs_callback
        self._efuncs[formrhs_callback.name] = formrhs_callback

    def _create_form_rhs_body(self, injectsolve, body, solver_objs, objs):
        linsolve_expr = injectsolve.expr.rhs

        dmda = solver_objs['callbackdm']

        snes_get_dm = petsc_call('SNESGetDM', [solver_objs['snes'], Byref(dmda)])

        b_arr = linsolve_expr.arrays['b_tmp']

        vec_get_array = petsc_call(
            'VecGetArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body, solver_objs)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(dummyctx._C_symbol)]
        )

        vec_restore_array = petsc_call(
            'VecRestoreArray', [solver_objs['b_local'], Byref(b_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            snes_get_dm,
            dm_get_app_context,
            vec_get_array,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, dummyctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(petsc_func_begin_user,),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, dummyctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}
        formrhs_body = Uxreplace(subs).visit(formrhs_body)

        self._struct_params.extend(fields)

        return formrhs_body

    def _local_struct(self, solver_objs):
        """
        This is the struct used within callback functions,
        usually accessed via DMGetApplicationContext.
        """
        solver_objs['localctx'] = petsc_struct(
            dummyctx.name,
            self.filtered_struct_params,
            solver_objs['Jac'].name+'_ctx',
            liveness='eager'
        )

    def _main_struct(self, solver_objs):
        """
        This is the struct initialised inside the main kernel and
        attached to the DM via DMSetApplicationContext.
        """
        solver_objs['mainctx'] = petsc_struct(
            self.sregistry.make_name(prefix='ctx'),
            self.filtered_struct_params,
            solver_objs['Jac'].name+'_ctx'
        )

    def _make_struct_callback(self, solver_objs, objs):
        mainctx = solver_objs['mainctx']
        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, mainctx), i._C_symbol)
            for i in mainctx.callback_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=(petsc_func_begin_user,),
            retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
        )
        struct_callback = Callable(
            self.sregistry.make_name(prefix='PopulateMatContext_'),
            struct_callback_body, objs['err'],
            parameters=[mainctx]
        )
        self._efuncs[struct_callback.name] = struct_callback
        self._struct_callback = struct_callback

    def _dummy_fields(self, iet, solver_objs):
        # Place all context data required by the shell routines into a struct
        fields = [f.function for f in FindSymbols('basics').visit(iet)]
        fields = [f for f in fields if not isinstance(f.function, (PETScArray, Temp))]
        fields = [
            f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
        ]
        return fields

    def _uxreplace_efuncs(self):
        mapper = {}
        visitor = Uxreplace({dummyctx: self.solver_objs['localctx']})
        for k, v in self._efuncs.items():
            mapper.update({k: visitor.visit(v)})
        return mapper


class BaseObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """

    def __init__(self, injectsolve, sregistry=None, **kwargs):
        self.sregistry = sregistry
        self.solver_objs = self._build(injectsolve)

    def _build(self, injectsolve):
        """
        Constructs the core dictionary of solver objects and allows
        subclasses to extend or modify it via `_extend_build`.

        Returns:
            dict: A dictionary containing the following objects:
                - 'Jac' (Mat): A matrix representing the jacobian.
                - 'x_global' (GlobalVec): The global solution vector.
                - 'x_local' (LocalVec): The local solution vector.
                - 'b_global': (GlobalVec) Global RHS vector `b`, where `F(x) = b`.
                - 'b_local': (LocalVec) Local RHS vector `b`, where `F(x) = b`.
                - 'ksp': (KSP) Krylov solver object that manages the linear solver.
                - 'pc': (PC) Preconditioner object.
                - 'snes': (SNES) Nonlinear solver object.
                - 'F_global': (GlobalVec) Global residual vector `F`, where `F(x) = b`.
                - 'F_local': (LocalVec) Local residual vector `F`, where `F(x) = b`.
                - 'Y_global': (GlobalVector) The output vector populated by the
                   matrix-free `MyMatShellMult` callback function.
                - 'Y_local': (LocalVector) The output vector populated by the matrix-free
                   `MyMatShellMult` callback function.
                - 'X_global': (GlobalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'X_local': (LocalVec) Current guess for the solution,
                   required by the FormFunction callback.
                - 'localsize' (PetscInt): The local length of the solution vector.
                - 'start_ptr' (StartPtr): A pointer to the beginning of the solution array
                   that will be updated at each time step.
                - 'dmda' (DM): The DMDA object associated with this solve, linked to
                   the SNES object via `SNESSetDM`.
                - 'callbackdm' (CallbackDM): The DM object accessed within callback
                   functions via `SNESGetDM`.
        """
        target = injectsolve.expr.rhs.target
        sreg = self.sregistry
        base_dict = {
            'Jac': Mat(sreg.make_name(prefix='J_')),
            'x_global': GlobalVec(sreg.make_name(prefix='x_global_')),
            'x_local': LocalVec(sreg.make_name(prefix='x_local_'), liveness='eager'),
            'b_global': GlobalVec(sreg.make_name(prefix='b_global_')),
            'b_local': LocalVec(sreg.make_name(prefix='b_local_')),
            'ksp': KSP(sreg.make_name(prefix='ksp_')),
            'pc': PC(sreg.make_name(prefix='pc_')),
            'snes': SNES(sreg.make_name(prefix='snes_')),
            'F_global': GlobalVec(sreg.make_name(prefix='F_global_')),
            'F_local': LocalVec(sreg.make_name(prefix='F_local_'), liveness='eager'),
            'Y_global': GlobalVec(sreg.make_name(prefix='Y_global_')),
            'Y_local': LocalVec(sreg.make_name(prefix='Y_local_'), liveness='eager'),
            'X_global': GlobalVec(sreg.make_name(prefix='X_global_')),
            'X_local': LocalVec(sreg.make_name(prefix='X_local_'), liveness='eager'),
            'localsize': PetscInt(sreg.make_name(prefix='localsize_')),
            'start_ptr': StartPtr(sreg.make_name(prefix='start_ptr_'), target.dtype),
            'dmda': DM(sreg.make_name(prefix='da_'), liveness='eager',
                       stencil_width=target.space_order),
            'callbackdm': CallbackDM(sreg.make_name(prefix='dm_'),
                                     liveness='eager', stencil_width=target.space_order),
        }
        return self._extend_build(base_dict, injectsolve)

    def _extend_build(self, base_dict, injectsolve):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of solver objects.
        """
        return base_dict


class BaseSetup:
    def __init__(self, solver_objs, objs, injectsolve, cbbuilder):
        self.calls = self._setup(solver_objs, objs, injectsolve, cbbuilder)

    def _setup(self, solver_objs, objs, injectsolve, cbbuilder):
        dmda = solver_objs['dmda']

        solver_params = injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(solver_objs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [solver_objs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(solver_objs['Jac'])])

        # NOTE: Assuming all solves are linear for now.
        snes_set_type = petsc_call('SNESSetType', [solver_objs['snes'], 'SNESKSPONLY'])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [solver_objs['snes'], solver_objs['Jac'],
                                solver_objs['Jac'], 'MatMFFDComputeJacobian', Null]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['x_global'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(solver_objs['b_global'])])

        local_b = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['b_local'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [solver_objs['snes'], Byref(solver_objs['ksp'])])

        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [solver_objs['ksp'], solver_params['ksp_rtol'],
                                 solver_params['ksp_atol'], solver_params['ksp_divtol'],
                                 solver_params['ksp_max_it']]
        )

        ksp_set_type = petsc_call(
            'KSPSetType', [solver_objs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )

        ksp_get_pc = petsc_call(
            'KSPGetPC', [solver_objs['ksp'], Byref(solver_objs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [solver_objs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [solver_objs['ksp']])

        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [solver_objs['Jac'], 'MATOP_MULT',
             MatVecCallback(cbbuilder.matvec_callback.name, void, void)]
        )

        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [solver_objs['snes'], Null,
             FormFunctionCallback(cbbuilder.formfunc_callback.name, void, void), Null]
        )

        dmda_calls = self._create_dmda_calls(dmda, objs)

        mainctx = solver_objs['mainctx']

        call_struct_callback = petsc_call(
            cbbuilder.struct_callback.name, [Byref(mainctx)]
        )
        calls_set_app_ctx = [
            petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])
        ]
        calls = [call_struct_callback] + calls_set_app_ctx + [BlankLine]

        base_setup = dmda_calls + (
            snes_create,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            snes_set_type,
            global_x,
            global_b,
            local_b,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            matvec_operation,
            formfunc_operation,
        ) + tuple(calls)

        extended_setup = self._extend_setup(solver_objs, objs, injectsolve, cbbuilder)
        return base_setup + tuple(extended_setup)

    def _extend_setup(self, solver_objs, objs, injectsolve, cbbuilder):
        """
        Hook for subclasses to add additional setup calls.
        """
        return []

    def _create_dmda_calls(self, dmda, objs):
        dmda_create = self._create_dmda(dmda, objs)
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        return dmda_create, dm_setup, dm_mat_type

    def _create_dmda(self, dmda, objs):
        grid = objs['grid']

        nspace_dims = len(grid.dimensions)

        # MPI communicator
        args = [objs['comm']]

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
        args.append(1)
        # "Stencil width" -> size of overlap
        args.append(dmda.stencil_width)
        args.extend([Null]*nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call('DMDACreate%sd' % nspace_dims, args)

        return dmda


class Solver:
    def __init__(self, solver_objs, objs, injectsolve, iters, cbbuilder,
                 timedep=None, **kwargs):
        self.timedep = timedep
        self.calls = self._execute_solve(solver_objs, objs, injectsolve, iters, cbbuilder)
        self.spatial_body = self._spatial_loop_nest(iters, injectsolve)

        space_iter, = self.spatial_body
        self.mapper = {space_iter: self.calls}

    def _execute_solve(self, solver_objs, objs, injectsolve, iters, cbbuilder):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        struct_assignment = self.timedep.assign_time_iters(solver_objs['mainctx'])

        rhs_callback = cbbuilder.formrhs_callback

        dmda = solver_objs['dmda']

        rhs_call = petsc_call(rhs_callback.name, list(rhs_callback.parameters))

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(solver_objs['x_local'])])

        vec_replace_array = self.timedep.replace_array(solver_objs)

        dm_local_to_global_x = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['x_local'], 'INSERT_VALUES',
                                solver_objs['x_global']]
        )

        dm_local_to_global_b = petsc_call(
            'DMLocalToGlobal', [dmda, solver_objs['b_local'], 'INSERT_VALUES',
                                solver_objs['b_global']]
        )

        snes_solve = petsc_call('SNESSolve', [
            solver_objs['snes'], solver_objs['b_global'], solver_objs['x_global']]
        )

        dm_global_to_local_x = petsc_call('DMGlobalToLocal', [
            dmda, solver_objs['x_global'], 'INSERT_VALUES', solver_objs['x_local']]
        )

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
            local_x
        ) + vec_replace_array + (
            dm_local_to_global_x,
            dm_local_to_global_b,
            snes_solve,
            dm_global_to_local_x,
            BlankLine,
        )
        return List(body=run_solver_calls)

    def _spatial_loop_nest(self, iters, injectsolve):
        spatial_body = []
        for tree in retrieve_iteration_tree(iters[0]):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
            if injectsolve in FindNodes(PetscMetaData).visit(root):
                spatial_body.append(root)
        return spatial_body


class NonTimeDependent:
    def __init__(self, injectsolve, iters, **kwargs):
        self.injectsolve = injectsolve
        self.iters = iters
        self.kwargs = kwargs
        self.origin_to_moddim = self._origin_to_moddim_mapper(iters)
        self.time_idx_to_symb = injectsolve.expr.rhs.time_mapper

    @property
    def is_target_time(self):
        return False

    @property
    def target(self):
        return self.injectsolve.expr.rhs.target

    def _origin_to_moddim_mapper(self, iters):
        return {}

    def uxreplace_time(self, body):
        return body

    def replace_array(self, solver_objs):
        """
        VecReplaceArray() is a PETSc function that allows replacing the array
        of a `Vec` with a user provided array.
        https://petsc.org/release/manualpages/Vec/VecReplaceArray/

        This function is used to replace the array of the PETSc solution `Vec`
        with the array from the `Function` object representing the target.

        Examples
        --------
        >>> self.target
        f1(x, y)
        >>> call = replace_array(solver_objs)
        >>> print(call)
        PetscCall(VecReplaceArray(x_local_0,f1_vec->data));
        """
        field_from_ptr = FieldFromPointer(
            self.target.function._C_field_data, self.target.function._C_symbol
        )
        vec_replace_array = (petsc_call(
            'VecReplaceArray', [solver_objs['x_local'], field_from_ptr]
        ),)
        return vec_replace_array

    def assign_time_iters(self, struct):
        return []


class TimeDependent(NonTimeDependent):
    """
    A class for managing time-dependent solvers.

    This includes scenarios where the target is not directly a `TimeFunction`,
    but depends on other functions that are.

    Outline of time loop abstraction with PETSc:

    - At PETScSolve, time indices are replaced with temporary `Symbol` objects
      via a mapper (e.g., {t: tau0, t + dt: tau1}) to prevent the time loop
      from being generated in the callback functions. These callbacks, needed
      for each `SNESSolve` at every time step, don't require the time loop, but
      may still need access to data from other time steps.
    - All `Function` objects are passed through the initial lowering via the
      `LinearSolveExpr` object, ensuring the correct time loop is generated
      in the main kernel.
    - Another mapper is created based on the modulo dimensions
      generated by the `LinearSolveExpr` object in the main kernel
      (e.g., {time: time, t: t0, t + 1: t1}).
    - These two mappers are used to generate a final mapper `symb_to_moddim`
      (e.g. {tau0: t0, tau1: t1}) which is used at the IET level to
      replace the temporary `Symbol` objects in the callback functions with
      the correct modulo dimensions.
    - Modulo dimensions are updated in the matrix context struct at each time
      step and can be accessed in the callback functions where needed.
    """
    @property
    def is_target_time(self):
        return any(i.is_Time for i in self.target.dimensions)

    @property
    def time_spacing(self):
        return self.target.grid.stepping_dim.spacing

    @property
    def target_time(self):
        target_time = [
            i for i, d in zip(self.target.indices, self.target.dimensions)
            if d.is_Time
        ]
        assert len(target_time) == 1
        target_time = target_time.pop()
        return target_time

    @property
    def symb_to_moddim(self):
        """
        Maps temporary `Symbol` objects created during `PETScSolve` to their
        corresponding modulo dimensions (e.g. creates {tau0: t0, tau1: t1}).
        """
        mapper = {
            v: k.xreplace({self.time_spacing: 1, -self.time_spacing: -1})
            for k, v in self.time_idx_to_symb.items()
        }
        return {symb: self.origin_to_moddim[mapper[symb]] for symb in mapper}

    def uxreplace_time(self, body):
        return Uxreplace(self.symb_to_moddim).visit(body)

    def _origin_to_moddim_mapper(self, iters):
        """
        Creates a mapper of the origin of the time dimensions to their corresponding
        modulo dimensions from a list of `Iteration` objects.

        Examples
        --------
        >>> iters
        (<WithProperties[affine,sequential]::Iteration time[t0,t1]; (time_m, time_M, 1)>,
         <WithProperties[affine,parallel,parallel=]::Iteration x; (x_m, x_M, 1)>)
        >>> _origin_to_moddim_mapper(iters)
        {time: time, t: t0, t + 1: t1}
        """
        time_iter = [i for i in iters if any(d.is_Time for d in i.dimensions)]
        mapper = {}

        if not time_iter:
            return mapper

        for i in time_iter:
            for d in i.dimensions:
                if d.is_Modulo:
                    mapper[d.origin] = d
                elif d.is_Time:
                    mapper[d] = d
        return mapper

    def replace_array(self, solver_objs):
        """
        In the case that the actual target is time-dependent e.g a `TimeFunction`,
        a pointer to the first element in the array that will be updated during
        the time step is passed to VecReplaceArray().

        Examples
        --------
        >>> self.target
        f1(time + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = (time + 1)*localsize_0 + (float*)(f1_vec->data);
        PetscCall(VecReplaceArray(x_local_0,start_ptr_0));

        >>> self.target
        f1(t + dt, x, y)
        >>> calls = replace_array(solver_objs)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(x_local_0,&(localsize_0)));
        float * start_ptr_0 = t1*localsize_0 + (float*)(f1_vec->data);
        """
        if self.is_target_time:
            mapper = {self.time_spacing: 1, -self.time_spacing: -1}
            target_time = self.target_time.xreplace(mapper)

            try:
                target_time = self.origin_to_moddim[target_time]
            except KeyError:
                pass

            start_ptr = solver_objs['start_ptr']

            vec_get_size = petsc_call(
                'VecGetSize', [solver_objs['x_local'], Byref(solver_objs['localsize'])]
            )

            field_from_ptr = FieldFromPointer(
                self.target.function._C_field_data, self.target.function._C_symbol
            )

            expr = DummyExpr(
                start_ptr, cast_mapper[(self.target.dtype, '*')](field_from_ptr) +
                Mul(target_time, solver_objs['localsize']), init=True
            )

            vec_replace_array = petsc_call(
                'VecReplaceArray', [solver_objs['x_local'], start_ptr]
            )
            return (vec_get_size, expr, vec_replace_array)
        else:
            return super().replace_array(solver_objs)

    def assign_time_iters(self, struct):
        """
        Assign required time iterators to the struct.
        These iterators are updated at each timestep in the main kernel
        for use in callback functions.

        Examples
        --------
        >>> struct
        ctx
        >>> struct.fields
        [h_x, x_M, x_m, f1(t, x), t0, t1]
        >>> assigned = assign_time_iters(struct)
        >>> print(assigned[0])
        ctx.t0 = t0;
        >>> print(assigned[1])
        ctx.t1 = t1;
        """
        to_assign = [
            f for f in struct.fields if (f.is_Dimension and (f.is_Time or f.is_Modulo))
        ]
        time_iter_assignments = [
            DummyExpr(FieldFromComposite(field, struct), field)
            for field in to_assign
        ]
        return time_iter_assignments


Null = Macro('NULL')
void = 'void'
dummyctx = Symbol('lctx')
dummyptr = DummyArg('dummy')


# TODO: Don't use c.Line here?
petsc_func_begin_user = c.Line('PetscFunctionBeginUser;')
