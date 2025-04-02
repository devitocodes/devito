from collections import OrderedDict
from functools import cached_property

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, FindNodes,
                           retrieve_iteration_tree, filter_iterations, Iteration)
from devito.symbolics import (Byref, FieldFromPointer, cast, VOID,
                              FieldFromComposite, IntDiv, Deref, Mod)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import Temp, Dimension
from devito.tools import filter_ordered

from devito.petsc.types import PETScArray
from devito.petsc.iet.nodes import (PETScCallable, FormFunctionCallback,
                                    MatShellSetOp, PetscMetaData)
from devito.petsc.iet.utils import petsc_call, petsc_struct
from devito.petsc.utils import solver_mapper
from devito.petsc.types import (DM, Mat, CallbackVec, Vec, KSP, PC, SNES,
                                PetscInt, StartPtr, PointerIS, PointerDM, VecScatter,
                                DMCast, JacobianStructCast, JacobianStruct,
                                SubMatrixStruct, CallbackDM)


class CBBuilder:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __init__(self, **kwargs):

        self.rcompile = kwargs.get('rcompile', None)
        self.sregistry = kwargs.get('sregistry', None)
        self.concretize_mapper = kwargs.get('concretize_mapper', {})
        self.timedep = kwargs.get('timedep')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.injectsolve = kwargs.get('injectsolve')

        self._efuncs = OrderedDict()
        self._struct_params = []

        self._main_matvec_callback = None
        self._main_formfunc_callback = None
        self._user_struct_callback = None
        # TODO: Test pickling. The mutability of these lists
        # could cause issues when pickling?
        self._matvecs = []
        self._formfuncs = []
        self._formrhs = []
        self._initialguesses = []

        self._make_core()
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
    def main_matvec_callback(self):
        """
        This is the matvec callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))...));`
        """
        return self._matvecs[0]

    @property
    def main_formfunc_callback(self):
        return self._formfuncs[0]

    @property
    def matvecs(self):
        return self._matvecs

    @property
    def formfuncs(self):
        return self._formfuncs

    @property
    def formrhs(self):
        return self._formrhs

    @property
    def initialguesses(self):
        return self._initialguesses

    @property
    def user_struct_callback(self):
        return self._user_struct_callback

    def _make_core(self):
        fielddata = self.injectsolve.expr.rhs.fielddata
        self._make_matvec(fielddata, fielddata.matvecs)
        self._make_formfunc(fielddata)
        self._make_formrhs(fielddata)
        if fielddata.initialguess:
            self._make_initialguess(fielddata)
        self._make_user_struct_callback()

    def _make_matvec(self, fielddata, matvecs, prefix='MatMult'):
        # Compile matvec `eqns` into an IET via recursive compilation
        irs_matvec, _ = self.rcompile(matvecs,
                                      options={'mpi': False}, sregistry=self.sregistry,
                                      concretize_mapper=self.concretize_mapper)
        body_matvec = self._create_matvec_body(List(body=irs_matvec.uiet.body),
                                               fielddata)

        objs = self.objs
        cb = PETScCallable(
            self.sregistry.make_name(prefix=prefix),
            body_matvec,
            retval=objs['err'],
            parameters=(objs['J'], objs['X'], objs['Y'])
        )
        self._matvecs.append(cb)
        self._efuncs[cb.name] = cb

    def _create_matvec_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']
        xlocal = objs['xloc']
        ylocal = objs['yloc']
        y_matvec = fielddata.arrays['y']
        x_matvec = fielddata.arrays['x']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)

        mat_get_dm = petsc_call('MatGetDM', [objs['J'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(xlocal)]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['X'],
                                     insert_vals, xlocal]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['X'], insert_vals, xlocal
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(ylocal)]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [ylocal, Byref(y_matvec._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [xlocal, Byref(x_matvec._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [ylocal, Byref(y_matvec._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [xlocal, Byref(x_matvec._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, ylocal, insert_vals, objs['Y']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, ylocal, insert_vals, objs['Y']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(xlocal)]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(ylocal)]
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
        dereference_funcs = [Dereference(i, ctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        matvec_body = CallableBody(
            List(body=body),
            init=(objs['begin_user'],),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for i in fields}
        matvec_body = Uxreplace(subs).visit(matvec_body)

        self._struct_params.extend(fields)
        return matvec_body

    def _make_formfunc(self, fielddata):
        formfuncs = fielddata.formfuncs
        # Compile formfunc `eqns` into an IET via recursive compilation
        irs_formfunc, _ = self.rcompile(
            formfuncs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body_formfunc = self._create_formfunc_body(
            List(body=irs_formfunc.uiet.body), fielddata
        )
        objs = self.objs
        cb = PETScCallable(
            self.sregistry.make_name(prefix='FormFunction'),
            body_formfunc,
            retval=objs['err'],
            parameters=(objs['snes'], objs['X'], objs['F'], objs['dummyptr'])
        )
        self._formfuncs.append(cb)
        self._efuncs[cb.name] = cb

    def _create_formfunc_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)
        self._struct_params.extend(fields)

        f_formfunc = fielddata.arrays['f']
        x_formfunc = fielddata.arrays['x']

        dm_cast = DummyExpr(dmda, DMCast(objs['dummyptr']), init=True)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(objs['xloc'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['X'],
                                     insert_vals, objs['xloc']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['X'], insert_vals, objs['xloc']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(objs['floc'])]
        )

        vec_get_array_y = petsc_call(
            'VecGetArray', [objs['floc'], Byref(f_formfunc._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [objs['xloc'], Byref(x_formfunc._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_y = petsc_call(
            'VecRestoreArray', [objs['floc'], Byref(f_formfunc._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [objs['xloc'], Byref(x_formfunc._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, objs['floc'], insert_vals, objs['F']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, objs['floc'], insert_vals, objs['F']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(objs['xloc'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(objs['floc'])]
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
            dm_cast,
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
        dereference_funcs = [Dereference(i, ctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formfunc_body = CallableBody(
            List(body=body),
            init=(objs['begin_user'],),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),))

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for i in fields}

        return Uxreplace(subs).visit(formfunc_body)

    def _make_formrhs(self, fielddata):
        formrhs = fielddata.formrhs
        sobjs = self.solver_objs

        # Compile formrhs `eqns` into an IET via recursive compilation
        irs_formrhs, _ = self.rcompile(
            formrhs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body_formrhs = self._create_form_rhs_body(
            List(body=irs_formrhs.uiet.body), fielddata
        )
        objs = self.objs
        cb = PETScCallable(
            self.sregistry.make_name(prefix='FormRHS'),
            body_formrhs,
            retval=objs['err'],
            parameters=(sobjs['callbackdm'], objs['B'])
        )
        self._formrhs.append(cb)
        self._efuncs[cb.name] = cb

    def _create_form_rhs_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        dm_get_local = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['blocal'])]
        )

        dm_global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['B'],
                                     insert_vals, sobjs['blocal']]
        )

        dm_global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['B'], insert_vals,
            sobjs['blocal']
        ])

        b_arr = fielddata.arrays['b']

        vec_get_array = petsc_call(
            'VecGetArray', [sobjs['blocal'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)
        self._struct_params.extend(fields)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['blocal'], insert_vals,
            objs['B']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['blocal'], insert_vals,
            objs['B']
        ])

        vec_restore_array = petsc_call(
            'VecRestoreArray', [sobjs['blocal'], Byref(b_arr._C_symbol)]
        )

        dm_restore_local_bvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(sobjs['blocal'])]
        )

        body = body._rebuild(body=body.body + (
            dm_local_to_global_begin, dm_local_to_global_end, vec_restore_array,
            dm_restore_local_bvec
        ))

        stacks = (
            dm_get_local,
            dm_global_to_local_begin,
            dm_global_to_local_end,
            vec_get_array,
            dm_get_app_context,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, ctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        formrhs_body = CallableBody(
            List(body=[body]),
            init=(objs['begin_user'],),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}

        return Uxreplace(subs).visit(formrhs_body)

    def _make_initialguess(self, fielddata):
        initguess = fielddata.initialguess
        sobjs = self.solver_objs

        # Compile initital guess `eqns` into an IET via recursive compilation
        irs, _ = self.rcompile(
            initguess, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body_init_guess = self._create_initial_guess_body(
            List(body=irs.uiet.body), fielddata
        )
        objs = self.objs
        cb = PETScCallable(
            self.sregistry.make_name(prefix='FormInitialGuess'),
            body_init_guess,
            retval=objs['err'],
            parameters=(sobjs['callbackdm'], objs['xloc'])
        )
        self._initialguesses.append(cb)
        self._efuncs[cb.name] = cb

    def _create_initial_guess_body(self, body, fielddata):
        linsolve_expr = self.injectsolve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        x_arr = fielddata.arrays['x']

        vec_get_array = petsc_call(
            'VecGetArray', [objs['xloc'], Byref(x_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.timedep.uxreplace_time(body)

        fields = self._dummy_fields(body)
        self._struct_params.extend(fields)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        vec_restore_array = petsc_call(
            'VecRestoreArray', [objs['xloc'], Byref(x_arr._C_symbol)]
        )

        body = body._rebuild(body=body.body + (vec_restore_array,))

        stacks = (
            vec_get_array,
            dm_get_app_context,
            dm_get_local_info
        )

        # Dereference function data in struct
        dereference_funcs = [Dereference(i, ctx) for i in
                             fields if isinstance(i.function, AbstractFunction)]

        body = CallableBody(
            List(body=[body]),
            init=(objs['begin_user'],),
            stacks=stacks+tuple(dereference_funcs),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}

        return Uxreplace(subs).visit(body)

    def _make_user_struct_callback(self):
        """
        This is the struct initialised inside the main kernel and
        attached to the DM via DMSetApplicationContext.
        # TODO: this could be common between all PETScSolves instead?
        """
        mainctx = self.solver_objs['userctx'] = petsc_struct(
            self.sregistry.make_name(prefix='ctx'),
            self.filtered_struct_params,
            self.sregistry.make_name(prefix='UserCtx'),
        )
        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, mainctx), i._C_symbol)
            for i in mainctx.callback_fields
        ]
        struct_callback_body = CallableBody(
            List(body=body), init=(self.objs['begin_user'],),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )
        cb = Callable(
            self.sregistry.make_name(prefix='PopulateUserContext'),
            struct_callback_body, self.objs['err'],
            parameters=[mainctx]
        )
        self._efuncs[cb.name] = cb
        self._user_struct_callback = cb

    def _dummy_fields(self, iet):
        # Place all context data required by the shell routines into a struct
        fields = [f.function for f in FindSymbols('basics').visit(iet)]
        fields = [f for f in fields if not isinstance(f.function, (PETScArray, Temp))]
        fields = [
            f for f in fields if not (f.is_Dimension and not (f.is_Time or f.is_Modulo))
        ]
        return fields

    def _uxreplace_efuncs(self):
        sobjs = self.solver_objs
        luserctx = petsc_struct(
            sobjs['userctx'].name,
            self.filtered_struct_params,
            sobjs['userctx'].pname,
            modifier=' *'
        )
        mapper = {}
        visitor = Uxreplace({self.objs['dummyctx']: luserctx})
        for k, v in self._efuncs.items():
            mapper.update({k: visitor.visit(v)})
        return mapper


class CCBBuilder(CBBuilder):
    def __init__(self, **kwargs):
        self._submatrices_callback = None
        super().__init__(**kwargs)

    @property
    def submatrices_callback(self):
        return self._submatrices_callback

    @property
    def submatrices(self):
        return self.injectsolve.expr.rhs.fielddata.submatrices

    @property
    def main_matvec_callback(self):
        """
        This is the matvec callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));`
        """
        return self._main_matvec_callback

    @property
    def main_formfunc_callback(self):
        return self._main_formfunc_callback

    def _make_core(self):
        injectsolve = self.injectsolve
        targets = injectsolve.expr.rhs.fielddata.targets
        all_fielddata = injectsolve.expr.rhs.fielddata

        for t in targets:
            data = all_fielddata.get_field_data(t)
            self._make_formfunc(data)
            self._make_formrhs(data)

            row_matvecs = all_fielddata.submatrices.submatrices[t]
            for submat, mtvs in row_matvecs.items():
                if mtvs['matvecs']:
                    self._make_matvec(data, mtvs['matvecs'], prefix=f'{submat}_MatMult')

        self._make_user_struct_callback()
        self._make_whole_matvec()
        self._make_whole_formfunc()
        self._create_submatrices()
        self._efuncs['PopulateMatContext'] = self.objs['dummyefunc']

    def _make_whole_matvec(self):
        objs = self.objs
        body = self._whole_matvec_body()

        cb = PETScCallable(
            self.sregistry.make_name(prefix='WholeMatMult'),
            List(body=body),
            retval=objs['err'],
            parameters=(objs['J'], objs['X'], objs['Y'])
        )
        self._main_matvec_callback = cb
        self._efuncs[cb.name] = cb

    def _whole_matvec_body(self):
        objs = self.objs
        sobjs = self.solver_objs

        jctx = objs['ljacctx']
        ctx_main = petsc_call('MatShellGetContext', [objs['J'], Byref(jctx)])

        nonzero_submats = self.submatrices.nonzero_submatrix_keys

        calls = ()
        for sm in nonzero_submats:
            idx = self.submatrices.submat_to_index[sm]
            ctx = sobjs[f'{sm}ctx']
            X = sobjs[f'{sm}X']
            Y = sobjs[f'{sm}Y']
            rows = objs['rows'].base
            cols = objs['cols'].base
            sm_indexed = objs['Submats'].indexed[idx]

            calls += (
                DummyExpr(sobjs[sm], FieldFromPointer(sm_indexed, jctx)),
                petsc_call('MatShellGetContext', [sobjs[sm], Byref(ctx)]),
                petsc_call(
                    'VecGetSubVector',
                    [objs['X'], Deref(FieldFromPointer(cols, ctx)), Byref(X)]
                ),
                petsc_call(
                    'VecGetSubVector',
                    [objs['Y'], Deref(FieldFromPointer(rows, ctx)), Byref(Y)]
                ),
                petsc_call('MatMult', [sobjs[sm], X, Y]),
                petsc_call(
                    'VecRestoreSubVector',
                    [objs['X'], Deref(FieldFromPointer(cols, ctx)), Byref(X)]
                ),
                petsc_call(
                    'VecRestoreSubVector',
                    [objs['Y'], Deref(FieldFromPointer(rows, ctx)), Byref(Y)]
                ),
            )
        return CallableBody(
            List(body=(ctx_main, BlankLine) + calls),
            init=(objs['begin_user'],),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

    def _make_whole_formfunc(self):
        objs = self.objs
        body = self._whole_formfunc_body()

        cb = PETScCallable(
            self.sregistry.make_name(prefix='WholeFormFunc'),
            List(body=body),
            retval=objs['err'],
            parameters=(objs['snes'], objs['X'], objs['F'], objs['dummyptr'])
        )
        self._main_formfunc_callback = cb
        self._efuncs[cb.name] = cb

    def _whole_formfunc_body(self):
        objs = self.objs
        sobjs = self.solver_objs

        ljacctx = objs['ljacctx']
        struct_cast = DummyExpr(ljacctx, JacobianStructCast(objs['dummyptr']))
        X = objs['X']
        F = objs['F']

        targets = self.injectsolve.expr.rhs.fielddata.targets

        deref_subdms = Dereference(objs['LocalSubdms'], ljacctx)
        deref_fields = Dereference(objs['LocalFields'], ljacctx)

        calls = ()
        for i, t in enumerate(targets):
            field_ptr = FieldFromPointer(objs['LocalFields'].indexed[i], ljacctx)
            x_name = f'Xglobal{t.name}'
            f_name = f'Fglobal{t.name}'

            calls += (
                petsc_call('VecGetSubVector', [X, field_ptr, Byref(sobjs[x_name])]),
                petsc_call('VecGetSubVector', [F, field_ptr, Byref(sobjs[f_name])]),
                petsc_call(self.formfuncs[i].name, [
                    objs['snes'], sobjs[x_name], sobjs[f_name],
                    VOID(objs['LocalSubdms'].indexed[i], stars='*')
                ]),
                petsc_call('VecRestoreSubVector', [X, field_ptr, Byref(sobjs[x_name])]),
                petsc_call('VecRestoreSubVector', [F, field_ptr, Byref(sobjs[f_name])]),
            )
        return CallableBody(
            List(body=calls + (BlankLine,)),
            init=(objs['begin_user'],),
            stacks=(struct_cast, deref_subdms, deref_fields),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

    def _create_submatrices(self):
        body = self._submat_callback_body()
        objs = self.objs
        params = (
            objs['J'],
            objs['nfields'],
            objs['irow'],
            objs['icol'],
            objs['matreuse'],
            objs['Submats'],
        )
        cb = PETScCallable(
            self.sregistry.make_name(prefix='MatCreateSubMatrices'),
            body,
            retval=objs['err'],
            parameters=params
        )
        self._submatrices_callback = cb
        self._efuncs[cb.name] = cb

    def _submat_callback_body(self):
        objs = self.objs
        sobjs = self.solver_objs

        n_submats = DummyExpr(
            objs['nsubmats'], Mul(objs['nfields'], objs['nfields'])
        )

        malloc_submats = petsc_call('PetscCalloc1', [objs['nsubmats'], objs['Submats']])

        mat_get_dm = petsc_call('MatGetDM', [objs['J'], Byref(sobjs['callbackdm'])])

        dm_get_app = petsc_call(
            'DMGetApplicationContext', [sobjs['callbackdm'], Byref(objs['dummyctx'])]
        )

        get_ctx = petsc_call('MatShellGetContext', [objs['J'], Byref(objs['ljacctx'])])

        Null = objs['Null']
        dm_get_info = petsc_call(
            'DMDAGetInfo', [
                sobjs['callbackdm'], Null, Byref(sobjs['M']), Byref(sobjs['N']),
                Null, Null, Null, Null, Byref(objs['dof']), Null, Null, Null, Null, Null
            ]
        )
        subblock_rows = DummyExpr(objs['subblockrows'], Mul(sobjs['M'], sobjs['N']))
        subblock_cols = DummyExpr(objs['subblockcols'], Mul(sobjs['M'], sobjs['N']))

        ptr = DummyExpr(objs['submat_arr']._C_symbol, Deref(objs['Submats']), init=True)

        mat_create = petsc_call('MatCreate', [self.objs['comm'], Byref(objs['block'])])

        mat_set_sizes = petsc_call(
            'MatSetSizes', [
                objs['block'], 'PETSC_DECIDE', 'PETSC_DECIDE',
                objs['subblockrows'], objs['subblockcols']
            ]
        )

        mat_set_type = petsc_call('MatSetType', [objs['block'], 'MATSHELL'])

        malloc = petsc_call('PetscMalloc1', [1, Byref(objs['subctx'])])
        i = Dimension(name='i')

        row_idx = DummyExpr(objs['rowidx'], IntDiv(i, objs['dof']))
        col_idx = DummyExpr(objs['colidx'], Mod(i, objs['dof']))

        deref_subdm = Dereference(objs['Subdms'], objs['ljacctx'])

        set_rows = DummyExpr(
            FieldFromPointer(objs['rows'].base, objs['subctx']),
            Byref(objs['irow'].indexed[objs['rowidx']])
        )
        set_cols = DummyExpr(
            FieldFromPointer(objs['cols'].base, objs['subctx']),
            Byref(objs['icol'].indexed[objs['colidx']])
        )
        dm_set_ctx = petsc_call(
            'DMSetApplicationContext', [
                objs['Subdms'].indexed[objs['rowidx']], objs['dummyctx']
            ]
        )
        matset_dm = petsc_call('MatSetDM', [
            objs['block'], objs['Subdms'].indexed[objs['rowidx']]
        ])

        set_ctx = petsc_call('MatShellSetContext', [objs['block'], objs['subctx']])

        mat_setup = petsc_call('MatSetUp', [objs['block']])

        assign_block = DummyExpr(objs['submat_arr'].indexed[i], objs['block'])

        iter_body = (
            mat_create,
            mat_set_sizes,
            mat_set_type,
            malloc,
            row_idx,
            col_idx,
            set_rows,
            set_cols,
            dm_set_ctx,
            matset_dm,
            set_ctx,
            mat_setup,
            assign_block
        )

        upper_bound = objs['nsubmats'] - 1
        iteration = Iteration(List(body=iter_body), i, upper_bound)

        nonzero_submats = self.submatrices.nonzero_submatrix_keys
        matvec_lookup = {mv.name.split('_')[0]: mv for mv in self.matvecs}

        matmult_op = [
            petsc_call(
                'MatShellSetOperation',
                [
                    objs['submat_arr'].indexed[self.submatrices.submat_to_index[sb]],
                    'MATOP_MULT',
                    MatShellSetOp(matvec_lookup[sb].name, void, void),
                ],
            )
            for sb in nonzero_submats if sb in matvec_lookup
        ]

        body = [
            n_submats,
            malloc_submats,
            mat_get_dm,
            dm_get_app,
            dm_get_info,
            subblock_rows,
            subblock_cols,
            ptr,
            BlankLine,
            iteration,
        ] + matmult_op

        return CallableBody(
            List(body=tuple(body)),
            init=(objs['begin_user'],),
            stacks=(get_ctx, deref_subdm),
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )


class BaseObjectBuilder:
    """
    A base class for constructing objects needed for a PETSc solver.
    Designed to be extended by subclasses, which can override the `_extend_build`
    method to support specific use cases.
    """
    def __init__(self, **kwargs):
        self.injectsolve = kwargs.get('injectsolve')
        self.objs = kwargs.get('objs')
        self.sregistry = kwargs.get('sregistry')
        self.fielddata = self.injectsolve.expr.rhs.fielddata
        self.solver_objs = self._build()

    def _build(self):
        """
        # TODO: update docs
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
        targets = self.fielddata.targets
        base_dict = {
            'Jac': Mat(sreg.make_name(prefix='J')),
            'xglobal': Vec(sreg.make_name(prefix='xglobal')),
            'xlocal': Vec(sreg.make_name(prefix='xlocal')),
            'bglobal': Vec(sreg.make_name(prefix='bglobal')),
            'blocal': CallbackVec(sreg.make_name(prefix='blocal')),
            'ksp': KSP(sreg.make_name(prefix='ksp')),
            'pc': PC(sreg.make_name(prefix='pc')),
            'snes': SNES(sreg.make_name(prefix='snes')),
            'localsize': PetscInt(sreg.make_name(prefix='localsize')),
            'dmda': DM(sreg.make_name(prefix='da'), dofs=len(targets)),
            'callbackdm': CallbackDM(sreg.make_name(prefix='dm')),
        }
        self._target_dependent(base_dict)
        return self._extend_build(base_dict)

    def _target_dependent(self, base_dict):
        """
        '_ptr' (StartPtr): A pointer to the beginning of the solution array
        that will be updated at each time step.
        """
        sreg = self.sregistry
        target = self.fielddata.target
        base_dict[f'{target.name}_ptr'] = StartPtr(
            sreg.make_name(prefix=f'{target.name}_ptr'), target.dtype
        )

    def _extend_build(self, base_dict):
        """
        Subclasses can override this method to extend or modify the
        base dictionary of solver objects.
        """
        return base_dict


class CoupledObjectBuilder(BaseObjectBuilder):
    def _extend_build(self, base_dict):
        injectsolve = self.injectsolve
        sreg = self.sregistry
        objs = self.objs
        targets = self.fielddata.targets

        base_dict['fields'] = PointerIS(
            name=sreg.make_name(prefix='fields'), nindices=len(targets)
        )
        base_dict['subdms'] = PointerDM(
            name=sreg.make_name(prefix='subdms'), nindices=len(targets)
        )
        base_dict['nfields'] = PetscInt(sreg.make_name(prefix='nfields'))

        space_dims = len(self.fielddata.grid.dimensions)

        dim_labels = ["M", "N", "P"]
        base_dict.update({
            dim_labels[i]: PetscInt(dim_labels[i]) for i in range(space_dims)
        })

        submatrices = injectsolve.expr.rhs.fielddata.submatrices
        submatrix_keys = submatrices.submatrix_keys

        base_dict['jacctx'] = JacobianStruct(
            name=sreg.make_name(prefix=objs['ljacctx'].name),
            fields=objs['ljacctx'].fields,
        )

        for key in submatrix_keys:
            base_dict[key] = Mat(name=key)
            base_dict[f'{key}ctx'] = SubMatrixStruct(
                name=f'{key}ctx',
                fields=objs['subctx'].fields,
            )
            base_dict[f'{key}X'] = CallbackVec(f'{key}X')
            base_dict[f'{key}Y'] = CallbackVec(f'{key}Y')
            base_dict[f'{key}F'] = CallbackVec(f'{key}F')

        return base_dict

    def _target_dependent(self, base_dict):
        sreg = self.sregistry
        targets = self.fielddata.targets
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


class BaseSetup:
    def __init__(self, **kwargs):
        self.injectsolve = kwargs.get('injectsolve')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.cbbuilder = kwargs.get('cbbuilder')
        self.fielddata = self.injectsolve.expr.rhs.fielddata
        self.calls = self._setup()

    @property
    def snes_ctx(self):
        """
        The [optional] context for private data for the function evaluation routine.
        https://petsc.org/main/manualpages/SNES/SNESSetFunction/
        """
        return VOID(self.solver_objs['dmda'], stars='*')

    def _setup(self):
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['dmda']

        solver_params = self.injectsolve.expr.rhs.solver_parameters

        snes_create = petsc_call('SNESCreate', [objs['comm'], Byref(sobjs['snes'])])

        snes_set_dm = petsc_call('SNESSetDM', [sobjs['snes'], dmda])

        create_matrix = petsc_call('DMCreateMatrix', [dmda, Byref(sobjs['Jac'])])

        # NOTE: Assuming all solves are linear for now
        snes_set_type = petsc_call('SNESSetType', [sobjs['snes'], 'SNESKSPONLY'])

        snes_set_jac = petsc_call(
            'SNESSetJacobian', [sobjs['snes'], sobjs['Jac'],
                                sobjs['Jac'], 'MatMFFDComputeJacobian', objs['Null']]
        )

        global_x = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['xglobal'])])

        local_x = petsc_call('DMCreateLocalVector',
                             [dmda, Byref(sobjs['xlocal'])])

        get_local_size = petsc_call('VecGetSize',
                                    [sobjs['xlocal'], Byref(sobjs['localsize'])])

        global_b = petsc_call('DMCreateGlobalVector',
                              [dmda, Byref(sobjs['bglobal'])])

        snes_get_ksp = petsc_call('SNESGetKSP',
                                  [sobjs['snes'], Byref(sobjs['ksp'])])

        ksp_set_tols = petsc_call(
            'KSPSetTolerances', [sobjs['ksp'], solver_params['ksp_rtol'],
                                 solver_params['ksp_atol'], solver_params['ksp_divtol'],
                                 solver_params['ksp_max_it']]
        )

        ksp_set_type = petsc_call(
            'KSPSetType', [sobjs['ksp'], solver_mapper[solver_params['ksp_type']]]
        )

        ksp_get_pc = petsc_call(
            'KSPGetPC', [sobjs['ksp'], Byref(sobjs['pc'])]
        )

        # Even though the default will be jacobi, set to PCNONE for now
        pc_set_type = petsc_call('PCSetType', [sobjs['pc'], 'PCNONE'])

        ksp_set_from_ops = petsc_call('KSPSetFromOptions', [sobjs['ksp']])

        matvec = self.cbbuilder.main_matvec_callback
        matvec_operation = petsc_call(
            'MatShellSetOperation',
            [sobjs['Jac'], 'MATOP_MULT', MatShellSetOp(matvec.name, void, void)]
        )
        formfunc = self.cbbuilder.main_formfunc_callback
        formfunc_operation = petsc_call(
            'SNESSetFunction',
            [sobjs['snes'], objs['Null'], FormFunctionCallback(formfunc.name, void, void),
             self.snes_ctx]
        )

        dmda_calls = self._create_dmda_calls(dmda)

        mainctx = sobjs['userctx']

        call_struct_callback = petsc_call(
            self.cbbuilder.user_struct_callback.name, [Byref(mainctx)]
        )

        # TODO: maybe don't need to explictly set this
        mat_set_dm = petsc_call('MatSetDM', [sobjs['Jac'], dmda])

        calls_set_app_ctx = petsc_call('DMSetApplicationContext', [dmda, Byref(mainctx)])

        base_setup = dmda_calls + (
            snes_create,
            snes_set_dm,
            create_matrix,
            snes_set_jac,
            snes_set_type,
            global_x,
            local_x,
            get_local_size,
            global_b,
            snes_get_ksp,
            ksp_set_tols,
            ksp_set_type,
            ksp_get_pc,
            pc_set_type,
            ksp_set_from_ops,
            matvec_operation,
            formfunc_operation,
            call_struct_callback,
            mat_set_dm,
            calls_set_app_ctx,
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
        dm_setup = petsc_call('DMSetUp', [dmda])
        dm_mat_type = petsc_call('DMSetMatType', [dmda, 'MATSHELL'])
        return dmda_create, dm_setup, dm_mat_type

    def _create_dmda(self, dmda):
        objs = self.objs
        grid = self.fielddata.grid
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
        args.append(dmda.dofs)
        # "Stencil width" -> size of overlap
        stencil_width = self.fielddata.space_order
        args.append(stencil_width)
        args.extend([objs['Null']]*nspace_dims)

        # The distributed array object
        args.append(Byref(dmda))

        # The PETSc call used to create the DMDA
        dmda = petsc_call(f'DMDACreate{nspace_dims}d', args)

        return dmda


class CoupledSetup(BaseSetup):
    @property
    def snes_ctx(self):
        return Byref(self.solver_objs['jacctx'])

    def _extend_setup(self):
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['dmda']
        create_field_decomp = petsc_call(
            'DMCreateFieldDecomposition',
            [dmda, Byref(sobjs['nfields']), objs['Null'], Byref(sobjs['fields']),
             Byref(sobjs['subdms'])]
        )
        submat_cb = self.cbbuilder.submatrices_callback
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

        targets = self.fielddata.targets

        deref_dms = [
            DummyExpr(sobjs[f'da{t.name}'], sobjs['subdms'].indexed[i])
            for i, t in enumerate(targets)
        ]

        xglobals = [petsc_call(
            'DMCreateGlobalVector',
            [sobjs[f'da{t.name}'], Byref(sobjs[f'xglobal{t.name}'])]
        ) for t in targets]

        bglobals = [petsc_call(
            'DMCreateGlobalVector',
            [sobjs[f'da{t.name}'], Byref(sobjs[f'bglobal{t.name}'])]
        ) for t in targets]

        return (
            create_field_decomp,
            matop_create_submats_op,
            call_coupled_struct_callback,
            shell_set_ctx,
            create_submats
        ) + tuple(deref_dms) + tuple(xglobals) + tuple(bglobals)


class Solver:
    def __init__(self, **kwargs):
        self.injectsolve = kwargs.get('injectsolve')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.iters = kwargs.get('iters')
        self.cbbuilder = kwargs.get('cbbuilder')
        self.timedep = kwargs.get('timedep')
        # TODO: Should/could _execute_solve be a cached_property?
        self.calls = self._execute_solve()

    def _execute_solve(self):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        sobjs = self.solver_objs
        target = self.injectsolve.expr.rhs.fielddata.target

        struct_assignment = self.timedep.assign_time_iters(sobjs['userctx'])

        rhs_callback = self.cbbuilder.formrhs[0]

        dmda = sobjs['dmda']

        rhs_call = petsc_call(rhs_callback.name, [sobjs['dmda'], sobjs['bglobal']])

        vec_replace_array = self.timedep.replace_array(target)

        if self.cbbuilder.initialguesses:
            initguess = self.cbbuilder.initialguesses[0]
            initguess_call = petsc_call(initguess.name, [dmda, sobjs['xlocal']])
        else:
            initguess_call = None

        dm_local_to_global_x = petsc_call(
            'DMLocalToGlobal', [dmda, sobjs['xlocal'], insert_vals,
                                sobjs['xglobal']]
        )

        snes_solve = petsc_call('SNESSolve', [
            sobjs['snes'], sobjs['bglobal'], sobjs['xglobal']]
        )

        dm_global_to_local_x = petsc_call('DMGlobalToLocal', [
            dmda, sobjs['xglobal'], insert_vals, sobjs['xlocal']]
        )

        vec_reset_array = self.timedep.reset_array(target)

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
        ) + vec_replace_array + (
            initguess_call,
            dm_local_to_global_x,
            snes_solve,
            dm_global_to_local_x,
            vec_reset_array,
            BlankLine,
        )
        return List(body=run_solver_calls)

    @cached_property
    def spatial_body(self):
        spatial_body = []
        # TODO: remove the iters[0]
        for tree in retrieve_iteration_tree(self.iters[0]):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
            if self.injectsolve in FindNodes(PetscMetaData).visit(root):
                spatial_body.append(root)
        spatial_body, = spatial_body
        return spatial_body


class CoupledSolver(Solver):
    def _execute_solve(self):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        sobjs = self.solver_objs

        struct_assignment = self.timedep.assign_time_iters(sobjs['userctx'])

        rhs_callbacks = self.cbbuilder.formrhs

        xglob = sobjs['xglobal']
        bglob = sobjs['bglobal']

        targets = self.injectsolve.expr.rhs.fielddata.targets

        # TODO: optimise the ccode generated here
        pre_solve = ()
        post_solve = ()

        for i, (c, t) in enumerate(zip(rhs_callbacks, targets)):
            name = t.name
            dm = sobjs[f'da{name}']
            target_xloc = sobjs[f'xlocal{name}']
            target_xglob = sobjs[f'xglobal{name}']
            target_bglob = sobjs[f'bglobal{name}']
            field = sobjs['fields'].indexed[i]
            s = sobjs[f'scatter{name}']

            pre_solve += (
                petsc_call(c.name, [dm, target_bglob]),
                petsc_call('DMCreateLocalVector', [dm, Byref(target_xloc)]),
                self.timedep.replace_array(t),
                petsc_call(
                    'DMLocalToGlobal',
                    [dm, target_xloc, insert_vals, target_xglob]
                ),
                petsc_call(
                    'VecScatterCreate',
                    [xglob, field, target_xglob, self.objs['Null'], Byref(s)]
                ),
                petsc_call(
                    'VecScatterBegin',
                    [s, target_xglob, xglob, insert_vals, sreverse]
                ),
                petsc_call(
                    'VecScatterEnd',
                    [s, target_xglob, xglob, insert_vals, sreverse]
                ),
                petsc_call(
                    'VecScatterBegin',
                    [s, target_bglob, bglob, insert_vals, sreverse]
                ),
                petsc_call(
                    'VecScatterEnd',
                    [s, target_bglob, bglob, insert_vals, sreverse]
                ),
            )

            post_solve += (
                petsc_call(
                    'VecScatterBegin',
                    [s, xglob, target_xglob, insert_vals, sforward]
                ),
                petsc_call(
                    'VecScatterEnd',
                    [s, xglob, target_xglob, insert_vals, sforward]
                ),
                petsc_call(
                    'DMGlobalToLocal',
                    [dm, target_xglob, insert_vals, target_xloc]
                )
            )

        snes_solve = (petsc_call('SNESSolve', [sobjs['snes'], bglob, xglob]),)

        return List(
            body=(
                (struct_assignment,)
                + pre_solve
                + snes_solve
                + post_solve
                + (BlankLine,)
            )
        )


class NonTimeDependent:
    def __init__(self, **kwargs):
        self.injectsolve = kwargs.get('injectsolve')
        self.iters = kwargs.get('iters')
        self.sobjs = kwargs.get('solver_objs')
        self.kwargs = kwargs
        self.origin_to_moddim = self._origin_to_moddim_mapper(self.iters)
        self.time_idx_to_symb = self.injectsolve.expr.rhs.time_mapper

    def _origin_to_moddim_mapper(self, iters):
        return {}

    def uxreplace_time(self, body):
        return body

    def replace_array(self, target):
        """
        VecReplaceArray() is a PETSc function that allows replacing the array
        of a `Vec` with a user provided array.
        https://petsc.org/release/manualpages/Vec/VecReplaceArray/

        This function is used to replace the array of the PETSc solution `Vec`
        with the array from the `Function` object representing the target.

        Examples
        --------
        >>> target
        f1(x, y)
        >>> call = replace_array(target)
        >>> print(call)
        PetscCall(VecReplaceArray(xlocal0,f1_vec->data));
        """
        sobjs = self.sobjs

        field_from_ptr = FieldFromPointer(
            target.function._C_field_data, target.function._C_symbol
        )
        xlocal = sobjs.get(f'xlocal{target.name}', sobjs['xlocal'])
        return (petsc_call('VecPlaceArray', [xlocal, field_from_ptr]),)

    def reset_array(self, target):
        """
        """
        sobjs = self.sobjs
        xlocal = sobjs.get(f'xlocal{target.name}', sobjs['xlocal'])
        return (
            petsc_call('VecResetArray', [xlocal])
        )

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
    def time_spacing(self):
        return self.injectsolve.expr.rhs.grid.stepping_dim.spacing

    @cached_property
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

    def is_target_time(self, target):
        return any(i.is_Time for i in target.dimensions)

    def target_time(self, target):
        target_time = [
            i for i, d in zip(target.indices, target.dimensions)
            if d.is_Time
        ]
        assert len(target_time) == 1
        target_time = target_time.pop()
        return target_time

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

    def replace_array(self, target):
        """
        In the case that the actual target is time-dependent e.g a `TimeFunction`,
        a pointer to the first element in the array that will be updated during
        the time step is passed to VecReplaceArray().

        Examples
        --------
        >>> target
        f1(time + dt, x, y)
        >>> calls = replace_array(target)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(xlocal0,&(localsize0)));
        float * f1_ptr0 = (time + 1)*localsize0 + (float*)(f1_vec->data);
        PetscCall(VecReplaceArray(xlocal0,f1_ptr0));

        >>> target
        f1(t + dt, x, y)
        >>> calls = replace_array(target)
        >>> print(List(body=calls))
        PetscCall(VecGetSize(xlocal0,&(localsize0)));
        float * f1_ptr0 = t1*localsize0 + (float*)(f1_vec->data);
        PetscCall(VecReplaceArray(xlocal0,f1_ptr0));
        """
        sobjs = self.sobjs

        if self.is_target_time(target):
            mapper = {self.time_spacing: 1, -self.time_spacing: -1}

            target_time = self.target_time(target).xreplace(mapper)
            target_time = self.origin_to_moddim.get(target_time, target_time)

            xlocal = sobjs.get(f'xlocal{target.name}', sobjs['xlocal'])
            start_ptr = sobjs[f'{target.name}_ptr']

            caster = cast(target.dtype, '*')
            return (
                DummyExpr(
                    start_ptr,
                    caster(
                        FieldFromPointer(target._C_field_data, target._C_symbol)
                    ) + Mul(target_time, sobjs['localsize']),
                    init=True
                ),
                petsc_call('VecPlaceArray', [xlocal, start_ptr])
            )
        return super().replace_array(target)

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


void = 'void'
insert_vals = 'INSERT_VALUES'
sreverse = 'SCATTER_REVERSE'
sforward = 'SCATTER_FORWARD'
