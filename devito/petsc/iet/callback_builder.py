from collections import OrderedDict

from devito.ir.iet import (Call, FindSymbols, List, Uxreplace, CallableBody,
                           Dereference, DummyExpr, BlankLine, Callable, Iteration,
                           PointerCast, Definition)
from devito.symbolics import (Byref, FieldFromPointer, IntDiv, Deref, Mod, String, Null)
from devito.symbolics.unevaluation import Mul
from devito.types.basic import AbstractFunction
from devito.types import Dimension
from devito.tools import filter_ordered

from devito.petsc.iet.nodes import PETScCallable, MatShellSetOp
from devito.petsc.iet.utils import (petsc_call, void, get_user_struct_fields)
from devito.petsc.types import DMCast, MainUserStruct, CallbackUserStruct
from devito.petsc.iet.object_builder import objs
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.types.strings import InsertMode


class BaseCallback:
    """
    Build IET routines to generate PETSc callback functions.
    """
    def __init__(self, **kwargs):

        self.rcompile = kwargs.get('rcompile', None)
        self.sregistry = kwargs.get('sregistry', None)
        self.concretize_mapper = kwargs.get('concretize_mapper', {})
        self.time_dependence = kwargs.get('time_dependence')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.inject_solve = kwargs.get('inject_solve')
        self.solve_expr = self.inject_solve.expr.rhs

        self._efuncs = OrderedDict()
        self._struct_params = []

        self._set_options_efunc = None
        self._clear_options_efunc = None
        self._main_matvec_callback = None
        self._user_struct_callback = None
        self._F_efunc = None
        self._b_efunc = None

        self._J_efuncs = []
        self._initial_guesses = []

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
        The matrix-vector callback for the full Jacobian.
        This is the function set in the main Kernel via:
            PetscCall(MatShellSetOperation(J, MATOP_MULT, (void (*)(void))...));
        The callback has the signature `(Mat, Vec, Vec)`.
        """
        return self._J_efuncs[0]

    @property
    def J_efuncs(self):
        """
        List of matrix-vector callbacks.
        Each callback has the signature `(Mat, Vec, Vec)`. Typically, this list
        contains a single element, but in mixed systems it can include multiple
        callbacks, one for each subblock.
        """
        return self._J_efuncs

    @property
    def initial_guesses(self):
        return self._initial_guesses

    @property
    def user_struct_callback(self):
        return self._user_struct_callback

    @property
    def solver_parameters(self):
        return self.solve_expr.solver_parameters

    @property
    def field_data(self):
        return self.solve_expr.field_data

    @property
    def formatted_prefix(self):
        return self.solve_expr.formatted_prefix

    @property
    def arrays(self):
        return self.field_data.arrays

    @property
    def target(self):
        return self.field_data.target

    def _make_core(self):
        self._make_options_callback()
        self._make_matvec(self.field_data.jacobian)
        self._make_formfunc()
        self._make_formrhs()
        if self.field_data.initial_guess.exprs:
            self._make_initial_guess()
        self._make_user_struct_callback()

    def _make_petsc_callable(self, prefix, body, parameters=()):
        return PETScCallable(
            self.sregistry.make_name(prefix=prefix),
            body,
            retval=self.objs['err'],
            parameters=parameters
        )

    def _make_callable_body(self, body, standalones=(), stacks=(), casts=()):
        return CallableBody(
            List(body=body),
            init=(petsc_func_begin_user,),
            standalones=standalones,
            stacks=stacks,
            casts=casts,
            retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
        )

    def _make_options_callback(self):
        """
        Create two callbacks: one to set PETSc options and one
        to clear them.
        Options are only set/cleared if they were not specifed via
        command line arguments.
        """
        params = self.solver_parameters
        prefix = self.inject_solve.expr.rhs.formatted_prefix

        set_body, clear_body = [], []

        for k, v in params.items():
            option = f'-{prefix}{k}'

            # TODO: Revisit use of a global variable here.
            # Consider replacing this with a call to `PetscGetArgs`, though
            # initial attempts failed, possibly because the argv pointer is
            # created in Python?..
            import devito.petsc.initialize
            if option in devito.petsc.initialize._petsc_clargs:
                # Ensures that the command line args take priority
                continue

            option_name = String(option)
            # For options without a value e.g `ksp_view`, pass Null
            option_value = Null if v is None else String(str(v))
            set_body.append(
                petsc_call('PetscOptionsSetValue', [Null, option_name, option_value])
            )
            clear_body.append(
                petsc_call('PetscOptionsClearValue', [Null, option_name])
            )

        set_body = self._make_callable_body(set_body)
        clear_body = self._make_callable_body(clear_body)

        set_callback = self._make_petsc_callable('SetPetscOptions', set_body)
        clear_callback = self._make_petsc_callable('ClearPetscOptions', clear_body)

        self._set_options_efunc = set_callback
        self._efuncs[set_callback.name] = set_callback
        self._clear_options_efunc = clear_callback
        self._efuncs[clear_callback.name] = clear_callback

    def _make_matvec(self, jacobian, prefix='MatMult'):
        # Compile `matvecs` into an IET via recursive compilation
        matvecs = jacobian.matvecs
        irs, _ = self.rcompile(
            matvecs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body = self._create_matvec_body(
            List(body=irs.uiet.body), jacobian
        )
        objs = self.objs
        cb = self._make_petsc_callable(
            prefix, body, parameters=(objs['J'], objs['X'], objs['Y'])
        )
        self._J_efuncs.append(cb)
        self._efuncs[cb.name] = cb

    def _create_matvec_body(self, body, jacobian):
        linsolve_expr = self.inject_solve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']
        xlocal = objs['xloc']
        ylocal = objs['yloc']
        y_matvec = self.arrays[jacobian.row_target]['y']
        x_matvec = self.arrays[jacobian.col_target]['x']

        body = self.time_dependence.uxreplace_time(body)

        fields = get_user_struct_fields(body)

        mat_get_dm = petsc_call('MatGetDM', [objs['J'], Byref(dmda)])

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        zero_y_memory = zero_vector(objs['Y']) if jacobian.zero_memory else None

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(xlocal)]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['X'],
                                     InsertMode.insert_values, xlocal]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['X'], InsertMode.insert_values, xlocal
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(ylocal)]
        )

        zero_ylocal_memory = zero_vector(ylocal)

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
            dmda, ylocal, InsertMode.add_values, objs['Y']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, ylocal, InsertMode.add_values, objs['Y']
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
            zero_y_memory,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            zero_ylocal_memory,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        derefs = dereference_funcs(ctx, fields)

        # Force the struct definition to appear at the very start, since
        # stacks, allocs etc may rely on its information
        struct_definition = [
            Definition(ctx), Definition(dmda), mat_get_dm, dm_get_app_context
        ]

        body = self._make_callable_body(
            body, standalones=struct_definition, stacks=stacks+derefs
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for i in fields}
        body = Uxreplace(subs).visit(body)

        self._struct_params.extend(fields)
        return body

    def _make_formfunc(self):
        objs = self.objs
        F_exprs = self.field_data.residual.F_exprs
        # Compile `F_exprs` into an IET via recursive compilation
        irs, _ = self.rcompile(
            F_exprs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body_formfunc = self._create_formfunc_body(
            List(body=irs.uiet.body)
        )
        parameters = (objs['snes'], objs['X'], objs['F'], objs['dummyptr'])
        cb = self._make_petsc_callable('FormFunction', body_formfunc, parameters)

        self._F_efunc = cb
        self._efuncs[cb.name] = cb

    def _create_formfunc_body(self, body):
        linsolve_expr = self.inject_solve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs
        arrays = self.arrays
        target = self.target

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        body = self.time_dependence.uxreplace_time(body)

        fields = get_user_struct_fields(body)
        self._struct_params.extend(fields)

        f_formfunc = arrays[target]['f']
        x_formfunc = arrays[target]['x']

        dm_cast = DummyExpr(dmda, DMCast(objs['dummyptr']), init=True)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        zero_f_memory = zero_vector(objs['F'])

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(objs['xloc'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['X'],
                                     InsertMode.insert_values, objs['xloc']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['X'], InsertMode.insert_values, objs['xloc']
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
            dmda, objs['floc'], InsertMode.add_values, objs['F']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, objs['floc'], InsertMode.add_values, objs['F']
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
            zero_f_memory,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_y,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        derefs = dereference_funcs(ctx, fields)

        # Force the struct definition to appear at the very start, since
        # stacks, allocs etc may rely on its information
        struct_definition = [Definition(ctx), dm_cast, dm_get_app_context]

        body = self._make_callable_body(
            body, standalones=struct_definition, stacks=stacks+derefs
        )
        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for i in fields}

        return Uxreplace(subs).visit(body)

    def _make_formrhs(self):
        b_exprs = self.field_data.residual.b_exprs
        sobjs = self.solver_objs

        # Compile `b_exprs` into an IET via recursive compilation
        irs, _ = self.rcompile(
            b_exprs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body = self._create_form_rhs_body(
            List(body=irs.uiet.body)
        )
        objs = self.objs
        cb = self._make_petsc_callable(
            'FormRHS', body, parameters=(sobjs['callbackdm'], objs['B'])
        )
        self._b_efunc = cb
        self._efuncs[cb.name] = cb

    def _create_form_rhs_body(self, body):
        linsolve_expr = self.inject_solve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs
        target = self.target

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        dm_get_local = petsc_call(
            'DMGetLocalVector', [dmda, Byref(sobjs['blocal'])]
        )

        dm_global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['B'],
                                     InsertMode.insert_values, sobjs['blocal']]
        )

        dm_global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['B'], InsertMode.insert_values,
            sobjs['blocal']
        ])

        b_arr = self.field_data.arrays[target]['b']

        vec_get_array = petsc_call(
            'VecGetArray', [sobjs['blocal'], Byref(b_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.time_dependence.uxreplace_time(body)

        fields = get_user_struct_fields(body)
        self._struct_params.extend(fields)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, sobjs['blocal'], InsertMode.insert_values,
            objs['B']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, sobjs['blocal'], InsertMode.insert_values,
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
            dm_get_local_info
        )

        # Dereference function data in struct
        derefs = dereference_funcs(ctx, fields)

        # Force the struct definition to appear at the very start, since
        # stacks, allocs etc may rely on its information
        struct_definition = [Definition(ctx), dm_get_app_context]

        body = self._make_callable_body(
            [body], standalones=struct_definition, stacks=stacks+derefs
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}

        return Uxreplace(subs).visit(body)

    def _make_initial_guess(self):
        exprs = self.field_data.initial_guess.exprs
        sobjs = self.solver_objs
        objs = self.objs

        # Compile initital guess `eqns` into an IET via recursive compilation
        irs, _ = self.rcompile(
            exprs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body = self._create_initial_guess_body(
            List(body=irs.uiet.body)
        )
        cb = self._make_petsc_callable(
            'FormInitialGuess', body, parameters=(sobjs['callbackdm'], objs['xloc'])
        )
        self._initial_guesses.append(cb)
        self._efuncs[cb.name] = cb

    def _create_initial_guess_body(self, body):
        linsolve_expr = self.inject_solve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs
        target = self.target

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        x_arr = self.field_data.arrays[target]['x']

        vec_get_array = petsc_call(
            'VecGetArray', [objs['xloc'], Byref(x_arr._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        body = self.time_dependence.uxreplace_time(body)

        fields = get_user_struct_fields(body)
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
            dm_get_local_info
        )

        # Dereference function data in struct
        derefs = dereference_funcs(ctx, fields)

        # Force the struct definition to appear at the very start, since
        # stacks, allocs etc may rely on its information
        struct_definition = [Definition(ctx), dm_get_app_context]

        body = self._make_callable_body(
            body, standalones=struct_definition, stacks=stacks+derefs
        )

        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for
                i in fields if not isinstance(i.function, AbstractFunction)}

        return Uxreplace(subs).visit(body)

    def _make_user_struct_callback(self):
        """
        This is the struct initialised inside the main kernel and
        attached to the DM via DMSetApplicationContext.
        """
        mainctx = self.solver_objs['userctx'] = MainUserStruct(
            name=self.sregistry.make_name(prefix='ctx'),
            pname=self.sregistry.make_name(prefix='UserCtx'),
            fields=self.filtered_struct_params,
            liveness='lazy',
            modifier=None
        )
        body = [
            DummyExpr(FieldFromPointer(i._C_symbol, mainctx), i._C_symbol)
            for i in mainctx.callback_fields
        ]
        struct_callback_body = self._make_callable_body(body)
        cb = Callable(
            self.sregistry.make_name(prefix='PopulateUserContext'),
            struct_callback_body, self.objs['err'],
            parameters=[mainctx]
        )
        self._efuncs[cb.name] = cb
        self._user_struct_callback = cb

    def _uxreplace_efuncs(self):
        sobjs = self.solver_objs
        callback_user_struct = CallbackUserStruct(
            name=sobjs['userctx'].name,
            pname=sobjs['userctx'].pname,
            fields=self.filtered_struct_params,
            liveness='lazy',
            modifier=' *',
            parent=sobjs['userctx']
        )
        mapper = {}
        visitor = Uxreplace({self.objs['dummyctx']: callback_user_struct})
        for k, v in self._efuncs.items():
            mapper.update({k: visitor.visit(v)})
        return mapper


class CoupledCallback(BaseCallback):
    def __init__(self, **kwargs):
        self._submatrices_callback = None
        super().__init__(**kwargs)

    @property
    def submatrices_callback(self):
        return self._submatrices_callback

    @property
    def jacobian(self):
        return self.inject_solve.expr.rhs.field_data.jacobian

    @property
    def main_matvec_callback(self):
        """
        This is the matrix-vector callback associated with the whole Jacobian i.e
        is set in the main kernel via
        `PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))MyMatShellMult));`
        """
        return self._main_matvec_callback

    def _make_core(self):
        for sm in self.field_data.jacobian.nonzero_submatrices:
            self._make_matvec(sm, prefix=f'{sm.name}_MatMult')

        self._make_options_callback()
        self._make_whole_matvec()
        self._make_whole_formfunc()
        self._make_user_struct_callback()
        self._create_submatrices()
        self._efuncs['PopulateMatContext'] = self.objs['dummyefunc']

    def _make_whole_matvec(self):
        objs = self.objs
        body = self._whole_matvec_body()

        parameters = (objs['J'], objs['X'], objs['Y'])
        cb = self._make_petsc_callable(
            'WholeMatMult', List(body=body), parameters=parameters
        )
        self._main_matvec_callback = cb
        self._efuncs[cb.name] = cb

    def _whole_matvec_body(self):
        objs = self.objs
        sobjs = self.solver_objs

        jctx = objs['ljacctx']
        ctx_main = petsc_call('MatShellGetContext', [objs['J'], Byref(jctx)])

        nonzero_submats = self.jacobian.nonzero_submatrices

        zero_y_memory = zero_vector(objs['Y'])

        calls = ()
        for sm in nonzero_submats:
            name = sm.name
            ctx = sobjs[f'{name}ctx']
            X = sobjs[f'{name}X']
            Y = sobjs[f'{name}Y']
            rows = objs['rows'].base
            cols = objs['cols'].base
            sm_indexed = objs['Submats'].indexed[sm.linear_idx]

            calls += (
                DummyExpr(sobjs[name], FieldFromPointer(sm_indexed, jctx)),
                petsc_call('MatShellGetContext', [sobjs[name], Byref(ctx)]),
                petsc_call(
                    'VecGetSubVector',
                    [objs['X'], Deref(FieldFromPointer(cols, ctx)), Byref(X)]
                ),
                petsc_call(
                    'VecGetSubVector',
                    [objs['Y'], Deref(FieldFromPointer(rows, ctx)), Byref(Y)]
                ),
                petsc_call('MatMult', [sobjs[name], X, Y]),
                petsc_call(
                    'VecRestoreSubVector',
                    [objs['X'], Deref(FieldFromPointer(cols, ctx)), Byref(X)]
                ),
                petsc_call(
                    'VecRestoreSubVector',
                    [objs['Y'], Deref(FieldFromPointer(rows, ctx)), Byref(Y)]
                ),
            )
        body = (ctx_main, zero_y_memory, BlankLine) + calls
        return self._make_callable_body(body)

    def _make_whole_formfunc(self):
        objs = self.objs
        F_exprs = self.field_data.residual.F_exprs
        # Compile `F_exprs` into an IET via recursive compilation
        irs, _ = self.rcompile(
            F_exprs, options={'mpi': False}, sregistry=self.sregistry,
            concretize_mapper=self.concretize_mapper
        )
        body = self._whole_formfunc_body(List(body=irs.uiet.body))

        parameters = (objs['snes'], objs['X'], objs['F'], objs['dummyptr'])
        cb = self._make_petsc_callable(
            'WholeFormFunc', body, parameters=parameters
        )

        self._F_efunc = cb
        self._efuncs[cb.name] = cb

    def _whole_formfunc_body(self, body):
        linsolve_expr = self.inject_solve.expr.rhs
        objs = self.objs
        sobjs = self.solver_objs

        dmda = sobjs['callbackdm']
        ctx = objs['dummyctx']

        body = self.time_dependence.uxreplace_time(body)

        fields = get_user_struct_fields(body)
        self._struct_params.extend(fields)

        # Process body with bundles for residual callback
        bundles = sobjs['bundles']
        fbundle = bundles['f']
        xbundle = bundles['x']

        body = self.residual_bundle(body, bundles)

        dm_cast = DummyExpr(dmda, DMCast(objs['dummyptr']), init=True)

        dm_get_app_context = petsc_call(
            'DMGetApplicationContext', [dmda, Byref(ctx._C_symbol)]
        )

        zero_f_memory = zero_vector(objs['F'])

        dm_get_local_xvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(objs['xloc'])]
        )

        global_to_local_begin = petsc_call(
            'DMGlobalToLocalBegin', [dmda, objs['X'],
                                     InsertMode.insert_values, objs['xloc']]
        )

        global_to_local_end = petsc_call('DMGlobalToLocalEnd', [
            dmda, objs['X'], InsertMode.insert_values, objs['xloc']
        ])

        dm_get_local_yvec = petsc_call(
            'DMGetLocalVector', [dmda, Byref(objs['floc'])]
        )

        vec_get_array_f = petsc_call(
            'VecGetArray', [objs['floc'], Byref(fbundle.vector._C_symbol)]
        )

        vec_get_array_x = petsc_call(
            'VecGetArray', [objs['xloc'], Byref(xbundle.vector._C_symbol)]
        )

        dm_get_local_info = petsc_call(
            'DMDAGetLocalInfo', [dmda, Byref(linsolve_expr.localinfo)]
        )

        vec_restore_array_f = petsc_call(
            'VecRestoreArray', [objs['floc'], Byref(fbundle.vector._C_symbol)]
        )

        vec_restore_array_x = petsc_call(
            'VecRestoreArray', [objs['xloc'], Byref(xbundle.vector._C_symbol)]
        )

        dm_local_to_global_begin = petsc_call('DMLocalToGlobalBegin', [
            dmda, objs['floc'], InsertMode.add_values, objs['F']
        ])

        dm_local_to_global_end = petsc_call('DMLocalToGlobalEnd', [
            dmda, objs['floc'], InsertMode.add_values, objs['F']
        ])

        dm_restore_local_xvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(objs['xloc'])]
        )

        dm_restore_local_yvec = petsc_call(
            'DMRestoreLocalVector', [dmda, Byref(objs['floc'])]
        )

        body = body._rebuild(
            body=body.body +
            (vec_restore_array_f,
             vec_restore_array_x,
             dm_local_to_global_begin,
             dm_local_to_global_end,
             dm_restore_local_xvec,
             dm_restore_local_yvec)
        )

        stacks = (
            zero_f_memory,
            dm_get_local_xvec,
            global_to_local_begin,
            global_to_local_end,
            dm_get_local_yvec,
            vec_get_array_f,
            vec_get_array_x,
            dm_get_local_info
        )

        # Dereference function data in struct
        derefs = dereference_funcs(ctx, fields)

        # Force the struct definition to appear at the very start, since
        # stacks, allocs etc may rely on its information
        struct_definition = [Definition(ctx), dm_cast, dm_get_app_context]

        f_soa = PointerCast(fbundle)
        x_soa = PointerCast(xbundle)

        formfunc_body = self._make_callable_body(
            body,
            standalones=struct_definition,
            stacks=stacks+derefs,
            casts=(f_soa, x_soa),
        )
        # Replace non-function data with pointer to data in struct
        subs = {i._C_symbol: FieldFromPointer(i._C_symbol, ctx) for i in fields}

        return Uxreplace(subs).visit(formfunc_body)

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
        cb = self._make_petsc_callable(
            'MatCreateSubMatrices', body, parameters=params)

        self._submatrices_callback = cb
        self._efuncs[cb.name] = cb

    def _submat_callback_body(self):
        objs = self.objs
        sobjs = self.solver_objs

        n_submats = DummyExpr(
            objs['nsubmats'], Mul(objs['nfields'], objs['nfields'])
        )

        malloc_submats = petsc_call(
            'PetscCalloc1', [objs['nsubmats'], objs['Submats']._C_symbol]
        )

        mat_get_dm = petsc_call('MatGetDM', [objs['J'], Byref(sobjs['callbackdm'])])

        dm_get_app = petsc_call(
            'DMGetApplicationContext', [sobjs['callbackdm'], Byref(objs['dummyctx'])]
        )

        get_ctx = petsc_call('MatShellGetContext', [objs['J'], Byref(objs['ljacctx'])])

        dm_get_info = petsc_call(
            'DMDAGetInfo', [
                sobjs['callbackdm'], Null, Byref(sobjs['M']), Byref(sobjs['N']),
                Null, Null, Null, Null, Byref(objs['dof']), Null, Null, Null, Null, Null
            ]
        )
        subblock_rows = DummyExpr(objs['subblockrows'], Mul(sobjs['M'], sobjs['N']))
        subblock_cols = DummyExpr(objs['subblockcols'], Mul(sobjs['M'], sobjs['N']))

        ptr = DummyExpr(
            objs['submat_arr']._C_symbol, Deref(objs['Submats']._C_symbol), init=True
        )

        mat_create = petsc_call('MatCreate', [sobjs['comm'], Byref(objs['block'])])

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

        nonzero_submats = self.jacobian.nonzero_submatrices
        matvec_lookup = {mv.name.split('_')[0]: mv for mv in self.J_efuncs}

        matmult_op = [
            petsc_call(
                'MatShellSetOperation',
                [
                    objs['submat_arr'].indexed[sb.linear_idx],
                    'MATOP_MULT',
                    MatShellSetOp(matvec_lookup[sb.name].name, void, void),
                ],
            )
            for sb in nonzero_submats if sb.name in matvec_lookup
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
        return self._make_callable_body(tuple(body), stacks=(get_ctx, deref_subdm))

    def residual_bundle(self, body, bundles):
        """
        Replaces PetscArrays in `body` with PetscBundle struct field accesses
        (e.g., f_v[ix][iy] -> f_bundle[ix][iy].v).
        Example:
            f_v[ix][iy] = x_v[ix][iy];
            f_u[ix][iy] = x_u[ix][iy];
        becomes:
            f_bundle[ix][iy].v = x_bundle[ix][iy].v;
            f_bundle[ix][iy].u = x_bundle[ix][iy].u;
        NOTE: This is used because the data is interleaved for
        multi-component DMDAs in PETSc.
        """
        mapper = bundles['bundle_mapper']
        indexeds = FindSymbols('indexeds').visit(body)
        subs = {}

        for i in indexeds:
            if i.base in mapper:
                bundle = mapper[i.base]
                index = bundles['target_indices'][i.function.target]
                index = (index,) + i.indices
                subs[i] = bundle.__getitem__(index)

        body = Uxreplace(subs).visit(body)
        return body


def populate_matrix_context(efuncs):
    if not objs['dummyefunc'] in efuncs.values():
        return

    subdms_expr = DummyExpr(
        FieldFromPointer(objs['Subdms']._C_symbol, objs['ljacctx']),
        objs['Subdms']._C_symbol
    )
    fields_expr = DummyExpr(
        FieldFromPointer(objs['Fields']._C_symbol, objs['ljacctx']),
        objs['Fields']._C_symbol
    )
    body = CallableBody(
        List(body=[subdms_expr, fields_expr]),
        init=(petsc_func_begin_user,),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    name = 'PopulateMatContext'
    efuncs[name] = Callable(
        name, body, objs['err'],
        parameters=[objs['ljacctx'], objs['Subdms'], objs['Fields']]
    )


def dereference_funcs(struct, fields):
    """
    Dereference AbstractFunctions from a struct.
    """
    return tuple(
        [Dereference(i, struct) for i in
         fields if isinstance(i.function, AbstractFunction)]
    )


def zero_vector(vec):
    """
    Set all entries of a PETSc vector to zero.
    """
    return petsc_call('VecSet', [vec, 0.0])
