from functools import cached_property

from devito.ir.iet import (
    BlankLine, FindNodes, retrieve_iteration_tree, filter_iterations
)
from devito.symbolics import (Byref, Null)

from devito.petsc.iet.nodes import PetscMetaData
from devito.petsc.iet.utils import petsc_call, insert_vals, sreverse, sforward


class Solver:
    def __init__(self, **kwargs):
        self.inject_solve = kwargs.get('inject_solve')
        self.objs = kwargs.get('objs')
        self.solver_objs = kwargs.get('solver_objs')
        self.iters = kwargs.get('iters')
        self.cbbuilder = kwargs.get('cbbuilder')
        self.time_dependence = kwargs.get('time_dependence')
        self.calls = self._execute_solve()

    def _execute_solve(self):
        """
        Assigns the required time iterators to the struct and executes
        the necessary calls to execute the SNES solver.
        """
        sobjs = self.solver_objs
        target = self.inject_solve.expr.rhs.field_data.target

        struct_assignment = self.time_dependence.assign_time_iters(sobjs['userctx'])

        b_efunc = self.cbbuilder._b_efunc

        dmda = sobjs['dmda']

        rhs_call = petsc_call(b_efunc.name, [sobjs['dmda'], sobjs['bglobal']])

        vec_place_array = self.time_dependence.place_array(target)

        if self.cbbuilder.initial_guesses:
            initguess = self.cbbuilder.initial_guesses[0]
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

        vec_reset_array = self.time_dependence.reset_array(target)

        run_solver_calls = (struct_assignment,) + (
            rhs_call,
        ) + vec_place_array + (
            initguess_call,
            dm_local_to_global_x,
            snes_solve,
            dm_global_to_local_x,
            vec_reset_array,
            BlankLine,
        )
        return run_solver_calls

    @cached_property
    def spatial_body(self):
        spatial_body = []
        # TODO: remove the iters[0]
        for tree in retrieve_iteration_tree(self.iters[0]):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)
            if root:
                root = root[0]
                if self.inject_solve in FindNodes(PetscMetaData).visit(root):
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
        xglob = sobjs['xglobal']

        struct_assignment = self.time_dependence.assign_time_iters(sobjs['userctx'])
        targets = self.inject_solve.expr.rhs.field_data.targets

        # TODO: optimise the ccode generated here
        pre_solve = ()
        post_solve = ()

        for i, t in enumerate(targets):
            name = t.name
            dm = sobjs[f'da{name}']
            target_xloc = sobjs[f'xlocal{name}']
            target_xglob = sobjs[f'xglobal{name}']
            field = sobjs['fields'].indexed[i]
            s = sobjs[f'scatter{name}']

            pre_solve += (
                # TODO: Need to call reset array
                self.time_dependence.place_array(t),
                petsc_call(
                    'DMLocalToGlobal',
                    [dm, target_xloc, insert_vals, target_xglob]
                ),
                petsc_call(
                    'VecScatterCreate',
                    [xglob, field, target_xglob, Null, Byref(s)]
                ),
                petsc_call(
                    'VecScatterBegin',
                    [s, target_xglob, xglob, insert_vals, sreverse]
                ),
                petsc_call(
                    'VecScatterEnd',
                    [s, target_xglob, xglob, insert_vals, sreverse]
                ),
                BlankLine,
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

        snes_solve = (petsc_call('SNESSolve', [sobjs['snes'], Null, xglob]),)

        return (struct_assignment,) + pre_solve + snes_solve + post_solve + (BlankLine,)
