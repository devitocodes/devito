import cgen as c
import numpy as np
from functools import cached_property

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (
    Transformer, MapNodes, Iteration, CallableBody, List, Call, FindNodes, Section,
    FindSymbols, DummyExpr, Uxreplace, Dereference
)
from devito.symbolics import Byref, Macro, Null, FieldFromPointer
from devito.types.basic import DataSymbol
import devito.logger as dl

from devito.petsc.types import (
    MultipleFieldData, Initialize, Finalize, ArgvSymbol, MainUserStruct,
    CallbackUserStruct
)
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.iet.nodes import PetscMetaData
from devito.petsc.utils import core_metadata, petsc_languages
from devito.petsc.iet.callback_builder import (
    BaseCallback, CoupledCallback, populate_matrix_context
)
from devito.petsc.iet.object_builder import BaseObjectBuilder, CoupledObjectBuilder, objs
from devito.petsc.iet.setup import BaseSetup, CoupledSetup, make_core_petsc_calls
from devito.petsc.iet.solver import Solver, CoupledSolver
from devito.petsc.iet.time_dependence import TimeDependent, NonTimeDependent
from devito.petsc.iet.logging import PetscLogger
from devito.petsc.iet.utils import petsc_call, get_user_struct_fields


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if PETScSolve was used
    inject_solve_mapper = MapNodes(Iteration, PetscMetaData,
                                   'groupby').visit(iet)

    if not inject_solve_mapper:
        return iet, {}

    if kwargs['language'] not in petsc_languages:
        raise ValueError(
            f"Expected 'language' to be one of "
            f"{petsc_languages}, but got '{kwargs['language']}'"
        )

    data = FindNodes(PetscMetaData).visit(iet)

    if any(filter(lambda i: isinstance(i.expr.rhs, Initialize), data)):
        return initialize(iet), core_metadata()

    if any(filter(lambda i: isinstance(i.expr.rhs, Finalize), data)):
        return finalize(iet), core_metadata()

    unique_grids = {i.expr.rhs.grid for (i,) in inject_solve_mapper.values()}
    # Assumption is that all solves are on the same grid
    if len(unique_grids) > 1:
        raise ValueError("All PETScSolves must use the same Grid, but multiple found.")
    grid = unique_grids.pop()
    devito_mpi = kwargs['options'].get('mpi', False)
    comm = grid.distributor._obj_comm if devito_mpi else 'PETSC_COMM_WORLD'

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, comm)

    setup = []
    subs = {}
    efuncs = {}

    # Map PETScSolve to its Section (for logging)
    section_mapper = MapNodes(Section, PetscMetaData, 'groupby').visit(iet)

    # Prefixes within the same `Operator` should not be duplicated
    prefixes = [d.expr.rhs.user_prefix for d in data if d.expr.rhs.user_prefix]
    duplicates = {p for p in prefixes if prefixes.count(p) > 1}

    if duplicates:
        dup_list = ", ".join(repr(p) for p in sorted(duplicates))
        raise ValueError(
            f"The following `options_prefix` values are duplicated "
            f"among your PETScSolves. Ensure each one is unique: {dup_list}"
        )

    # List of `Call`s to clear options from the global PETSc options database,
    # executed at the end of the Operator.
    clear_options = []

    for iters, (inject_solve,) in inject_solve_mapper.items():

        builder = Builder(inject_solve, iters, comm, section_mapper, **kwargs)

        setup.extend(builder.solver_setup.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update({builder.solve.spatial_body: builder.calls})

        efuncs.update(builder.cbbuilder.efuncs)

        clear_options.extend((petsc_call(
            builder.cbbuilder._clear_options_efunc.name, []
        ),))

    populate_matrix_context(efuncs)
    iet = Transformer(subs).visit(iet)
    body = core + tuple(setup) + iet.body.body + tuple(clear_options)
    body = iet.body._rebuild(body=body)
    iet = iet._rebuild(body=body)
    metadata = {**core_metadata(), 'efuncs': tuple(efuncs.values())}
    return iet, metadata


@iet_pass
def rebuild_petsc_struct(iet, mapper, **kwargs):
    """
    Rebuild the `CallbackUserStruct` (the child) and it's parent to include any
    new fields introduced by the `place_definitions` and `place_casts` passes.

    - The parent struct is the one registered in the main kernel via
    `DMSetApplicationContext`.
    - The child struct (`CallbackUserStruct`) is used to access the parent
    through `DMGetApplicationContext`.
    """
    # Get the old `CallbackUserStruct`
    old_child_struct = set([
        i for i in FindSymbols().visit(iet) if isinstance(i, CallbackUserStruct)
    ])

    if not old_child_struct:
        return iet, {}

    # There is only a single `CallbackUserStruct` in each iet
    assert len(old_child_struct) == 1

    old_child_struct = old_child_struct.pop()
    old_parent_struct = old_child_struct.parent

    # Collect any new fields that have been introduced since the struct was
    # previously built
    new_fields = [
        f for f in get_user_struct_fields(iet) if f not in old_child_struct.fields
    ]
    all_fields = old_child_struct.fields + new_fields

    # Rebuild the child struct
    new_child_struct = old_child_struct._rebuild(fields=all_fields)
    mapper[old_child_struct] = new_child_struct

    # Rebuild the parent struct
    new_parent_struct = old_parent_struct._rebuild(fields=all_fields)
    mapper[old_parent_struct] = new_parent_struct

    # Uxreplace old structs with new ones
    new_body = Uxreplace(mapper).visit(iet.body)

    # Dereference the new fields and insert them as `standalones` at the top of
    # the body. This ensures they are defined before any casts/allocs etc introduced
    # by the `place_definitions` and `place_casts` passes.
    derefs = tuple([Dereference(i, new_child_struct) for i in new_fields])
    new_body = new_body._rebuild(standalones=new_body.standalones + derefs)

    return iet._rebuild(body=new_body), {}


@iet_pass
def update_user_context_callback(iet, mapper, **kwargs):
    """
    """
    if not iet.name.startswith("PopulateUserContext"):
        return iet, {}

    # Update the body of the `PopulateUserContext` callback to initialize any
    # new fields in the struct `ctx`. For example, if the symbol `x_size` was
    # added, the body must now include an assignment like `ctx->x_size = x_size;`.
    old_user_ctx = [i for i in iet.parameters if isinstance(i, MainUserStruct)].pop()
    new_user_ctx = mapper[old_user_ctx]
    new_body = [
        DummyExpr(FieldFromPointer(i._C_symbol, new_user_ctx), i._C_symbol)
        for i in new_user_ctx.callback_fields
    ]
    new_body = iet.body._rebuild(body=new_body)
    iet = iet._rebuild(body=new_body, parameters=(new_user_ctx,))

    return iet, {}


def initialize(iet):
    # should be int because the correct type for argc is a C int
    # and not a int32
    argc = DataSymbol(name='argc', dtype=np.int32)
    argv = ArgvSymbol(name='argv')
    Help = Macro('help')

    help_string = c.Line(r'static char help[] = "This is help text.\n";')

    init_body = petsc_call('PetscInitialize', [Byref(argc), Byref(argv), Null, Help])
    init_body = CallableBody(
        body=(petsc_func_begin_user, help_string, init_body),
        retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
    )
    return iet._rebuild(body=init_body)


def finalize(iet):
    finalize_body = petsc_call('PetscFinalize', [])
    finalize_body = CallableBody(
        body=(petsc_func_begin_user, finalize_body),
        retstmt=(Call('PetscFunctionReturn', arguments=[0]),)
    )
    return iet._rebuild(body=finalize_body)


class Builder:
    """
    This class is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.
    The class will be extended to accommodate different solver types by
    returning subclasses of the objects initialised in __init__,
    depending on the properties of `inject_solve`.
    """
    def __init__(self, inject_solve, iters, comm, section_mapper, **kwargs):
        self.inject_solve = inject_solve
        self.objs = objs
        self.iters = iters
        self.comm = comm
        self.section_mapper = section_mapper
        self.get_info = inject_solve.expr.rhs.get_info
        self.kwargs = kwargs
        self.coupled = isinstance(inject_solve.expr.rhs.field_data, MultipleFieldData)
        self.common_kwargs = {
            'inject_solve': self.inject_solve,
            'objs': self.objs,
            'iters': self.iters,
            'comm': self.comm,
            'section_mapper': self.section_mapper,
            **self.kwargs
        }
        self.common_kwargs['solver_objs'] = self.object_builder.solver_objs
        self.common_kwargs['time_dependence'] = self.time_dependence
        self.common_kwargs['cbbuilder'] = self.cbbuilder
        self.common_kwargs['logger'] = self.logger

    @cached_property
    def object_builder(self):
        return (
            CoupledObjectBuilder(**self.common_kwargs)
            if self.coupled else
            BaseObjectBuilder(**self.common_kwargs)
        )

    @cached_property
    def time_dependence(self):
        mapper = self.inject_solve.expr.rhs.time_mapper
        time_class = TimeDependent if mapper else NonTimeDependent
        return time_class(**self.common_kwargs)

    @cached_property
    def cbbuilder(self):
        return CoupledCallback(**self.common_kwargs) \
            if self.coupled else BaseCallback(**self.common_kwargs)

    @cached_property
    def solver_setup(self):
        return CoupledSetup(**self.common_kwargs) \
            if self.coupled else BaseSetup(**self.common_kwargs)

    @cached_property
    def solve(self):
        return CoupledSolver(**self.common_kwargs) \
            if self.coupled else Solver(**self.common_kwargs)

    @cached_property
    def logger(self):
        log_level = dl.logger.level
        return PetscLogger(
            log_level, self.get_info, **self.common_kwargs
        )

    @cached_property
    def calls(self):
        return List(body=self.solve.calls+self.logger.calls)
