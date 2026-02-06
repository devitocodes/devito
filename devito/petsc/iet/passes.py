import cgen as c
import numpy as np
from functools import cached_property

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (
    Transformer, MapNodes, Iteration, CallableBody, List, Call, FindNodes, Section,
    FindSymbols, DummyExpr, Uxreplace, Dereference
)
from devito.symbolics import Byref, Macro, Null, FieldFromPointer
from devito.types.basic import DataSymbol, LocalType
from devito.types.misc import FIndexed
import devito.logger
from devito.passes.iet.linearization import linearize_accesses, Tracker

from devito.petsc.types import (
    MultipleFieldData, Initialize, Finalize, ArgvSymbol, MainUserStruct,
    CallbackUserStruct
)
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.iet.nodes import PetscMetaData, petsc_call
from devito.petsc.config import core_metadata, petsc_languages
from devito.petsc.iet.callbacks import (
    BaseCallbackBuilder, CoupledCallbackBuilder, populate_matrix_context,
    get_user_struct_fields
)
from devito.petsc.iet.type_builder import (
    BaseTypeBuilder, CoupledTypeBuilder, ConstrainedBCTypeBuilder, objs
)
from devito.petsc.iet.builder import (
    BuilderBase, CoupledBuilder, ConstrainedBCBuilder, make_core_petsc_calls
)
from devito.petsc.iet.solve import Solve, CoupledSolve
from devito.petsc.iet.time_dependence import TimeDependent, TimeIndependent
from devito.petsc.iet.logging import PetscLogger


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if `petscsolve` was used
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
    # Assumption is that all solves are on the same `Grid`
    if len(unique_grids) > 1:
        raise ValueError(
            "All `petscsolve` calls must use the same `Grid`, "
            "but multiple `Grid`s were found."
        )
    grid = unique_grids.pop()
    devito_mpi = kwargs['options'].get('mpi', False)
    comm = grid.distributor._obj_comm if devito_mpi else 'PETSC_COMM_WORLD'

    # Create core PETSc calls (not specific to each `petscsolve`)
    core = make_core_petsc_calls(objs, comm)

    setup = []
    subs = {}
    efuncs = {}

    # Map each `PetscMetaData` to its Section (for logging)
    section_mapper = MapNodes(Section, PetscMetaData, 'groupby').visit(iet)

    # Prefixes within the same `Operator` should not be duplicated
    prefixes = [d.expr.rhs.user_prefix for d in data if d.expr.rhs.user_prefix]
    duplicates = {p for p in prefixes if prefixes.count(p) > 1}

    if duplicates:
        dup_list = ", ".join(repr(p) for p in sorted(duplicates))
        raise ValueError(
            "The following `options_prefix` values are duplicated "
            f"among your `petscsolve` calls. Ensure each one is unique: {dup_list}"
        )

    # List of `Call`s to clear options from the global PETSc options database,
    # executed at the end of the Operator.
    clear_options = []

    for iters, (inject_solve,) in inject_solve_mapper.items():

        solver = BuildSolver(inject_solve, iters, comm, section_mapper, **kwargs)

        setup.extend(solver.builder.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update({solver.solve.spatial_body: solver.calls})

        efuncs.update(solver.callback_builder.efuncs)

        clear_options.extend((petsc_call(
            solver.callback_builder._clear_options_efunc.name, []
        ),))

    populate_matrix_context(efuncs)
    iet = Transformer(subs).visit(iet)
    body = core + tuple(setup) + iet.body.body + tuple(clear_options)
    body = iet.body._rebuild(body=body)
    iet = iet._rebuild(body=body)
    metadata = {**core_metadata(), 'efuncs': tuple(efuncs.values())}
    return iet, metadata


def lower_petsc_symbols(iet, **kwargs):
    """
    The `place_definitions` and `place_casts` passes may introduce new
    symbols, which must be incorporated into
    the relevant PETSc structs. To update the structs, this method then
    applies two additional passes: `rebuild_child_user_struct` and
    `rebuild_parent_user_struct`.
    """
    callback_struct_mapper = {}
    # Rebuild `CallbackUserStruct` and update iet accordingly
    rebuild_child_user_struct(iet, mapper=callback_struct_mapper)
    # Rebuild `MainUserStruct` and update iet accordingly
    rebuild_parent_user_struct(iet, mapper=callback_struct_mapper)

    iet = linear_indices(iet, **kwargs)


@iet_pass
def linear_indices(iet, **kwargs):
    """
    """
    if not iet.name.startswith("SetPointBCs"):
        return iet, {}

    if kwargs['options']['index-mode'] == 'int32':
        dtype = np.int32
    else:
        dtype = np.int64

    tracker = Tracker('basic', dtype, kwargs['sregistry'])

    indexeds = [
        i for i in FindSymbols('indexeds').visit(iet)
        if not isinstance(i.function, LocalType)
    ]
    candidates = {i.function.name for i in indexeds}
    key = lambda f: f.name in candidates

    iet = linearize_accesses(iet, key0=key, tracker=tracker)

    indexeds = [
        i for i in FindSymbols('indexeds').visit(iet)
        if i.function.name in candidates
    ]
    mapper_findexeds = {i: linear_index(i) for i in indexeds}

    return Uxreplace(mapper_findexeds).visit(iet), {}


def linear_index(i):
    if isinstance(i, FIndexed):
        return i.linear_index
    # 1D case
    assert len(i.indices) == 1
    return i.indices[0]


@iet_pass
def rebuild_child_user_struct(iet, mapper, **kwargs):
    """
    Rebuild each `CallbackUserStruct` (the child struct) to include any
    new fields introduced by the `place_definitions` and `place_casts` passes.
    Also, update the iet accordingly (e.g., dereference the new fields).
    - `CallbackUserStruct` is used to access information
    in PETSc callback functions via `DMGetApplicationContext`.
    """
    old_struct = set([
        i for i in FindSymbols().visit(iet) if isinstance(i, CallbackUserStruct)
    ])

    if not old_struct:
        return iet, {}

    # There is a unique `CallbackUserStruct` in each callback
    assert len(old_struct) == 1
    old_struct = old_struct.pop()

    # Collect any new fields that have been introduced since the struct was
    # previously built
    new_fields = [
        f for f in get_user_struct_fields(iet) if f not in old_struct.fields
    ]
    all_fields = old_struct.fields + new_fields

    # Rebuild the struct
    new_struct = old_struct._rebuild(fields=all_fields)
    mapper[old_struct] = new_struct

    # Replace old struct with the new one
    new_body = Uxreplace(mapper).visit(iet.body)

    # Dereference the new fields and insert them as `standalones` at the top of
    # the body. This ensures they are defined before any casts/allocs etc introduced
    # by the `place_definitions` and `place_casts` passes.
    derefs = tuple([Dereference(i, new_struct) for i in new_fields])
    new_body = new_body._rebuild(standalones=new_body.standalones + derefs)

    return iet._rebuild(body=new_body), {}


@iet_pass
def rebuild_parent_user_struct(iet, mapper, **kwargs):
    """
    Rebuild each `MainUserStruct` (the parent struct) so that it stays in sync
    with its corresponding `CallbackUserStruct` (the child struct). Any IET that
    references a parent struct is also updated — either the `PopulateUserContext`
    callback or the main Kernel, where the parent struct is registered
    via `DMSetApplicationContext`.
    """
    if not mapper:
        return iet, {}

    parent_struct_mapper = {
        v.parent: v.parent._rebuild(fields=v.fields) for v in mapper.values()
    }

    if not iet.name.startswith("PopulateUserContext"):
        new_body = Uxreplace(parent_struct_mapper).visit(iet.body)
        return iet._rebuild(body=new_body), {}

    old_struct = [i for i in iet.parameters if isinstance(i, MainUserStruct)]
    assert len(old_struct) == 1
    old_struct = old_struct.pop()

    new_struct = parent_struct_mapper[old_struct]

    new_body = [
        DummyExpr(FieldFromPointer(i._C_symbol, new_struct), i._C_symbol)
        for i in new_struct.callback_fields
    ]
    new_body = iet.body._rebuild(body=new_body)
    return iet._rebuild(body=new_body, parameters=(new_struct,)), {}


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


class BuildSolver:
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
        self.constrain_bc = inject_solve.expr.rhs.field_data.constrain_bc
        self.common_kwargs = {
            'inject_solve': self.inject_solve,
            'objs': self.objs,
            'iters': self.iters,
            'comm': self.comm,
            'section_mapper': self.section_mapper,
            **self.kwargs
        }
        self.common_kwargs['solver_objs'] = self.type_builder.solver_objs
        self.common_kwargs['time_dependence'] = self.time_dependence
        self.common_kwargs['callback_builder'] = self.callback_builder
        self.common_kwargs['logger'] = self.logger

    @cached_property
    def type_builder(self):
        if self.coupled and self.constrain_bc:
            return NotImplementedError
        elif self.coupled:
            return CoupledTypeBuilder(**self.common_kwargs)
        elif self.constrain_bc:
            return ConstrainedBCTypeBuilder(**self.common_kwargs)
        else:
            return BaseTypeBuilder(**self.common_kwargs)

    @cached_property
    def time_dependence(self):
        mapper = self.inject_solve.expr.rhs.time_mapper
        time_class = TimeDependent if mapper else TimeIndependent
        return time_class(**self.common_kwargs)

    @cached_property
    def callback_builder(self):
        return CoupledCallbackBuilder(**self.common_kwargs) \
            if self.coupled else BaseCallbackBuilder(**self.common_kwargs)

    @cached_property
    def builder(self):
        if self.coupled and self.constrain_bc:
            # TODO: implement CoupledConstrainedBCBuilder
            return NotImplementedError
        elif self.coupled:
            return CoupledBuilder(**self.common_kwargs)
        elif self.constrain_bc:
            return ConstrainedBCBuilder(**self.common_kwargs)
        else:
            return BuilderBase(**self.common_kwargs)

    @cached_property
    def solve(self):
        return CoupledSolve(**self.common_kwargs) \
            if self.coupled else Solve(**self.common_kwargs)

    @cached_property
    def logger(self):
        log_level = devito.logger.logger.level
        return PetscLogger(
            log_level, get_info=self.get_info, **self.common_kwargs
        )

    @cached_property
    def calls(self):
        return List(body=self.solve.calls+self.logger.calls)
