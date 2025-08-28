import cgen as c
import numpy as np
from functools import cached_property

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (
    Transformer, MapNodes, Iteration, CallableBody, List, Call, FindNodes, Section
)
from devito.symbolics import Byref, Macro, Null
from devito.types.basic import DataSymbol
import devito.logger as dl

from devito.petsc.types import MultipleFieldData, Initialize, Finalize, ArgvSymbol
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.iet.nodes import PetscMetaData
from devito.petsc.utils import core_metadata, petsc_languages
from devito.petsc.iet.callback_builder import BaseCallback, CoupledCallback, populate_matrix_context
from devito.petsc.iet.object_builder import BaseObjectBuilder, CoupledObjectBuilder, objs
from devito.petsc.iet.setup import BaseSetup, CoupledSetup, make_core_petsc_calls
from devito.petsc.iet.solver import Solver, CoupledSolver
from devito.petsc.iet.time_dependence import TimeDependent, NonTimeDependent
from devito.petsc.iet.logging import PetscLogger
from devito.petsc.iet.utils import petsc_call


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
