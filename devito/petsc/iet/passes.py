import cgen as c
import numpy as np
from functools import cached_property

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, BlankLine,
                           DummyExpr, CallableBody, List, Call, Callable,
                           FindNodes, Section)
from devito.symbolics import Byref, Macro, FieldFromPointer
from devito.types import Symbol, Scalar
from devito.types.basic import DataSymbol
from devito.tools import frozendict
from devito.petsc.types import (PetscMPIInt, PetscErrorCode, MultipleFieldData,
                                PointerIS, Mat, CallbackVec, Vec, CallbackMat, SNES,
                                DummyArg, PetscInt, PointerDM, PointerMat, MatReuse,
                                CallbackPointerIS, CallbackPointerDM, JacobianStruct,
                                SubMatrixStruct, Initialize, Finalize, ArgvSymbol)
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.iet.nodes import PetscMetaData
from devito.petsc.utils import core_metadata, petsc_languages
from devito.petsc.iet.routines import (CBBuilder, CCBBuilder, BaseObjectBuilder,
                                       CoupledObjectBuilder, BaseSetup, CoupledSetup,
                                       Solver, CoupledSolver, TimeDependent,
                                       NonTimeDependent)
from devito.petsc.iet.logging import PetscLogger
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi

import devito.logger as dl


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

    for iters, (inject_solve,) in inject_solve_mapper.items():

        builder = Builder(inject_solve, objs, iters, comm, section_mapper, **kwargs)

        setup.extend(builder.solversetup.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update({builder.solve.spatial_body: builder.calls})

        efuncs.update(builder.cbbuilder.efuncs)

    populate_matrix_context(efuncs, objs)

    iet = Transformer(subs).visit(iet)

    body = core + tuple(setup) + iet.body.body
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


def make_core_petsc_calls(objs, comm):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [comm, Byref(objs['size'])])
    return call_mpi, BlankLine


class Builder:
    """
    This class is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.
    The class will be extended to accommodate different solver types by
    returning subclasses of the objects initialised in __init__,
    depending on the properties of `inject_solve`.
    """
    def __init__(self, inject_solve, objs, iters, comm, section_mapper, **kwargs):
        self.inject_solve = inject_solve
        self.objs = objs
        self.iters = iters
        self.comm = comm
        self.section_mapper = section_mapper
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
        self.common_kwargs['solver_objs'] = self.objbuilder.solver_objs
        self.common_kwargs['time_dependence'] = self.time_dependence
        self.common_kwargs['cbbuilder'] = self.cbbuilder
        self.common_kwargs['logger'] = self.logger

    @cached_property
    def objbuilder(self):
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
        return CCBBuilder(**self.common_kwargs) \
            if self.coupled else CBBuilder(**self.common_kwargs)

    @cached_property
    def solversetup(self):
        return CoupledSetup(**self.common_kwargs) \
            if self.coupled else BaseSetup(**self.common_kwargs)

    @cached_property
    def solve(self):
        return CoupledSolver(**self.common_kwargs) \
            if self.coupled else Solver(**self.common_kwargs)

    @cached_property
    def logger(self):
        log_level = dl.logger.level
        return PetscLogger(log_level, **self.common_kwargs)

    @cached_property
    def calls(self):
        return List(body=self.solve.calls+self.logger.calls)


def populate_matrix_context(efuncs, objs):
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
        init=(objs['begin_user'],),
        retstmt=tuple([Call('PetscFunctionReturn', arguments=[0])])
    )
    name = 'PopulateMatContext'
    efuncs[name] = Callable(
        name, body, objs['err'],
        parameters=[objs['ljacctx'], objs['Subdms'], objs['Fields']]
    )


subdms = PointerDM(name='subdms')
fields = PointerIS(name='fields')
submats = PointerMat(name='submats')
rows = PointerIS(name='rows')
cols = PointerIS(name='cols')


# A static dict containing shared symbols and objects that are not
# unique to each PETScSolve.
# Many of these objects are used as arguments in callback functions to make
# the C code cleaner and more modular. This is also a step toward leveraging
# Devito's `reuse_efuncs` functionality, allowing reuse of efuncs when
# they are semantically identical.
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
    'nsubmats': Scalar('nsubmats', dtype=np.int32),
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
    'Null': Macro('NULL'),
    'dummyctx': Symbol('lctx'),
    'dummyptr': DummyArg('dummy'),
    'dummyefunc': Symbol('dummyefunc'),
    'dof': PetscInt('dof'),
    'begin_user': c.Line('PetscFunctionBeginUser;'),
})

# Move to macros file?
Null = Macro('NULL')
