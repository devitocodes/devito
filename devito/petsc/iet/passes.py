import cgen as c
import numpy as np

from devito.passes.iet.engine import iet_pass
from devito.ir.iet import (Transformer, MapNodes, Iteration, BlankLine,
                           FindNodes, Call, CallableBody)
from devito.symbolics import Byref, Macro
from devito.types.basic import DataSymbol
from devito.petsc.types import (PetscMPIInt, PetscErrorCode, Initialize,
                                Finalize, ArgvSymbol)
from devito.petsc.types.macros import petsc_func_begin_user
from devito.petsc.iet.nodes import PetscMetaData
from devito.petsc.utils import core_metadata
from devito.petsc.iet.routines import (CallbackBuilder, BaseObjectBuilder, BaseSetup,
                                       Solver, TimeDependent, NonTimeDependent)
from devito.petsc.iet.utils import petsc_call, petsc_call_mpi


@iet_pass
def lower_petsc(iet, **kwargs):
    # Check if PETScSolve was used
    injectsolve_mapper = MapNodes(Iteration, PetscMetaData,
                                  'groupby').visit(iet)

    if not injectsolve_mapper:
        return iet, {}

    metadata = core_metadata()

    data = FindNodes(PetscMetaData).visit(iet)

    if any(filter(lambda i: isinstance(i.expr.rhs, Initialize), data)):
        return initialize(iet), metadata

    if any(filter(lambda i: isinstance(i.expr.rhs, Finalize), data)):
        return finalize(iet), metadata

    targets = [i.expr.rhs.target for (i,) in injectsolve_mapper.values()]

    # Assumption is that all targets have the same grid so can use any target here
    objs = build_core_objects(targets[-1], **kwargs)

    # Create core PETSc calls (not specific to each PETScSolve)
    core = make_core_petsc_calls(objs, **kwargs)

    setup = []
    subs = {}
    efuncs = {}

    for iters, (injectsolve,) in injectsolve_mapper.items():

        builder = Builder(injectsolve, objs, iters, **kwargs)

        setup.extend(builder.solversetup.calls)

        # Transform the spatial iteration loop with the calls to execute the solver
        subs.update(builder.solve.mapper)

        efuncs.update(builder.cbbuilder.efuncs)

    iet = Transformer(subs).visit(iet)

    body = core + tuple(setup) + (BlankLine,) + iet.body.body
    body = iet.body._rebuild(body=body)
    iet = iet._rebuild(body=body)
    metadata.update({'efuncs': tuple(efuncs.values())})

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


def make_core_petsc_calls(objs, **kwargs):
    call_mpi = petsc_call_mpi('MPI_Comm_size', [objs['comm'], Byref(objs['size'])])

    return call_mpi, BlankLine


def build_core_objects(target, **kwargs):
    communicator = 'PETSC_COMM_WORLD'

    return {
        'size': PetscMPIInt(name='size'),
        'comm': communicator,
        'err': PetscErrorCode(name='err'),
        'grid': target.grid
    }


class Builder:
    """
    This class is designed to support future extensions, enabling
    different combinations of solver types, preconditioning methods,
    and other functionalities as needed.

    The class will be extended to accommodate different solver types by
    returning subclasses of the objects initialised in __init__,
    depending on the properties of `injectsolve`.
    """
    def __init__(self, injectsolve, objs, iters, **kwargs):

        # Determine the time dependency class
        time_mapper = injectsolve.expr.rhs.time_mapper
        timedep = TimeDependent if time_mapper else NonTimeDependent
        self.timedep = timedep(injectsolve, iters, **kwargs)

        # Objects
        self.objbuilder = BaseObjectBuilder(injectsolve, **kwargs)
        self.solver_objs = self.objbuilder.solver_objs

        # Callbacks
        self.cbbuilder = CallbackBuilder(
            injectsolve, objs, self.solver_objs, timedep=self.timedep,
            **kwargs
        )

        # Solver setup
        self.solversetup = BaseSetup(
            self.solver_objs, objs, injectsolve, self.cbbuilder
        )

        # Execute the solver
        self.solve = Solver(
            self.solver_objs, objs, injectsolve, iters,
            self.cbbuilder, timedep=self.timedep
        )


# Move these to types folder
Null = Macro('NULL')
void = 'void'
