import pytest

import numpy as np
import os
import re
import sympy as sp

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, Operator,
                    configuration, norm, switchconfig, SubDomain, sin)
from devito.operator.profiling import PerformanceSummary
from devito.ir.iet import (Call, ElementalFunction,
                           FindNodes, retrieve_iteration_tree)
from devito.types import Constant, LocalCompositeObject
from devito.passes.iet.languages.C import CDataManager
from devito.petsc.types import (DM, Mat, Vec, PetscMPIInt, KSP,
                                PC, KSPConvergedReason, PETScArray,
                                FieldData, MultipleFieldData,
                                SubMatrixBlock)
from devito.petsc.solve import petscsolve, EssentialBC
from devito.petsc.iet.nodes import Expression
from devito.petsc.initialize import PetscInitialize
from devito.petsc.logging import PetscSummary
from devito.petsc.solver_parameters import linear_solve_defaults


@pytest.fixture(scope='session')
def command_line():

    # Random prefixes to validate command line argument parsing
    prefix = (
        'd17weqroeg', 'riabfodkj5', 'fir8o3lsak',
        'zwejklqn25', 'qtr2vfvwiu')

    petsc_option = (
        ('ksp_rtol',),
        ('ksp_rtol', 'ksp_atol'),
        ('ksp_rtol', 'ksp_atol', 'ksp_divtol', 'ksp_max_it'),
        ('ksp_type',),
        ('ksp_divtol', 'ksp_type')
    )
    value = (
        (1e-8,),
        (1e-11, 1e-15),
        (1e-3, 1e-10, 50000, 2000),
        ('cg',),
        (22000, 'richardson'),
    )
    argv = []
    expected = {}
    for p, opt, val in zip(prefix, petsc_option, value, strict=True):
        for o, v in zip(opt, val, strict=True):
            argv.extend([f'-{p}_{o}', str(v)])
        expected[p] = zip(opt, val)
    return argv, expected


@pytest.fixture(scope='session', autouse=True)
def petsc_initialization(command_line):
    argv, _ = command_line
    # TODO: Temporary workaround until PETSc is automatically
    # initialized
    configuration['compiler'] = 'custom'
    os.environ['CC'] = 'mpicc'
    PetscInitialize(argv)


@skipif('petsc')
@pytest.mark.parallel(mode=[1, 2, 4, 6])
def test_petsc_initialization_parallel(mode):
    configuration['compiler'] = 'custom'
    os.environ['CC'] = 'mpicc'
    PetscInitialize()


@skipif('petsc')
def test_petsc_local_object():
    """
    Test C++ support for PETSc LocalObjects.
    """
    lo0 = DM('da', stencil_width=1)
    lo1 = Mat('A')
    lo2 = Vec('x')
    lo3 = PetscMPIInt('size')
    lo4 = KSP('ksp')
    lo5 = PC('pc')
    lo6 = KSPConvergedReason('reason')

    iet = Call('foo', [lo0, lo1, lo2, lo3, lo4, lo5, lo6])
    iet = ElementalFunction('foo', iet, parameters=())

    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert 'DM da;' in str(iet)
    assert 'Mat A;' in str(iet)
    assert 'Vec x;' in str(iet)
    assert 'PetscMPIInt size;' in str(iet)
    assert 'KSP ksp;' in str(iet)
    assert 'PC pc;' in str(iet)
    assert 'KSPConvergedReason reason;' in str(iet)


@skipif('petsc')
def test_petsc_subs():
    """
    Test support for PETScArrays in substitutions.
    """
    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=2)
    f2 = Function(name='f2', grid=grid, space_order=2)

    arr = PETScArray(name='arr', target=f2)

    eqn = Eq(f1, f2.laplace)
    eqn_subs = eqn.subs(f2, arr)

    assert str(eqn) == 'Eq(f1(x, y), Derivative(f2(x, y), (x, 2))' +  \
        ' + Derivative(f2(x, y), (y, 2)))'

    assert str(eqn_subs) == 'Eq(f1(x, y), Derivative(arr(x, y), (x, 2))' +  \
        ' + Derivative(arr(x, y), (y, 2)))'

    assert str(eqn_subs.rhs.evaluate) == '-2.0*arr(x, y)/h_x**2' + \
        ' + arr(x - h_x, y)/h_x**2 + arr(x + h_x, y)/h_x**2 - 2.0*arr(x, y)/h_y**2' + \
        ' + arr(x, y - h_y)/h_y**2 + arr(x, y + h_y)/h_y**2'


@skipif('petsc')
def test_petsc_solve():
    """
    Test `petscsolve`.
    """
    grid = Grid((2, 2), dtype=np.float64)

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)

    petsc = petscsolve(eqn, f)

    with switchconfig(language='petsc'):
        op = Operator(petsc, opt='noop')

    callable_roots = [meta_call.root for meta_call in op._func_table.values()]

    matvec_efunc = [root for root in callable_roots if root.name == 'MatMult0']

    b_efunc = [root for root in callable_roots if root.name == 'FormRHS0']

    action_expr = FindNodes(Expression).visit(matvec_efunc[0])
    rhs_expr = FindNodes(Expression).visit(b_efunc[0])

    # TODO: Investigate why there are double brackets here
    # TODO: The output is technically "correct" but there are redundant operations that
    # have not been cancelled out / simplified
    assert str(action_expr[-1].expr.rhs) == (
        '(x_f[x + 1, y + 2]/((ctx0->h_x*ctx0->h_x))'
        ' - 2.0*x_f[x + 2, y + 2]/(ctx0->h_x*ctx0->h_x)'
        ' + x_f[x + 3, y + 2]/((ctx0->h_x*ctx0->h_x))'
        ' + x_f[x + 2, y + 1]/((ctx0->h_y*ctx0->h_y))'
        ' - 2.0*x_f[x + 2, y + 2]/(ctx0->h_y*ctx0->h_y)'
        ' + x_f[x + 2, y + 3]/((ctx0->h_y*ctx0->h_y)))*ctx0->h_x*ctx0->h_y'
    )

    assert str(rhs_expr[-1].expr.rhs) == 'ctx0->h_x*ctx0->h_y*g[x + 2, y + 2]'

    # Check the iteration bounds are correct.
    assert op.arguments().get('x_m') == 0
    assert op.arguments().get('y_m') == 0
    assert op.arguments().get('y_M') == 1
    assert op.arguments().get('x_M') == 1

    assert len(retrieve_iteration_tree(op)) == 0

    # TODO: Remove pragmas from PETSc callback functions
    assert len(matvec_efunc[0].parameters) == 3


@skipif('petsc')
def test_multiple_petsc_solves():
    """
    Test multiple `petscsolve` calls, passed to a single `Operator`.
    """
    grid = Grid((2, 2), dtype=np.float64)

    f1 = Function(name='f1', grid=grid, space_order=2)
    g1 = Function(name='g1', grid=grid, space_order=2)

    f2 = Function(name='f2', grid=grid, space_order=2)
    g2 = Function(name='g2', grid=grid, space_order=2)

    eqn1 = Eq(f1.laplace, g1)
    eqn2 = Eq(f2.laplace, g2)

    petsc1 = petscsolve(eqn1, f1, options_prefix='pde1')
    petsc2 = petscsolve(eqn2, f2, options_prefix='pde2')

    with switchconfig(language='petsc'):
        op = Operator([petsc1, petsc2], opt='noop')

    callable_roots = [meta_call.root for meta_call in op._func_table.values()]

    # One FormRHS, MatShellMult, FormFunction, PopulateMatContext, SetPetscOptions
    # and ClearPetscOptions per solve.
    # TODO: Some efuncs are not reused where reuse is possible — investigate.
    assert len(callable_roots) == 12


@skipif('petsc')
def test_petsc_cast():
    """
    Test casting of PETScArray.
    """
    grid1 = Grid((2), dtype=np.float64)
    grid2 = Grid((2, 2), dtype=np.float64)
    grid3 = Grid((4, 5, 6), dtype=np.float64)

    f1 = Function(name='f1', grid=grid1, space_order=2)
    f2 = Function(name='f2', grid=grid2, space_order=4)
    f3 = Function(name='f3', grid=grid3, space_order=6)

    eqn1 = Eq(f1.laplace, 10)
    eqn2 = Eq(f2.laplace, 10)
    eqn3 = Eq(f3.laplace, 10)

    petsc1 = petscsolve(eqn1, f1)
    petsc2 = petscsolve(eqn2, f2)
    petsc3 = petscsolve(eqn3, f3)

    with switchconfig(language='petsc'):
        op1 = Operator(petsc1)
        op2 = Operator(petsc2)
        op3 = Operator(petsc3)

    assert 'PetscScalar * x_f1 = ' + \
        '(PetscScalar (*)) x_f1_vec;' in str(op1.ccode)
    assert 'PetscScalar (* x_f2)[info.gxm] = ' + \
        '(PetscScalar (*)[info.gxm]) x_f2_vec;' in str(op2.ccode)
    assert 'PetscScalar (* x_f3)[info.gym][info.gxm] = ' + \
        '(PetscScalar (*)[info.gym][info.gxm]) x_f3_vec;' in str(op3.ccode)


@skipif('petsc')
def test_dmda_create():

    grid1 = Grid((2), dtype=np.float64)
    grid2 = Grid((2, 2), dtype=np.float64)
    grid3 = Grid((4, 5, 6), dtype=np.float64)

    f1 = Function(name='f1', grid=grid1, space_order=2)
    f2 = Function(name='f2', grid=grid2, space_order=4)
    f3 = Function(name='f3', grid=grid3, space_order=6)

    eqn1 = Eq(f1.laplace, 10)
    eqn2 = Eq(f2.laplace, 10)
    eqn3 = Eq(f3.laplace, 10)

    petsc1 = petscsolve(eqn1, f1)
    petsc2 = petscsolve(eqn2, f2)
    petsc3 = petscsolve(eqn3, f3)

    with switchconfig(language='petsc'):
        op1 = Operator(petsc1, opt='noop')
        op2 = Operator(petsc2, opt='noop')
        op3 = Operator(petsc3, opt='noop')

    assert 'PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        '2,1,2,NULL,&da0));' in str(op1)

    assert 'PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        'DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,2,2,1,1,1,4,NULL,NULL,&da0));' \
        in str(op2)

    assert 'PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        'DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,6,5,4' + \
        ',1,1,1,1,6,NULL,NULL,NULL,&da0));' in str(op3)


class TestStruct:
    @skipif('petsc')
    def test_cinterface_petsc_struct(self):

        grid = Grid(shape=(11, 11), dtype=np.float64)
        f = Function(name='f', grid=grid, space_order=2)
        eq = Eq(f.laplace, 10)
        petsc = petscsolve(eq, f)

        name = "foo"

        with switchconfig(language='petsc'):
            op = Operator(petsc, name=name)

        # Trigger the generation of a .c and a .h files
        ccode, hcode = op.cinterface(force=True)

        dirname = op._compiler.get_jit_dir()
        assert os.path.isfile(os.path.join(dirname, "%s.c" % name))
        assert os.path.isfile(os.path.join(dirname, "%s.h" % name))

        ccode = str(ccode)
        hcode = str(hcode)

        assert 'include "%s.h"' % name in ccode

        # The public `struct UserCtx` only appears in the header file
        assert 'struct UserCtx0\n{' not in ccode
        assert 'struct UserCtx0\n{' in hcode

    @skipif('petsc')
    def test_temp_arrays_in_struct(self):

        grid = Grid(shape=(11, 11, 11), dtype=np.float64)

        u = TimeFunction(name='u', grid=grid, space_order=2)
        x, y, _ = grid.dimensions

        eqn = Eq(u.forward, sin(sp.pi*(x+y)/3.), subdomain=grid.interior)
        petsc = petscsolve(eqn, target=u.forward)

        with switchconfig(log_level='DEBUG', language='petsc'):
            op = Operator(petsc)
            # Check that it runs
            op.apply(time_M=3)

        assert 'ctx0->x_size = x_size;' in str(op.ccode)
        assert 'ctx0->y_size = y_size;' in str(op.ccode)

        assert 'const PetscInt y_size = ctx0->y_size;' in str(op.ccode)
        assert 'const PetscInt x_size = ctx0->x_size;' in str(op.ccode)

    @skipif('petsc')
    def test_parameters(self):

        grid = Grid((2, 2), dtype=np.float64)

        f1 = Function(name='f1', grid=grid, space_order=2)
        g1 = Function(name='g1', grid=grid, space_order=2)

        mu1 = Constant(name='mu1', value=2.0)
        mu2 = Constant(name='mu2', value=2.0)

        eqn1 = Eq(f1.laplace, g1*mu1)
        petsc1 = petscsolve(eqn1, f1)

        eqn2 = Eq(f1, g1*mu2)

        with switchconfig(language='petsc'):
            op = Operator([eqn2, petsc1])

        arguments = op.arguments()

        # Check mu1 and mu2 in arguments
        assert 'mu1' in arguments
        assert 'mu2' in arguments

        # Check mu1 and mu2 in op.parameters
        assert mu1 in op.parameters
        assert mu2 in op.parameters

        # Check PETSc struct not in op.parameters
        assert all(not isinstance(i, LocalCompositeObject) for i in op.parameters)

    @skipif('petsc')
    def test_field_order(self):
        """Verify that the order of fields in the user struct is fixed for
        `identical` Operator instances.
        """
        grid = Grid(shape=(11, 11, 11), dtype=np.float64)
        f = TimeFunction(name='f', grid=grid, space_order=2)
        x, y, _ = grid.dimensions
        t = grid.time_dim
        eq = Eq(f.dt, f.laplace + t*0.005 + sin(sp.pi*(x+y)/3.), subdomain=grid.interior)
        petsc = petscsolve(eq, f.forward)

        with switchconfig(language='petsc'):
            op1 = Operator(petsc, name="foo1")
            op2 = Operator(petsc, name="foo2")

        op1_user_struct = op1._func_table['PopulateUserContext0'].root.parameters[0]
        op2_user_struct = op2._func_table['PopulateUserContext0'].root.parameters[0]

        assert len(op1_user_struct.fields) == len(op2_user_struct.fields)
        assert len(op1_user_struct.callback_fields) == \
            len(op1_user_struct.callback_fields)
        assert str(op1_user_struct.fields) == str(op2_user_struct.fields)


@skipif('petsc')
def test_callback_arguments():
    """
    Test the arguments of each callback function.
    """
    grid = Grid((2, 2), dtype=np.float64)

    f1 = Function(name='f1', grid=grid, space_order=2)
    g1 = Function(name='g1', grid=grid, space_order=2)

    eqn1 = Eq(f1.laplace, g1)

    petsc1 = petscsolve(eqn1, f1)

    with switchconfig(language='petsc'):
        op = Operator(petsc1)

    mv = op._func_table['MatMult0'].root
    ff = op._func_table['FormFunction0'].root

    assert len(mv.parameters) == 3
    assert len(ff.parameters) == 4

    assert str(mv.parameters) == '(J, X, Y)'
    assert str(ff.parameters) == '(snes, X, F, dummy)'


@skipif('petsc')
def test_apply():

    grid = Grid(shape=(13, 13), dtype=np.float64)

    pn = Function(name='pn', grid=grid, space_order=2)
    rhs = Function(name='rhs', grid=grid, space_order=2)
    mu = Constant(name='mu', value=2.0, dtype=np.float64)

    eqn = Eq(pn.laplace*mu, rhs, subdomain=grid.interior)

    petsc = petscsolve(eqn, pn)

    with switchconfig(language='petsc'):
        # Build the op
        op = Operator(petsc)

        # Check the Operator runs without errors
        op.apply()

    # Verify that users can override `mu`
    mu_new = Constant(name='mu_new', value=4.0, dtype=np.float64)
    op.apply(mu=mu_new)


@skipif('petsc')
def test_petsc_frees():

    grid = Grid((2, 2), dtype=np.float64)

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)
    petsc = petscsolve(eqn, f)

    with switchconfig(language='petsc'):
        op = Operator(petsc)

    frees = op.body.frees

    # Check the frees appear in the following order
    assert str(frees[0]) == 'PetscCall(VecDestroy(&bglobal0));'
    assert str(frees[1]) == 'PetscCall(VecDestroy(&xglobal0));'
    assert str(frees[2]) == 'PetscCall(VecDestroy(&xlocal0));'
    assert str(frees[3]) == 'PetscCall(MatDestroy(&J0));'
    assert str(frees[4]) == 'PetscCall(SNESDestroy(&snes0));'
    assert str(frees[5]) == 'PetscCall(DMDestroy(&da0));'


@skipif('petsc')
def test_calls_to_callbacks():

    grid = Grid((2, 2), dtype=np.float64)

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)
    petsc = petscsolve(eqn, f)

    with switchconfig(language='petsc'):
        op = Operator(petsc)

    ccode = str(op.ccode)

    assert '(void (*)(void))MatMult0' in ccode
    assert 'PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));' in ccode


@skipif('petsc')
def test_start_ptr():
    """
    Verify that a pointer to the start of the memory address is correctly
    generated for TimeFunction objects. This pointer should indicate the
    beginning of the multidimensional array that will be overwritten at
    the current time step.
    This functionality is crucial for VecReplaceArray operations, as it ensures
    that the correct memory location is accessed and modified during each time step.
    """
    grid = Grid((11, 11), dtype=np.float64)
    u1 = TimeFunction(name='u1', grid=grid, space_order=2)
    eq1 = Eq(u1.dt, u1.laplace, subdomain=grid.interior)
    petsc1 = petscsolve(eq1, u1.forward)

    with switchconfig(language='petsc'):
        op1 = Operator(petsc1)

    # Verify the case with modulo time stepping
    assert ('PetscScalar * u1_ptr0 = t1*localsize0 + '
            '(PetscScalar*)(u1_vec->data);') in str(op1)

    # Verify the case with no modulo time stepping
    u2 = TimeFunction(name='u2', grid=grid, space_order=2, save=5)
    eq2 = Eq(u2.dt, u2.laplace, subdomain=grid.interior)
    petsc2 = petscsolve(eq2, u2.forward)

    with switchconfig(language='petsc'):
        op2 = Operator(petsc2)

    assert ('PetscScalar * u2_ptr0 = (time + 1)*localsize0 + '
            '(PetscScalar*)(u2_vec->data);') in str(op2)


class TestTimeLoop:
    @skipif('petsc')
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_time_dimensions(self, dim):
        """
        Verify the following:
        - Modulo dimensions are correctly assigned and updated in the PETSc struct
        at each time step.
        - Only assign/update the modulo dimensions required by any of the
        PETSc callback functions.
        """
        shape = tuple(11 for _ in range(dim))
        grid = Grid(shape=shape, dtype=np.float64)

        # Modulo time stepping
        u1 = TimeFunction(name='u1', grid=grid, space_order=2)
        v1 = Function(name='v1', grid=grid, space_order=2)
        eq1 = Eq(v1.laplace, u1)
        petsc1 = petscsolve(eq1, v1)

        with switchconfig(language='petsc'):
            op1 = Operator(petsc1)
            op1.apply(time_M=3)
        body1 = str(op1.body)
        rhs1 = str(op1._func_table['FormRHS0'].root.ccode)

        assert 'ctx0.t0 = t0' in body1
        assert 'ctx0.t1 = t1' not in body1
        assert 'ctx0->t0' in rhs1
        assert 'ctx0->t1' not in rhs1

        # Non-modulo time stepping
        u2 = TimeFunction(name='u2', grid=grid, space_order=2, save=5)
        v2 = Function(name='v2', grid=grid, space_order=2, save=5)
        eq2 = Eq(v2.laplace, u2)
        petsc2 = petscsolve(eq2, v2)

        with switchconfig(language='petsc'):
            op2 = Operator(petsc2)
            op2.apply(time_M=3)
        body2 = str(op2.body)
        rhs2 = str(op2._func_table['FormRHS0'].root.ccode)

        assert 'ctx0.time = time' in body2
        assert 'ctx0->time' in rhs2

        # Modulo time stepping with more than one time step
        # used in one of the callback functions
        eq3 = Eq(v1.laplace, u1 + u1.forward)
        petsc3 = petscsolve(eq3, v1)

        with switchconfig(language='petsc'):
            op3 = Operator(petsc3)
            op3.apply(time_M=3)
        body3 = str(op3.body)
        rhs3 = str(op3._func_table['FormRHS0'].root.ccode)

        assert 'ctx0.t0 = t0' in body3
        assert 'ctx0.t1 = t1' in body3
        assert 'ctx0->t0' in rhs3
        assert 'ctx0->t1' in rhs3

        # Multiple petsc solves within the same time loop
        v2 = Function(name='v2', grid=grid, space_order=2)
        eq4 = Eq(v1.laplace, u1)
        petsc4 = petscsolve(eq4, v1)
        eq5 = Eq(v2.laplace, u1)
        petsc5 = petscsolve(eq5, v2)

        with switchconfig(language='petsc'):
            op4 = Operator([petsc4, petsc5])
            op4.apply(time_M=3)
        body4 = str(op4.body)

        assert 'ctx0.t0 = t0' in body4
        assert body4.count('ctx0.t0 = t0') == 1

    @skipif('petsc')
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_trivial_operator(self, dim):
        """
        Test trivial time-dependent problems with `petscsolve`.
        """
        # create shape based on dimension
        shape = tuple(4 for _ in range(dim))
        grid = Grid(shape=shape, dtype=np.float64)
        u = TimeFunction(name='u', grid=grid, save=3)

        eqn = Eq(u.forward, u + 1)

        petsc = petscsolve(eqn, target=u.forward)

        with switchconfig(log_level='DEBUG'):
            op = Operator(petsc, language='petsc')
            op.apply()

        assert np.all(u.data[0] == 0.)
        assert np.all(u.data[1] == 1.)
        assert np.all(u.data[2] == 2.)

    @skipif('petsc')
    @pytest.mark.parametrize('dim', [1, 2, 3])
    def test_time_dim(self, dim):
        """
        Verify the time loop abstraction
        when a mixture of TimeDimensions and time dependent
        SteppingDimensions are used
        """
        shape = tuple(4 for _ in range(dim))
        grid = Grid(shape=shape, dtype=np.float64)
        # Use modoulo time stepping, i.e don't pass the save argument
        u = TimeFunction(name='u', grid=grid)
        # Use grid.time_dim in the equation, as well as the TimeFunction itself
        petsc = petscsolve(Eq(u.forward, u + 1 + grid.time_dim), target=u.forward)

        with switchconfig():
            op = Operator(petsc, language='petsc')
            op.apply(time_M=1)

        body = str(op.body)
        rhs = str(op._func_table['FormRHS0'].root.ccode)

        # Check both ctx0.t0 and ctx0.time are assigned since they are both used
        # in the callback functions, specifically in FormRHS0
        assert 'ctx0.t0 = t0' in body
        assert 'ctx0.time = time' in body
        assert 'ctx0->t0' in rhs
        assert 'ctx0->time' in rhs

        # Check the ouput is as expected given two time steps have been
        # executed (time_M=1)
        assert np.all(u.data[1] == 1.)
        assert np.all(u.data[0] == 3.)


@skipif('petsc')
def test_solve_output():
    """
    Verify that `petscsolve` returns the correct output for
    simple cases e.g. forming the identity matrix.
    """
    grid = Grid(shape=(11, 11), dtype=np.float64)

    u = Function(name='u', grid=grid, space_order=2)
    v = Function(name='v', grid=grid, space_order=2)

    # Solving Ax=b where A is the identity matrix
    v.data[:] = 5.0
    eqn = Eq(u, v)
    petsc = petscsolve(eqn, target=u)

    with switchconfig(language='petsc'):
        op = Operator(petsc)
        # Check the solve function returns the correct output
        op.apply()

    assert np.allclose(u.data, v.data)


class TestEssentialBCs:
    @skipif('petsc')
    def test_essential_bcs(self):
        """
        Verify that `petscsolve` returns the correct output with
        essential boundary conditions (`EssentialBC`).
        """
        # SubDomains used for essential boundary conditions
        # should not overlap.
        class SubTop(SubDomain):
            name = 'subtop'

            def define(self, dimensions):
                x, y = dimensions
                return {x: x, y: ('right', 1)}
        sub1 = SubTop()

        class SubBottom(SubDomain):
            name = 'subbottom'

            def define(self, dimensions):
                x, y = dimensions
                return {x: x, y: ('left', 1)}
        sub2 = SubBottom()

        class SubLeft(SubDomain):
            name = 'subleft'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 1), y: ('middle', 1, 1)}
        sub3 = SubLeft()

        class SubRight(SubDomain):
            name = 'subright'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('right', 1), y: ('middle', 1, 1)}
        sub4 = SubRight()

        subdomains = (sub1, sub2, sub3, sub4)
        grid = Grid(shape=(11, 11), subdomains=subdomains, dtype=np.float64)

        u = Function(name='u', grid=grid, space_order=2)
        v = Function(name='v', grid=grid, space_order=2)

        # Solving Ax=b where A is the identity matrix
        v.data[:] = 5.0
        eqn = Eq(u, v, subdomain=grid.interior)

        bcs = [EssentialBC(u, 1., subdomain=sub1)]  # top
        bcs += [EssentialBC(u, 2., subdomain=sub2)]  # bottom
        bcs += [EssentialBC(u, 3., subdomain=sub3)]  # left
        bcs += [EssentialBC(u, 4., subdomain=sub4)]  # right

        petsc = petscsolve([eqn]+bcs, target=u)

        with switchconfig(language='petsc'):
            op = Operator(petsc)
            op.apply()

        # Check u is equal to v on the interior
        assert np.allclose(u.data[1:-1, 1:-1], v.data[1:-1, 1:-1])
        # Check u satisfies the boundary conditions
        assert np.allclose(u.data[1:-1, -1], 1.0)  # top
        assert np.allclose(u.data[1:-1, 0], 2.0)  # bottom
        assert np.allclose(u.data[0, 1:-1], 3.0)  # left
        assert np.allclose(u.data[-1, 1:-1], 4.0)  # right


@skipif('petsc')
def test_jacobian():

    class SubLeft(SubDomain):
        name = 'subleft'

        def define(self, dimensions):
            x, = dimensions
            return {x: ('left', 1)}

    class SubRight(SubDomain):
        name = 'subright'

        def define(self, dimensions):
            x, = dimensions
            return {x: ('right', 1)}

    sub1 = SubLeft()
    sub2 = SubRight()

    grid = Grid(shape=(11,), subdomains=(sub1, sub2), dtype=np.float64)

    e = Function(name='e', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)

    bc_1 = EssentialBC(e, 1.0, subdomain=sub1)
    bc_2 = EssentialBC(e, 2.0, subdomain=sub2)

    eq1 = Eq(e.laplace + e, f + 2.0)

    petsc = petscsolve([eq1, bc_1, bc_2], target=e)

    jac = petsc.rhs.field_data.jacobian

    assert jac.row_target == e
    assert jac.col_target == e

    # 2 symbolic expressions for each each EssentialBC (One ZeroRow and one ZeroColumn).
    # NOTE: This is likely to change when PetscSection + DMDA is supported
    assert len(jac.matvecs) == 5
    # TODO: I think some internals are preventing symplification here?
    assert str(jac.scdiag) == 'h_x*(1 - 2.0/h_x**2)'

    assert all(isinstance(m, EssentialBC) for m in jac.matvecs[:4])
    assert not isinstance(jac.matvecs[-1], EssentialBC)


@skipif('petsc')
def test_residual():
    class SubLeft(SubDomain):
        name = 'subleft'

        def define(self, dimensions):
            x, = dimensions
            return {x: ('left', 1)}

    class SubRight(SubDomain):
        name = 'subright'

        def define(self, dimensions):
            x, = dimensions
            return {x: ('right', 1)}

    sub1 = SubLeft()
    sub2 = SubRight()

    grid = Grid(shape=(11,), subdomains=(sub1, sub2), dtype=np.float64)

    e = Function(name='e', grid=grid, space_order=2)
    f = Function(name='f', grid=grid, space_order=2)

    bc_1 = EssentialBC(e, 1.0, subdomain=sub1)
    bc_2 = EssentialBC(e, 2.0, subdomain=sub2)

    eq1 = Eq(e.laplace + e, f + 2.0)

    petsc = petscsolve([eq1, bc_1, bc_2], target=e)

    res = petsc.rhs.field_data.residual

    assert res.target == e
    # NOTE: This is likely to change when PetscSection + DMDA is supported
    assert len(res.F_exprs) == 5
    assert len(res.b_exprs) == 3

    assert not res.time_mapper
    assert str(res.scdiag) == 'h_x*(1 - 2.0/h_x**2)'


class TestCoupledLinear:
    # The coupled interface can be used even for uncoupled problems, meaning
    # the equations will be solved within a single matrix system.
    # These tests use simple problems to validate functionality, but they help
    # ensure correctness in code generation.
    # TODO: Add more comprehensive tests for fully coupled problems.
    # TODO: Add subdomain tests, time loop, multiple coupled etc.

    @pytest.mark.parametrize('eq1, eq2, so', [
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '2'),
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '4'),
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '6'),
        ('Eq(e.laplace, f + 5.)', 'Eq(g.laplace, h + 5.)', '2'),
        ('Eq(e.laplace, f + 5.)', 'Eq(g.laplace, h + 5.)', '4'),
        ('Eq(e.laplace, f + 5.)', 'Eq(g.laplace, h + 5.)', '6'),
        ('Eq(e.dx, e + 2.*f)', 'Eq(g.dx, g + 2.*h)', '2'),
        ('Eq(e.dx, e + 2.*f)', 'Eq(g.dx, g + 2.*h)', '4'),
        ('Eq(e.dx, e + 2.*f)', 'Eq(g.dx, g + 2.*h)', '6'),
        ('Eq(f.dx, e.dx + e + e.laplace)', 'Eq(h.dx, g.dx + g + g.laplace)', '2'),
        ('Eq(f.dx, e.dx + e + e.laplace)', 'Eq(h.dx, g.dx + g + g.laplace)', '4'),
        ('Eq(f.dx, e.dx + e + e.laplace)', 'Eq(h.dx, g.dx + g + g.laplace)', '6'),
    ])
    @skipif('petsc')
    def test_coupled_vs_non_coupled(self, eq1, eq2, so):
        """
        Test that solving multiple **uncoupled** equations separately
        vs. together with `petscsolve` yields the same result.
        This test is non time-dependent.
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=eval(so))
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        f.data[:] = 5.
        h.data[:] = 5.

        eq1 = eval(eq1)
        eq2 = eval(eq2)

        # Non-coupled
        petsc1 = petscsolve(eq1, target=e)
        petsc2 = petscsolve(eq2, target=g)

        with switchconfig(language='petsc'):
            op1 = Operator([petsc1, petsc2], opt='noop')
            op1.apply()

        enorm1 = norm(e)
        gnorm1 = norm(g)

        # Reset
        e.data[:] = 0
        g.data[:] = 0

        # Coupled
        petsc3 = petscsolve({e: [eq1], g: [eq2]})

        with switchconfig(language='petsc'):
            op2 = Operator(petsc3, opt='noop')
            op2.apply()

        enorm2 = norm(e)
        gnorm2 = norm(g)

        print('enorm1:', enorm1)
        print('enorm2:', enorm2)
        assert np.isclose(enorm1, enorm2, atol=1e-14)
        assert np.isclose(gnorm1, gnorm2, atol=1e-14)

        callbacks1 = [meta_call.root for meta_call in op1._func_table.values()]
        callbacks2 = [meta_call.root for meta_call in op2._func_table.values()]

        # Solving for multiple fields within the same matrix system requires
        # less callback functions than solving them separately.
        # TODO: As noted in the other test, some efuncs are not reused
        # where reuse is possible, investigate.
        assert len(callbacks1) == 12
        assert len(callbacks2) == 8

        # Check field_data type
        field0 = petsc1.rhs.field_data
        field1 = petsc2.rhs.field_data
        field2 = petsc3.rhs.field_data

        assert isinstance(field0, FieldData)
        assert isinstance(field1, FieldData)
        assert isinstance(field2, MultipleFieldData)

    @skipif('petsc')
    def test_coupled_structs(self):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = Eq(e + 5, f)
        eq2 = Eq(g + 10, h)

        petsc = petscsolve({f: [eq1], h: [eq2]})

        name = "foo"

        with switchconfig(language='petsc'):
            op = Operator(petsc, name=name)

        # Trigger the generation of a .c and a .h files
        ccode, hcode = op.cinterface(force=True)

        dirname = op._compiler.get_jit_dir()
        assert os.path.isfile(os.path.join(dirname, f"{name}.c"))
        assert os.path.isfile(os.path.join(dirname, f"{name}.h"))

        ccode = str(ccode)
        hcode = str(hcode)

        assert f'include "{name}.h"' in ccode

        # The public `struct JacobianCtx` only appears in the header file
        assert 'struct JacobianCtx\n{' not in ccode
        assert 'struct JacobianCtx\n{' in hcode

        # The public `struct SubMatrixCtx` only appears in the header file
        assert 'struct SubMatrixCtx\n{' not in ccode
        assert 'struct SubMatrixCtx\n{' in hcode

        # The public `struct UserCtx0` only appears in the header file
        assert 'struct UserCtx0\n{' not in ccode
        assert 'struct UserCtx0\n{' in hcode

        # The public struct Field0 only appears in the header file
        assert 'struct Field0\n{' not in ccode
        assert 'struct Field0\n{' in hcode

    @pytest.mark.parametrize('n_fields', [2, 3, 4, 5, 6])
    @skipif('petsc')
    def test_coupled_frees(self, n_fields):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=f'u{i}', grid=grid, space_order=2)
                     for i in range(n_fields + 1)]
        *solved_funcs, h = functions

        equations = [Eq(func.laplace, h) for func in solved_funcs]
        petsc = petscsolve({func: [eq] for func, eq in zip(solved_funcs, equations)})

        with switchconfig(language='petsc'):
            op = Operator(petsc, opt='noop')

        frees = op.body.frees

        # IS Destroy calls
        for i in range(n_fields):
            assert str(frees[i]) == f'PetscCall(ISDestroy(&fields0[{i}]));'
        assert str(frees[n_fields]) == 'PetscCall(PetscFree(fields0));'

        # DM Destroy calls
        for i in range(n_fields):
            assert str(frees[n_fields + 1 + i]) == \
                f'PetscCall(DMDestroy(&subdms0[{i}]));'
        assert str(frees[n_fields*2 + 1]) == 'PetscCall(PetscFree(subdms0));'

    @skipif('petsc')
    def test_dmda_dofs(self):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = Eq(e.laplace, h)
        eq2 = Eq(f.laplace, h)
        eq3 = Eq(g.laplace, h)

        petsc1 = petscsolve({e: [eq1]})
        petsc2 = petscsolve({e: [eq1], f: [eq2]})
        petsc3 = petscsolve({e: [eq1], f: [eq2], g: [eq3]})

        with switchconfig(language='petsc'):
            op1 = Operator(petsc1, opt='noop')
            op2 = Operator(petsc2, opt='noop')
            op3 = Operator(petsc3, opt='noop')

        # Check the number of dofs in the DMDA for each field
        assert 'PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
            'DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,11,11,1,1,1,2,NULL,NULL,&da0));' \
            in str(op1)

        assert 'PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
            'DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,11,11,1,1,2,2,NULL,NULL,&da0));' \
            in str(op2)

        assert 'PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
            'DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,11,11,1,1,3,2,NULL,NULL,&da0));' \
            in str(op3)

    @skipif('petsc')
    def test_mixed_jacobian(self):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = Eq(e.laplace, f)
        eq2 = Eq(g.laplace, h)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        jacobian = petsc.rhs.field_data.jacobian

        j00 = jacobian.get_submatrix(0, 0)
        j01 = jacobian.get_submatrix(0, 1)
        j10 = jacobian.get_submatrix(1, 0)
        j11 = jacobian.get_submatrix(1, 1)

        # Check type of each submatrix is a SubMatrixBlock
        assert isinstance(j00, SubMatrixBlock)
        assert isinstance(j01, SubMatrixBlock)
        assert isinstance(j10, SubMatrixBlock)
        assert isinstance(j11, SubMatrixBlock)

        assert j00.name == 'J00'
        assert j01.name == 'J01'
        assert j10.name == 'J10'
        assert j11.name == 'J11'

        assert j00.row_target == e
        assert j01.row_target == e
        assert j10.row_target == g
        assert j11.row_target == g

        assert j00.col_target == e
        assert j01.col_target == g
        assert j10.col_target == e
        assert j11.col_target == g

        assert j00.row_idx == 0
        assert j01.row_idx == 0
        assert j10.row_idx == 1
        assert j11.row_idx == 1

        assert j00.col_idx == 0
        assert j01.col_idx == 1
        assert j10.col_idx == 0
        assert j11.col_idx == 1

        assert j00.linear_idx == 0
        assert j01.linear_idx == 1
        assert j10.linear_idx == 2
        assert j11.linear_idx == 3

        # Check the number of submatrices
        assert jacobian.n_submatrices == 4

        # Technically a non-coupled problem, so the only non-zero submatrices
        # should be the diagonal ones i.e J00 and J11
        nonzero_submats = jacobian.nonzero_submatrices
        assert len(nonzero_submats) == 2
        assert j00 in nonzero_submats
        assert j11 in nonzero_submats
        assert j01 not in nonzero_submats
        assert j10 not in nonzero_submats
        assert not j01.matvecs
        assert not j10.matvecs

        # Compatible scaling to reduce condition number of jacobian
        assert str(j00.matvecs[0]) == 'Eq(y_e(x, y),' \
            + ' h_x*h_y*(Derivative(x_e(x, y), (x, 2)) + Derivative(x_e(x, y), (y, 2))))'

        assert str(j11.matvecs[0]) == 'Eq(y_g(x, y),' \
            + ' h_x*h_y*(Derivative(x_g(x, y), (x, 2)) + Derivative(x_g(x, y), (y, 2))))'

        # Check the col_targets
        assert j00.col_target == e
        assert j01.col_target == g
        assert j10.col_target == e
        assert j11.col_target == g

    @pytest.mark.parametrize('eq1, eq2, j01_matvec, j10_matvec', [
        ('Eq(-e.laplace, g)', 'Eq(-g.laplace, e)',
         'Eq(y_e(x, y), -h_x*h_y*x_g(x, y))',
         'Eq(y_g(x, y), -h_x*h_y*x_e(x, y))'),
        ('Eq(-e.laplace, 2.*g)', 'Eq(-g.laplace, 2.*e)',
         'Eq(y_e(x, y), -2.0*h_x*h_y*x_g(x, y))',
         'Eq(y_g(x, y), -2.0*h_x*h_y*x_e(x, y))'),
        ('Eq(-e.laplace, g.dx)', 'Eq(-g.laplace, e.dx)',
         'Eq(y_e(x, y), -h_x*h_y*Derivative(x_g(x, y), x))',
         'Eq(y_g(x, y), -h_x*h_y*Derivative(x_e(x, y), x))'),
        ('Eq(-e.laplace, g.dx + g)', 'Eq(-g.laplace, e.dx + e)',
         'Eq(y_e(x, y), h_x*h_y*(-x_g(x, y) - Derivative(x_g(x, y), x)))',
         'Eq(y_g(x, y), h_x*h_y*(-x_e(x, y) - Derivative(x_e(x, y), x)))'),
        ('Eq(e, g.dx + g)', 'Eq(g, e.dx + e)',
         'Eq(y_e(x, y), h_x*h_y*(-x_g(x, y) - Derivative(x_g(x, y), x)))',
         'Eq(y_g(x, y), h_x*h_y*(-x_e(x, y) - Derivative(x_e(x, y), x)))'),
        ('Eq(e, g.dx + g.dy)', 'Eq(g, e.dx + e.dy)',
         'Eq(y_e(x, y), h_x*h_y*(-Derivative(x_g(x, y), x) - Derivative(x_g(x, y), y)))',
         'Eq(y_g(x, y), h_x*h_y*(-Derivative(x_e(x, y), x) - Derivative(x_e(x, y), y)))'),
        ('Eq(g, -e.laplace)', 'Eq(e, -g.laplace)',
         'Eq(y_e(x, y), h_x*h_y*x_g(x, y))',
         'Eq(y_g(x, y), h_x*h_y*x_e(x, y))'),
        ('Eq(e + g, e.dx + 2.*g.dx)', 'Eq(g + e, g.dx + 2.*e.dx)',
         'Eq(y_e(x, y), h_x*h_y*(x_g(x, y) - 2.0*Derivative(x_g(x, y), x)))',
         'Eq(y_g(x, y), h_x*h_y*(x_e(x, y) - 2.0*Derivative(x_e(x, y), x)))'),
    ])
    @skipif('petsc')
    def test_coupling(self, eq1, eq2, j01_matvec, j10_matvec):
        """
        Test linear coupling between fields, where the off-diagonal
        Jacobian submatrices are nonzero.
        """
        grid = Grid(shape=(9, 9), dtype=np.float64)

        e = Function(name='e', grid=grid, space_order=2)
        g = Function(name='g', grid=grid, space_order=2)

        eq1 = eval(eq1)
        eq2 = eval(eq2)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        jacobian = petsc.rhs.field_data.jacobian

        j01 = jacobian.get_submatrix(0, 1)
        j10 = jacobian.get_submatrix(1, 0)

        assert j01.col_target == g
        assert j10.col_target == e

        assert str(j01.matvecs[0]) == j01_matvec
        assert str(j10.matvecs[0]) == j10_matvec

    @pytest.mark.parametrize('eq1, eq2, so, scale', [
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '2', '-2.0*h_x/h_x**2'),
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '4', '-2.5*h_x/h_x**2'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '2', 'h_x*(1 - 2.0/h_x**2)'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '4', 'h_x*(1 - 2.5/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '2',
         'h_x*(5.0 - 2.0/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '4',
         'h_x*(5.0 - 2.5/h_x**2)'),
        ('Eq(e.dx + e + e.laplace, f)', 'Eq(g.dx + g + g.laplace, h.dx)', '2',
         'h_x*(1 + 1/h_x - 2.0/h_x**2)'),
        ('Eq(e.dx + e + e.laplace, f)', 'Eq(g.dx + g + g.laplace, h.dx)', '4',
         'h_x*(1 - 2.5/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '2',
         'h_x*(1 - 4.0/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '4',
         'h_x*(1 - 5.0/h_x**2)'),
    ])
    @skipif('petsc')
    def test_jacobian_scaling_1D(self, eq1, eq2, so, scale):
        """
        Test the computation of diagonal scaling in a 1D Jacobian system.

        This scaling would be applied to the boundary rows of the matrix
        if essential boundary conditions were enforced in the solver.
        Its purpose is to reduce the condition number of the matrix.
        """
        grid = Grid(shape=(9,), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=eval(so))
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = eval(eq1)
        eq2 = eval(eq2)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        jacobian = petsc.rhs.field_data.jacobian

        j00 = jacobian.get_submatrix(0, 0)
        j11 = jacobian.get_submatrix(1, 1)

        assert str(j00.scdiag) == scale
        assert str(j11.scdiag) == scale

    @pytest.mark.parametrize('eq1, eq2, so, scale', [
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '2',
         'h_x*h_y*(-2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '4',
         'h_x*h_y*(-2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '2',
         'h_x*h_y*(1 - 2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '4',
         'h_x*h_y*(1 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '2',
         'h_x*h_y*(5.0 - 2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '4',
         'h_x*h_y*(5.0 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.dx + e.dy + e + e.laplace, f)', 'Eq(g.dx + g.dy + g + g.laplace, h)',
         '2', 'h_x*h_y*(1 + 1/h_y - 2.0/h_y**2 + 1/h_x - 2.0/h_x**2)'),
        ('Eq(e.dx + e.dy + e + e.laplace, f)', 'Eq(g.dx + g.dy + g + g.laplace, h)',
         '4', 'h_x*h_y*(1 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '2',
         'h_x*h_y*(1 - 4.0/h_y**2 - 4.0/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '4',
         'h_x*h_y*(1 - 5.0/h_y**2 - 5.0/h_x**2)'),
    ])
    @skipif('petsc')
    def test_jacobian_scaling_2D(self, eq1, eq2, so, scale):
        """
        Test the computation of diagonal scaling in a 2D Jacobian system.

        This scaling would be applied to the boundary rows of the matrix
        if essential boundary conditions were enforced in the solver.
        Its purpose is to reduce the condition number of the matrix.
        """
        grid = Grid(shape=(9, 9), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=eval(so))
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = eval(eq1)
        eq2 = eval(eq2)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        jacobian = petsc.rhs.field_data.jacobian

        j00 = jacobian.get_submatrix(0, 0)
        j11 = jacobian.get_submatrix(1, 1)

        assert str(j00.scdiag) == scale
        assert str(j11.scdiag) == scale

    @pytest.mark.parametrize('eq1, eq2, so, scale', [
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '2',
         'h_x*h_y*h_z*(-2.0/h_z**2 - 2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace, f)', 'Eq(g.laplace, h)', '4',
         'h_x*h_y*h_z*(-2.5/h_z**2 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '2',
         'h_x*h_y*h_z*(1 - 2.0/h_z**2 - 2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace + e, f)', 'Eq(g.laplace + g, h)', '4',
         'h_x*h_y*h_z*(1 - 2.5/h_z**2 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '2',
         'h_x*h_y*h_z*(5.0 - 2.0/h_z**2 - 2.0/h_y**2 - 2.0/h_x**2)'),
        ('Eq(e.laplace + 5.*e, f)', 'Eq(g.laplace + 5.*g, h)', '4',
         'h_x*h_y*h_z*(5.0 - 2.5/h_z**2 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(e.dx + e.dy + e.dz + e + e.laplace, f)',
         'Eq(g.dx + g.dy + g.dz + g + g.laplace, h)', '2',
         'h_x*h_y*h_z*(1 + 1/h_z - 2.0/h_z**2 + 1/h_y - 2.0/h_y**2 + ' +
         '1/h_x - 2.0/h_x**2)'),
        ('Eq(e.dx + e.dy + e.dz + e + e.laplace, f)',
         'Eq(g.dx + g.dy + g.dz + g + g.laplace, h)', '4',
         'h_x*h_y*h_z*(1 - 2.5/h_z**2 - 2.5/h_y**2 - 2.5/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '2',
         'h_x*h_y*h_z*(1 - 4.0/h_z**2 - 4.0/h_y**2 - 4.0/h_x**2)'),
        ('Eq(2.*e.laplace + e, f)', 'Eq(2*g.laplace + g, h)', '4',
         'h_x*h_y*h_z*(1 - 5.0/h_z**2 - 5.0/h_y**2 - 5.0/h_x**2)'),
    ])
    @skipif('petsc')
    def test_jacobian_scaling_3D(self, eq1, eq2, so, scale):
        """
        Test the computation of diagonal scaling in a 3D Jacobian system.

        This scaling would be applied to the boundary rows of the matrix
        if essential boundary conditions were enforced in the solver.
        Its purpose is to reduce the condition number of the matrix.
        """
        grid = Grid(shape=(9, 9, 9), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=eval(so))
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = eval(eq1)
        eq2 = eval(eq2)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        jacobian = petsc.rhs.field_data.jacobian

        j00 = jacobian.get_submatrix(0, 0)
        j11 = jacobian.get_submatrix(1, 1)

        assert str(j00.scdiag) == scale
        assert str(j11.scdiag) == scale

    @skipif('petsc')
    def test_residual_bundle(self):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = Eq(e.laplace, h)
        eq2 = Eq(f.laplace, h)
        eq3 = Eq(g.laplace, h)

        petsc1 = petscsolve({e: [eq1]})
        petsc2 = petscsolve({e: [eq1], f: [eq2]})
        petsc3 = petscsolve({e: [eq1], f: [eq2], g: [eq3]})

        with switchconfig(language='petsc'):
            op1 = Operator(petsc1, opt='noop', name='op1')
            op2 = Operator(petsc2, opt='noop', name='op2')
            op3 = Operator(petsc3, opt='noop', name='op3')

        # Check pointers to array of Field structs. Note this is only
        # required when dof>1 when constructing the multi-component DMDA.
        f_aos = 'struct Field0 (* f_bundle)[info.gxm] = ' \
            + '(struct Field0 (*)[info.gxm]) f_bundle_vec;'
        x_aos = 'struct Field0 (* x_bundle)[info.gxm] = ' \
            + '(struct Field0 (*)[info.gxm]) x_bundle_vec;'

        for op in (op1, op2, op3):
            ccode = str(op.ccode)
            assert f_aos in ccode
            assert x_aos in ccode

        assert 'struct Field0\n{\n  PetscScalar e;\n}' \
            in str(op1.ccode)
        assert 'struct Field0\n{\n  PetscScalar e;\n  PetscScalar f;\n}' \
            in str(op2.ccode)
        assert 'struct Field0\n{\n  PetscScalar e;\n  PetscScalar f;\n  ' \
            + 'PetscScalar g;\n}' in str(op3.ccode)

    @skipif('petsc')
    def test_residual_callback(self):
        """
        Check that the main residual callback correctly accesses the
        target fields in the bundle.
        """
        grid = Grid(shape=(9, 9), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        eq1 = Eq(e.laplace, f)
        eq2 = Eq(g.laplace, h)

        petsc = petscsolve({e: [eq1], g: [eq2]})

        with switchconfig(language='petsc'):
            op = Operator(petsc)

        # Check the residual callback
        residual = op._func_table['WholeFormFunc0'].root

        exprs = FindNodes(Expression).visit(residual)
        exprs = [str(e) for e in exprs]

        assert 'f_bundle[x + 2][y + 2].e = (r4*x_bundle[x + 1][y + 2].e + ' + \
            'r4*x_bundle[x + 3][y + 2].e + r5*x_bundle[x + 2][y + 1].e + r5*' + \
            'x_bundle[x + 2][y + 3].e - 2.0*(r4*x_bundle[x + 2][y + 2].e + r5*' + \
            'x_bundle[x + 2][y + 2].e) - f[x + 2][y + 2])*ctx0->h_x*ctx0->h_y;' in exprs

        assert 'f_bundle[x + 2][y + 2].g = (r4*x_bundle[x + 1][y + 2].g + ' + \
            'r4*x_bundle[x + 3][y + 2].g + r5*x_bundle[x + 2][y + 1].g + r5*' + \
            'x_bundle[x + 2][y + 3].g - 2.0*(r4*x_bundle[x + 2][y + 2].g + r5*' + \
            'x_bundle[x + 2][y + 2].g) - h[x + 2][y + 2])*ctx0->h_x*ctx0->h_y;' in exprs

    @skipif('petsc')
    def test_essential_bcs(self):
        """
        Test mixed problem with SubDomains
        """
        class SubTop(SubDomain):
            name = 'subtop'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 1), y: ('right', 1)}

        sub1 = SubTop()

        grid = Grid(shape=(9, 9), subdomains=(sub1,), dtype=np.float64)

        u = Function(name='u', grid=grid, space_order=2)
        v = Function(name='v', grid=grid, space_order=2)
        f = Function(name='f', grid=grid, space_order=2)

        eqn1 = Eq(-v.laplace, f, subdomain=grid.interior)
        eqn2 = Eq(-u.laplace, v, subdomain=grid.interior)

        bc_u = [EssentialBC(u, 0., subdomain=sub1)]
        bc_v = [EssentialBC(v, 0., subdomain=sub1)]

        petsc = petscsolve({v: [eqn1]+bc_v, u: [eqn2]+bc_u})

        with switchconfig(language='petsc'):
            op = Operator(petsc)

        # Test scaling
        J00 = op._func_table['J00_MatMult0'].root

        # Essential BC row
        assert 'a1[ix + 2][iy + 2] = (2.0/((o0->h_y*o0->h_y))' \
            ' + 2.0/((o0->h_x*o0->h_x)))*o0->h_x*o0->h_y*a0[ix + 2][iy + 2];' in str(J00)
        # Check zeroing of essential BC columns
        assert 'a0[ix + 2][iy + 2] = 0.0;' in str(J00)
        # Interior loop
        assert 'a1[ix + 2][iy + 2] = (2.0*(r0*a0[ix + 2][iy + 2] ' \
            '+ r1*a0[ix + 2][iy + 2]) - (r0*a0[ix + 1][iy + 2] + ' \
            'r0*a0[ix + 3][iy + 2] + r1*a0[ix + 2][iy + 1] ' \
            '+ r1*a0[ix + 2][iy + 3]))*o0->h_x*o0->h_y;' in str(J00)

        # J00 and J11 are semantically identical so check efunc reuse
        assert len(op._func_table.values()) == 9
        # J00_MatMult0 is reused (in replace of J11_MatMult0)
        create = op._func_table['MatCreateSubMatrices0'].root
        assert 'MatShellSetOperation(submat_arr[0],' \
            + 'MATOP_MULT,(void (*)(void))J00_MatMult0)' in str(create)
        assert 'MatShellSetOperation(submat_arr[3],' \
            + 'MATOP_MULT,(void (*)(void))J00_MatMult0)' in str(create)

    # TODO: Test mixed, time dependent solvers


class TestMPI:
    # TODO: Add test for DMDACreate() in parallel

    @pytest.mark.parametrize('nx, unorm', [
        (17, 7.441506654790017),
        (33, 10.317652759863675),
        (65, 14.445123374862874),
        (129, 20.32492895656658),
        (257, 28.67050632840985)
    ])
    @skipif('petsc')
    @pytest.mark.parallel(mode=[2, 4, 8])
    def test_laplacian_1d(self, nx, unorm, mode):
        """
        """
        configuration['compiler'] = 'custom'
        os.environ['CC'] = 'mpicc'
        PetscInitialize()

        class SubSide(SubDomain):
            def __init__(self, side='left', grid=None):
                self.side = side
                self.name = f'sub{side}'
                super().__init__(grid=grid)

            def define(self, dimensions):
                x, = dimensions
                return {x: (self.side, 1)}

        grid = Grid(shape=(nx,), dtype=np.float64)
        sub1, sub2 = [SubSide(side=s, grid=grid) for s in ('left', 'right')]

        u = Function(name='u', grid=grid, space_order=2)
        f = Function(name='f', grid=grid, space_order=2)

        u0 = Constant(name='u0', value=-1.0, dtype=np.float64)
        u1 = Constant(name='u1', value=-np.exp(1.0), dtype=np.float64)

        eqn = Eq(-u.laplace, f, subdomain=grid.interior)

        X = np.linspace(0, 1.0, nx).astype(np.float64)
        f.data[:] = np.float64(np.exp(X))

        # Create boundary condition expressions using subdomains
        bcs = [EssentialBC(u, u0, subdomain=sub1)]
        bcs += [EssentialBC(u, u1, subdomain=sub2)]

        petsc = petscsolve([eqn] + bcs, target=u, solver_parameters={'ksp_rtol': 1e-10})

        op = Operator(petsc, language='petsc')
        op.apply()

        # Expected norm computed "manually" from sequential run
        # What rtol and atol should be used?
        assert np.isclose(norm(u), unorm, rtol=1e-13, atol=1e-13)


class TestLogging:

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_logging(self, log_level):
        """Verify PetscSummary output when the log level is 'PERF' or 'DEBUG."""
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f']]
        e, f = functions
        f.data[:] = 5.0
        eq = Eq(e.laplace, f)

        petsc = petscsolve(eq, target=e, options_prefix='poisson')

        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(petsc)
            summary = op.apply()

        # One PerformanceSummary
        assert len(summary) == 1

        # Access the PetscSummary
        petsc_summary = summary.petsc

        assert isinstance(summary, PerformanceSummary)
        assert isinstance(petsc_summary, PetscSummary)

        # One section with a single solver
        assert len(petsc_summary) == 1

        entry0 = petsc_summary.get_entry('section0', 'poisson')
        entry1 = petsc_summary[('section0', 'poisson')]
        assert entry0 == entry1
        assert entry0.SNESGetIterationNumber == 1

        snesits0 = petsc_summary.SNESGetIterationNumber
        snesits1 = petsc_summary['SNESGetIterationNumber']
        # Check case insensitive key access
        snesits2 = petsc_summary['snesgetiterationnumber']
        snesits3 = petsc_summary['SNESgetiterationNumber']

        assert snesits0 == snesits1 == snesits2 == snesits3

        assert len(snesits0) == 1
        key, value = next(iter(snesits0.items()))
        assert str(key) == "PetscKey(name='section0', options_prefix='poisson')"
        assert value == 1

        # Test logging KSPGetTolerances. Since no overrides have been applied,
        # the tolerances should match the default linear values.
        tols = entry0.KSPGetTolerances
        assert tols['rtol'] == linear_solve_defaults['ksp_rtol']
        assert tols['atol'] == linear_solve_defaults['ksp_atol']
        assert tols['divtol'] == linear_solve_defaults['ksp_divtol']
        assert tols['max_it'] == linear_solve_defaults['ksp_max_it']

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_logging_multiple_solves(self, log_level):
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        e.data[:] = 5.0
        f.data[:] = 6.0

        eq1 = Eq(g.laplace, e)
        eq2 = Eq(h, f + 5.0)

        solver1 = petscsolve(eq1, target=g, options_prefix='poisson1')
        solver2 = petscsolve(eq2, target=h, options_prefix='poisson2')

        with switchconfig(language='petsc', log_level=log_level):
            op = Operator([solver1, solver2])
            summary = op.apply()

        petsc_summary = summary.petsc
        # One PetscKey, PetscEntry for each solver
        assert len(petsc_summary) == 2

        entry1 = petsc_summary.get_entry('section0', 'poisson1')
        entry2 = petsc_summary.get_entry('section1', 'poisson2')

        assert len(petsc_summary.KSPGetIterationNumber) == 2
        assert len(petsc_summary.SNESGetIterationNumber) == 2

        assert entry1.KSPGetIterationNumber == 16
        assert entry1.SNESGetIterationNumber == 1
        assert entry2.KSPGetIterationNumber == 1
        assert entry2.SNESGetIterationNumber == 1

        # Test key access to PetscEntry
        assert entry1['KSPGetIterationNumber'] == 16
        assert entry1['SNESGetIterationNumber'] == 1
        # Case insensitive key access
        assert entry1['kspgetiterationnumber'] == 16

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_logging_user_prefixes(self, log_level):
        """
        Verify that `PetscSummary` uses the user provided `options_prefix` when given.
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        pde1 = Eq(e.laplace, f)
        pde2 = Eq(g.laplace, h)

        petsc1 = petscsolve(pde1, target=e, options_prefix='pde1')
        petsc2 = petscsolve(pde2, target=g, options_prefix='pde2')

        with switchconfig(language='petsc', log_level=log_level):
            op = Operator([petsc1, petsc2])
            summary = op.apply()

        petsc_summary = summary.petsc

        # Check that the prefix is correctly set in the PetscSummary
        key_strings = [f"{key.name}:{key.options_prefix}" for key in petsc_summary.keys()]
        assert set(key_strings) == {"section0:pde1", "section1:pde2"}

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_logging_default_prefixes(self, log_level):
        """
        Verify that `PetscSummary` uses the default options prefix
        provided by Devito if no user `options_prefix` is specified.
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)

        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f', 'g', 'h']]
        e, f, g, h = functions

        pde1 = Eq(e.laplace, f)
        pde2 = Eq(g.laplace, h)

        petsc1 = petscsolve(pde1, target=e)
        petsc2 = petscsolve(pde2, target=g)

        with switchconfig(language='petsc', log_level=log_level):
            op = Operator([petsc1, petsc2])
            summary = op.apply()

        petsc_summary = summary.petsc

        # Users should set a custom options_prefix if they want logging; otherwise,
        # the default automatically generated prefix is used in the `PetscSummary`.
        assert all(re.fullmatch(r"devito_\d+_", k.options_prefix) for k in petsc_summary)


class TestSolverParameters:

    @skipif('petsc')
    def setup_class(self):
        """
        Setup grid, functions and equations shared across
        tests in this class
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)
        self.e, self.f, self.g, self.h = [
            Function(name=n, grid=grid, space_order=2)
            for n in ['e', 'f', 'g', 'h']
        ]
        self.eq1 = Eq(self.e.laplace, self.f)
        self.eq2 = Eq(self.g.laplace, self.h)

    @skipif('petsc')
    def test_different_solver_params(self):
        # Explicitly set the solver parameters
        solver1 = petscsolve(
            self.eq1, target=self.e, solver_parameters={'ksp_rtol': '1e-10'}
        )
        # Use solver parameter defaults
        solver2 = petscsolve(self.eq2, target=self.g)

        with switchconfig(language='petsc'):
            op = Operator([solver1, solver2])

        assert 'SetPetscOptions0' in op._func_table
        assert 'SetPetscOptions1' in op._func_table

        assert '_ksp_rtol","1e-10"' \
            in str(op._func_table['SetPetscOptions0'].root)

        assert '_ksp_rtol","1e-05"' \
            in str(op._func_table['SetPetscOptions1'].root)

    @skipif('petsc')
    def test_options_prefix(self):
        solver1 = petscsolve(self.eq1, self.e,
                             solver_parameters={'ksp_rtol': '1e-10'},
                             options_prefix='poisson1')
        solver2 = petscsolve(self.eq2, self.g,
                             solver_parameters={'ksp_rtol': '1e-12'},
                             options_prefix='poisson2')

        with switchconfig(language='petsc'):
            op = Operator([solver1, solver2])

        # Check the options prefix has been correctly set for each snes solver
        assert 'PetscCall(SNESSetOptionsPrefix(snes0,"poisson1_"));' in str(op)
        assert 'PetscCall(SNESSetOptionsPrefix(snes1,"poisson2_"));' in str(op)

        # Test the options prefix has be correctly applied to the solver options
        assert 'PetscCall(PetscOptionsSetValue(NULL,"-poisson1_ksp_rtol","1e-10"));' \
            in str(op._func_table['SetPetscOptions0'].root)

        assert 'PetscCall(PetscOptionsSetValue(NULL,"-poisson2_ksp_rtol","1e-12"));' \
            in str(op._func_table['SetPetscOptions1'].root)

    @skipif('petsc')
    def test_options_no_value(self):
        """
        Test solver parameters that do not require a value, such as
        `snes_view` and `ksp_view`.
        """
        solver = petscsolve(
            self.eq1, target=self.e, solver_parameters={'snes_view': None},
            options_prefix='solver1'
        )
        with switchconfig(language='petsc'):
            op = Operator(solver)
            op.apply()

        assert 'PetscCall(PetscOptionsSetValue(NULL,"-solver1_snes_view",NULL));' \
            in str(op._func_table['SetPetscOptions0'].root)

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_tolerances(self, log_level):
        params = {
            'ksp_rtol': 1e-12,
            'ksp_atol': 1e-20,
            'ksp_divtol': 1e3,
            'ksp_max_it': 100
        }
        solver = petscsolve(
            self.eq1, target=self.e, solver_parameters=params,
            options_prefix='solver'
        )

        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(solver)
            tmp = op.apply()

        petsc_summary = tmp.petsc
        entry = petsc_summary.get_entry('section0', 'solver')
        tolerances = entry.KSPGetTolerances

        # Test that the tolerances have been set correctly and therefore
        # appear as expected in the `PetscSummary`.
        assert tolerances['rtol'] == params['ksp_rtol']
        assert tolerances['atol'] == params['ksp_atol']
        assert tolerances['divtol'] == params['ksp_divtol']
        assert tolerances['max_it'] == params['ksp_max_it']

    @skipif('petsc')
    def test_clearing_options(self):
        # Explicitly set the solver parameters
        solver1 = petscsolve(
            self.eq1, target=self.e, solver_parameters={'ksp_rtol': '1e-10'}
        )
        # Use the solver parameter defaults
        solver2 = petscsolve(self.eq2, target=self.g)

        with switchconfig(language='petsc'):
            op = Operator([solver1, solver2])

        assert 'ClearPetscOptions0' in op._func_table
        assert 'ClearPetscOptions1' in op._func_table

    @skipif('petsc')
    def test_error_if_same_prefix(self):
        """
        Test an error is raised if the same options prefix is used
        for two different solvers within the same Operator.
        """
        solver1 = petscsolve(
            self.eq1, target=self.e, options_prefix='poisson',
            solver_parameters={'ksp_rtol': '1e-10'}
        )
        solver2 = petscsolve(
            self.eq2, target=self.g, options_prefix='poisson',
            solver_parameters={'ksp_rtol': '1e-12'}
        )
        with switchconfig(language='petsc'):
            with pytest.raises(ValueError):
                Operator([solver1, solver2])

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_multiple_operators(self, log_level):
        """
        Verify that solver parameters are set correctly when multiple `Operator`s
        are created with `petscsolve` calls sharing the same `options_prefix`.

        Note: Using the same `options_prefix` within a single `Operator` is not allowed
        (see previous test), but the same prefix can be used across
        different `Operator`s (although not advised).
        """
        # Create two `petscsolve` calls with the same `options_prefix``
        solver1 = petscsolve(
            self.eq1, target=self.e, options_prefix='poisson',
            solver_parameters={'ksp_rtol': '1e-10'}
        )
        solver2 = petscsolve(
            self.eq2, target=self.g, options_prefix='poisson',
            solver_parameters={'ksp_rtol': '1e-12'}
        )
        with switchconfig(language='petsc', log_level=log_level):
            op1 = Operator(solver1)
            op2 = Operator(solver2)
            summary1 = op1.apply()
            summary2 = op2.apply()

        petsc_summary1 = summary1.petsc
        entry1 = petsc_summary1.get_entry('section0', 'poisson')

        petsc_summary2 = summary2.petsc
        entry2 = petsc_summary2.get_entry('section0', 'poisson')

        assert entry1.KSPGetTolerances['rtol'] == 1e-10
        assert entry2.KSPGetTolerances['rtol'] == 1e-12

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_command_line_priority_tols_1(self, command_line, log_level):
        """
        Test solver tolerances specifed via the command line
        take precedence over those specified in the defaults.
        """
        prefix = 'd17weqroeg'
        _, expected = command_line

        solver1 = petscsolve(
            self.eq1, target=self.e,
            options_prefix=prefix
        )
        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(solver1)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', prefix)
        for opt, val in expected[prefix]:
            assert entry.KSPGetTolerances[opt.removeprefix('ksp_')] == val

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_command_line_priority_tols_2(self, command_line, log_level):
        prefix = 'riabfodkj5'
        _, expected = command_line

        solver1 = petscsolve(
            self.eq1, target=self.e,
            options_prefix=prefix
        )
        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(solver1)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', prefix)
        for opt, val in expected[prefix]:
            assert entry.KSPGetTolerances[opt.removeprefix('ksp_')] == val

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_command_line_priority_tols3(self, command_line, log_level):
        """
        Test solver tolerances specifed via the command line
        take precedence over those specified by the `solver_parameters` dict.
        """
        prefix = 'fir8o3lsak'
        _, expected = command_line

        # Set solver parameters that differ both from the defaults and from the
        # values provided on the command line for this prefix (see the `command_line`
        # fixture).
        params = {
            'ksp_rtol': 1e-13,
            'ksp_atol': 1e-35,
            'ksp_divtol': 300000,
            'ksp_max_it': 500
        }

        solver1 = petscsolve(
            self.eq1, target=self.e,
            solver_parameters=params,
            options_prefix=prefix
        )
        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(solver1)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', prefix)
        for opt, val in expected[prefix]:
            assert entry.KSPGetTolerances[opt.removeprefix('ksp_')] == val

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_command_line_priority_ksp_type(self, command_line, log_level):
        """
        Test the solver parameter 'ksp_type' specified via the command line
        take precedence over the one specified in the `solver_parameters` dict.
        """
        prefix = 'zwejklqn25'
        _, expected = command_line

        # Set `ksp_type`` in the solver parameters, which should be overridden
        # by the command line value (which is set to `cg` -
        # see the `command_line` fixture).
        params = {'ksp_type': 'richardson'}

        solver1 = petscsolve(
            self.eq1, target=self.e,
            solver_parameters=params,
            options_prefix=prefix
        )
        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(solver1)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', prefix)
        for _, val in expected[prefix]:
            assert entry.KSPGetType == val
            assert not entry.KSPGetType == params['ksp_type']

    @skipif('petsc')
    def test_command_line_priority_ccode(self, command_line):
        """
        Verify that if an option is set via the command line,
        the corresponding entry in `linear_solve_defaults` or `solver_parameters`
        is not set or cleared in the generated code. (The command line option
        will have already been set in the global PetscOptions database
        during PetscInitialize().)
        """
        prefix = 'qtr2vfvwiu'

        solver = petscsolve(
            self.eq1, target=self.e,
            # Specify a solver parameter that is not set via the
            # command line (see the `command_line` fixture for this prefix).
            solver_parameters={'ksp_rtol': '1e-10'},
            options_prefix=prefix
        )
        with switchconfig(language='petsc'):
            op = Operator(solver)

        set_options_callback = str(op._func_table['SetPetscOptions0'].root)
        clear_options_callback = str(op._func_table['ClearPetscOptions0'].root)

        # Check that the `ksp_rtol` option IS set and cleared explicitly
        # since it is NOT set via the command line.
        assert f'PetscOptionsSetValue(NULL,"-{prefix}_ksp_rtol","1e-10")' \
            in set_options_callback
        assert f'PetscOptionsClearValue(NULL,"-{prefix}_ksp_rtol")' \
            in clear_options_callback

        # Check that the `ksp_divtol` and `ksp_type` options are NOT set
        # or cleared explicitly since they ARE set via the command line.
        assert f'PetscOptionsSetValue(NULL,"-{prefix}_div_tol",' \
            not in set_options_callback
        assert f'PetscOptionsSetValue(NULL,"-{prefix}_ksp_type",' \
            not in set_options_callback
        assert f'PetscOptionsClearValue(NULL,"-{prefix}_div_tol"));' \
            not in clear_options_callback
        assert f'PetscOptionsClearValue(NULL,"-{prefix}_ksp_type"));' \
            not in clear_options_callback

        # Check that options specifed by the `linear_solver_defaults`
        # are still set and cleared
        assert f'PetscOptionsSetValue(NULL,"-{prefix}_ksp_atol",' \
            in set_options_callback
        assert f'PetscOptionsClearValue(NULL,"-{prefix}_ksp_atol"));' \
            in clear_options_callback


class TestHashing:

    @skipif('petsc')
    def test_solveexpr(self):
        grid = Grid(shape=(11, 11), dtype=np.float64)
        functions = [Function(name=n, grid=grid, space_order=2)
                     for n in ['e', 'f']]
        e, f = functions
        eq = Eq(e.laplace, f)

        # Two `petscsolve` calls with different `options_prefix` values
        # should hash differently.
        petsc1 = petscsolve(eq, target=e, options_prefix='poisson1')
        petsc2 = petscsolve(eq, target=e, options_prefix='poisson2')

        assert hash(petsc1.rhs) != hash(petsc2.rhs)
        assert petsc1.rhs != petsc2.rhs

        # Two `petscsolve` calls with the same `options_prefix` but
        # different `solver_parameters` should hash differently.
        petsc3 = petscsolve(
            eq, target=e, solver_parameters={'ksp_type': 'cg'},
            options_prefix='poisson3'
        )
        petsc4 = petscsolve(
            eq, target=e, solver_parameters={'ksp_type': 'richardson'},
            options_prefix='poisson3'
        )
        assert hash(petsc3.rhs) != hash(petsc4.rhs)


class TestGetInfo:
    """
    Test the `get_info` (optional) argument to `petscsolve`.

    This argument can be used independently of the `log_level` to retrieve
    specific information about the solve, such as the number of KSP
    iterations to converge.
    """
    @skipif('petsc')
    def setup_class(self):
        """
        Setup grid, functions and equations shared across
        tests in this class
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)
        self.e, self.f, self.g, self.h = [
            Function(name=n, grid=grid, space_order=2)
            for n in ['e', 'f', 'g', 'h']
        ]
        self.eq1 = Eq(self.e.laplace, self.f)
        self.eq2 = Eq(self.g.laplace, self.h)

    @skipif('petsc')
    def test_get_info(self):
        get_info = ['kspgetiterationnumber', 'snesgetiterationnumber']
        petsc = petscsolve(
            self.eq1, target=self.e, options_prefix='pde1', get_info=get_info
        )
        with switchconfig(language='petsc'):
            op = Operator(petsc)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', 'pde1')

        # Verify that the entry contains only the requested info
        # (since logging is not set)
        assert len(entry) == 2
        assert hasattr(entry, "KSPGetIterationNumber")
        assert hasattr(entry, "SNESGetIterationNumber")

    @skipif('petsc')
    @pytest.mark.parametrize('log_level', ['PERF', 'DEBUG'])
    def test_get_info_with_logging(self, log_level):
        """
        Test that `get_info` works correctly when logging is enabled.
        """
        get_info = ['kspgetiterationnumber']
        petsc = petscsolve(
            self.eq1, target=self.e, options_prefix='pde1', get_info=get_info
        )
        with switchconfig(language='petsc', log_level=log_level):
            op = Operator(petsc)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', 'pde1')

        # With logging enabled, the entry should include both the
        # requested KSP iteration number and additional PETSc info
        # (e.g., SNES iteration count logged at PERF/DEBUG).
        assert len(entry) > 1
        assert hasattr(entry, "KSPGetIterationNumber")
        assert hasattr(entry, "SNESGetIterationNumber")

    @skipif('petsc')
    def test_different_solvers(self):
        """
        Test that `get_info` works correctly when multiple solvers are used
        within the same Operator.
        """
        # Create two `petscsolve` calls with different `get_info` arguments

        get_info_1 = ['kspgetiterationnumber']
        get_info_2 = ['snesgetiterationnumber']

        solver1 = petscsolve(
            self.eq1, target=self.e, options_prefix='pde1', get_info=get_info_1
        )
        solver2 = petscsolve(
            self.eq2, target=self.g, options_prefix='pde2', get_info=get_info_2
        )
        with switchconfig(language='petsc'):
            op = Operator([solver1, solver2])
            summary = op.apply()

        petsc_summary = summary.petsc

        assert len(petsc_summary) == 2
        assert len(petsc_summary.KSPGetIterationNumber) == 1
        assert len(petsc_summary.SNESGetIterationNumber) == 1

        entry1 = petsc_summary.get_entry('section0', 'pde1')
        entry2 = petsc_summary.get_entry('section1', 'pde2')

        assert hasattr(entry1, "KSPGetIterationNumber")
        assert not hasattr(entry1, "SNESGetIterationNumber")

        assert not hasattr(entry2, "KSPGetIterationNumber")
        assert hasattr(entry2, "SNESGetIterationNumber")

    @skipif('petsc')
    def test_case_insensitive(self):
        """
        Test that `get_info` is case insensitive
        """
        # Create a list with mixed cases
        get_info = ['KSPGetIterationNumber', 'snesgetiterationnumber']
        petsc = petscsolve(
            self.eq1, target=self.e, options_prefix='pde1', get_info=get_info
        )
        with switchconfig(language='petsc'):
            op = Operator(petsc)
            summary = op.apply()

        petsc_summary = summary.petsc
        entry = petsc_summary.get_entry('section0', 'pde1')

        assert hasattr(entry, "KSPGetIterationNumber")
        assert hasattr(entry, "SNESGetIterationNumber")

    @skipif('petsc')
    def test_get_ksp_type(self):
        """
        Test that `get_info` can retrieve the KSP type as
        a string.
        """
        get_info = ['kspgettype']
        solver1 = petscsolve(
            self.eq1, target=self.e, options_prefix='poisson1', get_info=get_info
        )
        solver2 = petscsolve(
            self.eq1, target=self.e, options_prefix='poisson2',
            solver_parameters={'ksp_type': 'cg'}, get_info=get_info
        )
        with switchconfig(language='petsc'):
            op = Operator([solver1, solver2])
            summary = op.apply()

        petsc_summary = summary.petsc
        entry1 = petsc_summary.get_entry('section0', 'poisson1')
        entry2 = petsc_summary.get_entry('section1', 'poisson2')

        assert hasattr(entry1, "KSPGetType")
        # Check the type matches the default in linear_solve_defaults
        # since it has not been overridden
        assert entry1.KSPGetType == linear_solve_defaults['ksp_type']
        assert entry1['KSPGetType'] == linear_solve_defaults['ksp_type']
        assert entry1['kspgettype'] == linear_solve_defaults['ksp_type']

        # Test that the KSP type default is correctly overridden by the
        # solver_parameters dictionary passed to solver2
        assert hasattr(entry2, "KSPGetType")
        assert entry2.KSPGetType == 'cg'
        assert entry2['KSPGetType'] == 'cg'
        assert entry2['kspgettype'] == 'cg'


class TestPrinter:

    @skipif('petsc')
    def test_petsc_pi(self):
        """
        Test that sympy.pi is correctly translated to PETSC_PI in the
        generated code.
        """
        grid = Grid(shape=(11, 11), dtype=np.float64)
        e = Function(name='e', grid=grid)
        eq = Eq(e, sp.pi)

        petsc = petscsolve(eq, target=e)

        with switchconfig(language='petsc'):
            op = Operator(petsc)

        assert 'PETSC_PI' in str(op.ccode)
        assert 'M_PI' not in str(op.ccode)


class TestPetscSection:
    """
    These tests validate the use of `PetscSection` (from PETSc) to constrain essential
    boundary nodes by removing them from the linear solver, rather than keeping them in
    the system as trivial equations.

    Users specify essential boundary conditions via the `EssentialBC` equation,
    with a specifed `SubDomain`. When `constrain_bcs=True` is passed to `petscsolve`,
    the Devito compiler generates code that removes these degrees of freedom from
    the linear system. A PETSc requirement is that each MPI rank identifies ALL
    constrained nodes within its local data region, including non-owned (halo) nodes.

    To achieve this, the compiler creates new `EssentialBC`-like equations with
    modified (sub)dimensions (to extend the loop bounds), which are used in two
    callback functions to constrain the nodes. No non-owned (halo) data is indexed
    into - the loops are only used to specify the constrained "local" indices
    on each rank.

    Tests in this class use the following notation:
    - `x`   : a grid point
    - `[]`  : the `SubDomain` specified by the `EssentialBC` (the constrained region)
    - `|`   : an MPI rank boundary
    """
    # first test that the loop generated is correct symbolically..

    # TODO: loop bound modification only needs to happen for subdomain 'middle' type
    # so ensure this happens - by construction left and right subdomains do not
    # cross ranks instead of doing the manual loop bound check - grab the actual
    # iteration from the generated code I think...?

    def _get_loop_bounds(self, shape, so, subdomain):
        grid = Grid(
            shape=shape,
            subdomains=(subdomain,),
            dtype=np.float64
        )

        u = Function(name='u', grid=grid, space_order=so)
        v = Function(name='u', grid=grid, space_order=so)
        bc = Function(name='bc', grid=grid, space_order=so)

        eq = Eq(u, v, subdomain=grid.interior)
        bc = EssentialBC(u, bc, subdomain=subdomain)

        solver = petscsolve([eq, bc], u, constrain_bcs=True)

        with switchconfig(language='petsc'):
            op = Operator(solver)

        args = op.arguments()
        rank = grid.distributor.myrank

        bounds = []
        for _, dim in enumerate(grid.dimensions):
            lb = max(args[f'i{dim.name}_min0'], args[f'{dim.name}_m'] - so)
            ub = min(args[f'i{dim.name}_max0'], args[f'{dim.name}_M'] + so)
            bounds.append((lb, ub))

        return rank, tuple(bounds)

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6, 8])
    def test_constrain_indices_1d_left_halo2(self, mode):
        """halo_size=2, n=24, constrain left side of grid"""

        # 1 rank:
        #                        0
        # [x x x x x x x x x x x x x x x x x x x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 19)}

        # 2 ranks:
        #            0                       1
        # [x x x x x x x x x x x x|x x x x x x x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 13), 1: (-2, 7)}

        # 4 ranks:
        #       0           1           2            3
        # [x x x x x x|x x x x x x|x x x x x x|x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 7), 1: (-2, 7), 2: (-2, 7), 3: (-2, 1)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # [x x x x|x x x x|x x x x|x x x x|x x x x]|x x x x
        # Expected bounds per rank:
        # {0: (0, 5), 1: (-2, 5), 2: (-2, 5), 3: (-2, 5), 4: (-2, 3), 5: (-2, -1)}

        # 8 ranks:
        #    0     1     2     3     4     5     6      7
        # [x x x|x x x|x x x|x x x|x x x|x x x|x x] x|x x x
        # Expected bounds per rank:
        # {0: (0, 4), 1: (-2, 4), 2: (-2, 4), 3: (-2, 4), 4: (-2, 4),
        #  5: (-2, 4), 6: (-2, 1), 7: (-2, -2)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 0, 4)}

        sub = Middle()

        n = 24
        so = 2

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (0, 19),
            },
            2: {
                0: (0, 13),
                1: (-2, 7),
            },
            4: {
                0: (0, 7),
                1: (-2, 7),
                2: (-2, 7),
                3: (-2, 1),
            },
            6: {
                0: (0, 5),
                1: (-2, 5),
                2: (-2, 5),
                3: (-2, 5),
                4: (-2, 3),
                5: (-2, -1)
            },
            8: {
                0: (0, 4),
                1: (-2, 4),
                2: (-2, 4),
                3: (-2, 4),
                4: (-2, 4),
                5: (-2, 4),
                6: (-2, 1),
                7: (-2, -2)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6])
    def test_constrain_indices_1d_left_halo4(self, mode):
        """halo_size=4, n=24, constrain left side of grid"""

        # 1 rank:
        #                        0
        # [x x x x x x x x x x x x x x x x x x x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 19)}

        # 2 ranks:
        #            0                       1
        # [x x x x x x x x x x x x|x x x x x x x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 15), 1: (-4, 7)}

        # 4 ranks:
        #       0           1           2            3
        # [x x x x x x|x x x x x x|x x x x x x|x x] x x x x
        # Expected bounds per rank:
        # {0: (0, 9), 1: (-4, 9), 2: (-4, 7), 3: (-4, 1)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # [x x x x|x x x x|x x x x|x x x x|x x x x]|x x x x
        # Expected bounds per rank:
        # {0: (0, 7), 1: (-4, 7), 2: (-4, 7), 3: (-4, 7), 4: (-4, 3), 5: (-4, -1)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 0, 4)}

        sub = Middle()

        n = 24
        so = 4

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (0, 19),
            },
            2: {
                0: (0, 15),
                1: (-4, 7),
            },
            4: {
                0: (0, 9),
                1: (-4, 9),
                2: (-4, 7),
                3: (-4, 1),
            },
            6: {
                0: (0, 7),
                1: (-4, 7),
                2: (-4, 7),
                3: (-4, 7),
                4: (-4, 3),
                5: (-4, -1)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6, 8])
    def test_constrain_indices_1d_right_halo2(self, mode):
        """halo_size=2, n=24, constrain right side of grid"""

        # 1 rank:
        #                        0
        # x x x [x x x x x x x x x x x x x x x x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 23)}

        # 2 ranks:
        #            0                       1
        # x x x [x x x x x x x x x|x x x x x x x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 13), 1: (-2, 11)}

        # 4 ranks:
        #       0           1           2            3
        # x x x [x x x|x x x x x x|x x x x x x|x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 7), 1: (-2, 7), 2: (-2, 7), 3: (-2, 5)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # x x x [x|x x x x|x x x x|x x x x|x x x x|x x x x]
        # Expected bounds per rank:
        # {0: (3, 5), 1: (-1, 5), 2: (-2, 5), 3: (-2, 5), 4: (-2, 5), 5: (-2, 3)}

        # 8 ranks:
        #    0     1     2     3     4     5     6      7
        # x x x[|x x x|x x x|x x x|x x x|x x x|x x x|x x x]
        # Expected bounds per rank:
        # {0: (3, 4), 1: (0, 4), 2: (-2, 4), 3: (-2, 4), 4: (-2, 4),
        #  5: (-2, 4), 6: (-2, 4), 7: (-2, 2)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 3, 0)}

        sub = Middle()

        n = 24
        so = 2

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (3, 23),
            },
            2: {
                0: (3, 13),
                1: (-2, 11),
            },
            4: {
                0: (3, 7),
                1: (-2, 7),
                2: (-2, 7),
                3: (-2, 5),
            },
            6: {
                0: (3, 5),
                1: (-1, 5),
                2: (-2, 5),
                3: (-2, 5),
                4: (-2, 5),
                5: (-2, 3)
            },
            8: {
                0: (3, 4),
                1: (0, 4),
                2: (-2, 4),
                3: (-2, 4),
                4: (-2, 4),
                5: (-2, 4),
                6: (-2, 4),
                7: (-2, 2)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6])
    def test_constrain_indices_1d_right_halo4(self, mode):
        """halo_size=4, n=24, constrain right side of grid"""

        # 1 rank:
        #                        0
        # x x x [x x x x x x x x x x x x x x x x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 23)}

        # 2 ranks:
        #            0                       1
        # x x x [x x x x x x x x x|x x x x x x x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 15), 1: (-4, 11)}

        # 4 ranks:
        #       0           1           2            3
        # x x x [x x x|x x x x x x|x x x x x x|x x x x x x]
        # Expected bounds per rank:
        # {0: (3, 9), 1: (-3, 9), 2: (-4, 9), 3: (-4, 5)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # x x x [x|x x x x|x x x x|x x x x|x x x x|x x x x]
        # Expected bounds per rank:
        # {0: (3, 7), 1: (-1, 7), 2: (-4, 7), 3: (-4, 7), 4: (-4, 7), 5: (-4, 3)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 3, 0)}

        sub = Middle()

        n = 24
        so = 2

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (3, 23),
            },
            2: {
                0: (3, 13),
                1: (-2, 11),
            },
            4: {
                0: (3, 7),
                1: (-2, 7),
                2: (-2, 7),
                3: (-2, 5),
            },
            6: {
                0: (3, 5),
                1: (-1, 5),
                2: (-2, 5),
                3: (-2, 5),
                4: (-2, 5),
                5: (-2, 3)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6, 8])
    def test_constrain_indices_1d_middle_halo2(self, mode):
        """halo_size=2, n=24, constrain middle of grid"""

        # 1 rank:
        #                        0
        # x x x x x x x x [x x x x x x x x x x x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 18)}

        # 2 ranks:
        #            0                       1
        # x x x x x x x x [x x x x|x x x x x x x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 13), 1: (-2, 6)}

        # 4 ranks:
        #       0           1           2            3
        # x x x x x x|x x [x x x x|x x x x x x|x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 7), 1: (2, 7), 2: (-2, 6), 3: (-2, 0)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # x x x x|x x x x[|x x x x|x x x x|x x x] x|x x x x
        # Expected bounds per rank:
        # {0: (8, 5), 1: (4, 5), 2: (0, 5), 3: (-2, 5), 4: (-2, 2), 5: (-2, -2)}

        # 8 ranks:
        #   0     1     2      3     4     5     6      7
        # x x x|x x x|x x [x|x x x|x x x|x x x|x] x x|x x x
        # Expected bounds per rank:
        # {0: (8, 4), 1: (5, 4), 2: (2, 4), 3: (-1, 4), 4: (-2, 4),
        #  5: (-2, 3), 6: (-2, 0), 7: (-2, -3)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 8, 5)}

        sub = Middle()

        n = 24
        so = 2

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (8, 18),
            },
            2: {
                0: (8, 13),
                1: (-2, 6),
            },
            4: {
                0: (8, 7),
                1: (2, 7),
                2: (-2, 6),
                3: (-2, 0),
            },
            6: {
                0: (8, 5),
                1: (4, 5),
                2: (0, 5),
                3: (-2, 5),
                4: (-2, 2),
                5: (-2, -2)
            },
            8: {
                0: (8, 4),
                1: (5, 4),
                2: (2, 4),
                3: (-1, 4),
                4: (-2, 4),
                5: (-2, 3),
                6: (-2, 0),
                7: (-2, -3)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    @skipif('petsc')
    @pytest.mark.parallel(mode=[1, 2, 4, 6])
    def test_constrain_indices_1d_middle_halo4(self, mode):
        """halo_size=4, n=24, constrain middle of grid"""

        # 1 rank:
        #                        0
        # x x x x x x x x [x x x x x x x x x x x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 18)}

        # 2 ranks:
        #            0                       1
        # x x x x x x x x [x x x x|x x x x x x x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 15), 1: (-4, 6)}

        # 4 ranks:
        #       0           1           2            3
        # x x x x x x|x x [x x x x|x x x x x x|x] x x x x x
        # Expected bounds per rank:
        # {0: (8, 9), 1: (2, 9), 2: (-4, 6), 3: (-4, 0)}

        # 6 ranks:
        #     0       1       2       3       4        5
        # x x x x|x x x x[|x x x x|x x x x|x x x] x|x x x x
        # Expected bounds per rank:
        # {0: (8, 7), 1: (4, 7), 2: (0, 7), 3: (-4, 6), 4: (-4, 2), 5: (-4, -2)}

        class Middle(SubDomain):
            name = 'submiddle'

            def define(self, dimensions):
                x, = dimensions
                return {x: ('middle', 8, 5)}

        sub = Middle()

        n = 24
        so = 4

        rank, bounds = self._get_loop_bounds(
            shape=(n,),
            so=so,
            subdomain=sub
        )
        actual = bounds[0]

        expected = {
            1: {
                0: (8, 18),
            },
            2: {
                0: (8, 15),
                1: (-4, 6),
            },
            4: {
                0: (8, 9),
                1: (2, 9),
                2: (-4, 6),
                3: (-4, 0),
            },
            6: {
                0: (8, 7),
                1: (4, 7),
                2: (0, 7),
                3: (-4, 6),
                4: (-4, 2),
                5: (-4, -2)
            }
        }[mode]

        assert actual == expected[rank], \
            f"rank {rank}: expected {expected[rank]}, got {actual}"

    # TODO: add 2d and 3d tests
