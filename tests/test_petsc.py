import numpy as np
import os
import pytest

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, Operator, switchconfig,
                    configuration)
from devito.ir.iet import (Call, ElementalFunction, Definition, DummyExpr,
                           FindNodes, retrieve_iteration_tree)
from devito.types import Constant, CCompositeObject
from devito.passes.iet.languages.C import CDataManager
from devito.petsc.types import (DM, Mat, LocalVec, PetscMPIInt, KSP,
                                PC, KSPConvergedReason, PETScArray,
                                LinearSolveExpr)
from devito.petsc.solve import PETScSolve, separate_eqn, centre_stencil
from devito.petsc.iet.nodes import Expression
from devito.petsc.initialize import PetscInitialize


@skipif('petsc')
def test_petsc_initialization():
    # TODO: Temporary workaround until PETSc is automatically
    # initialized
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
    lo2 = LocalVec('x')
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
def test_petsc_functions():
    """
    Test C++ support for PETScArrays.
    """
    grid = Grid((2, 2))
    x, y = grid.dimensions

    f0 = Function(name='f', grid=grid, space_order=2, dtype=np.float32)
    f1 = Function(name='f', grid=grid, space_order=2, dtype=np.float64)

    ptr0 = PETScArray(name='ptr0', target=f0)
    ptr1 = PETScArray(name='ptr1', target=f0, is_const=True)
    ptr2 = PETScArray(name='ptr2', target=f1, is_const=True)

    defn0 = Definition(ptr0)
    defn1 = Definition(ptr1)
    defn2 = Definition(ptr2)

    expr = DummyExpr(ptr0.indexed[x, y], ptr1.indexed[x, y] + 1)

    assert str(defn0) == 'float *restrict ptr0_vec;'
    assert str(defn1) == 'const float *restrict ptr1_vec;'
    assert str(defn2) == 'const double *restrict ptr2_vec;'
    assert str(expr) == 'ptr0[x][y] = ptr1[x][y] + 1;'


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
    Test PETScSolve.
    """
    grid = Grid((2, 2))

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)

    petsc = PETScSolve(eqn, f)

    with switchconfig(openmp=False):
        op = Operator(petsc, opt='noop')

    callable_roots = [meta_call.root for meta_call in op._func_table.values()]

    matvec_callback = [root for root in callable_roots if root.name == 'MyMatShellMult_0']

    formrhs_callback = [root for root in callable_roots if root.name == 'FormRHS_0']

    action_expr = FindNodes(Expression).visit(matvec_callback[0])
    rhs_expr = FindNodes(Expression).visit(formrhs_callback[0])

    assert str(action_expr[-1].expr.rhs) == \
        'x_matvec_f[x + 1, y + 2]/lctx->h_x**2' + \
        ' - 2.0*x_matvec_f[x + 2, y + 2]/lctx->h_x**2' + \
        ' + x_matvec_f[x + 3, y + 2]/lctx->h_x**2' + \
        ' + x_matvec_f[x + 2, y + 1]/lctx->h_y**2' + \
        ' - 2.0*x_matvec_f[x + 2, y + 2]/lctx->h_y**2' + \
        ' + x_matvec_f[x + 2, y + 3]/lctx->h_y**2'

    assert str(rhs_expr[-1].expr.rhs) == 'g[x + 2, y + 2]'

    # Check the iteration bounds are correct.
    assert op.arguments().get('x_m') == 0
    assert op.arguments().get('y_m') == 0
    assert op.arguments().get('y_M') == 1
    assert op.arguments().get('x_M') == 1

    assert len(retrieve_iteration_tree(op)) == 0

    # TODO: Remove pragmas from PETSc callback functions
    assert len(matvec_callback[0].parameters) == 3


@skipif('petsc')
def test_multiple_petsc_solves():
    """
    Test multiple PETScSolves.
    """
    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=2)
    g1 = Function(name='g1', grid=grid, space_order=2)

    f2 = Function(name='f2', grid=grid, space_order=2)
    g2 = Function(name='g2', grid=grid, space_order=2)

    eqn1 = Eq(f1.laplace, g1)
    eqn2 = Eq(f2.laplace, g2)

    petsc1 = PETScSolve(eqn1, f1)
    petsc2 = PETScSolve(eqn2, f2)

    with switchconfig(openmp=False):
        op = Operator(petsc1+petsc2, opt='noop')

    callable_roots = [meta_call.root for meta_call in op._func_table.values()]

    # One FormRHS, MatShellMult, FormFunction, PopulateMatContext per solve
    assert len(callable_roots) == 8


@skipif('petsc')
def test_petsc_cast():
    """
    Test casting of PETScArray.
    """
    grid1 = Grid((2))
    grid2 = Grid((2, 2))
    grid3 = Grid((4, 5, 6))

    f1 = Function(name='f1', grid=grid1, space_order=2)
    f2 = Function(name='f2', grid=grid2, space_order=4)
    f3 = Function(name='f3', grid=grid3, space_order=6)

    eqn1 = Eq(f1.laplace, 10)
    eqn2 = Eq(f2.laplace, 10)
    eqn3 = Eq(f3.laplace, 10)

    petsc1 = PETScSolve(eqn1, f1)
    petsc2 = PETScSolve(eqn2, f2)
    petsc3 = PETScSolve(eqn3, f3)

    with switchconfig(openmp=False):
        op1 = Operator(petsc1, opt='noop')
        op2 = Operator(petsc2, opt='noop')
        op3 = Operator(petsc3, opt='noop')

    cb1 = [meta_call.root for meta_call in op1._func_table.values()]
    cb2 = [meta_call.root for meta_call in op2._func_table.values()]
    cb3 = [meta_call.root for meta_call in op3._func_table.values()]

    assert 'float (*restrict x_matvec_f1) = ' + \
        '(float (*)) x_matvec_f1_vec;' in str(cb1[0])
    assert 'float (*restrict x_matvec_f2)[info.gxm] = ' + \
        '(float (*)[info.gxm]) x_matvec_f2_vec;' in str(cb2[0])
    assert 'float (*restrict x_matvec_f3)[info.gym][info.gxm] = ' + \
        '(float (*)[info.gym][info.gxm]) x_matvec_f3_vec;' in str(cb3[0])


@skipif('petsc')
def test_LinearSolveExpr():

    grid = Grid((2, 2))

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f, g.laplace)

    linsolveexpr = LinearSolveExpr(eqn.rhs, target=f)

    # Check the target
    assert linsolveexpr.target == f
    # Check the solver parameters
    assert linsolveexpr.solver_parameters == \
        {'ksp_type': 'gmres', 'pc_type': 'jacobi', 'ksp_rtol': 1e-05,
         'ksp_atol': 1e-50, 'ksp_divtol': 100000.0, 'ksp_max_it': 10000}


@skipif('petsc')
def test_dmda_create():

    grid1 = Grid((2))
    grid2 = Grid((2, 2))
    grid3 = Grid((4, 5, 6))

    f1 = Function(name='f1', grid=grid1, space_order=2)
    f2 = Function(name='f2', grid=grid2, space_order=4)
    f3 = Function(name='f3', grid=grid3, space_order=6)

    eqn1 = Eq(f1.laplace, 10)
    eqn2 = Eq(f2.laplace, 10)
    eqn3 = Eq(f3.laplace, 10)

    petsc1 = PETScSolve(eqn1, f1)
    petsc2 = PETScSolve(eqn2, f2)
    petsc3 = PETScSolve(eqn3, f3)

    with switchconfig(openmp=False):
        op1 = Operator(petsc1, opt='noop')
        op2 = Operator(petsc2, opt='noop')
        op3 = Operator(petsc3, opt='noop')

    assert 'PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        '2,1,2,NULL,&(da_0)));' in str(op1)

    assert 'PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        'DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,2,2,1,1,1,4,NULL,NULL,&(da_0)));' \
        in str(op2)

    assert 'PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,' + \
        'DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,6,5,4' + \
        ',1,1,1,1,6,NULL,NULL,NULL,&(da_0)));' in str(op3)


@skipif('petsc')
def test_cinterface_petsc_struct():

    grid = Grid(shape=(11, 11))
    f = Function(name='f', grid=grid, space_order=2)
    eq = Eq(f.laplace, 10)
    petsc = PETScSolve(eq, f)

    name = "foo"
    with switchconfig(openmp=False):
        op = Operator(petsc, name=name)

    # Trigger the generation of a .c and a .h files
    ccode, hcode = op.cinterface(force=True)

    dirname = op._compiler.get_jit_dir()
    assert os.path.isfile(os.path.join(dirname, "%s.c" % name))
    assert os.path.isfile(os.path.join(dirname, "%s.h" % name))

    ccode = str(ccode)
    hcode = str(hcode)

    assert 'include "%s.h"' % name in ccode

    # The public `struct MatContext` only appears in the header file
    assert 'struct J_0_ctx\n{' not in ccode
    assert 'struct J_0_ctx\n{' in hcode


@skipif('petsc')
@pytest.mark.parametrize('eqn, target, expected', [
    ('Eq(f1.laplace, g1)',
     'f1', ('g1(x, y)', 'Derivative(f1(x, y), (x, 2)) + Derivative(f1(x, y), (y, 2))')),
    ('Eq(g1, f1.laplace)',
     'f1', ('-g1(x, y)', '-Derivative(f1(x, y), (x, 2)) - Derivative(f1(x, y), (y, 2))')),
    ('Eq(g1, f1.laplace)', 'g1',
     ('Derivative(f1(x, y), (x, 2)) + Derivative(f1(x, y), (y, 2))', 'g1(x, y)')),
    ('Eq(f1 + f1.laplace, g1)', 'f1', ('g1(x, y)',
     'f1(x, y) + Derivative(f1(x, y), (x, 2)) + Derivative(f1(x, y), (y, 2))')),
    ('Eq(g1.dx + f1.dx, g1)', 'f1',
     ('g1(x, y) - Derivative(g1(x, y), x)', 'Derivative(f1(x, y), x)')),
    ('Eq(g1.dx + f1.dx, g1)', 'g1',
     ('-Derivative(f1(x, y), x)', '-g1(x, y) + Derivative(g1(x, y), x)')),
    ('Eq(f1 * g1.dx, g1)', 'g1', ('0', 'f1(x, y)*Derivative(g1(x, y), x) - g1(x, y)')),
    ('Eq(f1 * g1.dx, g1)', 'f1', ('g1(x, y)', 'f1(x, y)*Derivative(g1(x, y), x)')),
    ('Eq((f1 * g1.dx).dy, f1)', 'f1',
     ('0', '-f1(x, y) + Derivative(f1(x, y)*Derivative(g1(x, y), x), y)')),
    ('Eq((f1 * g1.dx).dy, f1)', 'g1',
     ('f1(x, y)', 'Derivative(f1(x, y)*Derivative(g1(x, y), x), y)')),
    ('Eq(f2.laplace, g2)', 'g2',
     ('-Derivative(f2(t, x, y), (x, 2)) - Derivative(f2(t, x, y), (y, 2))',
      '-g2(t, x, y)')),
    ('Eq(f2.laplace, g2)', 'f2', ('g2(t, x, y)',
     'Derivative(f2(t, x, y), (x, 2)) + Derivative(f2(t, x, y), (y, 2))')),
    ('Eq(f2.laplace, f2)', 'f2', ('0',
     '-f2(t, x, y) + Derivative(f2(t, x, y), (x, 2)) + Derivative(f2(t, x, y), (y, 2))')),
    ('Eq(f2*g2, f2)', 'f2', ('0', 'f2(t, x, y)*g2(t, x, y) - f2(t, x, y)')),
    ('Eq(f2*g2, f2)', 'g2', ('f2(t, x, y)', 'f2(t, x, y)*g2(t, x, y)')),
    ('Eq(g2*f2.laplace, f2)', 'g2', ('f2(t, x, y)',
     '(Derivative(f2(t, x, y), (x, 2)) + Derivative(f2(t, x, y), (y, 2)))*g2(t, x, y)')),
    ('Eq(f2.forward, f2)', 'f2.forward', ('f2(t, x, y)', 'f2(t + dt, x, y)')),
    ('Eq(f2.forward, f2)', 'f2', ('-f2(t + dt, x, y)', '-f2(t, x, y)')),
    ('Eq(f2.forward.laplace, f2)', 'f2.forward', ('f2(t, x, y)',
     'Derivative(f2(t + dt, x, y), (x, 2)) + Derivative(f2(t + dt, x, y), (y, 2))')),
    ('Eq(f2.forward.laplace, f2)', 'f2',
     ('-Derivative(f2(t + dt, x, y), (x, 2)) - Derivative(f2(t + dt, x, y), (y, 2))',
      '-f2(t, x, y)')),
    ('Eq(f2.laplace + f2.forward.laplace, g2)', 'f2.forward',
     ('g2(t, x, y) - Derivative(f2(t, x, y), (x, 2)) - Derivative(f2(t, x, y), (y, 2))',
      'Derivative(f2(t + dt, x, y), (x, 2)) + Derivative(f2(t + dt, x, y), (y, 2))')),
    ('Eq(g2.laplace, f2 + g2.forward)', 'g2.forward',
     ('f2(t, x, y) - Derivative(g2(t, x, y), (x, 2)) - Derivative(g2(t, x, y), (y, 2))',
      '-g2(t + dt, x, y)'))
])
def test_separate_eqn(eqn, target, expected):
    """
    Test the separate_eqn function.

    This function is called within PETScSolve to decompose the equation
    into the form F(x) = b. This is necessary to utilise the SNES
    interface in PETSc.
    """
    grid = Grid((2, 2))

    so = 2

    f1 = Function(name='f1', grid=grid, space_order=so)  # noqa
    g1 = Function(name='g1', grid=grid, space_order=so)  # noqa

    f2 = TimeFunction(name='f2', grid=grid, space_order=so)  # noqa
    g2 = TimeFunction(name='g2', grid=grid, space_order=so)  # noqa

    b, F, _ = separate_eqn(eval(eqn), eval(target))
    expected_b, expected_F = expected

    assert str(b) == expected_b
    assert str(F) == expected_F


@skipif('petsc')
@pytest.mark.parametrize('eqn, target, expected', [
    ('Eq(f1.laplace, g1).evaluate', 'f1',
        (
            'g1(x, y)',
            '-2.0*f1(x, y)/h_x**2 + f1(x - h_x, y)/h_x**2 + f1(x + h_x, y)/h_x**2 '
            '- 2.0*f1(x, y)/h_y**2 + f1(x, y - h_y)/h_y**2 + f1(x, y + h_y)/h_y**2'
        )),
    ('Eq(g1, f1.laplace).evaluate', 'f1',
        (
            '-g1(x, y)',
            '-(-2.0*f1(x, y)/h_x**2 + f1(x - h_x, y)/h_x**2 + f1(x + h_x, y)/h_x**2) '
            '- (-2.0*f1(x, y)/h_y**2 + f1(x, y - h_y)/h_y**2 + f1(x, y + h_y)/h_y**2)'
        )),
    ('Eq(g1, f1.laplace).evaluate', 'g1',
        (
            '-2.0*f1(x, y)/h_x**2 + f1(x - h_x, y)/h_x**2 + f1(x + h_x, y)/h_x**2 '
            '- 2.0*f1(x, y)/h_y**2 + f1(x, y - h_y)/h_y**2 + f1(x, y + h_y)/h_y**2',
            'g1(x, y)'
        )),
    ('Eq(f1 + f1.laplace, g1).evaluate', 'f1',
        (
            'g1(x, y)',
            '-2.0*f1(x, y)/h_x**2 + f1(x - h_x, y)/h_x**2 + f1(x + h_x, y)/h_x**2 - 2.0'
            '*f1(x, y)/h_y**2 + f1(x, y - h_y)/h_y**2 + f1(x, y + h_y)/h_y**2 + f1(x, y)'
        )),
    ('Eq(g1.dx + f1.dx, g1).evaluate', 'f1',
        (
            '-(-g1(x, y)/h_x + g1(x + h_x, y)/h_x) + g1(x, y)',
            '-f1(x, y)/h_x + f1(x + h_x, y)/h_x'
        )),
    ('Eq(g1.dx + f1.dx, g1).evaluate', 'g1',
        (
            '-(-f1(x, y)/h_x + f1(x + h_x, y)/h_x)',
            '-g1(x, y)/h_x + g1(x + h_x, y)/h_x - g1(x, y)'
        )),
    ('Eq(f1 * g1.dx, g1).evaluate', 'g1',
        (
            '0', '(-g1(x, y)/h_x + g1(x + h_x, y)/h_x)*f1(x, y) - g1(x, y)'
        )),
    ('Eq(f1 * g1.dx, g1).evaluate', 'f1',
        (
            'g1(x, y)', '(-g1(x, y)/h_x + g1(x + h_x, y)/h_x)*f1(x, y)'
        )),
    ('Eq((f1 * g1.dx).dy, f1).evaluate', 'f1',
        (
            '0', '(-1/h_y)*(-g1(x, y)/h_x + g1(x + h_x, y)/h_x)*f1(x, y) '
            '+ (-g1(x, y + h_y)/h_x + g1(x + h_x, y + h_y)/h_x)*f1(x, y + h_y)/h_y '
            '- f1(x, y)'
        )),
    ('Eq((f1 * g1.dx).dy, f1).evaluate', 'g1',
        (
            'f1(x, y)', '(-1/h_y)*(-g1(x, y)/h_x + g1(x + h_x, y)/h_x)*f1(x, y) + '
            '(-g1(x, y + h_y)/h_x + g1(x + h_x, y + h_y)/h_x)*f1(x, y + h_y)/h_y'
        )),
    ('Eq(f2.laplace, g2).evaluate', 'g2',
        (
            '-(-2.0*f2(t, x, y)/h_x**2 + f2(t, x - h_x, y)/h_x**2 + f2(t, x + h_x, y)'
            '/h_x**2) - (-2.0*f2(t, x, y)/h_y**2 + f2(t, x, y - h_y)/h_y**2 + '
            'f2(t, x, y + h_y)/h_y**2)', '-g2(t, x, y)'
        )),
    ('Eq(f2.laplace, g2).evaluate', 'f2',
        (
            'g2(t, x, y)', '-2.0*f2(t, x, y)/h_x**2 + f2(t, x - h_x, y)/h_x**2 + '
            'f2(t, x + h_x, y)/h_x**2 - 2.0*f2(t, x, y)/h_y**2 + f2(t, x, y - h_y)'
            '/h_y**2 + f2(t, x, y + h_y)/h_y**2'
        )),
    ('Eq(f2.laplace, f2).evaluate', 'f2',
        (
            '0', '-2.0*f2(t, x, y)/h_x**2 + f2(t, x - h_x, y)/h_x**2 + '
            'f2(t, x + h_x, y)/h_x**2 - 2.0*f2(t, x, y)/h_y**2 + f2(t, x, y - h_y)/h_y**2'
            ' + f2(t, x, y + h_y)/h_y**2 - f2(t, x, y)'
        )),
    ('Eq(g2*f2.laplace, f2).evaluate', 'g2',
        (
            'f2(t, x, y)', '(-2.0*f2(t, x, y)/h_x**2 + f2(t, x - h_x, y)/h_x**2 + '
            'f2(t, x + h_x, y)/h_x**2 - 2.0*f2(t, x, y)/h_y**2 + f2(t, x, y - h_y)/h_y**2'
            ' + f2(t, x, y + h_y)/h_y**2)*g2(t, x, y)'
        )),
    ('Eq(f2.forward.laplace, f2).evaluate', 'f2.forward',
        (
            'f2(t, x, y)', '-2.0*f2(t + dt, x, y)/h_x**2 + f2(t + dt, x - h_x, y)/h_x**2'
            ' + f2(t + dt, x + h_x, y)/h_x**2 - 2.0*f2(t + dt, x, y)/h_y**2 + '
            'f2(t + dt, x, y - h_y)/h_y**2 + f2(t + dt, x, y + h_y)/h_y**2'
        )),
    ('Eq(f2.forward.laplace, f2).evaluate', 'f2',
        (
            '-(-2.0*f2(t + dt, x, y)/h_x**2 + f2(t + dt, x - h_x, y)/h_x**2 + '
            'f2(t + dt, x + h_x, y)/h_x**2) - (-2.0*f2(t + dt, x, y)/h_y**2 + '
            'f2(t + dt, x, y - h_y)/h_y**2 + f2(t + dt, x, y + h_y)/h_y**2)',
            '-f2(t, x, y)'
        )),
    ('Eq(f2.laplace + f2.forward.laplace, g2).evaluate', 'f2.forward',
        (
            '-(-2.0*f2(t, x, y)/h_x**2 + f2(t, x - h_x, y)/h_x**2 + f2(t, x + h_x, y)/'
            'h_x**2) - (-2.0*f2(t, x, y)/h_y**2 + f2(t, x, y - h_y)/h_y**2 + '
            'f2(t, x, y + h_y)/h_y**2) + g2(t, x, y)', '-2.0*f2(t + dt, x, y)/h_x**2 + '
            'f2(t + dt, x - h_x, y)/h_x**2 + f2(t + dt, x + h_x, y)/h_x**2 - 2.0*'
            'f2(t + dt, x, y)/h_y**2 + f2(t + dt, x, y - h_y)/h_y**2 + '
            'f2(t + dt, x, y + h_y)/h_y**2'
        )),
    ('Eq(g2.laplace, f2 + g2.forward).evaluate', 'g2.forward',
        (
            '-(-2.0*g2(t, x, y)/h_x**2 + g2(t, x - h_x, y)/h_x**2 + '
            'g2(t, x + h_x, y)/h_x**2) - (-2.0*g2(t, x, y)/h_y**2 + g2(t, x, y - h_y)'
            '/h_y**2 + g2(t, x, y + h_y)/h_y**2) + f2(t, x, y)', '-g2(t + dt, x, y)'
        ))
])
def test_separate_eval_eqn(eqn, target, expected):
    """
    Test the separate_eqn function on pre-evaluated equations.
    This ensures that evaluated equations can be passed to PETScSolve,
    allowing users to modify stencils for specific boundary conditions,
    such as implementing free surface boundary conditions.
    """
    grid = Grid((2, 2))

    so = 2

    f1 = Function(name='f1', grid=grid, space_order=so)  # noqa
    g1 = Function(name='g1', grid=grid, space_order=so)  # noqa

    f2 = TimeFunction(name='f2', grid=grid, space_order=so)  # noqa
    g2 = TimeFunction(name='g2', grid=grid, space_order=so)  # noqa

    b, F, _ = separate_eqn(eval(eqn), eval(target))
    expected_b, expected_F = expected

    assert str(b) == expected_b
    assert str(F) == expected_F


@skipif('petsc')
@pytest.mark.parametrize('expr, so, target, expected', [
    ('f1.laplace', 2, 'f1', '-2.0*f1(x, y)/h_y**2 - 2.0*f1(x, y)/h_x**2'),
    ('f1 + f1.laplace', 2, 'f1',
     'f1(x, y) - 2.0*f1(x, y)/h_y**2 - 2.0*f1(x, y)/h_x**2'),
    ('g1.dx + f1.dx', 2, 'f1', '-f1(x, y)/h_x'),
    ('10 + f1.dx2', 2, 'g1', '0'),
    ('(f1 * g1.dx).dy', 2, 'f1',
     '(-1/h_y)*(-g1(x, y)/h_x + g1(x + h_x, y)/h_x)*f1(x, y)'),
    ('(f1 * g1.dx).dy', 2, 'g1', '-(-1/h_y)*f1(x, y)*g1(x, y)/h_x'),
    ('f2.laplace', 2, 'f2', '-2.0*f2(t, x, y)/h_y**2 - 2.0*f2(t, x, y)/h_x**2'),
    ('f2*g2', 2, 'f2', 'f2(t, x, y)*g2(t, x, y)'),
    ('g2*f2.laplace', 2, 'f2',
     '(-2.0*f2(t, x, y)/h_y**2 - 2.0*f2(t, x, y)/h_x**2)*g2(t, x, y)'),
    ('f2.forward', 2, 'f2.forward', 'f2(t + dt, x, y)'),
    ('f2.forward.laplace', 2, 'f2.forward',
     '-2.0*f2(t + dt, x, y)/h_y**2 - 2.0*f2(t + dt, x, y)/h_x**2'),
    ('f2.laplace + f2.forward.laplace', 2, 'f2.forward',
     '-2.0*f2(t + dt, x, y)/h_y**2 - 2.0*f2(t + dt, x, y)/h_x**2'),
    ('f2.laplace + f2.forward.laplace', 2,
     'f2', '-2.0*f2(t, x, y)/h_y**2 - 2.0*f2(t, x, y)/h_x**2'),
    ('f2.laplace', 4, 'f2', '-2.5*f2(t, x, y)/h_y**2 - 2.5*f2(t, x, y)/h_x**2'),
    ('f2.laplace + f2.forward.laplace', 4, 'f2.forward',
     '-2.5*f2(t + dt, x, y)/h_y**2 - 2.5*f2(t + dt, x, y)/h_x**2'),
    ('f2.laplace + f2.forward.laplace', 4, 'f2',
     '-2.5*f2(t, x, y)/h_y**2 - 2.5*f2(t, x, y)/h_x**2'),
    ('f2.forward*f2.forward.laplace', 4, 'f2.forward',
     '(-2.5*f2(t + dt, x, y)/h_y**2 - 2.5*f2(t + dt, x, y)/h_x**2)*f2(t + dt, x, y)')
])
def test_centre_stencil(expr, so, target, expected):
    """
    Test extraction of centre stencil from an equation.
    """
    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=so)  # noqa
    g1 = Function(name='g1', grid=grid, space_order=so)  # noqa

    f2 = TimeFunction(name='f2', grid=grid, space_order=so)  # noqa
    g2 = TimeFunction(name='g2', grid=grid, space_order=so)  # noqa

    centre = centre_stencil(eval(expr), eval(target))

    assert str(centre) == expected


@skipif('petsc')
def test_callback_arguments():
    """
    Test the arguments of each callback function.
    """
    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=2)
    g1 = Function(name='g1', grid=grid, space_order=2)

    eqn1 = Eq(f1.laplace, g1)

    petsc1 = PETScSolve(eqn1, f1)

    with switchconfig(openmp=False):
        op = Operator(petsc1)

    mv = op._func_table['MyMatShellMult_0'].root
    ff = op._func_table['FormFunction_0'].root

    assert len(mv.parameters) == 3
    assert len(ff.parameters) == 4

    assert str(mv.parameters) == '(J_0, X_global_0, Y_global_0)'
    assert str(ff.parameters) == '(snes_0, X_global_0, F_global_0, dummy)'


@skipif('petsc')
def test_petsc_struct():

    grid = Grid((2, 2))

    f1 = Function(name='f1', grid=grid, space_order=2)
    g1 = Function(name='g1', grid=grid, space_order=2)

    mu1 = Constant(name='mu1', value=2.0)
    mu2 = Constant(name='mu2', value=2.0)

    eqn1 = Eq(f1.laplace, g1*mu1)
    petsc1 = PETScSolve(eqn1, f1)

    eqn2 = Eq(f1, g1*mu2)

    with switchconfig(openmp=False):
        op = Operator([eqn2] + petsc1)

    arguments = op.arguments()

    # Check mu1 and mu2 in arguments
    assert 'mu1' in arguments
    assert 'mu2' in arguments

    # Check mu1 and mu2 in op.parameters
    assert mu1 in op.parameters
    assert mu2 in op.parameters

    # Check PETSc struct not in op.parameters
    assert all(not isinstance(i, CCompositeObject) for i in op.parameters)


@skipif('petsc')
def test_apply():

    grid = Grid(shape=(13, 13), dtype=np.float64)

    pn = Function(name='pn', grid=grid, space_order=2, dtype=np.float64)
    rhs = Function(name='rhs', grid=grid, space_order=2, dtype=np.float64)
    mu = Constant(name='mu', value=2.0)

    eqn = Eq(pn.laplace*mu, rhs, subdomain=grid.interior)

    petsc = PETScSolve(eqn, pn)

    # Build the op
    op = Operator(petsc)

    # Check the Operator runs without errors. Not verifying output for
    # now. Need to consolidate BC implementation
    op.apply()

    # Verify that users can override `mu`
    mu_new = Constant(name='mu_new', value=4.0)
    op.apply(mu=mu_new)


@skipif('petsc')
def test_petsc_frees():

    grid = Grid((2, 2))

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)
    petsc = PETScSolve(eqn, f)

    with switchconfig(openmp=False):
        op = Operator(petsc)

    frees = op.body.frees

    # Check the frees appear in the following order
    assert str(frees[0]) == 'PetscCall(VecDestroy(&(b_global_0)));'
    assert str(frees[1]) == 'PetscCall(VecDestroy(&(x_global_0)));'
    assert str(frees[2]) == 'PetscCall(MatDestroy(&(J_0)));'
    assert str(frees[3]) == 'PetscCall(SNESDestroy(&(snes_0)));'
    assert str(frees[4]) == 'PetscCall(DMDestroy(&(da_0)));'


@skipif('petsc')
def test_calls_to_callbacks():

    grid = Grid((2, 2))

    f = Function(name='f', grid=grid, space_order=2)
    g = Function(name='g', grid=grid, space_order=2)

    eqn = Eq(f.laplace, g)
    petsc = PETScSolve(eqn, f)

    with switchconfig(openmp=False):
        op = Operator(petsc)

    ccode = str(op.ccode)

    assert '(void (*)(void))MyMatShellMult_0' in ccode
    assert 'PetscCall(SNESSetFunction(snes_0,NULL,FormFunction_0,NULL));' in ccode


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
    grid = Grid((11, 11))
    u1 = TimeFunction(name='u1', grid=grid, space_order=2, dtype=np.float32)
    eq1 = Eq(u1.dt, u1.laplace, subdomain=grid.interior)
    petsc1 = PETScSolve(eq1, u1.forward)

    with switchconfig(openmp=False):
        op1 = Operator(petsc1)

    # Verify the case with modulo time stepping
    assert 'float * start_ptr_0 = t1*localsize_0 + (float*)(u1_vec->data);' in str(op1)

    # Verify the case with no modulo time stepping
    u2 = TimeFunction(name='u2', grid=grid, space_order=2, dtype=np.float32, save=5)
    eq2 = Eq(u2.dt, u2.laplace, subdomain=grid.interior)
    petsc2 = PETScSolve(eq2, u2.forward)

    with switchconfig(openmp=False):
        op2 = Operator(petsc2)

    assert 'float * start_ptr_0 = (time + 1)*localsize_0 + ' + \
        '(float*)(u2_vec->data);' in str(op2)


@skipif('petsc')
def test_time_loop():
    """
    Verify the following:
    - Modulo dimensions are correctly assigned and updated in the PETSc struct
    at each time step.
    - Only assign/update the modulo dimensions required by any of the
    PETSc callback functions.
    """
    grid = Grid((11, 11))

    # Modulo time stepping
    u1 = TimeFunction(name='u1', grid=grid, space_order=2)
    v1 = Function(name='v1', grid=grid, space_order=2)
    eq1 = Eq(v1.laplace, u1)
    petsc1 = PETScSolve(eq1, v1)
    with switchconfig(openmp=False):
        op1 = Operator(petsc1)
    body1 = str(op1.body)
    rhs1 = str(op1._func_table['FormRHS_0'].root.ccode)

    assert 'ctx0.t0 = t0' in body1
    assert 'ctx0.t1 = t1' not in body1
    assert 'lctx->t0' in rhs1
    assert 'lctx->t1' not in rhs1

    # Non-modulo time stepping
    u2 = TimeFunction(name='u2', grid=grid, space_order=2, save=5)
    v2 = Function(name='v2', grid=grid, space_order=2, save=5)
    eq2 = Eq(v2.laplace, u2)
    petsc2 = PETScSolve(eq2, v2)
    with switchconfig(openmp=False):
        op2 = Operator(petsc2)
    body2 = str(op2.body)
    rhs2 = str(op2._func_table['FormRHS_0'].root.ccode)

    assert 'ctx0.time = time' in body2
    assert 'lctx->time' in rhs2

    # Modulo time stepping with more than one time step
    # used in one of the callback functions
    eq3 = Eq(v1.laplace, u1 + u1.forward)
    petsc3 = PETScSolve(eq3, v1)
    with switchconfig(openmp=False):
        op3 = Operator(petsc3)
    body3 = str(op3.body)
    rhs3 = str(op3._func_table['FormRHS_0'].root.ccode)

    assert 'ctx0.t0 = t0' in body3
    assert 'ctx0.t1 = t1' in body3
    assert 'lctx->t0' in rhs3
    assert 'lctx->t1' in rhs3

    # Multiple petsc solves within the same time loop
    v2 = Function(name='v2', grid=grid, space_order=2)
    eq4 = Eq(v1.laplace, u1)
    petsc4 = PETScSolve(eq4, v1)
    eq5 = Eq(v2.laplace, u1)
    petsc5 = PETScSolve(eq5, v2)
    with switchconfig(openmp=False):
        op4 = Operator(petsc4 + petsc5)
    body4 = str(op4.body)

    assert 'ctx0.t0 = t0' in body4
    assert body4.count('ctx0.t0 = t0') == 1
