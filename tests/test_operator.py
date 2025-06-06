from itertools import permutations

import numpy as np
import sympy

import pytest
from conftest import assert_structure, skipif
from devito import (Grid, Eq, Operator, Constant, Function, TimeFunction,
                    SparseFunction, SparseTimeFunction, Dimension, error, SpaceDimension,
                    NODE, CELL, dimensions, configuration, TensorFunction,
                    TensorTimeFunction, VectorFunction, VectorTimeFunction,
                    div, grad, switchconfig, exp)
from devito import  Inc, Le, Lt, Ge, Gt  # noqa
from devito.exceptions import InvalidOperator
from devito.finite_differences.differentiable import diff2sympy
from devito.ir.equations import ClusterizedEq
from devito.ir.equations.algorithms import lower_exprs
from devito.ir.iet import (Callable, Conditional, Expression, Iteration, TimedList,
                           FindNodes, IsPerfectIteration, retrieve_iteration_tree,
                           FindSymbols)
from devito.ir.support import Any, Backward, Forward
from devito.passes.iet.languages.C import CDataManager
from devito.symbolics import ListInitializer, indexify, retrieve_indexed
from devito.tools import flatten, powerset, timed_region
from devito.types import Array, Barrier, CustomDimension, Indirection, Scalar, Symbol


def dimify(dimensions):
    assert isinstance(dimensions, str)
    return tuple(SpaceDimension(name=i) for i in dimensions.split())


def symbol(name, dimensions, value=0., shape=(3, 5), mode='function'):
    """Short-cut for symbol creation to test "function"
    and "indexed" API."""
    assert(mode in ['function', 'indexed'])
    s = Function(name=name, dimensions=dimensions, shape=shape)
    s.data_with_halo[:] = value
    return s.indexify() if mode == 'indexed' else s


class TestOperatorSetup:

    def test_platform_compiler_language(self):
        """
        Test code generation when ``platform``, ``compiler`` and ``language``
        are explicitly supplied to an Operator, thus bypassing the global values
        stored in ``configuration``.
        """
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        # Unrecognised platform name -> exception
        try:
            Operator(Eq(u, u + 1), platform='asga')
            assert False
        except InvalidOperator:
            assert True

        # Operator with auto-detected CPU platform (ie, `configuration['platform']`)
        op1 = Operator(Eq(u, u + 1))
        # Operator with preset platform
        op2 = Operator(Eq(u, u + 1), platform='nvidiaX')

        # Definitely should be
        assert str(op1) != str(op2)

        # `op2` should have OpenMP offloading code
        assert '#pragma omp target' in str(op2)

        # `op2` uses a user-supplied `platform`, so the Compiler gets rebuilt
        # to make sure it can JIT for the target platform
        assert op1._compiler is not op2._compiler

        # The compiler itself can also be passed explicitly ...
        Operator(Eq(u, u + 1), platform='nvidiaX', compiler='gcc')
        # ... but it will raise an exception if an unknown one
        try:
            Operator(Eq(u, u + 1), platform='nvidiaX', compiler='asf')
            assert False
        except InvalidOperator:
            assert True

        # Now with explicit platform *and* language
        op3 = Operator(Eq(u, u + 1), platform='nvidiaX', language='openacc')
        assert '#pragma acc parallel' in str(op3)
        assert op3._compiler is not configuration['compiler']
        assert (op3._compiler.__class__.__name__ ==
                configuration['compiler'].__class__.__name__)

        # Unsupported combination of `platform` and `language` should throw an error
        try:
            Operator(Eq(u, u + 1), platform='bdw', language='openacc')
            assert False
        except InvalidOperator:
            assert True

        # Check that local config takes precedence over global config
        op4 = switchconfig(language='openmp')(Operator)(Eq(u, u + 1), language='C')
        assert '#pragma omp for' not in str(op4)

    def test_opt_options(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        # Unknown pass
        try:
            Operator(Eq(u, u + 1), opt=('aaa'))
            assert False
        except InvalidOperator:
            assert True

        # Unknown optimization option
        try:
            Operator(Eq(u, u + 1), opt=('advanced', {'aaa': 1}))
            assert False
        except InvalidOperator:
            assert True

    def test_compiler_uniqueness(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        eqns = [Eq(u.forward, u + 1)]

        op0 = Operator(eqns)
        op1 = Operator(eqns)
        op2 = Operator(eqns, compiler='gcc')

        assert op0._compiler is not op1._compiler
        assert op0._compiler is not op2._compiler
        assert op1._compiler is not op2._compiler


class TestCodeGen:

    def test_parameters(self):
        """Tests code generation for Operator parameters."""
        grid = Grid(shape=(3,))
        a_dense = Function(name='a_dense', grid=grid)
        const = Constant(name='constant')
        eqn = Eq(a_dense, a_dense + 2.*const)
        op = Operator(eqn, opt=('advanced', {'openmp': False}))
        assert len(op.parameters) == 5
        assert op.parameters[0].name == 'a_dense'
        assert op.parameters[0].is_AbstractFunction
        assert op.parameters[1].name == 'constant'
        assert op.parameters[1].is_Symbol
        assert op.parameters[2].name == 'x_M'
        assert op.parameters[2].is_Symbol
        assert op.parameters[3].name == 'x_m'
        assert op.parameters[3].is_Symbol
        assert op.parameters[4].name == 'timers'
        assert op.parameters[4].is_Object
        assert 'a_dense[x + 1] = 2.0F*constant + a_dense[x + 1]' in str(op)

    @pytest.mark.parametrize('expr, so, to, expected', [
        ('Eq(u.forward,u+1)', 0, 1, 'Eq(u[t+1,x,y,z],u[t,x,y,z]+1)'),
        ('Eq(u.forward,u+1)', 1, 1, 'Eq(u[t+1,x+1,y+1,z+1],u[t,x+1,y+1,z+1]+1)'),
        ('Eq(u.forward,u+1)', 1, 2, 'Eq(u[t+1,x+1,y+1,z+1],u[t,x+1,y+1,z+1]+1)'),
        ('Eq(u.forward,u+u.backward + m)', 8, 2,
         'Eq(u[t+1,x+8,y+8,z+8],m[x,y,z]+u[t,x+8,y+8,z+8]+u[t-1,x+8,y+8,z+8])')
    ])
    def test_index_shifting(self, expr, so, to, expected):
        """Tests that array accesses get properly shifted based on the halo and
        padding regions extent."""
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)  # noqa
        m = Function(name='m', grid=grid, space_order=0)  # noqa

        expr = eval(expr)

        with timed_region('x'):
            expr = Operator._lower_exprs([expr], options={})[0]

        assert str(expr).replace(' ', '') == expected

    @pytest.mark.parametrize('expr, so, expected', [
        ('Lt(0.1*(g1 + g2), 0.2*(g1 + g2))', 0,
         '0.1*g1[x,y]+0.1*g2[x,y]<0.2*g1[x,y]+0.2*g2[x,y]'),
        ('Le(0.1*(g1 + g2), 0.2*(g1 + g2))', 1,
         '0.1*g1[x+1,y+1]+0.1*g2[x+1,y+1]<=0.2*g1[x+1,y+1]+0.2*g2[x+1,y+1]'),
        ('Ge(0.1*(g1 + g2), 0.2*(g1 + g2))', 2,
         '0.1*g1[x+2,y+2]+0.1*g2[x+2,y+2]>=0.2*g1[x+2,y+2]+0.2*g2[x+2,y+2]'),
        ('Gt(0.1*(g1 + g2), 0.2*(g1 + g2))', 4,
         '0.1*g1[x+4,y+4]+0.1*g2[x+4,y+4]>0.2*g1[x+4,y+4]+0.2*g2[x+4,y+4]'),
    ])
    def test_relationals_index_shifting(self, expr, so, expected):

        grid = Grid(shape=(3, 3))
        g1 = Function(name='g1', grid=grid, space_order=so)  # noqa
        g2 = Function(name='g2', grid=grid, space_order=so)  # noqa
        expr = eval(expr)
        expr = lower_exprs(expr)

        assert str(expr).replace(' ', '') == expected

    @pytest.mark.parametrize('expr,exp_uindices,exp_mods', [
        ('Eq(v.forward, u[0, x, y, z] + v + 1)', [(0, 5), (2, 5)], {'v': 5}),
        ('Eq(v.forward, u + v + 1)', [(0, 5), (2, 5), (0, 2)], {'v': 5, 'u': 2}),
    ])
    def test_multiple_steppers(self, expr, exp_uindices, exp_mods):
        """Tests generation of multiple, mixed time stepping indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid, time_order=4)  # noqa

        op = Operator(eval(expr), opt='noop')

        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1
        time_iter = time_iter[0]

        # Check uindices in Iteration header
        signatures = [(i._offset, i._modulo) for i in time_iter.uindices]
        assert len(signatures) == len(exp_uindices)
        exp_uindices = [(time + i, j) for i, j in exp_uindices]
        assert all(i in signatures for i in exp_uindices)

        # Check uindices within each TimeFunction
        exprs = [i.expr for i in FindNodes(Expression).visit(op)]
        assert(i.indices[i.function._time_position].modulo == exp_mods[i.function.name]
               for i in flatten(retrieve_indexed(i) for i in exprs))

    def test_lower_stepping_dims_with_mutiple_iterations(self):
        """
        Test lowering SteppingDimensions for a time dimension with
        more than one iteration loop with different ModuloDimensions.
        MFE for issue #1486
        """
        grid = Grid(shape=(4, 4))

        f = Function(name="f", grid=grid, space_order=4)
        g = Function(name="g", grid=grid, space_order=4)
        h = TimeFunction(name="h", grid=grid, space_order=4, time_order=2)

        f.data[:] = 0.0
        h.data[:] = 0.0

        eqn = [Eq(f, h + 1), Eq(g, f),
               Eq(h.forward, h + g + 1)]

        op = Operator(eqn)

        for iter in [i for i in FindNodes(Iteration).visit(op) if i.dim.is_Time]:
            exprtimeindices = set([a.indices[a.function._time_position] for
                                   expr in FindNodes(Expression).visit(iter) for
                                   a in retrieve_indexed(expr.expr) if
                                   isinstance(a.function, TimeFunction)])
            # Check if iteration time indices match with expressions time indices
            assert (exprtimeindices == set(iter.uindices))
            # Check if expressions time indices are modulo dimensions
            assert(all([i.is_Modulo for i in exprtimeindices]))

        op.apply(time_M=10)

        assert np.all(h.data[0, :] == 18)
        assert np.all(h.data[1, :] == 20)
        assert np.all(h.data[2, :] == 22)

    @skipif('device')
    def test_timedlist_wraps_time_if_parallel(self):
        """
        Test that if the time loop is parallel, then it must be wrapped by a
        Section (and consequently by a TimedList).
        """
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, save=3)

        op = Operator(Eq(u, u + 1), platform='intel64')

        assert op.body.body[1].body[0].is_Section
        assert isinstance(op.body.body[1].body[0].body[0], TimedList)
        timedlist = op.body.body[1].body[0].body[0]
        if 'openmp' in configuration['language']:
            ompreg = timedlist.body[0]
            assert ompreg.body[0].dim is grid.time_dim
        else:
            timedlist.body[0].dim is grid.time_dim

    def test_nested_lowering(self):
        """
        Tests that deeply nested (depth > 2) functions over subdomains are lowered.
        """
        grid = Grid(shape=(4, 4), dtype=np.int32)
        x, y = grid.dimensions
        x0, y0 = dimensions('x0 y0')

        u0 = Function(name="u0", grid=grid)
        u1 = Function(name="u1", shape=grid.shape, dimensions=(x0, y0), dtype=np.int32)
        u2 = Function(name="u2", grid=grid)

        u0.data[:2, :2] = 1
        u0.data[2:, 2:] = 2
        u1.data[:, :] = 1
        u2.data[:, :] = 1

        eq0 = Eq(u0, u0[u1[x0+1, y0+2], u2[x, u2]], subdomain=grid.interior)
        eq1 = Eq(u0, u0[u1[x0+1, y0+2], u2[x, u2[x, y]]], subdomain=grid.interior)

        op0 = Operator(eq0)
        op1 = Operator(eq1)
        op0.apply()

        # Check they indeed produced the same code
        assert str(op0.ccode) == str(op1.ccode)

        # Also check for numerical correctness
        assert np.all(u0.data[0, 3] == 0) and np.all(u0.data[3, 0] == 0)
        assert np.all(u0.data[:2, :2] == 1) and np.all(u0.data[1:3, 1:3] == 1)
        assert np.all(u0.data[2:3, 3] == 2) and np.all(u0.data[3, 2:3] == 2)

    def test_nested_lowering_indexify(self):
        """
        Tests that nested function are lowered if only used as index.
        """
        grid = Grid(shape=(4, 4), dtype=np.int32)
        x, y = grid.dimensions

        u0 = Function(name="u0", grid=grid)
        u1 = Function(name="u1", grid=grid)
        u2 = Function(name="u2", grid=grid)

        u0.data[:, :] = 2
        u1.data[:, :] = 1
        u2.data[:, :] = 1

        # Function as index only
        eq0 = Eq(u0._subs(x, u1), 2*u0)
        # Function as part of expression as index only
        eq1 = Eq(u0._subs(x, u1._subs(y, u2) + 1), 4*u0)

        op0 = Operator(eq0)
        op0.apply()
        op1 = Operator(eq1)
        op1.apply()
        assert np.all(np.all(u0.data[i, :] == 2) for i in [0, 3])
        assert np.all(u0.data[1, :] == 4)
        assert np.all(u0.data[2, :] == 8)

    def test_scalar_type(self):
        grid = Grid(shape=(4, 4), dtype=np.float32)
        u = Function(name='u', grid=grid, space_order=4)

        eq = Eq(u, u.laplace)
        op0 = Operator(eq)
        scalars = [s for s in FindSymbols().visit(op0) if s.name.startswith('r')]
        assert all(s.dtype == np.float32 for s in scalars)

        op1 = Operator(eq, opt=('advanced', {'scalar-min-type': np.float64}))
        scalars = [s for s in FindSymbols().visit(op1) if s.name.startswith('r')]
        assert all(s.dtype == np.float64 for s in scalars)


class TestArithmetic:

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a, a + b + 5.)', 10.),
        ('Eq(a, b - a)', 1.),
        ('Eq(a, 4 * (b * a))', 24.),
        ('Eq(a, (6. / b) + (8. * a))', 18.),
    ])
    @pytest.mark.parametrize('mode', ['function'])
    def test_flat(self, expr, result, mode):
        """Tests basic point-wise arithmetic on two-dimensional data"""
        i, j = dimify('i j')
        a = symbol(name='a', dimensions=(i, j), value=2., mode=mode)
        b = symbol(name='b', dimensions=(i, j), value=3., mode=mode)
        fa = a.base.function if mode == 'indexed' else a
        fb = b.base.function if mode == 'indexed' else b

        eqn = eval(expr)
        Operator(eqn)(a=fa, b=fb)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a, a + b + 5.)', 10.),
        ('Eq(a, b - a)', 1.),
        ('Eq(a, 4 * (b * a))', 24.),
        ('Eq(a, (6. / b) + (8. * a))', 18.),
    ])
    @pytest.mark.parametrize('mode', ['function', 'indexed'])
    def test_deep(self, expr, result, mode):
        """Tests basic point-wise arithmetic on multi-dimensional data"""
        i, j, k, l = dimify('i j k l')
        a = symbol(name='a', dimensions=(i, j, k, l), shape=(3, 5, 7, 6),
                   value=2., mode=mode)
        b = symbol(name='b', dimensions=(j, k), shape=(5, 7),
                   value=3., mode=mode)
        fa = a.base.function if mode == 'indexed' else a
        fb = b.base.function if mode == 'indexed' else b

        eqn = eval(expr)
        Operator(eqn)(a=fa, b=fb)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[j, l], a[j - 1 , l] + 1.)',
         np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[1]),
        ('Eq(a[j, l], a[j, l - 1] + 1.)',
         np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[0]),
    ])
    def test_indexed_increment(self, expr, result):
        """Tests point-wise increments with stencil offsets in one dimension"""
        j, l = dimify('j l')
        a = symbol(name='a', dimensions=(j, l), value=1., shape=(5, 6),
                   mode='indexed').base
        fa = a.function
        fa.data[:] = 0.

        eqn = eval(expr)
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[j, l], b[j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_stencil(self, expr, result):
        """Test point-wise arithmetic with stencil offsets across two
        functions in indexed expression format"""
        j, l = dimify('j l')
        a = symbol(name='a', dimensions=(j, l), value=0., shape=(5, 6),
                   mode='indexed').base
        fa = a.function
        b = symbol(name='b', dimensions=(j, l), value=2., shape=(5, 6),
                   mode='indexed').base
        fb = b.function

        eqn = eval(expr)
        Operator(eqn)(a=fa, b=fb)
        assert np.allclose(fa.data[1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[1, j, l], a[0, j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_buffered(self, expr, result):
        """Test point-wise arithmetic with stencil offsets across a single
        functions with buffering dimension in indexed expression format"""
        i, j, l = dimify('i j l')
        a = symbol(name='a', dimensions=(i, j, l), value=2., shape=(3, 5, 6),
                   mode='indexed').base
        fa = a.function

        eqn = eval(expr)
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[1, j, l], a[0, j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_open_loops(self, expr, result):
        """Test point-wise arithmetic with stencil offsets and open loop
        boundaries in indexed expression format"""
        i, j, l = dimify('i j l')
        a = Function(name='a', dimensions=(i, j, l), shape=(3, 5, 6))
        fa = a.function
        fa.data[0, :, :] = 2.

        eqn = eval(expr)
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    def test_indexed_w_indirections(self):
        """Test point-wise arithmetic with indirectly indexed Functions."""
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        p_poke = Dimension('p_src')
        d = Dimension('d')

        npoke = 1

        u = Function(name='u', grid=grid, space_order=0)
        coordinates = Function(name='coordinates', dimensions=(p_poke, d),
                               shape=(npoke, grid.dim), space_order=0, dtype=np.int32)
        coordinates.data[0, 0] = 4
        coordinates.data[0, 1] = 3

        poke_eq = Eq(u[coordinates[p_poke, 0], coordinates[p_poke, 1]], 1.0)
        op = Operator(poke_eq)
        op.apply()

        ix, iy = np.where(u.data == 1.)
        assert len(ix) == len(iy) == 1
        assert ix[0] == 4 and iy[0] == 3
        assert np.all(u.data[0:3] == 0.) and np.all(u.data[5:] == 0.)
        assert np.all(u.data[:, 0:3] == 0.) and np.all(u.data[:, 5:] == 0.)

    def test_constant_time_dense(self):
        """Test arithmetic between different data objects, namely Constant
        and Function."""
        i, j = dimify('i j')
        const = Constant(name='truc', value=2.)
        a = Function(name='a', shape=(20, 20), dimensions=(i, j))
        a.data[:] = 2.
        eqn = Eq(a, a + 2.*const)
        op = Operator(eqn)

        op.apply(a=a, truc=const)
        assert(np.allclose(a.data, 6.))

        # Applying a different constant still works
        op.apply(a=a, truc=Constant(name='truc2', value=3.))
        assert(np.allclose(a.data, 12.))

    def test_incs_same_lhs(self):
        """Test point-wise arithmetic with multiple increments expressed
        as different equations."""
        grid = Grid(shape=(10, 10))
        u = Function(name='u', grid=grid, space_order=0)
        op = Operator([Eq(u, u+1.0), Eq(u, u+2.0)])
        u.data[:] = 0.0
        op.apply()
        assert np.all(u.data[:] == 3)

    def test_sparsefunction_inject(self):
        """
        Test injection of a SparseFunction into a Function
        """
        grid = Grid(shape=(11, 11))
        u = Function(name='u', grid=grid, space_order=1)

        sf1 = SparseFunction(name='s', grid=grid, npoint=1)
        op = Operator(sf1.inject(u, expr=sf1))

        assert sf1.data.shape == (1, )
        sf1.coordinates.data[0, :] = (0.6, 0.6)
        sf1.data[0] = 5.0
        u.data[:] = 0.0

        op.apply()

        # This should be exactly on a point, all others 0
        assert u.data[6, 6] == pytest.approx(5.0)
        assert np.sum(u.data) == pytest.approx(5.0)

    def test_sparsefunction_interp(self):
        """
        Test interpolation of a SparseFunction from a Function
        """
        grid = Grid(shape=(11, 11))
        u = Function(name='u', grid=grid, space_order=1)

        sf1 = SparseFunction(name='s', grid=grid, npoint=1)
        op = Operator(sf1.interpolate(u))

        assert sf1.data.shape == (1, )
        sf1.coordinates.data[0, :] = (0.45, 0.45)
        sf1.data[:] = 0.0
        u.data[:] = 0.0
        u.data[4, 4] = 4.0

        op.apply()

        # Exactly in the middle of 4 points, only 1 nonzero is 4
        assert sf1.data[0] == pytest.approx(1.0)

    def test_sparsetimefunction_interp(self):
        """
        Test injection of a SparseTimeFunction into a TimeFunction
        """
        grid = Grid(shape=(11, 11))
        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=1)

        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)
        op = Operator(sf1.interpolate(u))

        assert sf1.data.shape == (5, 1)
        sf1.coordinates.data[0, :] = (0.45, 0.45)
        sf1.data[:] = 0.0
        u.data[:] = 0.0
        u.data[:, 4, 4] = 8*np.arange(5)+4

        # Because of time_order=2 this is probably the range we get anyway, but
        # to be sure...
        op.apply(time_m=1, time_M=3)

        # Exactly in the middle of 4 points, only 1 nonzero is 4
        assert np.all(sf1.data[:, 0] == pytest.approx([0.0, 3.0, 5.0, 7.0, 0.0]))

    def test_sparsetimefunction_inject(self):
        """
        Test injection of a SparseTimeFunction from a TimeFunction
        """
        grid = Grid(shape=(11, 11))
        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=1)

        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)
        op = Operator(sf1.inject(u, expr=3*sf1))

        assert sf1.data.shape == (5, 1)
        sf1.coordinates.data[0, :] = (0.45, 0.45)
        sf1.data[:, 0] = np.arange(5)
        u.data[:] = 0.0

        # Because of time_order=2 this is probably the range we get anyway, but
        # to be sure...
        op.apply(time_m=1, time_M=3)

        # Exactly in the middle of 4 points, only 1 nonzero is 4
        assert np.all(u.data[1, 4:6, 4:6] == pytest.approx(0.75))
        assert np.all(u.data[2, 4:6, 4:6] == pytest.approx(1.5))
        assert np.all(u.data[3, 4:6, 4:6] == pytest.approx(2.25))
        assert np.sum(u.data[:]) == pytest.approx(4*0.75+4*1.5+4*2.25)

    def test_sparsetimefunction_inject_dt(self):
        """
        Test injection of the time deivative of a SparseTimeFunction into a TimeFunction
        """
        grid = Grid(shape=(11, 11))
        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=1)

        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5, time_order=2)

        # This should end up as a central difference operator
        op = Operator(sf1.inject(u, expr=3*sf1.dt))

        assert sf1.data.shape == (5, 1)
        sf1.coordinates.data[0, :] = (0.45, 0.45)
        sf1.data[:, 0] = np.arange(5)
        u.data[:] = 0.0

        # Because of time_order=2 this is probably the range we get anyway, but
        # to be sure...
        op.apply(time_m=1, time_M=3, dt=1)

        # Exactly in the middle of 4 points, only 1 nonzero is 4
        assert np.all(u.data[1:4, 4:6, 4:6] == pytest.approx(0.75))
        assert np.sum(u.data[:]) == pytest.approx(12*0.75)

    @pytest.mark.parametrize('func1', [TensorFunction, TensorTimeFunction,
                                       VectorFunction, VectorTimeFunction])
    def test_tensor(self, func1):
        grid = Grid(tuple([5]*3))
        f1 = func1(name="f1", grid=grid)
        op1 = Operator(Eq(f1, f1.dx))
        op2 = Operator([Eq(f, f.dx) for f in f1.values()])
        assert str(op1.ccode) == str(op2.ccode)

    @pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
    def test_complex(self, dtype):
        grid = Grid((5, 5))
        x, y = grid.dimensions

        c = Constant(name='c', dtype=dtype)
        u = Function(name="u", grid=grid, dtype=dtype)

        eq = Eq(u, x + sympy.I*y + exp(sympy.I + x.spacing) * c)
        op = Operator(eq)
        op(c=1.0 + 2.0j)

        # Check against numpy
        dx = grid.spacing_map[x.spacing]
        xx, yy = np.meshgrid(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        npres = xx + 1j*yy + np.exp(1j + dx) * (1.0 + 2.0j)

        assert np.allclose(u.data, npres.T, rtol=1e-7, atol=0)


class TestAllocation:

    @pytest.mark.parametrize('shape', [(20, 20),
                                       (20, 20, 20),
                                       (20, 20, 20, 20)])
    def test_first_touch(self, shape):
        dimensions = dimify('i j k l')[:len(shape)]
        grid = Grid(shape=shape, dimensions=dimensions)
        m = Function(name='m', grid=grid, first_touch=True)
        assert(np.allclose(m.data, 0))
        m2 = Function(name='m2', grid=grid, first_touch=False)
        assert(np.allclose(m2.data, 0))
        assert(np.array_equal(m.data, m2.data))

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_staggered(self, ndim):
        """
        Test the "deformed" allocation for staggered functions
        """
        grid = Grid(shape=tuple([11]*ndim))
        for stagg in tuple(powerset(grid.dimensions))[1::] + (NODE, CELL):
            f = Function(name='f', grid=grid, staggered=stagg)
            assert f.data.shape == tuple([11]*ndim)
            # Add a non-staggered field to ensure that the auto-derived
            # dimension size arguments are at maximum
            g = Function(name='g', grid=grid)
            # Test insertion into a central point
            index = tuple(5 for _ in f.dimensions)
            set_f = Eq(f[index], 2.)
            set_g = Eq(g[index], 3.)

            Operator([set_f, set_g])()
            assert f.data[index] == 2.

    @pytest.mark.parametrize('ndim', [2, 3])
    def test_staggered_time(self, ndim):
        """
        Test the "deformed" allocation for staggered functions
        """
        grid = Grid(shape=tuple([11]*ndim))
        for stagg in tuple(powerset(grid.dimensions))[1::] + (NODE,):
            f = TimeFunction(name='f', grid=grid, staggered=stagg)
            assert f.data.shape[1:] == tuple([11]*ndim)
            # Add a non-staggered field to ensure that the auto-derived
            # dimension size arguments are at maximum
            g = TimeFunction(name='g', grid=grid)
            # Test insertion into a central point
            index = tuple([0] + [5 for _ in f.dimensions[1:]])
            set_f = Eq(f[index], 2.)
            set_g = Eq(g[index], 3.)

            Operator([set_f, set_g])()
            assert f.data[index] == 2.


class TestApplyArguments:

    def verify_arguments(self, arguments, expected):
        """
        Utility function to verify an argument dictionary against
        expected values.
        """
        for name, v in expected.items():
            if isinstance(v, (Function, SparseFunction)):
                condition = v._C_as_ndarray(arguments[name])[v._mask_domain] == v.data
                condition = condition.all()
            elif isinstance(arguments[name], range):
                condition = arguments[name].start <= v < arguments[name].stop
            else:
                condition = arguments[name] == v

            if not condition:
                error('Wrong argument %s: expected %s, got %s' %
                      (name, v, arguments[name]))
            assert condition

    def verify_parameters(self, parameters, expected):
        """
        Utility function to verify a parameter set against expected
        values.
        """
        boilerplate = ['timers']
        parameters = [p.name for p in parameters]
        for expi in expected:
            if expi not in parameters + boilerplate:
                error("Missing parameter: %s" % expi)
            assert expi in parameters + boilerplate
        extra = [p for p in parameters if p not in expected and p not in boilerplate]
        if len(extra) > 0:
            error("Redundant parameters: %s" % str(extra))
        assert len(extra) == 0

    def test_default_functions(self):
        """
        Test the default argument derivation for functions.
        """
        grid = Grid(shape=(5, 6, 7))
        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        op = Operator(Eq(f.forward, g + f), opt=('advanced', {'openmp': False}))

        expected = {
            'x_m': 0, 'x_M': 4,
            'y_m': 0, 'y_M': 5,
            'z_m': 0, 'z_M': 6,
            'f': f, 'g': g,
        }
        self.verify_arguments(op.arguments(time=4), expected)
        exp_parameters = ['f', 'g', 'x_m', 'x_M', 'y_m', 'y_M', 'z_m', 'z_M',
                          'x0_blk0_size', 'y0_blk0_size', 'time_m', 'time_M']
        self.verify_parameters(op.parameters, exp_parameters)

    def test_default_sparse_functions(self):
        """
        Test the default argument derivation for composite functions.
        """
        grid = Grid(shape=(5, 6, 7))
        f = TimeFunction(name='f', grid=grid)
        s = SparseTimeFunction(name='s', grid=grid, npoint=3, nt=4)
        s.coordinates.data[:, 0] = np.arange(0., 3.)
        s.coordinates.data[:, 1] = np.arange(1., 4.)
        s.coordinates.data[:, 2] = np.arange(2., 5.)
        op = Operator(s.interpolate(f))

        expected = {
            's': s, 's_coords': s.coordinates,
            # Default dimensions of the sparse data
            'p_s_size': 3, 'p_s_m': 0, 'p_s_M': 2,
            'd_size': 3, 'd_m': 0, 'd_M': 2,
            'time_size': 4, 'time_m': 0, 'time_M': 3,
        }
        self.verify_arguments(op.arguments(), expected)

    def test_override_function_size(self):
        """
        Test runtime size overrides for Function dimensions.

        Note: The current behaviour for size-only arguments seems
        ambiguous (eg. op(x=3, y=4), as it sets `dim_size` as well as
        `dim_end`. Since `dim_size` is used for the cast, we can get
        garbage results if it does not agree with the shape of the
        provided data. This should error out, or potentially we could
        set the corresponding size, while aliasing `dim` to `dim_e`?

        The same should be tested for TimeFunction once fixed.
        """
        grid = Grid(shape=(5, 6, 7))
        g = Function(name='g', grid=grid)

        op = Operator(Eq(g, 1.))
        args = {'x': 3, 'y': 4, 'z': 5}
        arguments = op.arguments(**args)
        expected = {
            'x_m': 0, 'x_M': 3,
            'y_m': 0, 'y_M': 4,
            'z_m': 0, 'z_M': 5,
            'g': g
        }
        self.verify_arguments(arguments, expected)
        # Verify execution
        op(**args)
        assert (g.data[4:] == 0.).all()
        assert (g.data[:, 5:] == 0.).all()
        assert (g.data[:, :, 6:] == 0.).all()
        assert (g.data[:4, :5, :6] == 1.).all()

    def test_override_function_subrange(self):
        """
        Test runtime start/end override for Function dimensions.
        """
        grid = Grid(shape=(5, 6, 7))
        g = Function(name='g', grid=grid)

        op = Operator(Eq(g, 1.))
        args = {'x_m': 1, 'x_M': 3, 'y_m': 2, 'y_M': 4, 'z_m': 3, 'z_M': 5}
        arguments = op.arguments(**args)
        expected = {
            'x_m': 1, 'x_M': 3,
            'y_m': 2, 'y_M': 4,
            'z_m': 3, 'z_M': 5,
            'g': g
        }
        self.verify_arguments(arguments, expected)
        # Verify execution
        op(**args)
        mask = np.ones((5, 6, 7), dtype=bool)
        mask[1:4, 2:5, 3:6] = False
        assert (g.data[mask] == 0.).all()
        assert (g.data[1:4, 2:5, 3:6] == 1.).all()

    def test_override_timefunction_subrange(self):
        """
        Test runtime start/end overrides for TimeFunction dimensions.
        """
        grid = Grid(shape=(5, 6, 7))
        f = TimeFunction(name='f', grid=grid, time_order=0)

        # Suppress opts to work around a know bug with GCC and OpenMP:
        # https://github.com/devitocodes/devito/issues/320
        op = Operator(Eq(f, 1.), opt=None)
        # TODO: Currently we require the `time` subrange to be set
        # explicitly. Ideally `t` would directly alias with `time`,
        # but this seems broken currently.
        args = {'x_m': 1, 'x_M': 3, 'y_m': 2, 'y_M': 4,
                'z_m': 3, 'z_M': 5, 't_m': 1, 't_M': 4}
        arguments = op.arguments(**args)
        expected = {
            'x_m': 1, 'x_M': 3,
            'y_m': 2, 'y_M': 4,
            'z_m': 3, 'z_M': 5,
            'time_m': 1, 'time_M': 4,
            'f': f
        }
        self.verify_arguments(arguments, expected)
        # Verify execution
        op(**args)
        mask = np.ones((1, 5, 6, 7), dtype=bool)
        mask[:, 1:4, 2:5, 3:6] = False
        assert (f.data[mask] == 0.).all()
        assert (f.data[:, 1:4, 2:5, 3:6] == 1.).all()

    def test_override_function_data(self):
        """
        Test runtime data overrides for Function symbols.
        """
        grid = Grid(shape=(5, 6, 7))
        a = Function(name='a', grid=grid)
        op = Operator(Eq(a, a + 3))

        # Run with default value
        a.data[:] = 1.
        op()
        assert (a.data[:] == 4.).all()

        # Override with symbol (different name)
        a1 = Function(name='a1', grid=grid)
        a1.data[:] = 2.
        op(a=a1)
        assert (a1.data[:] == 5.).all()

        # Override with symbol (same name as original)
        a2 = Function(name='a', grid=grid)
        a2.data[:] = 3.
        op(a=a2)
        assert (a2.data[:] == 6.).all()

        # Override with user-allocated numpy data
        a3 = np.zeros_like(a._data_allocated)
        a3[:] = 4.
        op(a=a3)
        assert (a3[a._mask_domain] == 7.).all()

    def test_override_timefunction_data(self):
        """
        Test runtime data overrides for TimeFunction symbols.
        """
        grid = Grid(shape=(5, 6, 7))
        a = TimeFunction(name='a', grid=grid, save=2)
        # Suppress opts to work around a know bug with GCC and OpenMP:
        # https://github.com/devitocodes/devito/issues/320
        op = Operator(Eq(a, a + 3), opt=None)

        # Run with default value
        a.data[:] = 1.
        op(time_m=0, time=1)
        assert (a.data[:] == 4.).all()

        # Override with symbol (different name)
        a1 = TimeFunction(name='a1', grid=grid, save=2)
        a1.data[:] = 2.
        op(time_m=0, time=1, a=a1)
        assert (a1.data[:] == 5.).all()

        # Override with symbol (same name as original)
        a2 = TimeFunction(name='a', grid=grid, save=2)
        a2.data[:] = 3.
        op(time_m=0, time=1, a=a2)
        assert (a2.data[:] == 6.).all()

        # Override with user-allocated numpy data
        a3 = np.zeros_like(a._data_allocated)
        a3[:] = 4.
        op(time_m=0, time=1, a=a3)
        assert (a3[a._mask_domain] == 7.).all()

    def test_dimension_size_infer(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        grid = Grid(shape=(3, 5, 7))
        a = Function(name='a', grid=grid)
        b = TimeFunction(name='b', grid=grid, save=nt)
        op = Operator(Eq(b, a))

        time = b.indices[0]
        op_arguments = op.arguments()
        assert(op_arguments[time.min_name] == 0)
        assert(op_arguments[time.max_name] == nt-1)

    def test_dimension_offset_adjust(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        grid = Grid(shape=shape, dimensions=(i, j, k))
        a = Function(name='a', grid=grid)
        b = TimeFunction(name='b', grid=grid, save=nt)
        time = b.indices[0]
        eqn = Eq(b[time + 1, i, j, k], b[time - 1, i, j, k]
                 + b[time, i, j, k] + a[i, j, k])
        op = Operator(eqn)
        op_arguments = op.arguments(time=nt-10)
        assert(op_arguments[time.min_name] == 1)
        assert(op_arguments[time.max_name] == nt - 10)

    def test_dimension_size_override(self):
        """Test explicit overrides for the leading time dimension"""
        grid = Grid(shape=(3, 5, 7))
        a = TimeFunction(name='a', grid=grid)
        one = Function(name='one', grid=grid)
        one.data[:] = 1.
        op = Operator(Eq(a.forward, a + one))

        # Test dimension override via the buffered dimenions
        a.data[0] = 0.
        op(a=a, t=5)
        assert(np.allclose(a.data[1], 5.))

        # Test dimension override via the parent dimenions
        a.data[0] = 0.
        op(a=a, time=4)
        assert(np.allclose(a.data[0], 4.))

    def test_override_sparse_data_fix_dim(self):
        """
        Ensure the arguments are derived correctly for an input SparseFunction.
        The dimensions are forced to be the same in this case to verify
        the aliasing on the SparseFunction name.
        """
        grid = Grid(shape=(10, 10))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

        original_coords = (1., 1.)
        new_coords = (2., 2.)
        p_dim = Dimension(name='p_src')
        src1 = SparseTimeFunction(name='src1', grid=grid, dimensions=(time, p_dim), nt=10,
                                  npoint=1, coordinates=original_coords, time_order=2)
        src2 = SparseTimeFunction(name='src2', grid=grid, dimensions=(time, p_dim),
                                  npoint=1, nt=10, coordinates=new_coords, time_order=2)
        op = Operator(src1.inject(u, src1))

        # Move the source from the location where the setup put it so we can test
        # whether the override picks up the original coordinates or the changed ones

        args = op.arguments(src1=src2, time=0)
        arg_name = src1.coordinates._arg_names[0]
        assert(np.array_equal(src2.coordinates._C_as_ndarray(args[arg_name]),
                              np.asarray((new_coords,))))

    def test_override_sparse_data_default_dim(self):
        """
        Ensure the arguments are derived correctly for an input SparseFunction.
        The dimensions are the defaults (name dependant 'p_name') in this case to verify
        the aliasing on the SparseFunction coordinates and dimensions.
        """
        grid = Grid(shape=(10, 10))
        original_coords = (1., 1.)
        new_coords = (2., 2.)
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)
        src1 = SparseTimeFunction(name='src1', grid=grid, npoint=1, nt=10,
                                  coordinates=original_coords, time_order=2)
        src2 = SparseTimeFunction(name='src2', grid=grid, npoint=1, nt=10,
                                  coordinates=new_coords, time_order=2)
        op = Operator(src1.inject(u, src1))

        # Move the source from the location where the setup put it so we can test
        # whether the override picks up the original coordinates or the changed ones

        args = op.arguments(src1=src2, t=0)
        arg_name = src1.coordinates._arg_names[0]
        assert(np.array_equal(src2.coordinates._C_as_ndarray(args[arg_name]),
                              np.asarray((new_coords,))))

    def test_argument_derivation_order(self, nt=100):
        """ Ensure the precedence order of arguments is respected
        Defaults < (overriden by) Tensor Arguments < Dimensions < Scalar Arguments
        """
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        grid = Grid(shape=shape, dimensions=(i, j, k))
        a = Function(name='a', grid=grid)
        b = TimeFunction(name='b', grid=grid, save=nt)
        time = b.indices[0]
        op = Operator(Eq(b, a))

        # Simple case, same as that tested above.
        # Repeated here for clarity of further tests.
        op_arguments = op.arguments()
        assert(op_arguments[time.min_name] == 0)
        assert(op_arguments[time.max_name] == nt-1)

        # Providing a tensor argument should infer the dimension size from its shape
        b1 = TimeFunction(name='b1', grid=grid, save=nt+1)
        op_arguments = op.arguments(b=b1)
        assert(op_arguments[time.min_name] == 0)
        assert(op_arguments[time.max_name] == nt)

        # Providing a dimension size explicitly should override the automatically inferred
        op_arguments = op.arguments(b=b1, time=nt - 1)
        assert(op_arguments[time.min_name] == 0)
        assert(op_arguments[time.max_name] == nt - 1)

        # Providing a scalar argument explicitly should override the automatically
        # inferred
        op_arguments = op.arguments(b=b1, time_M=nt - 2)
        assert(op_arguments[time.min_name] == 0)
        assert(op_arguments[time.max_name] == nt - 2)

    def test_derive_constant_value(self):
        """Ensure that values for Constant symbols are derived correctly."""
        grid = Grid(shape=(5, 6))
        f = Function(name='f', grid=grid)
        a = Constant(name='a', value=3.)
        Operator(Eq(f, a))()
        assert np.allclose(f.data, 3.)

        g = Function(name='g', grid=grid)
        b = Constant(name='b')
        op = Operator(Eq(g, b))
        b.data = 4.
        op()
        assert np.allclose(g.data, 4.)

    def test_argument_from_index_constant(self):
        nx, ny = 30, 30
        grid = Grid(shape=(nx, ny))
        x, y = grid.dimensions

        arbdim = Dimension('arb')
        u = TimeFunction(name='u', grid=grid, save=None, time_order=2, space_order=0)
        snap = Function(name='snap', dimensions=(arbdim, x, y), shape=(5, nx, ny),
                        space_order=0)

        save_t = Constant(name='save_t', dtype=np.int32)
        save_slot = Constant(name='save_slot', dtype=np.int32)

        expr = Eq(snap.subs(arbdim, save_slot), u.subs(grid.stepping_dim, save_t))
        op = Operator(expr)
        u.data[:] = 0.0
        snap.data[:] = 0.0
        u.data[0, 10, 10] = 1.0
        op.apply(save_t=0, save_slot=1)
        assert snap.data[1, 10, 10] == 1.0

    def test_argument_no_shifting(self):
        """Tests that there's no shifting in the written-to region when
        iteration bounds are prescribed."""
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions
        a = Function(name='a', grid=grid)
        a.data[:] = 1.

        # Try with an operator w/o stencil offsets
        op = Operator(Eq(a, a + a))
        op(x_m=3, x_M=7)
        assert (a.data[:3, :] == 1.).all()
        assert (a.data[3:7, :] == 2.).all()
        assert (a.data[8:, :] == 1.).all()

        # Try with an operator w/ stencil offsets
        a.data[:] = 1.
        op = Operator(Eq(a, a + (a[x-1, y] + a[x+1, y]) / 2.))
        op(x_m=3, x_M=7)
        assert (a.data[:3, :] == 1.).all()
        assert (a.data[3:7, :] >= 2.).all()
        assert (a.data[8:, :] == 1.).all()

    def test_argument_unknown(self):
        """Check that Operators deal with unknown runtime arguments."""
        grid = Grid(shape=(11, 11))
        a = Function(name='a', grid=grid)

        op = Operator(Eq(a, a + a))
        try:
            op.apply(b=3)
            assert False
        except ValueError:
            # `b` means nothing to `op`, so we end up here
            assert True

        try:
            configuration['ignore-unknowns'] = True
            op.apply(b=3)
            assert True
        except ValueError:
            # we should not end up here as we're now ignoring unknown arguments
            assert False
        finally:
            configuration['ignore-unknowns'] = configuration._defaults['ignore-unknowns']

    @pytest.mark.parametrize('so,to,pad,expected', [
        (0, 1, 0, (2, 4, 4, 4)),
        (2, 1, 0, (2, 8, 8, 8)),
        (4, 1, 0, (2, 12, 12, 12)),
        (4, 3, 0, (4, 12, 12, 12)),
        (4, 1, 3, (2, 15, 15, 15)),
        ((2, 5, 2), 1, 0, (2, 11, 11, 11)),
        ((2, 5, 4), 1, 3, (2, 16, 16, 16)),
    ])
    def test_function_dataobj(self, so, to, pad, expected):
        """
        Tests that the C-level structs from DiscreteFunctions are properly
        populated upon application of an Operator.
        """
        grid = Grid(shape=(4, 4, 4))

        u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to, padding=pad)

        op = Operator(Eq(u, 1), opt='noop')

        u_arg = op.arguments(time=0)['u']
        u_arg_shape = tuple(u_arg._obj.size[i] for i in range(u.ndim))

        assert u_arg_shape == expected

    def test_illegal_override(self):
        grid0 = Grid(shape=(11, 11))
        grid1 = Grid(shape=(13, 13))

        a0 = Function(name='a', grid=grid0)
        b0 = Function(name='b', grid=grid0)
        a1 = Function(name='a', grid=grid1)

        op = Operator(Eq(a0, a0 + b0 + 1))
        op.apply()

        try:
            op.apply(a=a1, b=b0)
            assert False
        except ValueError as e:
            assert 'Override' in e.args[0]  # Check it's hitting the right error msg
        except:
            assert False

    def test_incomplete_override(self):
        """
        Simulate a typical user error when one has to supply replacements for lots
        of Functions (a complex Operator) but at least one is forgotten.
        """
        grid0 = Grid(shape=(11, 11))
        grid1 = Grid(shape=(13, 13))

        a0 = Function(name='a', grid=grid0)
        a1 = Function(name='a', grid=grid1)
        b = Function(name='b', grid=grid0)

        op = Operator(Eq(a0, a0 + b + 1))
        op.apply()

        try:
            op.apply(a=a1)
            assert False
        except ValueError as e:
            assert 'Default' in e.args[0]  # Check it's hitting the right error msg
        except:
            assert False

    @pytest.mark.parallel(mode=1)
    def test_new_distributor(self, mode):
        """
        Test that `comm` and `nb` are correctly updated when a different distributor
        from that it was originally built with is required by an operator.
        Note that MPI is required to ensure `comm` and `nb` are included in op.objects.
        """
        from devito.mpi import MPI
        grid = Grid(shape=(10, 10), comm=MPI.COMM_SELF)
        grid2 = Grid(shape=(10, 10), comm=MPI.COMM_WORLD)

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u2 = TimeFunction(name='u2', grid=grid2, space_order=2)

        # Create some operator that requires MPI communication
        eqn = Eq(u.forward, u + u.laplace)
        op = Operator(eqn)
        assert op.arguments(u=u, time_M=0)['comm'] is grid.distributor._obj_comm.value
        assert (op.arguments(u=u, time_M=0)['nb'] is
                grid.distributor._obj_neighborhood.value)
        assert op.arguments(u=u2, time_M=0)['comm'] is grid2.distributor._obj_comm.value
        assert (op.arguments(u=u2, time_M=0)['nb'] is
                grid2.distributor._obj_neighborhood.value)

    def test_spacing_from_new_grid(self):
        """
        MFE for issue #1518.
        """
        grid = Grid(shape=(10, 10), extent=(9, 9))
        u = Function(name='u', grid=grid, space_order=1)

        # A bogus operator that just assigns the x spacing into the array
        # Note, grid.dimensions[0].spacing here is not a number, it's the symbol h_x
        op = Operator(Eq(u, grid.dimensions[0].spacing))

        # Create a new grid with different spacing, and a function defined on it
        grid2 = Grid(shape=(5, 5), extent=(9, 9))
        u2 = Function(name='u', grid=grid2, space_order=1)
        op(u=u2)

        # The h_x that was passed to the C code must be the one `grid2`, not `grid`
        assert u2.data[2, 2] == grid2.spacing[0]

    def test_loose_kwargs(self):
        grid = Grid(shape=(10, 10))

        x, y = grid.dimensions
        s = Symbol(name='s', dtype=np.int32)

        eq = Eq(s, x.symbolic_size + y.symbolic_size)

        op = Operator(eq)

        # Exception expected here because a binding for `x_size` and `y_size`
        # needs to be provided
        with pytest.raises(ValueError):
            op.arguments()

        # But the following should work perfectly fine
        op.arguments(x_size=2, y_size=2)


@skipif('device')
class TestDeclarator:

    def test_conditional_declarations(self):
        x = Dimension(name="x")
        a = Array(name='a', dimensions=(x,), dtype=np.int32, scope='stack')
        init_value = ListInitializer([0, 0])
        list_initialize = Expression(ClusterizedEq(Eq(a[x], init_value), ispace=None))
        iet = Conditional(x < 3, list_initialize, list_initialize)
        iet = Callable('test', iet, 'void')
        dm = CDataManager(sregistry=None)
        iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]
        for i in iet.body.body[0].children:
            assert len(i) == 1
            assert i[0].is_Expression
            assert i[0].expr.rhs is init_value

    def test_nested_scalar_assigns(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        s = Symbol(name='s', dtype=grid.dtype)

        eqns = [Eq(s, 0),
                Eq(s, s + f + 1)]

        op = Operator(eqns)

        exprs = FindNodes(Expression).visit(op)
        nlin = 2 if op._options['linearize'] else 0

        assert len(exprs) == 2 + nlin
        assert exprs[nlin].init
        assert 'float' in str(exprs[nlin])
        assert not exprs[1+nlin].init
        assert 'float' not in str(exprs[1+nlin])


class TestLoopScheduling:

    def test_permutations_without_deps(self):
        """
        Test that if none of the Function accesses in the equations use
        offsets, implying that there are no carried dependences, then no
        matter the order in which the equations are provided to an Operator
        the resulting loop nest is the same, and the input ordering of the
        equations is honored.
        """
        grid = Grid(shape=(4, 4, 4))

        ti0 = Function(name='ti0', grid=grid)
        ti1 = Function(name='ti1', grid=grid)
        tu = TimeFunction(name='tu', grid=grid)
        tv = TimeFunction(name='tv', grid=grid)

        eq1 = Eq(tu, tv*ti0 + ti0)
        eq2 = Eq(ti0, tu + 3.)
        eq3 = Eq(tv, ti0*ti1)
        op1 = Operator([eq1, eq2, eq3], opt='noop')
        op2 = Operator([eq2, eq1, eq3], opt='noop')
        op3 = Operator([eq3, eq2, eq1], opt='noop')

        trees = [retrieve_iteration_tree(i) for i in [op1, op2, op3]]
        assert all(len(i) == 1 for i in trees)
        trees = [i[0] for i in trees]
        for tree in trees:
            assert IsPerfectIteration().visit(tree[1])
            exprs = FindNodes(Expression).visit(tree[-1])
            assert len(exprs) == 3

    @pytest.mark.parametrize('exprs,fissioned,shared', [
        # 0) Trivial case
        (('Eq(u, 1)', 'Eq(v, u.dxl)'), '(1,x)', [0]),
        # 1) Anti-dependence along x
        (('Eq(u, 1)', 'Eq(v, u.dxr)'), '(1,x)', [0]),
        # 2, 3) As above, but with an additional Dimension-independent dependence
        (('Eq(u, v)', 'Eq(v, u.dxl)'), '(1,x)', [0]),
        (('Eq(u, v)', 'Eq(v, u.dxr)'), '(1,x)', [0]),
        # 4) Slightly more convoluted than above, as the additional dependence is
        # now carried along x
        (('Eq(u, v)', 'Eq(v, u.dxr)'), '(1,x)', [0]),
        # 5) No backward carried dependences, no storage related dependences
        (('Eq(us.forward, vs)', 'Eq(vs, us.dxl)'), '(0,time)', []),
        # 6) No backward carried dependences, no storage related dependences
        (('Eq(us.forward, vs)', 'Eq(vs, us.dxr)'), '(0,time)', []),
        # 7) Three fissionable Eqs
        (('Eq(u, u.dxl + v.dxr)', 'Eq(v, w.dxr)', 'Eq(w, u*w.dxl)'), '(1,x)', [0]),
        # 8) There are carried backward dependences, but not in the Dimension
        # that gets fissioned
        (('Eq(u.forward, u + v.dx)', 'Eq(v.forward, v + u.forward.dx)'), '(1,x)', [0])
    ])
    def test_fission_for_parallelism(self, exprs, fissioned, shared):
        """
        Test that expressions are scheduled to separate loops if this can
        turn one sequential loop into two parallel loops ("loop fission").
        """
        grid = Grid(shape=(3, 3))
        t = grid.stepping_dim  # noqa
        time = grid.time_dim  # noqa
        x, y = grid.dimensions  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        w = TimeFunction(name='w', grid=grid)  # noqa
        us = TimeFunction(name='u', grid=grid, save=5)  # noqa
        vs = TimeFunction(name='v', grid=grid, save=5)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        # `opt='noop'` is only to avoid loop blocking, hence making the asserts
        # below much simpler to write and understand
        op = Operator(eqns, opt='noop')

        # Fission expected
        trees = retrieve_iteration_tree(op)
        assert len(trees) == len(eqns)

        exp_depth, exp_dim = eval(fissioned)
        for i in trees:
            # Some outer loops may still be shared
            for j in shared:
                assert i[j] is trees[0][j]
            # Fission happened
            assert i[exp_depth].dim is exp_dim

    @pytest.mark.parametrize('exprs', [
        # 0) Storage related dependence
        ('Eq(u.forward, v)', 'Eq(v, u.dxl)'),
        # 1) Backward carried flow-dependence through `v`
        ('Eq(u, v.forward)', 'Eq(v, u)'),
        # 2) Backward carried flow-dependence through `vs`
        ('Eq(us.forward, vs)', 'Eq(vs.forward, us.dxl)'),
        # 3) Classic coupled forward-marching equations
        ('Eq(u.forward, u + u.backward + v)', 'Eq(v.forward, v + v.backward + u)'),
        # 4) Three non-fissionable Eqs
        ('Eq(u, v.dxl)', 'Eq(v, w.dxl)', 'Eq(w, u*w.dxl)')
    ])
    def test_no_fission_as_illegal(self, exprs):
        """
        Antithesis of `test_fission_for_parallelism`.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        w = TimeFunction(name='w', grid=grid)  # noqa
        us = TimeFunction(name='u', grid=grid, save=5)  # noqa
        vs = TimeFunction(name='v', grid=grid, save=5)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        op = Operator(eqns)

        # No fission expected
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

    @pytest.mark.parametrize('exprs,directions,expected,visit', [
        # 0) WAR 2->3, 3 fissioned to maximize parallelism
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti3[x,y,z])',
          'Eq(ti3[x,y,z], ti1[x,y,z+1] + 1.)'),
         '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 1) WAR 1->2, 2->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y,z+1])',
          'Eq(ti3[x,y,z], ti1[x,y,z-2] + 1.)'),
         '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 2) WAR 1->2, 2->3, RAW 2->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y,z+1])',
          'Eq(ti3[x,y,z], ti1[x,y,z-2] + ti1[x,y,z+2])'),
         '+++++', ['xyz', 'xyz', 'xyz'], 'xyzzz'),
        # 3) WAR 1->3
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti3[x,y,z])',
          'Eq(ti3[x,y,z], ti0[x,y,z+1] + 1.)'),
         '++++', ['xyz', 'xyz'], 'xyzz'),
        # 4) WAR 1->3
        # Like before, but the WAR is along `y`, an inner Dimension
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti3[x,y,z])',
          'Eq(ti3[x,y,z], ti0[x,y+1,z] + 1.)'),
         '+++++', ['xyz', 'xyz'], 'xyzyz'),
        # 5) WAR 1->2, 2->3; WAW 1->3
        # Similar to the cases above, but the last equation does not iterate over `z`
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y,z+2])',
          'Eq(ti0[x,y,0], ti0[x,y,0] + 1.)'),
         '++++', ['xyz', 'xyz', 'xy'], 'xyzz'),
        # 6) WAR 1->2; WAW 1->3
        # Basically like above, but with the time dimension. This should have no impact
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tu[t,x,y,0], tu[t,x,y,0] + 1.)'),
         '+++++', ['txyz', 'txyz', 'txy'], 'txyzz'),
        # 7) WAR 1->2, 2->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tw[t,x,y,z], tv[t,x,y,z-1] + 1.)'),
         '++++++', ['txyz', 'txyz', 'txyz'], 'txyzzz'),
        # 8) WAR 1->2; WAW 1->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x+2,y,z])',
          'Eq(tu[t,3,y,0], tu[t,3,y,0] + 1.)'),
         '++++++++', ['txyz', 'txyz', 'ty'], 'txyzxyzy'),
        # 9) RAW 1->2, WAR 2->3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z-2])',
          'Eq(tw[t,x,y,z], tv[t,x,y+1,z] + 1.)'),
         '++++++++', ['txyz', 'txyz', 'txyz'], 'txyzyzyz'),
        # 10) WAR 1->2; WAW 1->3
        (('Eq(tu[t-1,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tu[t-1,x,y,0], tu[t,x,y,0] + 1.)'),
         '-+++', ['txyz', 'txy'], 'txyz'),
        # 11) WAR 1->2
        (('Eq(tu[t-1,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2] + tu[t,x,y,z-2])',
          'Eq(tw[t,x,y,z], tv[t,x,y,z] + 2)'),
         '-+++', ['txyz'], 'txyz'),
        # 12) Time goes backward so that information flows in time
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(tv[t-1,x,y,z], tu[t,x,y,z+2])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
         '-+++', ['txyz'], 'txyz'),
        # 13) Time goes backward so that information flows in time, but the
        # first and last Eqs are interleaved by a completely independent
        # Eq. This results in three disjoint sets of loops
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(ti0[x,y,z], ti1[x,y,z+2])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
         '-++++++++++', ['txyz', 'xyz', 'txyz'], 'txyzxyztxyz'),
        # 14) Time goes backward so that information flows in time
        (('Eq(ti0[x,y,z], ti1[x,y,z+2])',
          'Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + ti0[x,y-1,z])'),
         '+++-+++', ['xyz', 'txyz'], 'xyztxyz'),
        # 15) WAR 2->1
        # Here the difference is that we're using SubDimensions
        (('Eq(tv[t,xi,yi,zi], tu[t,xi-1,yi,zi] + tu[t,xi+1,yi,zi])',
          'Eq(tu[t+1,xi,yi,zi], tu[t,xi,yi,zi] + tv[t,xi-1,yi,zi] + tv[t,xi+1,yi,zi])'),
         '+++++++', ['txyz', 'txyz'], 'txyzxyz'),
        # 16) RAW 3->1; expected=2
        # Time goes backward, but the third equation should get fused with
        # the first one, as the time dependence is loop-carried
        (('Eq(tv[t-1,x,y,z], tv[t,x-1,y,z] + tv[t,x+1,y,z])',
          'Eq(tv[t-1,z,z,z], tv[t-1,z,z,z] + 1)',
          'Eq(f[x,y,z], tu[t-1,x,y,z] + tu[t,x,y,z] + tu[t+1,x,y,z] + tv[t,x,y,z])'),
         '-++++', ['txyz', 'tz'], 'txyzz'),
        # 17) WAR 2->3, 2->4; expected=4
        (('Eq(tu[t+1,x,y,z], tu[t,x,y,z] + 1.)',
          'Eq(tu[t+1,y,y,y], tu[t+1,y,y,y] + tw[t+1,y,y,y])',
          'Eq(tw[t+1,z,z,z], tw[t+1,z,z,z] + 1.)',
          'Eq(tv[t+1,x,y,z], tu[t+1,x,y,z] + 1.)'),
         '+++++++++', ['txyz', 'ty', 'tz', 'txyz'], 'txyzyzxyz'),
        # 18) WAR 1->3; expected=3
        # 5 is expected to be moved before 4 but after 3, to be merged with 3
        (('Eq(tu[t+1,x,y,z], tv[t,x,y,z] + 1.)',
          'Eq(tv[t+1,x,y,z], tu[t,x,y,z] + 1.)',
          'Eq(tw[t+1,x,y,z], tu[t+1,x+1,y,z] + tu[t+1,x-1,y,z])',
          'Eq(f[x,x,z], tu[t,x,x,z] + tw[t,x,x,z])',
          'Eq(ti0[x,y,z], tw[t+1,x,y,z] + 1.)'),
         '++++++++', ['txyz', 'txyz', 'txz'], 'txyzxyzz'),
        # 19) WAR 1->3; expected=3
        # Cannot merge 1 with 3 otherwise we would break an anti-dependence
        (('Eq(tv[t+1,x,y,z], tu[t,x,y,z] + tu[t,x+1,y,z])',
          'Eq(tu[t+1,xi,yi,zi], tv[t+1,xi,yi,zi] + tv[t+1,xi+1,yi,zi])',
          'Eq(tw[t+1,x,y,z], tv[t+1,x,y,z] + tv[t+1,x+1,y,z])'),
         '++++++++++', ['txyz', 'txyz', 'txyz'], 'txyzxyzxyz'),
    ])
    def test_consistency_anti_dependences(self, exprs, directions, expected, visit):
        """
        Test that anti dependences end up generating multi loop nests, rather
        than a single loop nest enclosing all of the equations.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions  # noqa
        xi, yi, zi = grid.interior.dimensions  # noqa
        t = grid.stepping_dim  # noqa

        ti0 = Array(name='ti0', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti1 = Array(name='ti1', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        ti3 = Array(name='ti3', shape=grid.shape, dimensions=grid.dimensions)  # noqa
        f = Function(name='f', grid=grid)  # noqa
        tu = TimeFunction(name='tu', grid=grid)  # noqa
        tv = TimeFunction(name='tv', grid=grid)  # noqa
        tw = TimeFunction(name='tw', grid=grid)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        # Note: `topofuse` is a subset of `advanced` mode. We use it merely to
        # bypass 'blocking', which would complicate the asserts below
        op = Operator(eqns, opt=('topofuse', {'openmp': False, 'opt-comms': False}))

        trees = retrieve_iteration_tree(op)
        iters = FindNodes(Iteration).visit(op)
        assert len(trees) == len(expected)
        assert len(iters) == len(directions)
        # mapper just makes it quicker to write out the test parametrization
        mapper = {'time': 't'}
        assert ["".join(mapper.get(i.dim.name, i.dim.name) for i in j)
                for j in trees] == expected
        assert "".join(mapper.get(i.dim.name, i.dim.name) for i in iters) == visit
        # mapper just makes it quicker to write out the test parametrization
        mapper = {'+': Forward, '-': Backward, '*': Any}
        assert all(i.direction == mapper[j] for i, j in zip(iters, directions))

    def test_expressions_imperfect_loops(self):
        """
        Test that equations depending only on a subset of all indices
        appearing across all equations are placed within earlier loops
        in the loop nest tree.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        t0 = Constant(name='t0')
        t1 = Scalar(name='t1')
        e = Function(name='e', shape=(3,), dimensions=(x,), space_order=0)
        f = Function(name='f', shape=(3, 3), dimensions=(x, y), space_order=0)
        g = Function(name='g', grid=grid, space_order=0)
        h = Function(name='h', grid=grid, space_order=0)

        eq0 = Eq(t1, e*1.)
        eq1 = Eq(f, t0*3. + t1)
        eq2 = Eq(h, g + 4. + f*5.)
        op = Operator([eq0, eq1, eq2], opt='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3
        outer, middle, inner = trees
        assert len(outer) == 1 and len(middle) == 2 and len(inner) == 3
        assert outer[0] == middle[0] == inner[0]
        assert middle[1] == inner[1]
        assert outer[-1].nodes[0].exprs[0].expr.rhs == diff2sympy(indexify(eq0.rhs))
        assert (str(middle[-1].nodes[0].exprs[0].expr.rhs) ==
                str(diff2sympy(indexify(eq1.rhs))))
        assert (str(inner[-1].nodes[0].exprs[0].expr.rhs) ==
                str(diff2sympy(indexify(eq2.rhs))))

    def test_equations_emulate_bc(self):
        """
        Test that bc-like equations get inserted into the same loop nest
        as the "main" equations.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t0 = Scalar(name='t0')
        a = Function(name='a', grid=grid)
        b = TimeFunction(name='b', grid=grid, save=6)
        main = Eq(b[time + 1, x, y, z], b[time - 1, x, y, z] + a[x, y, z] + 3.*t0)
        bcs = [Eq(b[time, 0, y, z], 0.),
               Eq(b[time, x, 0, z], 0.),
               Eq(b[time, x, y, 0], 0.)]
        op = Operator([main] + bcs, opt='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 4
        assert all(id(trees[0][0]) == id(i[0]) for i in trees)

    def test_different_section_nests(self):
        grid = Grid((3, 3, 3))
        tu = TimeFunction(name='tu', grid=grid, space_order=4)
        t0 = Scalar(name='t0')
        t1 = Scalar(name='t1')
        ti0 = Array(name='ti0', shape=(3, 5, 7), dimensions=grid.dimensions,
                    scope='heap').indexify()
        eq1 = Eq(ti0, t0*3.)
        eq2 = Eq(tu, ti0 + t1*3.)
        op = Operator([eq1, eq2], opt='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert str(trees[0][-1].nodes[0].exprs[0].expr.rhs) == str(eq1.rhs)
        assert str(trees[1][-1].nodes[0].exprs[0].expr.rhs) == str(eq2.rhs)

    @pytest.mark.parametrize('exprs', [
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,0,z], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z-1] + t0*2.)', 'Eq(ti0[0,0,z], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,y,0], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,y,z], 0.)'],
    ])
    def test_directly_indexed_expression(self, exprs):
        """
        Test that equations using integer indices are inserted in the right
        loop nest, at the right loop nest depth.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions  # noqa

        ti0 = Function(name='ti0', grid=grid, space_order=0)  # noqa
        t0 = Scalar(name='t0')  # noqa

        eqs = [eval(exprs[0]), eval(exprs[1])]

        op = Operator(eqs, opt='noop')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert str(trees[0][-1].nodes[0].exprs[0].expr.rhs) == str(eqs[0].rhs)
        assert str(trees[1][-1].nodes[0].exprs[0].expr.rhs) == str(eqs[1].rhs)

    @pytest.mark.parametrize('shape', [(11, 11), (11, 11, 11)])
    def test_equations_mixed_functions(self, shape):
        """
        Test that equations using a mixture of Function and TimeFunction objects
        are embedded within the same time loop.
        """
        dims0 = Grid(shape).dimensions
        for dims in permutations(dims0):
            grid = Grid(shape=shape, dimensions=dims, dtype=np.float64)
            time = grid.time_dim
            a = TimeFunction(name='a', grid=grid, time_order=2, space_order=2)
            p_aux = Dimension(name='p_aux')
            b = Function(name='b', shape=shape + (10,), dimensions=dims + (p_aux,),
                         space_order=2, dtype=np.float64)
            b.data_with_halo[:] = 1.0
            b2 = Function(name='b2', shape=(10,) + shape, dimensions=(p_aux,) + dims,
                          space_order=2, dtype=np.float64)
            b2.data_with_halo[:] = 1.0
            eqns = [Eq(a.forward, a.laplace + 1.),
                    Eq(b, time*b*a + b)]
            eqns2 = [Eq(a.forward, a.laplace + 1.),
                     Eq(b2, time*b2*a + b2)]
            subs = {d.spacing: v for d, v in zip(dims0, [2.5, 1.5, 2.0][:grid.dim])}

            op = Operator(eqns, subs=subs, opt='noop')
            trees = retrieve_iteration_tree(op)
            assert len(trees) == 2
            assert all(trees[0][i] is trees[1][i] for i in range(3))

            op2 = Operator(eqns2, subs=subs, opt='noop')
            trees = retrieve_iteration_tree(op2)
            assert len(trees) == 2

            # Verify both operators produce the same result
            op(time=10)
            a.data_with_halo[:] = 0.
            op2(time=10)

            for i in range(10):
                assert(np.allclose(b2.data[i, ...].reshape(-1),
                                   b.data[..., i].reshape(-1),
                                   rtol=1e-9))

    def test_equations_mixed_timedim_stepdim(self):
        """"
        Test that two equations one using a TimeDimension the other a derived
        SteppingDimension end up in the same loop nest.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t = grid.stepping_dim
        u1 = TimeFunction(name='u1', grid=grid)
        u2 = TimeFunction(name='u2', grid=grid, save=2)
        eqn_1 = Eq(u1[t+1, x, y, z], u1[t, x, y, z] + 1.)
        eqn_2 = Eq(u2[time+1, x, y, z], u2[time, x, y, z] + 1.)
        op = Operator([eqn_1, eqn_2], opt='topofuse')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        assert len(trees[0][-1].nodes[0].exprs) == 2
        assert trees[0][-1].nodes[0].exprs[0].write == u1
        assert trees[0][-1].nodes[0].exprs[1].write == u2

    def test_flow_detection(self):
        """
        Test detection of spatial flow directions inside a time loop.

        Stencil uses values at new timestep as well as those at previous ones
        This forces an evaluation order onto x.
        Weights are:

               x=0     x=1     x=2     x=3
        t=n     2    ---3
                v   /
        t=n+1   o--+----4

        Flow dependency should traverse x in the negative direction

               x=2     x=3     x=4     x=5      x=6
        t=0             0   --- 0     -- 1    -- 0
                        v  /    v    /   v   /
        t=1            44 -+--- 11 -+--- 2--+ -- 0
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        u = TimeFunction(name='u', grid=grid, save=2, time_order=1, space_order=0)
        step = Eq(u.forward, 2*u
                  + 3*u.subs(x, x+x.spacing)
                  + 4*u.forward.subs(x, x+x.spacing))
        op = Operator(step)

        u.data[:] = 0.0
        u.data[0, 5, 5] = 1.0

        op.apply(time_M=0)
        assert u.data[1, 5, 5] == 2
        assert u.data[1, 4, 5] == 11
        assert u.data[1, 3, 5] == 44
        assert u.data[1, 2, 5] == 4*44
        assert u.data[1, 1, 5] == 4*4*44
        assert u.data[1, 0, 5] == 4*4*4*44
        assert np.all(u.data[1, 6:, :] == 0)
        assert np.all(u.data[1, :, 0:5] == 0)
        assert np.all(u.data[1, :, 6:] == 0)

    def test_scheduling_sparse_functions(self):
        """Tests loop scheduling in presence of sparse functions."""
        grid = Grid((10, 10))
        time = grid.time_dim

        u1 = TimeFunction(name="u1", grid=grid, save=10, time_order=2)
        u2 = TimeFunction(name="u2", grid=grid, time_order=2)
        sf1 = SparseTimeFunction(name='sf1', grid=grid, npoint=1, nt=10)
        sf2 = SparseTimeFunction(name='sf2', grid=grid, npoint=1, nt=10)

        # Deliberately inject into u1, rather than u1.forward, to create a WAR w/ eqn3
        eqn1 = Eq(u1.forward, u1 + 2.0 - u1.backward)
        eqn2 = sf1.inject(u1, expr=sf1)
        eqn3 = Eq(u2.forward, u2 + 2*u2.backward - u1.dt2)
        eqn4 = sf2.interpolate(u2)

        # Note: opts disabled only because with OpenMP otherwise there might be more
        # `trees` than 6
        op = Operator([eqn1] + eqn2 + [eqn3] + eqn4, opt=('noop', {'openmp': False}))
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 5
        # Time loop not shared due to the WAR
        assert trees[0][0].dim is time and trees[0][0] is trees[1][0]  # this IS shared
        assert trees[1][0] is not trees[3][0]
        assert trees[3][0].dim is time and trees[3][0] is trees[4][0]  # this IS shared

        # Now single, shared time loop expected
        eqn2 = sf1.inject(u1.forward, expr=sf1)
        op = Operator([eqn1] + eqn2 + [eqn3] + eqn4, opt=('noop', {'openmp': False}))
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 5
        assert all(trees[0][0] is i[0] for i in trees)

    def test_scheduling_with_free_dims(self):
        """Tests loop scheduling in presence of free dimensions."""
        grid = Grid((4, 4))
        time = grid.time_dim
        x, y = grid.dimensions

        u = TimeFunction(name="u", grid=grid)
        f = Function(name="f", grid=grid)

        eq0 = Eq(u.forward, u + 1)
        eq1 = Eq(f, time*2)

        # Note that `eq1` doesn't impose any constraint on the ordering of
        # the `time` Dimension w.r.t. the `grid` Dimensions, as `time` appears
        # as a free Dimension and not within an array access such as [time, x, y]
        op = Operator([eq0, eq1], opt='topofuse')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == 3
        assert tree[0].dim is time
        assert tree[1].dim is x
        assert tree[2].dim is y

    def test_issue_1897(self):
        grid = Grid(shape=(11, 11, 11))

        v = VectorTimeFunction(name='v', grid=grid, time_order=1, space_order=4)
        tau = TensorTimeFunction(name='tau', grid=grid, time_order=1, space_order=4)

        eqns = [
            Eq(v.forward, div(tau) + v),
            Eq(tau.forward, grad(v.forward) + tau)
        ]

        op = Operator(eqns)

        assert_structure(
            op,
            ['t,x0_blk0,y0_blk0,x,y,z', 't,x1_blk0,y1_blk0,x,y,z'],
            't,x0_blk0,y0_blk0,x,y,z,x1_blk0,y1_blk0,x,y,z'
        )

    def test_barrier_halts_topofuse(self):
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        time = grid.time_dim

        f = Function(name='f', grid=grid)
        tu = TimeFunction(name='tu', grid=grid)
        tv = TimeFunction(name='tv', grid=grid)

        eqns0 = [Eq(tv, 0),
                 Eq(tv[t-1, z, z, z], 1),
                 Eq(tu, f + 1)]

        # No surprises here -- the third equation gets swapped with the second
        # one so as to be fused with the first equation
        op0 = Operator(eqns0, opt=('advanced', {'openmp': True}))
        assert_structure(op0, ['t,x,y,z', 't', 't,z'], 't,x,y,z,z')

        class DummyBarrier(sympy.Function, Barrier):
            pass

        eqns1 = list(eqns0)
        eqns1[1] = Eq(Symbol('dummy'), DummyBarrier(time))

        op1 = Operator(eqns1, opt=('advanced', {'openmp': True}))
        assert_structure(op1, ['t,x,y,z', 't', 't,x,y,z'], 't,x,y,z,x,y,z')

        # Again, but now a swap is performed *before* the barrier so it's legal
        eqns2 = list(eqns0)
        eqns2.append(eqns1[1])

        op2 = Operator(eqns2, opt=('advanced', {'openmp': True}))
        assert_structure(op2, ['t,x,y,z', 't', 't,z'], 't,x,y,z,z')

    def test_array_shared_w_topofuse(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions
        i = Dimension('i')

        a0 = Array(name='a0', dimensions=(x, y), halo=((2, 2), (2, 2)),
                   scope='shared')
        a1 = Array(name='a1', dimensions=(x, y), halo=((2, 2), (2, 2)),
                   scope='shared')
        w = Array(name='w', dimensions=(i,))
        s = Symbol(name='s')

        eqns = [Eq(a1, 1),
                Inc(s, w*a1[x+2, y], implicit_dims=(i, x, y)),
                Eq(a0, 2)]

        # For thread-shared Arrays, WAR dependencies shouldn't prevent topo-fusion
        # opportunities, since they're not really WAR's as classic Lamport
        # theory would tag
        op = Operator(eqns, opt=('advanced', {'openmp': True}))
        assert_structure(op, ['x,y', 'i,x,y'], 'x,y,i,x,y')

    def test_topofuse_w_numeric_dim(self):
        r = Dimension('r')
        i = CustomDimension('i', 0, 3)

        a = Array(name='a', dimensions=(i,))
        b = Array(name='b', dimensions=(r,))
        f = Function(name='f', dimensions=(r, i), shape=(3, 4))
        g = Function(name='g', dimensions=(r, i), shape=(3, 4))

        eqns = [Eq(a[i], r, implicit_dims=(r, i)),
                Eq(f, 1),
                Eq(b[r], 2),
                Eq(g, a[4])]

        op = Operator(eqns)

        assert_structure(op, ['r,i', 'r'], 'r,i')

    @pytest.mark.parametrize('eqns, expected, exp_trees, exp_iters', [
        (['Eq(u[0, x], 1)',
            'Eq(u[1, x], u[0, x + h_x] + u[0, x - h_x] - 2*u[0, x])'],
            np.array([[1., 1., 1.], [-1., 0., -1.]]),
            ['x', 'x'], 'x,x')
    ])
    def test_2194(self, eqns, expected, exp_trees, exp_iters):
        grid = Grid(shape=(3, ))
        u = TimeFunction(name='u', grid=grid)
        x = grid.dimensions[0]
        h_x = x.spacing  # noqa: F841

        for i, e in enumerate(list(eqns)):
            eqns[i] = eval(e)

        op = Operator(eqns)
        assert_structure(op, exp_trees, exp_iters)

        op.apply()
        assert(np.all(u.data[:] == expected[:]))

    @pytest.mark.parametrize('eqns, expected, exp_trees, exp_iters', [
        (['Eq(u[0, y], 1)', 'Eq(u[1, y], u[0, y + 1])'],
            np.array([[1., 1.], [1., 0.]]),
            ['y', 'y'], 'y,y'),
        (['Eq(u[0, y], 1)', 'Eq(u[1, y], u[0, 2])'],
            np.array([[1., 1.], [0., 0.]]),
            ['y', 'y'], 'y,y'),
        (['Eq(u[0, y], 1)', 'Eq(u[1, y], u[0, 1])'],
            np.array([[1., 1.], [1., 1.]]),
            ['y', 'y'], 'y,y'),
        (['Eq(u[0, y], 1)', 'Eq(u[1, y], u[0, y + 1])'],
            np.array([[1., 1.], [1., 0.]]),
            ['y', 'y'], 'y,y'),
        (['Eq(u[0, 1], 1)', 'Eq(u[x, y], u[0, y])'],
            np.array([[0., 1.], [0., 1.]]),
            ['xy'], 'x,y')
    ])
    def test_2194_v2(self, eqns, expected, exp_trees, exp_iters):
        grid = Grid(shape=(2, 2))
        u = Function(name='u', grid=grid)
        x, y = grid.dimensions

        for i, e in enumerate(list(eqns)):
            eqns[i] = eval(e)

        op = Operator(eqns)
        assert_structure(op, exp_trees, exp_iters)

        op.apply()
        assert(np.all(u.data[:] == expected[:]))


class TestInternals:

    def test_indirection(self):
        nt = 10
        grid = Grid(shape=(4, 4))
        time = grid.time_dim
        x, y = grid.dimensions

        f = TimeFunction(name='f', grid=grid, save=nt)
        g = TimeFunction(name='g', grid=grid)

        idx = time + 1
        s = Indirection(name='ofs0', mapped=idx)

        eqns = [
            Eq(s, idx),
            Eq(f[s, x, y], g + 3.)
        ]

        op = Operator(eqns)

        assert op._dspace[time].lower == 0
        assert op._dspace[time].upper == 1
        assert op.arguments()['time_M'] == nt - 2

        op()

        assert np.all(f.data[0] == 0.)
        assert np.all(f.data[i] == 3. for i in range(1, 10))
