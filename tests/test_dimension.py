from itertools import product
from copy import deepcopy

import numpy as np
from sympy import And, Or
import pytest

from conftest import assert_blocking, assert_structure, skipif, opts_tiling
from devito import (ConditionalDimension, Grid, Function, TimeFunction, floor,  # noqa
                    SparseFunction, SparseTimeFunction, Eq, Operator, Constant,
                    Dimension, DefaultDimension, SubDimension, switchconfig,
                    SubDomain, Lt, Le, Gt, Ge, Ne, Buffer, sin, SpaceDimension,
                    CustomDimension, dimensions, configuration, norm, Inc, sum)
from devito.ir.iet import (Conditional, Expression, Iteration, FindNodes,
                           FindSymbols, retrieve_iteration_tree)
from devito.ir.equations.algorithms import concretize_subdims
from devito.ir import SymbolRegistry
from devito.symbolics import indexify, retrieve_functions, IntDiv, INT
from devito.types import Array, StencilDimension, Symbol
from devito.types.basic import Scalar
from devito.types.dimension import AffineIndexAccessFunction, Thickness


class TestIndexAccessFunction:

    def test_basic(self):
        d = Dimension(name='x')

        expr = d + 1

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1
        assert expr.sds == ()

        s0 = Symbol(name='s0', dtype=np.int32)
        s1 = Symbol(name='s1', dtype=np.int32)

        expr = d + s0 + s1 + 1

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == s0 + s1 + 1
        assert expr.sds == ()

    def test_reversed(self):
        d = Dimension(name='x')

        expr = 1 + d

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1
        assert expr.sds == ()

        expr = d.symbolic_max + d

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs is d.symbolic_max
        assert expr.sds == ()

    def test_non_affine(self):
        grid = Grid(shape=(3,))
        x, = grid.dimensions

        f = Function(name='f', grid=grid)

        expr = x + f[x]

        assert expr.is_Add
        assert not isinstance(expr, AffineIndexAccessFunction)

    def test_stencil_dim(self):
        d = Dimension(name='x')
        sd = StencilDimension('i', 0, 1)

        expr = d + sd + 1

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1
        assert expr.sds == (sd,)

        s = Symbol(name='s')

        expr = sd + d + 1 + s

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1 + s
        assert expr.sds == (sd,)

        expr = sd + 1 + d

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1
        assert expr.sds == (sd,)

    def test_stencil_dim_multiple(self):
        d = Dimension(name='x')
        sd0 = StencilDimension('i0', 0, 1)
        sd1 = StencilDimension('i1', 0, 1)

        expr = d + sd0 + sd1 + 1

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1
        assert expr.sds == (sd0, sd1)

        s = Symbol(name='s')

        expr = sd0 + d + sd1 + 1 + s

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == 1 + s
        assert expr.sds == (sd0, sd1)

    def test_sub(self):
        d = Dimension(name='x')

        expr = d - 1

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == -1

        sd = StencilDimension('i', 0, 1)
        s = Symbol(name='s')

        expr = d + sd - 1 - s

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d is d
        assert expr.ofs == -1 - s
        assert expr.sds == (sd,)

        expr = d + 1 + sd - d

        assert isinstance(expr, AffineIndexAccessFunction)
        assert expr.d == 0
        assert expr.ofs == 1
        assert expr.sds == (sd,)


class TestBufferedDimension:

    def test_multi_buffer(self):
        grid = Grid((3, 3))
        f = TimeFunction(name="f", grid=grid)
        g = TimeFunction(name="g", grid=grid, save=Buffer(7))

        op = Operator([Eq(f.forward, 1), Eq(g, f.forward)])
        op(time_M=3)
        # f looped all time_order buffer and is 1 everywhere
        assert np.allclose(f.data, 1)
        # g looped indices 0 to 3, rest is still 0
        assert np.allclose(g.data[0:4], 1)
        assert np.allclose(g.data[4:], 0)

    def test_multi_buffer_long_time(self):
        grid = Grid((3, 3))
        time = grid.time_dim
        f = TimeFunction(name="f", grid=grid)
        g = TimeFunction(name="g", grid=grid, save=Buffer(7))

        op = Operator([Eq(f.forward, time), Eq(g, time+1)])
        op(time_M=20)
        # f[0] is time=19, f[1] is time=20
        assert np.allclose(f.data[0], 19)
        assert np.allclose(f.data[1], 20)
        # g is time 15 to 21 (loop twice the 7 buffer then 15->21)
        for i in range(7):
            assert np.allclose(g.data[i], 14+i+1)

    @pytest.mark.parametrize('time_order', [1, 2, 5])
    def test_modulo_dims_generation(self, time_order):
        grid = Grid((3, 3))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name="f", grid=grid, save=Buffer(2), time_order=time_order)

        # Irrespective of time_order -- Buffer takes precedence
        assert f.shape == (2, 3, 3)

        op = Operator([Eq(f[t+1, x, y], f + f[t-1, x, y] + 1)])

        # Check that only *two*, and not three (t-1, t, t+1)  ModuloDimensions
        # have been generated
        assert len([i for i in FindSymbols('dimensions').visit(op) if i.is_Modulo]) == 2

        op(time_M=3)
        assert np.all(f.data[0] == 4)
        assert np.all(f.data[1] == 2)

    def test_modulo_dims_generation_v2(self):
        grid = Grid((3, 3))

        f = TimeFunction(name="f", grid=grid, save=Buffer(5))

        op = Operator([Eq(f.forward, f + f.backward + 1)])

        assert len([i for i in FindSymbols('dimensions').visit(op) if i.is_Modulo]) == 3

        op(time_M=3)

        # NOTE: default is time_m=1, hence the first written entry is f.data[2]
        assert np.all(f.data[0:2] == 0)
        assert np.all(f.data[2] == 1)
        assert np.all(f.data[3] == 2)
        assert np.all(f.data[4] == 4)

    def test_degenerate_to_zero(self):
        """
        Check that if `save=Buffer(1)` is used, then the TimeFunction doesn't
        need any ModuloDimension for indexing.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, save=Buffer(1))

        eq = Eq(u.forward, u + 1)

        op = Operator(eq)

        assert len([i for i in FindSymbols('dimensions').visit(op) if i.is_Modulo]) == 0

        op.apply(time_M=9)

        assert np.all(u.data == 10)


class TestSubDimension:

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_interior(self, opt):
        """
        Tests application of an Operator consisting of a single equation
        over the ``interior`` subdomain.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid)

        eqn = [Eq(u.forward, u + 2, subdomain=interior)]

        op = Operator(eqn, opt=opt)
        op.apply(time_M=2)
        assert np.all(u.data[1, 1:-1, 1:-1, 1:-1] == 6.)
        assert np.all(u.data[1, :, 0] == 0.)
        assert np.all(u.data[1, :, -1] == 0.)
        assert np.all(u.data[1, :, :, 0] == 0.)
        assert np.all(u.data[1, :, :, -1] == 0.)

    def test_domain_vs_interior(self):
        """
        Tests application of an Operator consisting of two equations, one
        over the whole domain (default), and one over the ``interior`` subdomain.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid)  # noqa
        eqs = [Eq(u.forward, u + 1),
               Eq(u.forward, u.forward + 2, subdomain=interior)]

        op = Operator(eqs, opt='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2

        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, :, 0, :] == 1)
        assert np.all(u.data[1, :, -1, :] == 1)
        assert np.all(u.data[1, :, :, 0] == 1)
        assert np.all(u.data[1, :, :, -1] == 1)
        assert np.all(u.data[1, 1:3, 1:3, 1:3] == 3)

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_subdim_middle(self, opt):
        """
        Tests that instantiating SubDimensions using the classmethod
        constructors works correctly.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=1,
                                 thickness_right=1)
        assert xi.is_middle
        assert not xi.is_left
        assert not xi.is_right

        eqs = [Eq(u.forward, u + 1)]
        eqs = [e.subs(x, xi) for e in eqs]

        op = Operator(eqs, opt=opt)

        u.data[:] = 1.0
        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, 1:3, :, :] == 2)

    def test_symbolic_size(self):
        """Check the symbolic size of all possible SubDimensions is as expected."""
        grid = Grid(shape=(4,))
        x, = grid.dimensions
        thickness = 4

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        assert xleft.is_left
        assert not xleft.is_middle
        assert xleft.symbolic_size == xleft.thickness.left

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        assert xi.symbolic_size == (x.symbolic_max - x.symbolic_min -
                                    xi.thickness.left - xi.thickness.right + 1)

        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)
        assert xright.symbolic_size == xright.thickness.right

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_bcs(self, opt):
        """
        Tests application of an Operator consisting of multiple equations
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xleft = SubDimension.left(name='xleft', parent=x, thickness=thickness)
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)
        xright = SubDimension.right(name='xright', parent=x, thickness=thickness)
        assert xright.is_right
        assert not xright.is_middle
        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        t_in_centre = Eq(u[t+1, xi, yi], 1)
        leftbc = Eq(u[t+1, xleft, yi], u[t+1, xleft+1, yi] + 1)
        rightbc = Eq(u[t+1, xright, yi], u[t+1, xright-1, yi] + 1)

        op = Operator([t_in_centre, leftbc, rightbc], opt=opt)

        op.apply(time_m=1, time_M=1)

        assert np.all(u.data[0, :, 0:thickness] == 0.)
        assert np.all(u.data[0, :, -thickness:] == 0.)
        assert all(np.all(u.data[0, i, thickness:-thickness] == (thickness+1-i))
                   for i in range(thickness))
        assert all(np.all(u.data[0, -i, thickness:-thickness] == (thickness+2-i))
                   for i in range(1, thickness + 1))
        assert np.all(u.data[0, thickness:-thickness, thickness:-thickness] == 1.)

    def test_flow_detection_interior(self):
        """
        Test detection of flow directions when SubDimensions are used
        (in this test they are induced by the ``interior`` subdomain).

        Stencil uses values at new timestep as well as those at previous ones
        This forces an evaluation order onto x.
        Weights are:

               x=0     x=1     x=2     x=3
         t=N    2    ---3
                v   /
         t=N+1  o--+----4

        Flow dependency should traverse x in the negative direction

               x=2     x=3     x=4     x=5      x=6
        t=0             0   --- 0     -- 1    -- 0
                        v  /    v    /   v   /
        t=1            44 -+--- 11 -+--- 2--+ -- 0
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions

        interior = grid.interior

        u = TimeFunction(name='u', grid=grid, save=10, time_order=1, space_order=0)

        step = Eq(u.forward, 2*u
                  + 3*u.subs(x, x+x.spacing)
                  + 4*u.forward.subs(x, x+x.spacing),
                  subdomain=interior)
        op = Operator(step)

        u.data[0, 5, 5] = 1.0
        op.apply(time_M=0)
        assert u.data[1, 5, 5] == 2
        assert u.data[1, 4, 5] == 11
        assert u.data[1, 3, 5] == 44
        assert u.data[1, 2, 5] == 4*44
        assert u.data[1, 1, 5] == 4*4*44

        # This point isn't updated because of the `interior` selection
        assert u.data[1, 0, 5] == 0

        assert np.all(u.data[1, 6:, :] == 0)
        assert np.all(u.data[1, :, 0:5] == 0)
        assert np.all(u.data[1, :, 6:] == 0)

    @pytest.mark.parametrize('exprs,expected,', [
        # Carried dependence in both /t/ and /x/
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y])'], 'y'),
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y], subdomain=interior)'], 'y'),
        # Carried dependence in both /t/ and /y/
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y])'], 'x'),
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], subdomain=interior)'], 'x'),
        # Carried dependence in /y/, leading to separate /y/ loops, one
        # going forward, the other backward
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], subdomain=interior)',
          'Eq(u[t+1, x, y], u[t+1, x, y+1] + u[t, x, y], subdomain=interior)'], 'x'),
    ])
    def test_iteration_property_parallel(self, exprs, expected):
        """Tests detection of sequental and parallel Iterations when applying
        equations over different subdomains."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        t = grid.time_dim  # noqa

        interior = grid.interior  # noqa

        u = TimeFunction(name='u', grid=grid, save=10, time_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, opt='noop')
        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Sequential for i in iterations if i.dim.name != expected)
        assert all(i.is_Parallel for i in iterations if i.dim.name == expected)

    @skipif(['device'])
    @pytest.mark.parametrize('exprs,expected,', [
        # All parallel, the innermost Iteration gets vectorized
        (['Eq(u[time, x, yleft], u[time, x, yleft] + 1.)'], ['y']),
        # All outers are parallel, carried dependence in `yleft`, so the middle
        # Iteration over `x` gets vectorized
        (['Eq(u[time, x, yleft], u[time, x, yleft+1] + 1.)'], ['x']),
        # Only the middle Iteration is parallel, so no vectorization (the Iteration
        # is left non-vectorised for OpenMP parallelism)
        (['Eq(u[time+1, x, yleft], u[time, x, yleft+1] + u[time+1, x, yleft+1])'], [])
    ])
    def test_iteration_property_vector(self, exprs, expected):
        """Tests detection of vector Iterations when using subdimensions."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        time = grid.time_dim  # noqa

        # The leftmost 10 elements
        yleft = SubDimension.left(name='yleft', parent=y, thickness=10) # noqa

        u = TimeFunction(name='u', grid=grid, save=10, time_order=0, space_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, opt='simd')
        iterations = FindNodes(Iteration).visit(op)
        vectorized = [i.dim.name for i in iterations if i.is_Vectorized]
        assert set(vectorized) == set(expected)

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_subdimmiddle_parallel(self, opt):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # a 5 point stencil that can be computed in parallel
        centre = Eq(u[t+1, xi, yi], u[t, xi, yi] + u[t, xi-1, yi] +
                    u[t, xi+1, yi] + u[t, xi, yi-1] + u[t, xi, yi+1])

        u.data[0, 10, 10] = 1.0

        op = Operator([centre], opt=opt)

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim in [xi, yi])

        op.apply(time_m=0, time_M=0)

        assert np.all(u.data[1, 9:12, 10] == 1.0)
        assert np.all(u.data[1, 10, 9:12] == 1.0)

        # Other than those, it should all be 0
        u.data[1, 9:12, 10] = 0.0
        u.data[1, 10, 9:12] = 0.0
        assert np.all(u.data[1, :] == 0)

    def test_subdimleft_parallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        This tests that flow direction is not being automatically inferred
        from whether the subdimension is on the left or right boundary.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xl = SubDimension.left(name='xl', parent=x, thickness=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # Can be done in parallel
        eq = Eq(u[t+1, xl, yi], u[t, xl, yi] + 1)

        op = Operator([eq])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim in [xl, yi])

        op.apply(time_m=0, time_M=0)

        assert np.all(u.data[1, 0:thickness, 0:thickness] == 0)
        assert np.all(u.data[1, 0:thickness, -thickness:] == 0)
        assert np.all(u.data[1, 0:thickness, thickness:-thickness] == 1)
        assert np.all(u.data[1, thickness+1:, :] == 0)

    @pytest.mark.parametrize('thickness,flag', [
        (4, True),
        (8, False)
    ])
    def test_subdim_local_parallel(self, thickness, flag):
        """
        A variation of `test_subdimleft_parallel` where the thickness, whose
        value is statically known, explicitly appears in the equations.
        """
        grid = Grid(shape=(30, 30, 30))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)

        zl = SubDimension.left(name='zl', parent=z, thickness=thickness)

        eqns = [Eq(u[t, x, y, zl], u[t, x, y, 8 - zl]),
                Eq(v[t, x, y, zl], v[t, x, y, 8 - zl])]

        op = Operator(eqns)

        if flag:
            assert_structure(op, ['t,x,y,z'], 't,x,y,z')
        else:
            assert_structure(op, ['t,x,y,z', 't,x,y,z'], 't,x,y,z,z')

    def test_subdimmiddle_notparallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        Different from ``test_subdimmiddle_parallel`` because an interior
        dimension cannot be evaluated in parallel.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=0, time_order=1)

        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=thickness, thickness_right=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # flow dependencies in x and y which should force serial execution
        # in reverse direction
        centre = Eq(u[t+1, xi, yi], u[t, xi, yi] + u[t+1, xi+1, yi+1])
        u.data[0, 10, 10] = 1.0

        op = Operator([centre])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Sequential for i in iterations if i.dim == xi)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim == yi)

        op.apply(time_m=0, time_M=0)

        for i in range(4, 11):
            assert u.data[1, i, i] == 1.0
            u.data[1, i, i] = 0.0

        assert np.all(u.data[1, :] == 0)

    def test_subdimleft_notparallel(self):
        """
        Tests application of an Operator consisting of a subdimension
        defined over different sub-regions, explicitly created through the
        use of SubDimensions.

        This tests that flow direction is not being automatically inferred
        from whether the subdimension is on the left or right boundary.
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', save=None, grid=grid, space_order=1, time_order=0)

        xl = SubDimension.left(name='xl', parent=x, thickness=thickness)

        yi = SubDimension.middle(name='yi', parent=y,
                                 thickness_left=thickness, thickness_right=thickness)

        # Flows inward (i.e. forward) rather than outward
        eq = Eq(u[t+1, xl, yi], u[t+1, xl-1, yi] + 1)

        op = Operator([eq])

        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Affine and i.is_Sequential for i in iterations if i.dim == xl)
        assert all(i.is_Affine and i.is_Parallel for i in iterations if i.dim == yi)

        op.apply(time_m=1, time_M=1)

        assert all(np.all(u.data[0, :thickness, thickness+i] == [1, 2, 3, 4])
                   for i in range(12))
        assert np.all(u.data[0, thickness:] == 0)
        assert np.all(u.data[0, :, thickness+12:] == 0)

    def test_subdimmiddle_subdimleft_blocked(self):
        grid = Grid(shape=(20, 20, 20))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        thickness = 4

        u = TimeFunction(name='u', grid=grid)

        zl = SubDimension.left(name='zl', parent=z, thickness=thickness)
        zi = SubDimension.middle(name='zi', parent=z, thickness_left=thickness,
                                 thickness_right=thickness)

        eqns = [Eq(u[t+1, x, y, zi], u[t, x, y, zi] + 1),
                Eq(u[t+1, x, y, zl], u[t, x, y, zl] + 1)]

        op = Operator(eqns)

        # Check generated code -- expected loop blocking over x and y, with the
        # two z loops, zi and zl, within y
        # Note that the zi and zl iterators are renamed to remove clutter given the loop
        # structure
        assert_structure(op,
                         ['t,x0_blk0,y0_blk0,x,y,z', 't,x0_blk0,y0_blk0,x,y,z'],
                         't,x0_blk0,y0_blk0,x,y,z,z')

        op.apply(time_M=0)
        assert np.all(u.data[0] == 0)
        assert np.all(u.data[1, :, :, :-thickness] == 1)
        assert np.all(u.data[1, :, :, -thickness:] == 0)

    def test_subdim_blocked_w_partial_reuse(self):
        grid = Grid(shape=(12, 12, 12))
        x, y, z = grid.dimensions

        f = TimeFunction(name='f', grid=grid, space_order=4)

        zr = SubDimension.right(name='zr', parent=z, thickness=4)

        eqns = [Eq(f.forward, f.dx2 + 1.),
                Eq(f.forward, 0).subs(z, zr)]

        op = Operator(eqns)

        # Check generated code -- expected loop blocking
        # Note that loop structure means zr is renamed to z to remove clutter
        assert_structure(op,
                         ['t,x0_blk0,y0_blk0,x,y,z', 't,x0_blk0,y0_blk0,x,y,z'],
                         't,x0_blk0,y0_blk0,x,y,z,z')

    def test_subdim_fd(self):
        """
        Test that the FD shortcuts are handled correctly with SubDimensions
        """
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions

        u = TimeFunction(name='u', save=None, grid=grid, space_order=1, time_order=1)
        u.data[:] = 2.

        # Flows inward (i.e. forward) rather than outward
        eq = [Eq(u.forward, u.dx + u.dy, subdomain=grid.interior)]

        op = Operator(eq)

        op.apply(time_M=0)

        assert np.all(u.data[1, -1, :] == 2.)
        assert np.all(u.data[1, :, 0] == 2.)
        assert np.all(u.data[1, :, -1] == 2.)
        assert np.all(u.data[1, 0, :] == 2.)
        assert np.all(u.data[1, 1:18, 1:18] == 0.)

    def test_arrays_defined_over_subdims(self):
        """
        Check code generation when an Array uses a SubDimension.
        """
        grid = Grid(shape=(3,))
        x, = grid.dimensions
        xi, = grid.interior.dimensions

        f = Function(name='f', grid=grid)
        a = Array(name='a', dimensions=(xi,), dtype=grid.dtype)
        op = Operator([Eq(a[xi], 1), Eq(f, f + a[xi + 1], subdomain=grid.interior)],
                      opt=('advanced', {'openmp': False}))

        assert len(op.parameters) == 6
        # neither `x_size` nor `xi_size` are expected here
        assert not any(i.name in ('x_size', 'xi_size') for i in op.parameters)
        # Try running it -- regardless of what it will produce, this should run
        # ie, this checks this error isn't raised:
        # "ValueError: No value found for parameter xi_size"
        op()

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_expandingbox_like(self, opt):
        """
        Make sure SubDimensions aren't an obstacle to expanding boxes.
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions

        u = TimeFunction(name='u', grid=grid)
        xi = SubDimension.middle(name='xi', parent=x, thickness_left=2, thickness_right=2)
        yi = SubDimension.middle(name='yi', parent=y, thickness_left=2, thickness_right=2)

        eqn = Eq(u.forward, u + 1)
        eqn = eqn.subs({x: xi, y: yi})

        op = Operator(eqn, opt=opt)

        op.apply(time=3, x_m=2, x_M=5, y_m=2, y_M=5,
                 x_ltkn0=0, x_rtkn0=0, y_ltkn0=0, y_rtkn0=0)

        assert np.all(u.data[0, 2:-2, 2:-2] == 4.)
        assert np.all(u.data[1, 2:-2, 2:-2] == 3.)
        assert np.all(u.data[:, :2] == 0.)
        assert np.all(u.data[:, -2:] == 0.)
        assert np.all(u.data[:, :, :2] == 0.)
        assert np.all(u.data[:, :, -2:] == 0.)

    def test_standalone_thickness(self):
        x = Dimension('x')
        ix = SubDimension.left('ix', x, 2)
        f = Function(name="f", dimensions=(ix,), shape=(5,), dtype=np.int32)

        eqns = Eq(f[ix.symbolic_max], 1)

        op = Operator(eqns)
        assert 'x_ltkn0' in str(op.ccode)
        op(x_m=0)
        assert np.all(f.data == np.array([0, 1, 0, 0, 0]))


class TestConditionalDimension:

    """
    A collection of tests to check the correct functioning of ConditionalDimensions.
    """

    def test_basic(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.), Eq(usave, u)]
        op = Operator(eqns)
        op.apply()
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

    def test_overrides(self):
        # Check overrides
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        f1, f2 = 4, 5
        n1, n2 = (nt+f1-1)//f1, (nt+f2-1)//f2
        t1 = ConditionalDimension('t_sub1', parent=time, factor=f1)
        t2 = ConditionalDimension('t_sub2', parent=time, factor=f2)
        u1 = TimeFunction(name='usave1', grid=grid, save=n1, time_dim=t1)
        u2 = TimeFunction(name='usave2', grid=grid, save=n2, time_dim=t2)
        assert(t1 in u1.indices)
        assert(t2 in u2.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u1, u), Eq(u2, u)]
        op = Operator(eqns)
        op.apply(u=u, usave1=u1, usave2=u2, time_M=nt-2)

        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        for (uk, fk) in zip((u1, u2), (f1, f2)):
            assert np.all([np.allclose(uk.data[i], i*fk)
                           for i in range((nt+fk-1)//fk)])

    def test_overrides_newfact(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        f1, f2 = 4, 5
        n1, n2 = (nt+f1-1)//f1, (nt+f2-1)//f2
        t1 = ConditionalDimension('t_sub1', parent=time, factor=f1)
        t2 = ConditionalDimension('t_sub2', parent=time, factor=f2)
        u1 = TimeFunction(name='usave1', grid=grid, save=n1, time_dim=t1)
        u2 = TimeFunction(name='usave2', grid=grid, save=n2, time_dim=t2)
        assert(t1 in u1.indices)
        assert(t2 in u2.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u1, u)]
        op = Operator(eqns)
        op.apply(u=u, usave1=u1, time_M=nt-2)
        u.data.fill(0)
        op.apply(u=u, usave1=u2, time_M=nt-2)

        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        for (uk, fk) in zip((u1, u2), (f1, f2)):
            assert np.all([np.allclose(uk.data[i], i*fk)
                           for i in range((nt+fk-1)//fk)])

    def test_basic_shuffles(self):
        """
        Like ``test_basic``, but with different equation orderings. Nevertheless,
        we assert against the same exact values as in ``test_basic``, since we
        save `u`, not `u.forward`.
        """
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)

        # Shuffle 1
        eqns = [Eq(usave, u), Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

        # Shuffle 2
        usave.data[:] = 0.
        u.data[:] = 0.
        u2.data[:] = 0.
        eqns = [Eq(u.forward, u + 1.), Eq(usave, u), Eq(u2.forward, u2 + 1.)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_spacial_subsampling(self, opt):
        """
        Test conditional dimension for the spatial ones.
        This test saves u every two grid points :
        u2[x, y] = u[2*x, 2*y]
        """
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, save=nt)
        assert(grid.time_dim in u.indices)

        # Creates subsampled spatial dimensions and accordine grid
        dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                      for d in u.grid.dimensions])
        grid2 = Grid((6, 6), dimensions=dims, time_dimension=time)
        u2 = TimeFunction(name='u2', grid=grid2, save=nt)
        assert(time in u2.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2, u)]
        op = Operator(eqns, opt=opt)
        op.apply(time_M=nt-2)
        # Verify that u2[x,y]= u[2*x, 2*y]
        assert np.allclose(u.data[:-1, 0::2, 0::2], u2.data[:-1, :, :])

    def test_spacial_filtering(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        g.data[:] = [[-.7, -.8, 0, .4],
                     [-.3, -.5, 0, .6],
                     [.1, .2, -.1, .8],
                     [.5, .7, 0, .9]]

        condition = And(Ge(g, -0.5), Le(g, 0.5))
        cd = ConditionalDimension(name='cd1', parent=y, condition=condition)

        eqn = Eq(f, g, implicit_dims=cd)

        # NOTE: bypass heuristics and enforce loop blocking. Now `cd` is defined
        # along `y` but loop blocking will generate two different BlockDimensions,
        # `y0_blk0` and `Y`, in place of `y`. So the compiler must be smart
        # enough to stick `cd` inside `Y`
        opt = ('advanced', {'blockrelax': 'device-aware'})

        op = Operator(eqn, opt=opt)

        op()

        assert np.all(f.data == np.array([[0, 0, 0, .4],
                                          [-.3, -.5, 0, 0],
                                          [.1, .2, -.1, 0],
                                          [.5, 0, 0, 0]], dtype=f.dtype))

    def test_time_subsampling_fd(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        x, y = grid.dimensions
        time = grid.time_dim

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled, time_order=2)

        dx2 = {indexify(i) for i in retrieve_functions(usave.dt2.evaluate)}
        assert dx2 == {usave[time_subsampled - 1, x, y],
                       usave[time_subsampled + 1, x, y],
                       usave[time_subsampled, x, y]}

    def test_issue_1592(self):
        grid = Grid(shape=(11, 11))
        time = grid.time_dim
        time_sub = ConditionalDimension('t_sub', parent=time, factor=2)
        v = TimeFunction(name="v", grid=grid, space_order=4, time_dim=time_sub, save=5)
        w = Function(name="w", grid=grid, space_order=4)
        Operator(Eq(w, v.dx))(time=6)
        op = Operator(Eq(v.forward, v.dx))
        op.apply(time=6)
        exprs = FindNodes(Expression).visit(op)
        assert exprs[-1].expr.lhs.indices[0] == IntDiv(time, time_sub.symbolic_factor) + 1
        assert time_sub.factor == 2
        assert isinstance(time_sub.symbolic_factor, Scalar)

    def test_issue_1753(self):
        grid = Grid(shape=(3, 3, 3))
        f = TimeFunction(name='f', grid=grid)
        p = Function(name='p', grid=grid)
        p.data[0, 1, 0] = 1
        condition = Ge(p, 1)
        z_cond = ConditionalDimension(name='z_cond', parent=grid.dimensions[-1],
                                      condition=condition)
        eq_p = Eq(f.forward, 1, implicit_dims=z_cond)
        op = Operator([eq_p])
        op.apply(time_M=1)
        assert np.all(np.flatnonzero(f.data) == [3, 30])

    def test_issue_2273(self):
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        nt = 200
        bounds = (10, 100)
        factor = 5

        condition = And(Ge(time, bounds[0]), Le(time, bounds[1]))

        time_under = ConditionalDimension(name='timeu', parent=time,
                                          factor=factor, condition=condition)
        buffer_size = (bounds[1] - bounds[0] + factor) // factor + 1

        rec = SparseTimeFunction(name='rec', grid=grid, npoint=1, nt=nt,
                                 coordinates=[(.5, .5)])
        rec.data[:] = 1.0

        u = TimeFunction(name='u', grid=grid, space_order=2)
        usaved = TimeFunction(name='usaved', grid=grid, space_order=2,
                              time_dim=time_under, save=buffer_size)

        eq = [Eq(u.forward, u)] + rec.inject(field=u.forward, expr=rec) + [Eq(usaved, u)]

        op = Operator(eq)
        op(time_m=0, time_M=nt-1)
        expected = np.linspace(bounds[0], bounds[1], num=buffer_size-1)
        assert np.allclose(usaved.data[:-1, 5, 5], expected)

    def test_subsampled_fd(self):
        """
        Test that the FD shortcuts are handled correctly with ConditionalDimensions
        """
        grid = Grid(shape=(11, 11))
        time = grid.time_dim
        # Creates subsampled spatial dimensions and accordine grid
        dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                      for d in grid.dimensions])
        grid2 = Grid((6, 6), dimensions=dims, time_dimension=time)
        u2 = TimeFunction(name='u2', grid=grid2, space_order=2, time_order=1)
        u2.data.fill(2.)
        eqns = [Eq(u2.forward, u2.dx + u2.dy)]
        op = Operator(eqns)
        op.apply(time_M=0, x_M=11, y_M=11)
        # Verify that u2 contains subsampled fd values
        assert np.all(u2.data[0, :, :] == 2.)
        assert np.all(u2.data[1, 0, 0] == 0.)
        assert np.all(u2.data[1, -1, -1] == -20.)
        assert np.all(u2.data[1, 0, -1] == -10.)
        assert np.all(u2.data[1, -1, 0] == -10.)
        assert np.all(u2.data[1, 1:-1, 0] == 0.)
        assert np.all(u2.data[1, 0, 1:-1] == 0.)
        assert np.all(u2.data[1, 1:-1, -1] == -10.)
        assert np.all(u2.data[1, -1, 1:-1] == -10.)
        assert np.all(u2.data[1, 1:4, 1:4] == 0.)

    # This test generates an openmp loop form which makes older gccs upset
    @switchconfig(language='C')
    def test_nothing_in_negative(self):
        """Test the case where when the condition is false, there is nothing to do."""
        nt = 4
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', save=nt, grid=grid)
        assert(grid.time_dim in u.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(usave, u)]
        op = Operator(eqns)

        u.data[:] = 1.0
        usave.data[:] = 0.0
        op.apply(time_m=1, time_M=1)
        assert np.allclose(usave.data, 0.0)

        op.apply(time_m=0, time_M=0)
        assert np.allclose(usave.data, 1.0)

    def test_laplace(self):
        grid = Grid(shape=(20, 20, 20))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t = grid.stepping_dim
        tsave = ConditionalDimension(name='tsave', parent=time, factor=2)

        u = TimeFunction(name='u', grid=grid, save=None, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, time_dim=tsave,
                             time_order=0, space_order=0)

        steps = []
        # save of snapshot
        steps.append(Eq(usave, u))
        # standard laplace-like thing
        steps.append(Eq(u[t+1, x, y, z],
                        u[t, x, y, z] - u[t-1, x, y, z]
                        + u[t, x-1, y, z] + u[t, x+1, y, z]
                        + u[t, x, y-1, z] + u[t, x, y+1, z]
                        + u[t, x, y, z-1] + u[t, x, y, z+1]))

        op = Operator(steps)

        u.data[:] = 0.0
        u.data[0, 10, 10, 10] = 1.0
        op.apply(time_m=0, time_M=0)
        assert np.sum(u.data[0, :, :, :]) == 1.0
        assert np.sum(u.data[1, :, :, :]) == 7.0
        assert np.all(usave.data[0, :, :, :] == u.data[0, :, :, :])

    def test_as_expr(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave, time_subsampled * u)]
        op = Operator(eqns)
        op.apply(t=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor*i)
                      for i in range((nt+factor-1)//factor)])

    def test_shifted(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        t_sub_shift = Constant(name='t_sub_shift', dtype=np.int32)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave.subs(time_subsampled, time_subsampled - t_sub_shift), u)]
        op = Operator(eqns)

        # Starting at time_m=10, so time_subsampled - t_sub_shift is in range
        op.apply(time_m=10, time_M=nt-2, t_sub_shift=3)
        assert np.all(np.allclose(u.data[0], 8))
        assert np.all([np.allclose(u2.data[i], i - 10) for i in range(10, nt)])
        assert np.all([np.allclose(usave.data[i], 2+i*factor) for i in range(2)])

    def test_no_index(self):
        """Test behaviour when the ConditionalDimension is used as a symbol in
        an expression."""
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        v = Function(name='v', grid=grid)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        eqns = [Eq(u.forward, u + 1), Eq(v, v + u*u*time_subsampled)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        # expected result is 1024
        # v = u[0]**2 * 0 + u[4]**2 * 1 + u[8]**2 * 2 + u[12]**2 * 3 + u[16]**2 * 4
        # with u[t] = t
        # v = 16 * 1 + 64 * 2 + 144 * 3 + 256 * 4 = 1600
        assert np.all(np.allclose(v.data, 1600))

    def test_no_index_sparse(self):
        """Test behaviour when the ConditionalDimension is used as a symbol in
        an expression over sparse data objects."""
        grid = Grid(shape=(4, 4), extent=(3.0, 3.0))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=1)
        f.data[:] = 0.

        coordinates = [(0.5, 0.5), (0.5, 2.5), (2.5, 0.5), (2.5, 2.5)]
        sf = SparseFunction(name='sf', grid=grid, npoint=4, coordinates=coordinates)
        sf.data[:] = 1.
        sd = sf.dimensions[sf._sparse_position]

        # We want to write to `f` through `sf` so that we obtain the
        # following 4x4 grid (the '*' show the position of the sparse points)
        # We do that by emulating an injection
        #
        # 0 --- 0 --- 0 --- 0
        # |  *  |     |  *  |
        # 0 --- 1 --- 1 --- 0
        # |     |     |     |
        # 0 --- 1 --- 1 --- 0
        # |  *  |     |  *  |
        # 0 --- 0 --- 0 --- 0

        radius = 1
        indices = [(INT(floor(i)), INT(floor(i))+radius)
                   for i in sf._position_map.keys()]
        bounds = [i.symbolic_size - radius for i in grid.dimensions]

        eqs = [Eq(p, v) for (v, p) in sf._position_map.items()]
        for e, i in enumerate(product(*indices)):
            args = [j > 0 for j in i]
            args.extend([j < k for j, k in zip(i, bounds)])
            condition = And(*args, evaluate=False)
            cd = ConditionalDimension('sfc%d' % e, parent=sd, condition=condition)
            index = [time] + list(i)
            eqs.append(Eq(f[index], f[index] + sf[cd]))

        op = Operator(eqs)
        op.apply(time=0)

        assert np.all(f.data[0, 1:-1, 1:-1] == 1.)
        assert np.all(f.data[0, 0] == 0.)
        assert np.all(f.data[0, -1] == 0.)
        assert np.all(f.data[0, :, 0] == 0.)
        assert np.all(f.data[0, :, -1] == 0.)

    def test_no_index_symbolic(self):
        grid = Grid(shape=(10, 10, 10))
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid)

        v0 = Constant(name='v0', dtype=np.float32)
        v1 = Constant(name='v1', dtype=np.float32)
        condition = And(Ge(x, v0), Le(x, v1))
        cd = ConditionalDimension(name='cd', parent=x, condition=condition,
                                  indirect=True)

        eq = Eq(u.forward, u + 1, implicit_dims=cd)

        # Ensure both code generation and jitting work
        op = Operator(eq)
        op.cfunction

    @pytest.mark.parametrize('value', [0, 1])
    def test_constant_as_condition(self, value):
        x = Dimension('x')

        c = Constant(name="c", dtype=np.int8, value=value)
        cd = ConditionalDimension(name="cd", parent=x, condition=c)

        f = Function(name='f', dimensions=(x,), shape=(11,), dtype=np.int32)

        op = Operator(Eq(f, 1, implicit_dims=cd))
        op.apply()

        assert np.all(f.data == value)

    def test_symbolic_factor(self):
        """
        Test ConditionalDimension with symbolic factor (provided as a Constant).
        """
        g = Grid(shape=(4, 4, 4))

        u = TimeFunction(name='u', grid=g, time_order=0)

        fact = Constant(name='fact', dtype=np.int32, value=4)
        tsub = ConditionalDimension(name='tsub', parent=g.time_dim, factor=fact)
        usave = TimeFunction(name='usave', grid=g, time_dim=tsub, save=4)

        op = Operator([Eq(u, u + 1), Eq(usave, u)])

        op.apply(time=7)  # Use `fact`'s default value, 4
        assert np.all(usave.data[0] == 1)
        assert np.all(usave.data[1] == 5)

        u.data[:] = 0.
        op.apply(time=7, fact=2)
        assert np.all(usave.data[0] == 1)
        assert np.all(usave.data[1] == 3)
        assert np.all(usave.data[2] == 5)
        assert np.all(usave.data[3] == 7)

    def test_auto_override_factor(self):
        grid = Grid(shape=(4, 4, 4))
        time_dim = grid.time_dim

        fact = Constant(name='fact', dtype=np.int32, value=4)
        tsub = ConditionalDimension(name='tsub', parent=time_dim, factor=fact)

        u = TimeFunction(name='u', grid=grid, time_order=0)
        usave = TimeFunction(name='usave', grid=grid, time_dim=tsub, save=4)

        eqns = [Eq(u, u + 1),
                Eq(usave, u)]

        op = Operator(eqns)

        op.apply(time_M=7)  # Use `fact`'s default value, 4
        assert np.all(usave.data[0] == 1)
        assert np.all(usave.data[1] == 5)
        assert np.all(usave.data[2:] == 0)

        # Now try with a different set of input
        fact1 = Constant(name='fact', dtype=np.int32, value=3)
        tsub1 = ConditionalDimension(name='tsub', parent=time_dim, factor=fact1)

        u1 = TimeFunction(name='u', grid=grid, time_order=0)
        usave1 = TimeFunction(name='usave', grid=grid, time_dim=tsub1, save=4)

        op.apply(time_M=7, u=u1, usave=usave1)  # Expect to use tsub1's factor!
        assert np.all(usave1.data[0] == 1)
        assert np.all(usave1.data[1] == 4)
        assert np.all(usave1.data[2] == 7)
        assert np.all(usave1.data[3:] == 0)

    def test_implicit_dims(self):
        """
        Test ConditionalDimension as an implicit dimension for an equation.
        """

        # This test makes an Operator that should create a vector of increasing
        # integers, but stop incrementing when a certain stop value is reached

        shape = (50,)
        stop_value = 20

        time = Dimension(name='time')
        f = TimeFunction(name='f', shape=shape, dimensions=[time])

        # The condition to stop incrementing
        cond = ConditionalDimension(name='cond',
                                    parent=time, condition=f[time] < stop_value)

        eqs = [Eq(f.forward, f), Eq(f.forward, f.forward + 1, implicit_dims=[cond])]
        op = Operator(eqs)
        op.apply(time_M=shape[0] - 2)

        # Make the same calculation in python to assert the result
        F = np.zeros(shape[0])
        for i in range(shape[0]):
            F[i] = i if i < stop_value else stop_value

        assert np.all(f.data == F)

    def test_implict_dims_multiple(self):
        """Test supplying multiple ConditionalDimensions as implicit dimensions"""
        shape = (50,)
        start_value = 5
        stop_value = 20

        time = Dimension(name='time')
        f = TimeFunction(name='f', shape=shape, dimensions=[time])
        # The condition to start incrementing
        cond0 = ConditionalDimension(name='cond0',
                                     parent=time, condition=time > start_value)
        # The condition to stop incrementing
        cond1 = ConditionalDimension(name='cond1',
                                     parent=time, condition=time < stop_value)
        # Factor of 2
        cond2 = ConditionalDimension(name='cond2', parent=time, factor=2)

        eqs = [Eq(f.forward, f), Eq(f.forward, f.forward + 1,
                                    implicit_dims=[cond0, cond1, cond2])]
        op = Operator(eqs)
        op.apply(time_M=shape[0] - 2)

        # Make the same calculation in python to assert the result
        F = np.zeros(shape[0])
        val = 0
        for i in range(shape[0]):
            F[i] = val
            if i > start_value and i < stop_value and i % 2 == 0:
                val += 1

        assert np.all(f.data == F)

    def test_grouping(self):
        """
        Test that Clusters over the same set of ConditionalDimensions fall within
        the same Conditional. This is a follow up to issue #1610.
        """
        grid = Grid(shape=(10, 10))
        time = grid.time_dim
        cond = ConditionalDimension(name='cond', parent=time, condition=time < 5)

        u = TimeFunction(name='u', grid=grid, space_order=4)

        # We use a SubDomain only to keep the two Eqs separated
        eqns = [Eq(u.forward, u + 1, subdomain=grid.interior),
                Eq(u.forward, u.dx.dx + 1., implicit_dims=[cond])]

        op = Operator(eqns, opt=('advanced-fsg', {'cire-mingain': 1}))

        conds = FindNodes(Conditional).visit(op)
        assert len(conds) == 1
        assert len(retrieve_iteration_tree(conds[0].then_body)) == 2

    def test_grouping_v2(self):
        """
        Test that two equations sharing a strict subset of loops get scheduled
        within the same (also shared) ConditionalDimension.
        """
        grid = Grid(shape=(10, 10))
        time = grid.time_dim
        cond = ConditionalDimension(name='cond', condition=time < 5)

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(u.forward, u + 1., implicit_dims=cond),
                Eq(v.forward, v + 1., implicit_dims=cond, subdomain=grid.interior)]

        op = Operator(eqns)

        conds = FindNodes(Conditional).visit(op)
        assert len(conds) == 1
        assert len(retrieve_iteration_tree(conds[0].then_body)) == 2

    def test_stepping_dim_in_condition_lowering(self):
        """
        Check that the compiler performs lowering on conditions
        with TimeDimensions and generates the expected code::

        if (g[t][x + 1][y + 1] <= 10){          if (g[t0][x + 1][y + 1] <= 10){
            ...                          -->       ...
        }                                       }

        This test increments a function by one at every timestep until it is
        less-or-equal to 10 (g<=10) while although operator runs for 13 timesteps.
        """
        grid = Grid(shape=(4, 4))
        _, y = grid.dimensions

        ths = 10
        g = TimeFunction(name='g', grid=grid)

        ci = ConditionalDimension(name='ci', parent=y, condition=Le(g, ths))

        op = Operator(Eq(g.forward, g + 1, implicit_dims=ci))

        op.apply(time_M=ths+3)
        assert np.all(g.data[0, :, :] == ths)
        assert np.all(g.data[1, :, :] == ths + 1)
        assert 'if (g[t0][x + 1][y + 1] <= 10)\n'
        '{\n g[t1][x + 1][y + 1] = g[t0][x + 1][y + 1] + 1' in str(op.ccode)

    def test_expr_like_lowering(self):
        """
        Test the lowering of an expr-like ConditionalDimension's condition.
        This test makes an Operator that should indexify and lower the condition
        passed in the Conditional Dimension
        """

        grid = Grid(shape=(3, 3))
        g1 = Function(name='g1', grid=grid)
        g2 = Function(name='g2', grid=grid)

        g1.data[:] = 0.49
        g2.data[:] = 0.49
        x, y = grid.dimensions
        ci = ConditionalDimension(name='ci', parent=y, condition=Le((g1 + g2),
                                  1.01*(g1 + g2)))

        f = Function(name='f', shape=grid.shape, dimensions=(x, ci))
        Operator(Eq(f, g1+g2)).apply()

        assert np.all(f.data[:] == g1.data[:] + g2.data[:])

    @pytest.mark.parametrize('setup_rel, rhs, c1, c2, c3, c4', [
        # Relation, RHS, c1 to c4 used as indexes in assert
        (Lt, 3, 2, 4, 4, -1), (Le, 2, 2, 4, 4, -1), (Ge, 3, 4, 6, 1, 4),
        (Gt, 2, 4, 6, 1, 4), (Ne, 5, 2, 6, 1, 2)
    ])
    def test_relational_classes(self, setup_rel, rhs, c1, c2, c3, c4):
        """
        Test ConditionalDimension using conditions based on Relations over SubDomains.
        """

        class InnerDomain(SubDomain):
            name = 'inner'

            def define(self, dimensions):
                return {d: ('middle', 2, 2) for d in dimensions}

        inner_domain = InnerDomain()
        grid = Grid(shape=(8, 8), subdomains=(inner_domain,))
        g = Function(name='g', grid=grid)
        g2 = Function(name='g2', grid=grid)

        for i in [g, g2]:
            i.data[:4, :4] = 1
            i.data[4:, :4] = 2
            i.data[4:, 4:] = 3
            i.data[:4, 4:] = 4

        xi, yi = grid.subdomains['inner'].dimensions

        cond = setup_rel(0.25*g + 0.75*g2, rhs, subdomain=grid.subdomains['inner'])
        ci = ConditionalDimension(name='ci', parent=yi, condition=cond)
        f = Function(name='f', shape=grid.shape, dimensions=(xi, ci))

        eq1 = Eq(f, 0.4*g + 0.6*g2)
        eq2 = Eq(f, 5)

        Operator([eq1, eq2]).apply()
        assert np.all(f.data[2:6, c1:c2] == 5.)
        assert np.all(f.data[:, c3:c4] < 5.)

    def test_from_cond_to_param(self):
        """
        Test that Functions appearing in the condition of a ConditionalDimension
        but not explicitly in an Eq are actually part of the Operator input
        (stems from issue #1298).
        """
        grid = Grid(shape=(8, 8))
        x, y = grid.dimensions

        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        ci = ConditionalDimension(name='ci', parent=y, condition=Lt(g, 2 + h))
        f = Function(name='f', shape=grid.shape, dimensions=(x, ci))

        for _ in range(5):
            # issue #1298 was non deterministic
            Operator(Eq(f, 5)).apply()

    @skipif('device')
    def test_no_fusion_simple(self):
        """
        If ConditionalDimensions are present, then Clusters must not be fused so
        that ultimately Eqs get scheduled to different loop nests.
        """
        grid = Grid(shape=(4, 4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        # No ConditionalDimensions yet. Will be fused and optimized
        eqns = [Eq(f.forward, f + 1),
                Eq(h, f + 1),
                Eq(g, f + 1)]

        op = Operator(eqns)

        bns, _ = assert_blocking(op, {'x0_blk0'})

        exprs = FindNodes(Expression).visit(bns['x0_blk0'])
        assert len(exprs) == 4
        assert exprs[1].expr.rhs is exprs[0].output
        assert exprs[2].expr.rhs is exprs[0].output
        assert exprs[3].expr.rhs is exprs[0].output

        # Now with a ConditionalDimension. No fusion, no optimization
        ctime = ConditionalDimension(name='ctime', parent=time, condition=time > 4)

        eqns = [Eq(f.forward, f + 1),
                Eq(h, f + 1),
                Eq(g, f.dx + h + 1, implicit_dims=[ctime])]

        op = Operator(eqns)

        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0'})

        exprs = FindNodes(Expression).visit(bns['x0_blk0'])
        assert len(exprs) == 3
        assert exprs[1].expr.rhs is exprs[0].output
        assert exprs[2].expr.rhs is exprs[0].output
        exprs = FindNodes(Expression).visit(bns['x1_blk0'])
        assert len(exprs) == 1

    @skipif('device')
    def test_no_fusion_convoluted(self):
        """
        Conceptually like `test_no_fusion_simple`, but with more expressions
        and non-trivial data flow.
        """
        grid = Grid(shape=(4, 4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        ctime = ConditionalDimension(name='ctime', parent=time, condition=time > 4)

        eqns = [Eq(f.forward, f + 1),
                Eq(h, f + 1),
                Eq(g, f + h + 1, implicit_dims=[ctime]),
                Eq(f.forward, f + h + 1, implicit_dims=[ctime]),
                Eq(f.forward, f.dx + h + 1),
                Eq(g, f.dx + h + 1)]

        op = Operator(eqns)

        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0', 'x2_blk0'})

        exprs = FindNodes(Expression).visit(bns['x0_blk0'])
        assert len(exprs) == 3
        assert exprs[1].expr.rhs is exprs[0].output
        assert exprs[2].expr.rhs is exprs[0].output

        exprs = FindNodes(Expression).visit(bns['x1_blk0'])
        assert len(exprs) == 3

        exprs = FindNodes(Expression).visit(bns['x2_blk0'])
        assert len(exprs) == 3
        assert exprs[1].expr.rhs is exprs[0].output
        assert exprs[2].expr.rhs is exprs[0].output

    def test_affiness(self):
        """
        Test for issue #1616.
        """
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)

        eqns = [Eq(u.forward, u + 1.), Eq(usave, u)]

        op = Operator(eqns)

        iterations = [i for i in FindNodes(Iteration).visit(op) if i.dim is not time]
        assert all(i.is_Affine for i in iterations)

    @switchconfig(safe_math=True)
    def test_sparse_time_function(self):
        nt = 20

        shape = (21, 21, 21)
        origin = (0., 0., 0.)
        spacing = (1., 1., 1.)
        extent = tuple([d * (s - 1) for s, d in zip(shape, spacing)])
        grid = Grid(shape=shape, extent=extent, origin=origin)
        time = grid.time_dim
        x, y, z = grid.dimensions

        p = TimeFunction(name="p", grid=grid, time_order=2, space_order=4, save=nt)

        # Place source in the middle of the grid
        src_coords = np.empty((1, len(shape)), dtype=np.float32)
        src_coords[0, :] = [o + d * (s-1)//2 for o, d, s in zip(origin, spacing, shape)]
        src = SparseTimeFunction(name='src', grid=grid, npoint=1, nt=nt)
        src.data[:] = 1.
        src.coordinates.data[:] = src_coords[:]

        cd = ConditionalDimension(name="cd", parent=time,
                                  condition=And(Ge(time, 1), Le(time, 10)))

        src_term = src.inject(field=p.forward, expr=src*time, implicit_dims=cd)

        op = Operator(src_term)

        op.apply(time_m=1, time_M=nt-2, dt=1.0)

        assert np.all(p.data[0] == 0)
        # Note the endpoint of the range is 12 because we inject at p.forward
        for i in range(1, 12):
            assert p.data[i].sum() == i - 1
            assert p.data[i, 10, 10, 10] == i - 1
        for i in range(12, 20):
            assert np.all(p.data[i] == 0)

    def test_issue_1435(self):
        names = 't1 t2 t3 t4 t5 t6 t7 t8 t9 t10'
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = \
            tuple(SpaceDimension(i) for i in names.split())

        f0 = Function(name='f0', grid=Grid(shape=(2, 2, 4, 4),
                                           dimensions=(t1, t2, t3, t4)))
        f1 = Function(name='f1', grid=Grid(shape=(2, 2, 3, 3),
                                           dimensions=(t5, t6, t7, t8)))
        f2 = Function(name='f2', grid=f1.grid)

        cd = ConditionalDimension(name='cd', parent=t10,
                                  condition=Or(Gt(f0[t5, t6, t7 + t9,
                                                     t8 + t10],
                                                  f1[t5, t6, t7, t8]),
                                               And(~Ne(f0[t5, t6, t7 + t9,
                                                          t8 + t10],
                                                       f1[t5, t6, t7, t8]),
                                                   Lt(2 * t9 + t10,
                                                      f2[t5, t6, t7, t8]))))

        op = Operator([Eq(f1[t5, t6, t7, t8], f0[t5, t6, t7 + t9, t8 + t10],
                          implicit_dims=cd),
                       Eq(f2[t5, t6, t7, t8], 2 * t9 + t10, implicit_dims=cd)])

        # Check it compiles correctly! See issue report
        op.cfunction

    @pytest.mark.parametrize('factor', [
        4,
        Constant(name='factor', dtype=np.int32, value=4),
    ])
    def test_issue_1927(self, factor):
        """
        Ensure `time_M` is correctly inferred even in presence of TimeFunctions
        defined on ConditionalDimensions.
        """
        grid = Grid(shape=(4, 4))
        time = grid.time_dim
        save = 10

        time_sub = ConditionalDimension('t_sub', parent=time, factor=factor)

        f = TimeFunction(name='f', grid=grid, save=10, time_dim=time_sub)

        op = Operator(Eq(f, 1))

        assert op.arguments()['time_M'] == 4*save-1  # == min legal endpoint

        # Also no issues when supplying an override
        assert op.arguments(time_M=10)['time_M'] == 10
        assert op.arguments(time=10)['time_M'] == 10

        op()
        assert np.all(f.data == 1)

    def test_issue_1927_v2(self):
        size = 16
        factor = 4
        i = Dimension(name='i')

        ci = ConditionalDimension(name='ci', parent=i, factor=factor)
        g = Function(name='g', shape=(size,), dimensions=(i,))
        f = Function(name='f', shape=(int(size/factor),), dimensions=(ci,))

        op = Operator([Eq(f, g)])

        op.apply()

    def test_issue_2007(self):
        """
        Proxy for a Fourier integral. Main issue: conditional placed too deep
        in the tree. It's functionally OK, but performance-wise really bad.
        """
        freq = [2, 4, 8]
        nfreq = np.shape(freq)[0]
        grid = Grid(shape=(5, 5))
        time = grid.time_dim

        u = TimeFunction(name="u", grid=grid)

        # frequency dimension for holding fft proxy
        freq_dim = DefaultDimension(name='freq_dim', default_value=nfreq)
        f = Function(name="f", grid=grid, dimensions=(freq_dim,), shape=(nfreq,))

        # Load the frequencies into f, this will serve as a lower bound for the
        # proxy integral
        f.data[:] = np.array(freq[:])
        # Proxy Fourier integral holder
        ure = Function(name="ure", grid=grid,
                       dimensions=(freq_dim,) + u.indices[1:],
                       shape=(nfreq,) + u.shape[1:])

        # ConditionalDimension based on `f` to simulate bounds of Fourier integral
        ct = ConditionalDimension(name="ct", parent=time, condition=Ge(time, f))
        eqns = [
            Eq(u.forward, u+1),
            Eq(ure, ure + u, implicit_dims=ct)
        ]

        op = Operator(eqns)

        op.apply(time_M=10)

        assert np.all(ure.data[0] == 54)
        assert np.all(ure.data[1] == 49)
        assert np.all(ure.data[2] == 27)

        # Make sure the ConditionalDimension is at the right depth for performance
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3
        assert trees[1][1].nodes[0].is_Conditional

    def test_diff_guards_halts_topofuse(self):
        grid = Grid(shape=(4,))
        time = grid.time_dim

        s0 = Symbol(name='s0')
        s1 = Symbol(name='s1')
        s2 = Symbol(name='s2')

        cd0 = ConditionalDimension(name='cd0', parent=time, condition=Ge(time, 2))

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        eqns = [
            Eq(s0, time, implicit_dims=(cd0,)),
            Eq(s1, time, implicit_dims=(cd0,)),
            Eq(f, s0 + s1, implicit_dims=(cd0,)),
            Eq(s2, time),
            Eq(g, s2)
        ]

        op = Operator(eqns)

        op.cfunction

        assert_structure(op, ['t', 't,x', 't,x'], 't,x,x')

    def test_array_shared_w_topofuse(self):
        shape = (4, 4, 4)
        dims = dimensions('i x y')

        i, _, _ = dims
        cd0 = ConditionalDimension(name='cd0', parent=i, condition=Ge(i, 2))

        f = Function(name='f', dimensions=dims, shape=shape)
        a0 = Array(name='a0', dimensions=dims, scope='shared')
        a1 = Array(name='a1', dimensions=dims, scope='shared')

        eqns = [Eq(a0, 1),
                Eq(a0, 1, implicit_dims=(cd0,)),
                Eq(f, a0),
                Eq(a1, 1),
                Eq(a1, 2, implicit_dims=(cd0,))]

        op = Operator(eqns, opt=('advanced', {'openmp': True}))

        assert_structure(op, ['i,x,y', 'i', 'i,x,y', 'i,x,y'], 'i,x,y,x,y,x,y')

    def test_cond_notime(self):
        grid = Grid(shape=(10, 10))
        time = grid.time_dim

        time_under = ConditionalDimension(name='timeu', parent=time, factor=5)
        nt = 10

        u = TimeFunction(name='u', grid=grid, space_order=2)
        usaved = TimeFunction(name='usaved', grid=grid, space_order=2,
                              time_dim=time_under, save=nt//5+1)
        g = Function(name='g', grid=grid)

        op = Operator([Eq(usaved, u)])
        op(time_m=1, time_M=nt-1, dt=1)

        op = Operator([Inc(g, usaved)])
        op(time_m=1, time_M=nt-1, dt=1)
        assert norm(g, order=1) == norm(sum(usaved, dims=time_under), order=1)

    def test_cond_copy(self):
        grid = Grid((11, 11, 11))
        time = grid.time_dim

        cd = ConditionalDimension(name='tsub', parent=time, factor=5)
        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2, save=Buffer(2))
        u1 = TimeFunction(name='u1', grid=grid, space_order=0,
                          time_order=0, save=5, time_dim=cd)
        u2 = TimeFunction(name='u2', grid=grid, space_order=0,
                          time_order=0, save=5, time_dim=cd)

        # Mimic what happens when an operator is copied
        u12 = deepcopy(u1)
        u22 = deepcopy(u2)

        op = Operator([Eq(u.forward, u.laplace), Eq(u12, u), Eq(u22, u)])
        assert len([p for p in op.parameters if p.name == 'tsubf']) == 1

    def test_const_factor(self):
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        f1 = 4
        f2 = Constant(name='f2', dtype=np.int32, value=4)
        t1 = ConditionalDimension('t_sub', parent=time, factor=f1)
        t2 = ConditionalDimension('t_sub2', parent=time, factor=f2)

        assert isinstance(t1.symbolic_factor, Scalar)
        assert t1.factor == f1

        assert t2.symbolic_factor.is_Constant
        assert t2.factor == f2
        assert t2.factor.data == f1
        assert t2.spacing == t1.spacing

    def test_symbolic_factor_override_legacy(self):
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        fact = Constant(name='fact', dtype=np.int32, value=4)
        cd = ConditionalDimension(name='cd', parent=time, factor=fact)

        u = TimeFunction(name='u', grid=grid, time_order=0)
        usave = TimeFunction(name='usave', grid=grid, time_dim=cd, save=4)

        eqns = [Eq(usave, u),
                Eq(u.forward, u + 1)]

        op = Operator(eqns)

        op.apply()

        assert all(np.all(usave.data[i] == i*4) for i in range(4))

        # Now override the factor
        fact1 = Constant(name='fact1', dtype=np.int32, value=8)

        op.apply(time_M=31, fact=fact1)

        assert all(np.all(usave.data[i] == 16 + i*8) for i in range(4))

    def test_symbolic_factor_override(self):
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        cd = ConditionalDimension(name='cd', parent=time, factor=4)

        u = TimeFunction(name='u', grid=grid, time_order=0)
        usave = TimeFunction(name='usave', grid=grid, time_dim=cd, save=4)

        eqns = [Eq(usave, u),
                Eq(u.forward, u + 1)]

        op = Operator(eqns)

        op.apply()

        assert all(np.all(usave.data[i] == i*4) for i in range(4))

        # Now override the factor
        op.apply(time_M=31, **{cd.symbolic_factor.name: 8})

        assert all(np.all(usave.data[i] == 16 + i*8) for i in range(4))


class TestCustomDimension:

    def test_shifted_minmax(self):
        grid = Grid(shape=(4, 4))

        so = 4

        u = TimeFunction(name='u', grid=grid, space_order=so)
        v = TimeFunction(name='v', grid=grid, space_order=so)

        subdims = [CustomDimension(name=d.name, parent=d,
                                   symbolic_min=d.symbolic_min - so,
                                   symbolic_max=d.symbolic_max + so)
                   for d in grid.dimensions]

        eqn = Eq(v, u)
        eqn = eqn.xreplace(dict(zip(grid.dimensions, subdims)))

        op = Operator(eqn)

        u.data_with_halo[:] = 1.

        op.apply(time_M=1)

        assert np.all(v.data_with_halo[:] == 1)


class TestMashup:

    """
    Check the correct functioning of the compiler in presence of many Dimension types.
    """

    def test_topofusion_w_subdims_conddims(self):
        """
        Check that topological fusion works across guarded Clusters over different
        iteration spaces and in presence of anti-dependences.

        This test uses both SubDimensions (via SubDomains) and ConditionalDimensions.
        """
        grid = Grid(shape=(4, 4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, time_order=2)
        g = TimeFunction(name='g', grid=grid, time_order=2)
        h = TimeFunction(name='h', grid=grid, time_order=2)
        fsave = TimeFunction(name='fsave', grid=grid, time_order=2, save=5)
        gsave = TimeFunction(name='gsave', grid=grid, time_order=2, save=5)

        ctime = ConditionalDimension(name='ctime', parent=time, condition=time > 4)

        eqns = [Eq(f.forward, f + 1),
                Eq(g.forward, g + 1),
                Eq(fsave, f.dt2, implicit_dims=[ctime]),
                Eq(h, f.dx + g, subdomain=grid.interior),
                Eq(gsave, g.dt2, implicit_dims=[ctime])]

        op = Operator(eqns)

        # Check generated code -- expect the gsave equation to be scheduled together
        # in the same loop nest with the fsave equation
        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0', 'x2_blk0'})
        exprs = FindNodes(Expression).visit(bns['x0_blk0'])
        assert len(exprs) == 2
        assert exprs[0].write is f
        assert exprs[1].write is g

        exprs = FindNodes(Expression).visit(bns['x1_blk0'])
        assert len(exprs) == 1
        assert exprs[0].write is h

        exprs = FindNodes(Expression).visit(bns['x2_blk0'])
        assert len(exprs) == 2
        assert exprs[0].write is fsave
        assert exprs[1].write is gsave

    def test_topofusion_w_subdims_conddims_v2(self):
        """
        Like `test_topofusion_w_subdims_conddims` but with more SubDomains,
        so we expect fewer loop nests.
        """
        grid = Grid(shape=(4, 4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, time_order=2)
        g = TimeFunction(name='g', grid=grid, time_order=2)
        h = TimeFunction(name='h', grid=grid, time_order=2)
        fsave = TimeFunction(name='fsave', grid=grid, time_order=2, save=5)
        gsave = TimeFunction(name='gsave', grid=grid, time_order=2, save=5)

        ctime = ConditionalDimension(name='ctime', parent=time, condition=time > 4)

        eqns = [Eq(f.forward, f + 1, subdomain=grid.interior),
                Eq(g.forward, g + 1, subdomain=grid.interior),
                Eq(fsave, f.dt2, implicit_dims=[ctime]),
                Eq(h, f + g, subdomain=grid.interior),
                Eq(gsave, g.dt2, implicit_dims=[ctime])]

        op = Operator(eqns)

        # Check generated code -- expect the gsave equation to be scheduled together
        # in the same loop nest with the fsave equation
        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0'})
        assert len(FindNodes(Expression).visit(bns['x0_blk0'])) == 3
        exprs = FindNodes(Expression).visit(bns['x1_blk0'])
        assert len(exprs) == 2
        assert exprs[0].write is fsave
        assert exprs[1].write is gsave

    def test_topofusion_w_subdims_conddims_v3(self):
        """
        Like `test_topofusion_w_subdims_conddims_v2` but with an extra anti-dependence,
        which causes scheduling over more loop nests.
        """
        grid = Grid(shape=(4, 4, 4))
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, time_order=2, space_order=4)
        g = TimeFunction(name='g', grid=grid, time_order=2)
        h = TimeFunction(name='h', grid=grid, time_order=2)
        fsave = TimeFunction(name='fsave', grid=grid, time_order=2, save=5)
        gsave = TimeFunction(name='gsave', grid=grid, time_order=2, save=5)

        ctime = ConditionalDimension(name='ctime', parent=time, condition=time > 4)

        eqns = [Eq(f.forward, f + 1, subdomain=grid.interior),
                Eq(g.forward, g + 1, subdomain=grid.interior),
                Eq(fsave, f.dt2, implicit_dims=[ctime]),
                Eq(h, f.dt2.dx + g, subdomain=grid.interior),
                Eq(gsave, g.dt2, implicit_dims=[ctime])]

        op = Operator(eqns)

        # Check generated code -- expect the gsave equation to be scheduled together
        # in the same loop nest with the fsave equation
        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0', 'x2_blk0'})
        exprs = FindNodes(Expression).visit(bns['x0_blk0'])
        assert len(exprs) == 2
        assert exprs[0].write is f
        assert exprs[1].write is g

        exprs = FindNodes(Expression).visit(bns['x2_blk0'])
        assert len(exprs) == 2
        assert exprs[0].write is fsave
        assert exprs[1].write is gsave

        # Additional nest due to anti-dependence
        exprs = FindNodes(Expression).visit(bns['x1_blk0'])
        assert len(exprs) == 2
        assert exprs[1].write is h


class TestConcretization:
    """
    Class for testing renaming of SubDimensions and MultiSubDimensions
    during compilation.
    """

    def test_correct_thicknesses(self):
        """
        Check that thicknesses aren't created where they shouldn't be.
        """
        x = Dimension('x')
        ix0 = SubDimension.left('x', x, 2)
        ix1 = SubDimension.right('x', x, 2)
        ix2 = SubDimension.middle('x', x, 2, 2)

        rebuilt = concretize_subdims([ix0, ix1, ix2], sregistry=SymbolRegistry())

        assert rebuilt[0].is_left
        assert rebuilt[1].is_right
        assert rebuilt[2].is_middle

    def test_condition_concretization(self):
        """
        Check thicknesses in conditionals get concretized correctly, including in
        cases where there are multiple such conditions.
        """
        x = Dimension('x')

        ix0 = SubDimension.left('ix', x, 6)
        ix1 = SubDimension.middle('ix', x, 2, 1)

        cond0 = Gt(x, ix0.ltkn)
        cond1 = Gt(x, ix1.ltkn)

        cdim0 = ConditionalDimension('cdim0', parent=x, condition=cond0)
        cdim1 = ConditionalDimension('cdim1', parent=x, condition=cond1)

        f = Function(name='f', dimensions=(x,), shape=(11,))

        eqs = [Eq(f, 1, implicit_dims=cdim0), Eq(f, 1, implicit_dims=cdim1)]

        op = Operator(eqs)

        # Check correct conditionals are in the generated code
        assert "if (x > x_ltkn0)" in str(op.ccode)
        assert "if (x > x_ltkn1)" in str(op.ccode)

        expected_conditionals = [Gt(x, p) for p in op.parameters
                                 if isinstance(p, Thickness)]

        conditionals = FindNodes(Conditional).visit(op)
        assert len(conditionals) == 2
        # Check that the two conditions are concretized uniquely
        assert len({c.condition for c in conditionals}) == 2
        for c in conditionals:
            assert c.condition in expected_conditionals
