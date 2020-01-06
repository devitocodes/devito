import pytest
import numpy as np
from sympy import S

from conftest import EVAL, skipif  # noqa
from devito import (Eq, Inc, Grid, Constant, Function, TimeFunction, # noqa
                    Operator, Dimension, SubDimension, switchconfig)
from devito.ir.equations import DummyEq, LoweredEq
from devito.ir.equations.algorithms import dimension_sort
from devito.ir.iet import (Call, Conditional, Expression, Iteration, CGen, FindNodes,
                           FindSymbols, retrieve_iteration_tree, filter_iterations,
                           make_efunc)
from devito.ir.support.basic import (IterationInstance, TimedAccess, Scope,
                                     Vector, AFFINE, IRREGULAR)
from devito.ir.support.space import (NullInterval, Interval, Forward, Backward,
                                     IterationSpace)
from devito.types import Scalar, Symbol, Array
from devito.tools import as_tuple

pytestmark = skipif(['yask', 'ops'], whole_module=True)


class TestVectorHierarchy(object):

    @pytest.fixture
    def grid(self):
        return Grid((3, 3, 3))

    @pytest.fixture
    def x(self, grid):
        return grid.dimensions[0]

    @pytest.fixture
    def y(self, grid):
        return grid.dimensions[1]

    @pytest.fixture
    def z(self, grid):
        return grid.dimensions[2]

    @pytest.fixture
    def v_num(self):
        v2 = Vector(2, smart=True)
        v3 = Vector(3, smart=True)
        v4 = Vector(4)
        v11 = Vector(1, 1)
        v13 = Vector(1, 3)
        v23 = Vector(2, 3)
        return v2, v3, v4, v11, v13, v23

    @pytest.fixture
    def v_literal(self, x, y):
        vx = Vector(x)
        vxy = Vector(x, y)
        vx1y = Vector(x + 1, y)
        s = Scalar(name='s', nonnegative=True)
        vs3 = Vector(s + 3, smart=True)
        return vx, vxy, vx1y, vs3

    @pytest.fixture
    def fa(self, grid, x):
        return Array(name='fa', dimensions=(x,), shape=(3,)).indexed

    @pytest.fixture
    def fc(self, grid, x, y):
        return Array(name='fc', shape=(3, 5), dimensions=(x, y)).indexed

    @pytest.fixture
    def ii_num(self, fa, fc):
        fa4 = IterationInstance(fa[4])
        fc00 = IterationInstance(fc[0, 0])
        fc11 = IterationInstance(fc[1, 1])
        fc23 = IterationInstance(fc[2, 3])
        return fa4, fc00, fc11, fc23

    @pytest.fixture
    def ii_literal(self, fa, fc, x, y):
        fax = IterationInstance(fa[x])
        fcxy = IterationInstance(fc[x, y])
        fcx1y = IterationInstance(fc[x + 1, y])
        return fax, fcxy, fcx1y

    @pytest.fixture
    def ta_literal(self, fc, x, y):
        intervals = [Interval(x, 0, 0), Interval(y, 0, 0)]
        fwd_ispace = IterationSpace(intervals, directions={x: Forward, y: Forward})
        mixed_ispace = IterationSpace(intervals, directions={x: Backward, y: Forward})
        tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, fwd_ispace)
        tcxy_r0 = TimedAccess(fc[x, y], 'R', 0, fwd_ispace)
        tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, fwd_ispace)
        tcx1y_r1 = TimedAccess(fc[x + 1, y], 'R', 1, fwd_ispace)
        rev_tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, mixed_ispace)
        rev_tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, mixed_ispace)
        return tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1

    def test_vector_cmp(self, v_num, v_literal):
        v2, v3, v4, v11, v13, v23 = v_num
        vx, vxy, vx1y, vs3 = v_literal

        # Equality check (numeric, symbolic, mixed)
        assert v4 == v4
        assert v4 != v11
        assert vx == vx
        assert vx != v4
        assert vx != vxy
        assert vs3 != v4

        # Lexicographic comparison (numeric, symbolic, mixed)
        assert v3 < v4
        assert v11 < v23
        assert v11 <= v23
        assert v11 < v13
        assert v11 <= v13
        assert v23 > v11
        assert (vxy < vx1y) is True
        assert (vxy <= vx1y) is True
        assert (vx1y > vxy) is True
        assert (vx1y <= vxy) is False

        # Smart vector comparison
        # Note: `v3` and `vs3` use the "smart" mode
        assert (v3 < vs3) is False
        assert (vs3 < v3) is False
        assert v3 != vs3
        assert (v3 <= vs3) is True
        assert (vs3 <= v3) is False
        assert v2 < vs3
        assert v2 <= vs3
        assert vs3 > v2

    def test_iteration_instance_arithmetic(self, x, y, ii_num, ii_literal):
        """
        Tests arithmetic operations involving objects of type IterationInstance.
        """
        fa4, fc00, fc11, fc23 = ii_num
        fax, fcxy, fcx1y = ii_literal

        # Trivial arithmetic with numbers
        assert fc00 == 0
        assert fc23 != 0
        assert fc23.sum == 5
        assert (fc00 + fc11 + fc23)[0] == 3
        assert (fc00 + fc11 + fc23)[1] == 4

        # Trivial arithmetic with literals
        assert (fcxy + fcx1y)[0].subs(x, 2) == 5
        assert (fcxy + fcx1y)[1].subs(y, 4) == 8

        # Mixed arithmetic literals/numbers
        assert (fcx1y + fc11)[0].subs(x, 4) == 6
        assert (fcx1y + fc11)[1].subs(y, 4) == 5

        # Arithmetic between Vectors and numbers
        assert fc00 + 1 == (1, 1)
        assert 1 + fc00 == (1, 1)
        assert fc00 - 1 == (-1, -1)
        assert 1 - fc00 == (-1, -1)

        # Illegal ops
        for ii in [fax, fa4]:
            try:
                ii + fcx1y
                assert False
            except TypeError:
                pass
            except:
                assert False

    def test_iteration_instance_distance(self, ii_num, ii_literal):
        """
        Tests calculation of vector distance between objects of type IterationInstance.
        """
        _, fc00, fc11, fc23 = ii_num
        fax, fcxy, fcx1y = ii_literal

        # Distance with numbers
        assert fc11.distance(fc00) == (1, 1)
        assert fc23.distance(fc11) == (1, 2)
        assert fc11.distance(fc23) == (-1, -2)

        # Distance with matching literals
        assert fcxy.distance(fcx1y) == (-1, 0)
        assert fcx1y.distance(fcxy) == (1, 0)

        # Should fail due mismatching indices
        try:
            fcxy.distance(fax)
            assert False
        except TypeError:
            pass
        except:
            assert False

    def test_iteration_instance_cmp(self, ii_num, ii_literal):
        """
        Tests comparison of objects of type IterationInstance.
        """
        fa4, fc00, fc11, fc23 = ii_num
        fax, fcxy, fcx1y = ii_literal

        # Lexicographic comparison with numbers and same rank
        assert fc11 == fc11
        assert fc11 != fc23
        assert fc23 <= fc23
        assert not (fc23 < fc23)
        assert fc11 < fc23
        assert fc23 > fc00
        assert fc00 >= fc00

        # Lexicographic comparison with numbers but different rank should faxl
        try:
            fa4 > fc23
            assert False
        except TypeError:
            pass
        except:
            assert False

        # Lexicographic comparison with literals
        assert fcxy <= fcxy
        assert fcxy < fcx1y

    def test_timed_access_distance(self, x, y, ta_literal):
        """
        Tests comparison of objects of type TimedAccess.
        """
        tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1 = ta_literal

        # Simple distance calculations
        assert tcxy_w0.distance(tcxy_r0) == (0, 0)
        assert tcx1y1_r1.distance(tcxy_r0) == (1, 1)
        assert tcxy_r0.distance(tcx1y1_r1) == (-1, -1)
        assert tcx1y1_r1.distance(tcx1y_r1) == (0, 1)

        # Distance should go to infinity due to mismatching directions
        assert rev_tcxy_w0.distance(tcx1y_r1) == (S.Infinity,)
        assert tcx1y_r1.distance(rev_tcxy_w0) == (S.Infinity,)

        # Distance when both source and since go backwards along the x Dimension
        assert rev_tcxy_w0.distance(rev_tcx1y1_r1) == (1, -1)
        assert rev_tcx1y1_r1.distance(rev_tcxy_w0) == (-1, 1)

        # Distance up to provided dimension
        assert tcx1y1_r1.distance(tcxy_r0, x) == (1,)
        assert tcx1y1_r1.distance(tcxy_r0, y) == (1, 1)

    def test_timed_access_cmp(self, ta_literal):
        """
        Tests comparison of objects of type TimedAccess.
        """
        tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1 = ta_literal

        # Equality check
        assert tcxy_w0 == tcxy_w0
        assert (tcxy_w0 != tcxy_r0) is True  # Different mode R vs W
        assert tcxy_w0 != tcx1y1_r1
        assert tcxy_w0 != rev_tcxy_w0

        # Lexicographic comparison
        assert tcxy_r0 < tcx1y1_r1
        assert (tcxy_r0 > tcx1y1_r1) is False
        assert (tcxy_r0 >= tcx1y1_r1) is False
        assert tcx1y1_r1 > tcxy_r0
        assert tcx1y1_r1 >= tcxy_r0
        assert tcx1y_r1 > tcxy_w0
        assert tcx1y_r1 < tcx1y1_r1
        assert tcx1y1_r1 > tcx1y_r1

        # Lexicographic comparison with reverse direction
        assert rev_tcxy_w0 > rev_tcx1y1_r1
        assert rev_tcx1y1_r1 <= rev_tcxy_w0

        # Non-comparable due to different direction
        try:
            rev_tcxy_w0 > tcxy_r0
            assert False
        except TypeError:
            assert True
        except:
            assert False


class TestSpace(object):

    @pytest.fixture
    def grid(self):
        return Grid((3, 3, 3))

    @pytest.fixture
    def x(self, grid):
        return grid.dimensions[0]

    @pytest.fixture
    def y(self, grid):
        return grid.dimensions[1]

    def test_intervals_intersection(self, x, y):
        nullx = NullInterval(x)

        # All nulls
        assert nullx.intersection(nullx) == nullx

        nully = NullInterval(y)
        ix = Interval(x, -2, 2)
        iy = Interval(y, -2, 2)

        # Mixed nulls and defined
        assert nullx.intersection(ix) == nullx
        assert nullx.intersection(iy) == nullx
        assert nullx.intersection(iy) != nully
        assert nully.intersection(iy) == nully

        ix2 = Interval(x, -8, -3)
        ix3 = Interval(x, 3, 4)

        assert ix.intersection(ix2) == Interval(x, -2, -3)
        assert ix.intersection(ix3) == Interval(x, 3, 2)
        assert ix2.intersection(ix3) == Interval(x, 3, -3)
        assert ix.intersection(iy) == nullx
        assert iy.intersection(ix) == nully

        ix4 = Interval(x, 1, 4)
        ix5 = Interval(x, -3, 0)

        assert ix.intersection(ix4) == Interval(x, 1, 2)
        assert ix.intersection(ix5) == Interval(x, -2, 0)

        # Mixed symbolic and non-symbolic
        c = Constant(name='c')
        ix6 = Interval(x, c, c + 4)
        ix7 = Interval(x, c - 1, c + 5)

        assert ix6.intersection(ix7) == Interval(x, c, c + 4)
        assert ix7.intersection(ix6) == Interval(x, c, c + 4)

        # Symbolic with properties
        s = Scalar(name='s', nonnegative=True)
        ix8 = Interval(x, s - 2, s + 2)
        ix9 = Interval(x, s - 1, s + 1)

        assert ix.intersection(ix8) == Interval(x, s - 2, 2)
        assert ix8.intersection(ix) == Interval(x, s - 2, 2)
        assert ix8.intersection(ix9) == Interval(x, s - 1, s + 1)
        assert ix9.intersection(ix8) == Interval(x, s - 1, s + 1)

    def test_intervals_union(self, x, y):
        nullx = NullInterval(x)

        # All nulls
        assert nullx.union(nullx) == nullx

        ix = Interval(x, -2, 2)

        # Mixed nulls and defined
        assert nullx.union(ix) == ix
        assert ix.union(ix) == ix
        assert ix.union(nullx) == ix

        ix2 = Interval(x, 1, 4)
        ix3 = Interval(x, -3, 6)

        assert ix.union(ix2) == Interval(x, -2, 4)
        assert ix.union(ix3) == ix3
        assert ix2.union(ix3) == ix3

        ix4 = Interval(x, 4, 8)
        ix5 = Interval(x, -3, -3)
        ix6 = Interval(x, -10, -3)
        nully = NullInterval(y)
        iy = Interval(y, -2, 2)

        assert ix.union(ix4) == Interval(x, -2, 8)
        assert ix.union(ix5) == Interval(x, -3, 2)
        assert ix6.union(ix) == Interval(x, -10, 2)

        # The union of non-compatible Intervals isn't possible, and an exception
        # is expected
        ixs1 = Interval(x, -2, 2, stamp=1)

        for i, j in [(ix, nully), (ix, iy), (iy, ix), (ix, ixs1), (ixs1, ix)]:
            try:
                i.union(j)
                assert False  # Shouldn't arrive here
            except ValueError:
                assert True
            except:
                # No other types of exception expected
                assert False

        # Mixed symbolic and non-symbolic
        c = Constant(name='c')
        ix7 = Interval(x, c, c + 4)
        ix8 = Interval(x, c - 1, c + 5)

        assert ix7.union(ix8) == Interval(x, c - 1, c + 5)
        assert ix8.union(ix7) == Interval(x, c - 1, c + 5)

        # Symbolic with properties
        s = Scalar(name='s', nonnegative=True)
        ix9 = Interval(x, s - 2, s + 2)
        ix10 = Interval(x, s - 1, s + 1)

        assert ix.union(ix9) == Interval(x, -2, s + 2)
        assert ix9.union(ix) == Interval(x, -2, s + 2)
        assert ix9.union(ix10) == ix9
        assert ix10.union(ix9) == ix9

    def test_intervals_subtract(self, x, y):
        nullx = NullInterval(x)

        # All nulls
        assert nullx.subtract(nullx) == nullx

        ix = Interval(x, 2, -2)

        # Mixed nulls and defined on the same dimension
        assert nullx.subtract(ix) == nullx
        assert ix.subtract(ix) == Interval(x, 0, 0)
        assert ix.subtract(nullx) == ix

        ix2 = Interval(x, 4, -4)
        ix3 = Interval(x, 6, -6)

        # All defined same dimension
        assert ix2.subtract(ix) == ix
        assert ix.subtract(ix2) == Interval(x, -2, 2)
        assert ix3.subtract(ix) == ix2

        c = Constant(name='c')
        ix4 = Interval(x, c + 2, c + 4)
        ix5 = Interval(x, c + 1, c + 5)

        # All defined symbolic
        assert ix4.subtract(ix5) == Interval(x, 1, -1)
        assert ix5.subtract(ix4) == Interval(x, -1, 1)
        assert ix5.subtract(ix) == Interval(x, c - 1, c + 7)


class TestDependenceAnalysis(object):

    @pytest.fixture
    def grid(self):
        return Grid((3, 3, 3))

    @pytest.fixture
    def ti0(self, grid):
        return Array(name='ti0', shape=(3, 5, 7), dimensions=grid.dimensions).indexify()

    @pytest.fixture
    def ti1(self, grid):
        return Array(name='ti1', shape=(3, 5, 7), dimensions=grid.dimensions).indexify()

    @pytest.fixture
    def ti3(self, grid):
        return Array(name='ti3', shape=(3, 5, 7), dimensions=grid.dimensions).indexify()

    @pytest.fixture
    def fa(self, grid):
        return Array(name='fa', dimensions=(grid.dimensions[0],), shape=(3,)).indexed

    @pytest.mark.parametrize('indexed,expected', [
        ('u[x,y,z]', (AFFINE, AFFINE, AFFINE)),
        ('u[x+1,y,z-1]', (AFFINE, AFFINE, AFFINE)),
        ('u[x+1,3,z-1]', (AFFINE, AFFINE, AFFINE)),
        ('u[sx+1,y,z-1]', (AFFINE, AFFINE, AFFINE)),
        ('u[x+1,c,s]', (AFFINE, AFFINE, IRREGULAR)),
        ('u[x+1,c,sc]', (AFFINE, AFFINE, AFFINE)),
        ('u[x+1,c+1,sc*sc]', (AFFINE, AFFINE, AFFINE)),
        ('u[x*x+1,y,z]', (IRREGULAR, AFFINE, AFFINE)),
        ('u[x*y,y,z]', (IRREGULAR, AFFINE, AFFINE)),
        ('u[x + z,x + y,z*z]', (IRREGULAR, IRREGULAR, IRREGULAR)),
        ('u[x+1,u[2,2,2],z-1]', (AFFINE, IRREGULAR, AFFINE)),
        ('u[y,x,z]', (IRREGULAR, IRREGULAR, AFFINE)),
    ])
    def test_index_mode_detection(self, indexed, expected):
        """
        Test detection of IterationInstance access modes (AFFINE vs IRREGULAR).

        Proper detection of access mode is a prerequisite to any sort of
        data dependence analysis.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions  # noqa

        sx = SubDimension.middle('sx', x, 1, 1)  # noqa

        u = Function(name='u', grid=grid)  # noqa
        c = Constant(name='c')  # noqa
        sc = Scalar(name='sc', is_const=True)  # noqa
        s = Scalar(name='s')  # noqa

        ii = IterationInstance(eval(indexed))
        assert ii.index_mode == expected

    @pytest.mark.parametrize('expr,expected', [
        ('Eq(ti0[x,y,z], ti1[x,y,z])', None),
        ('Eq(ti0[x,y,z], ti0[x,y,z])', 'flow,indep,None,regular'),
        ('Eq(ti0[x,y,z], ti0[x,y,z])', 'flow,inplace,None,regular'),
        ('Eq(ti0[x,y,z], ti0[x,y,z-1])', 'flow,carried,z,regular'),
        ('Eq(ti0[x,y,z], ti0[x-1,y,z-1])', 'flow,carried,x,regular'),
        ('Eq(ti0[x,y,z], ti0[x-1,y,z+1])', 'flow,carried,x,regular'),
        ('Eq(ti0[x,y,z], ti0[x+1,y+2,z])', 'anti,carried,x,regular'),
        ('Eq(ti0[x,y,z], ti0[x,y+2,z-3])', 'anti,carried,y,regular'),
        ('Eq(ti0[x,y,z], ti0[fa[x],y,z])', 'all,carried,x,irregular'),
        ('Eq(ti0[x,y,z], ti0[fa[x],y,fa[z]])', 'all,carried,x,irregular'),
        ('Eq(ti0[x,fa[y],z], ti0[x,y,z])', 'all,carried,y,irregular'),
        ('Eq(ti0[x,y,z], ti0[x-1,fa[y],z])', 'flow,carried,x,regular'),
    ])
    def test_single_eq(self, expr, expected, ti0, ti1, fa, grid):
        """
        Tests data dependences within a single equation consisting of only two Indexeds.

        ``expected`` is a comma-separated word consisting of four pieces of information:

            * if it's a flow, anti, or output dependence
            * if it's loop-carried or loop-independent
            * the dimension causing the dependence
            * whether it's direct or indirect (i.e., through A[B[i]])
        """
        expr = LoweredEq(EVAL(expr, ti0.base, ti1.base, fa))

        # Force innatural flow, only to stress the compiler to see if it was
        # capable of detecting anti-dependences
        expr.ispace._directions = {i: Forward for i in expr.ispace.directions}

        scope = Scope(expr)
        deps = scope.d_all
        if expected is None:
            assert len(deps) == 0
            return
        else:
            type, mode, exp_cause, regular = expected.split(',')
            if type == 'all':
                assert len(deps) == 2
            else:
                assert len(deps) == 1
        dep = list(deps)[0]

        # Check type
        types = ['flow', 'anti']
        if type != 'all':
            types.remove(type)
            assert len(getattr(scope, 'd_%s' % type)) == 1
            assert all(len(getattr(scope, 'd_%s' % i)) == 0 for i in types)
        else:
            assert all(len(getattr(scope, 'd_%s' % i)) == 1 for i in types)

        # Check mode
        assert getattr(dep, 'is_%s' % mode)()

        # Check cause
        if exp_cause == 'None':
            assert not dep.cause
            return
        else:
            assert len(dep.cause) == 1
            cause = set(dep.cause).pop()
            assert cause.name == exp_cause

        # Check mode restricted to the cause
        assert getattr(dep, 'is_%s' % mode)(cause)
        non_causes = [i for i in grid.dimensions if i is not cause]
        assert all(not getattr(dep, 'is_%s' % mode)(i) for i in non_causes)

        # Check if it's regular or irregular
        assert getattr(dep.source, 'is_%s' % regular) or\
            getattr(dep.sink, 'is_%s' % regular)

    @pytest.mark.parametrize('exprs,expected', [
        # Trivial flow dep
        (['Eq(ti0[x,y,z], ti1[x,y,z])',
          'Eq(ti3[x,y,z], ti0[x,y,z])'],
         ['ti0,flow,set()']),
        # One flow dep, one anti dep
        (['Eq(ti0[x,y,z], ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y,z])'],
         ['ti0,flow,set()', 'ti1,anti,set()']),
        # One output dep, two identical flow deps
        (['Eq(ti3[x+1,y,z], ti1[x,y,z])',
          'Eq(ti3[x+1,y,z], ti3[x,y,z])'],
         ['ti3,output,set()', 'ti3,flow,{x}', 'ti3,flow,{x}']),
        # One flow independent dep, two flow carried flow deps
        (['Eq(ti0[x,y,z], ti0[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y-1,z])',
          'Eq(ti3[x,y,z], ti0[x-2,y,z])'],
         ['ti0,flow,set()', 'ti0,flow,{y}', 'ti0,flow,{x}']),
        # An indirect dep, conservatively assumed flow and anti
        (['Eq(ti0[x,y,z], ti1[x,y,z])',
          'Eq(ti3[x,y,z], ti0[fa[x],y,z])'],
         ['ti0,flow,{x}', 'ti0,anti,{x}']),
        # A direct anti dep "masking" the indirect dep in an inner dimension
        (['Eq(ti0[x,y,z], ti1[x,y,z])',
          'Eq(ti3[x,y,z], ti0[x+1,fa[y],z])'],
         ['ti0,anti,{x}']),
        # Conservatively assume dependences due to "complex" affine index functions
        (['Eq(ti0[x,y,z], ti1[x,2*y,z])',
          'Eq(ti1[x,3*y,z], ti0[x+1,y,z])'],
         ['ti1,flow,{y}', 'ti1,anti,{y}', 'ti0,anti,{x}']),
        # Data indices don't match iteration indices, so conservatively assume
        # all sort of deps
        (['Eq(ti0[x,y,z], ti1[x,y,z])',
          'Eq(ti3[x,y,z], ti0[y+1,y,y])'],
         ['ti0,flow,{x}', 'ti0,anti,{x}']),
    ])
    def test_multiple_eqs(self, exprs, expected, ti0, ti1, ti3, fa):
        """
        Tests data dependences across ordered sequences of equations representing
        a scope.

        ``expected`` is a list of comma-separated words, each word representing a
        dependence in the scope and consisting of three pieces of information:

            * the name of the function inducing a dependence
            * if it's a flow, anti, or output dependence
            * the dimension causing the dependence
        """
        exprs = [LoweredEq(i) for i in EVAL(exprs, ti0.base, ti1.base, ti3.base, fa)]
        expected = [tuple(i.split(',')) for i in expected]

        # Force innatural flow, only to stress the compiler to see if it was
        # capable of detecting anti-dependences
        for i in exprs:
            i.ispace._directions = {i: Forward for i in i.ispace.directions}

        scope = Scope(exprs)
        assert len(scope.d_all) == len(expected)

        for i in ['flow', 'anti', 'output']:
            for dep in getattr(scope, 'd_%s' % i):
                item = (dep.function.name, i, str(set(dep.cause)))
                assert item in expected
                expected.remove(item)

        # Sanity check: we did find all of the expected dependences
        assert len(expected) == 0


class TestIETConstruction(object):

    @pytest.fixture
    def grid(self):
        return Grid((3, 3, 3))

    @pytest.fixture
    def fc(self, grid):
        return Array(name='fc', dimensions=(grid.dimensions[0], grid.dimensions[1]),
                     shape=(3, 5)).indexed

    def test_conditional(self, fc, grid):
        x, y, _ = grid.dimensions
        then_body = Expression(DummyEq(fc[x, y], fc[x, y] + 1))
        else_body = Expression(DummyEq(fc[x, y], fc[x, y] + 2))
        conditional = Conditional(x < 3, then_body, else_body)
        assert str(conditional) == """\
if (x < 3)
{
  fc[x][y] = fc[x][y] + 1;
}
else
{
  fc[x][y] = fc[x][y] + 2;
}"""

    @pytest.mark.parametrize("exprs,nfuncs,ntimeiters,nests", [
        (('Eq(v[t+1,x,y], v[t,x,y] + 1)',), (1,), (2,), ('xy',)),
        (('Eq(v[t,x,y], v[t,x-1,y] + 1)', 'Eq(v[t,x,y], v[t,x+1,y] + u[x,y])'),
         (1, 2), (1, 1), ('xy', 'xy'))
    ])
    @switchconfig(openmp=False)
    def test_make_efuncs(self, exprs, nfuncs, ntimeiters, nests):
        """Test construction of ElementalFunctions."""
        exprs = list(as_tuple(exprs))

        grid = Grid(shape=(10, 10))
        t = grid.stepping_dim  # noqa
        x, y = grid.dimensions  # noqa

        u = Function(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)

        # We create one ElementalFunction for each Iteration nest over space dimensions
        efuncs = []
        for n, tree in enumerate(retrieve_iteration_tree(op)):
            root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
            efuncs.append(make_efunc('f%d' % n, root))

        assert len(efuncs) == len(nfuncs) == len(ntimeiters) == len(nests)

        for efunc, nf, nt, nest in zip(efuncs, nfuncs, ntimeiters, nests):
            # Check the `efunc` parameters
            assert all(i in efunc.parameters for i in (x.symbolic_min, x.symbolic_max))
            assert all(i in efunc.parameters for i in (y.symbolic_min, y.symbolic_max))
            functions = FindSymbols().visit(efunc)
            assert len(functions) == nf
            assert all(i in efunc.parameters for i in functions)
            timeiters = [i for i in FindSymbols('free-symbols').visit(efunc)
                         if isinstance(i, Dimension) and i.is_Time]
            assert len(timeiters) == nt
            assert all(i in efunc.parameters for i in timeiters)
            assert len(efunc.parameters) == 4 + len(functions) + len(timeiters)

            # Check the loop nest structure
            trees = retrieve_iteration_tree(efunc)
            assert len(trees) == 1
            tree = trees[0]
            assert all(i.dim.name == j for i, j in zip(tree, nest))

            assert efunc.make_call()

    def test_nested_calls_cgen(self):
        call = Call('foo', [
            Call('bar', [])
        ])

        code = CGen().visit(call)

        assert str(code) == 'foo(bar());'

    @pytest.mark.parametrize('mode,expected', [
        ('free-symbols', '["f", "x"]'),
        ('symbolics', '["f"]')
    ])
    def test_find_symbols_nested(self, mode, expected):
        grid = Grid(shape=(4, 4, 4))
        call = Call('foo', [
            Call('bar', [
                Symbol(name='x'),
                Call('baz', [Function(name='f', grid=grid)])
            ])
        ])

        found = FindSymbols(mode).visit(call)

        assert [f.name for f in found] == eval(expected)


class TestAnalysis(object):

    @pytest.mark.parametrize('exprs,atomic,parallel', [
        (['Inc(u[gp[p, 0]+rx, gp[p, 1]+ry], cx*cy*src)'],
         ['p', 'rx', 'ry'], []),
        (['Eq(rcv, 0)', 'Inc(rcv, cx*cy)'],
         ['rx', 'ry'], ['time', 'p']),
        (['Eq(v.forward, u+1)', 'Eq(rcv, 0)',
          'Inc(rcv, v[t, gp[p, 0]+rx, gp[p, 1]+ry]*cx*cy)'],
         ['rx', 'ry'], ['x', 'y', 'p']),
        (['Eq(v.forward, v[t+1, x+1, y]+v[t, x, y]+v[t, x+1, y])'],
         [], ['y']),
        (['Eq(v.forward, v[t+1, x-1, y]+v[t, x, y]+v[t, x-1, y])'],
         [], ['y']),
        (['Eq(v.forward, v[t+1, x, y+1]+v[t, x, y]+v[t, x, y+1])'],
         [], ['x']),
        (['Eq(v.forward, v[t+1, x, y-1]+v[t, x, y]+v[t, x, y-1])'],
         [], ['x']),
        (['Eq(v.forward, v+1)', 'Inc(u, v)'],
         [], ['x', 'y'])
    ])
    def test_iteration_parallelism_2d(self, exprs, atomic, parallel):
        """Tests detection of PARALLEL_* properties."""
        grid = Grid(shape=(10, 10))
        time = grid.time_dim  # noqa
        t = grid.stepping_dim  # noqa
        x, y = grid.dimensions  # noqa

        p = Dimension(name='p')
        d = Dimension(name='d')
        rx = Dimension(name='rx')
        ry = Dimension(name='ry')

        u = Function(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid, save=None)  # noqa

        cx = Function(name='coeff_x', dimensions=(p, rx), shape=(1, 2))  # noqa
        cy = Function(name='coeff_y', dimensions=(p, ry), shape=(1, 2))  # noqa

        gp = Function(name='gridpoints', dimensions=(p, d), shape=(1, 2))  # noqa
        src = Function(name='src', dimensions=(p,), shape=(1,))  # noqa
        rcv = Function(name='rcv', dimensions=(time, p), shape=(100, 1), space_order=0)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, dle='openmp')

        iters = FindNodes(Iteration).visit(op)
        assert all(i.is_ParallelAtomic for i in iters if i.dim.name in atomic)
        assert all(not i.is_ParallelAtomic for i in iters if i.dim.name not in atomic)
        assert all(i.is_Parallel for i in iters if i.dim.name in parallel)
        assert all(not i.is_Parallel for i in iters if i.dim.name not in parallel)

    @pytest.mark.parametrize('exprs,atomic,parallel', [
        (['Inc(u[gp[p, 0]+rx, gp[p, 1]+ry, gp[p, 2]+rz], cx*cy*cz*src)'],
         ['p', 'rx', 'ry', 'rz'], []),
        (['Eq(rcv, 0)', 'Inc(rcv, cx*cy*cz)'],
         ['rx', 'ry', 'rz'], ['time', 'p']),
        (['Eq(v.forward, u+1)', 'Eq(rcv, 0)',
          'Inc(rcv, v[t, gp[p, 0]+rx, gp[p, 1]+ry, gp[p, 2]+rz]*cx*cy*cz)'],
         ['rx', 'ry', 'rz'], ['x', 'y', 'z', 'p']),
        (['Eq(v.forward, v[t+1, x+1, y, z]+v[t, x, y, z]+v[t, x+1, y, z])'],
         [], ['y', 'z']),
        (['Eq(v.forward, v[t+1, x-1, y, z]+v[t, x, y, z]+v[t, x-1, y, z])'],
         [], ['y', 'z']),
        (['Eq(v.forward, v[t+1, x, y+1, z]+v[t, x, y, z]+v[t, x, y+1, z])'],
         [], ['x', 'z']),
        (['Eq(v.forward, v[t+1, x, y-1, z]+v[t, x, y, z]+v[t, x, y-1, z])'],
         [], ['x', 'z']),
        (['Eq(v.forward, v[t+1, x, y, z+1]+v[t, x, y, z]+v[t, x, y, z+1])'],
         [], ['x', 'y']),
        (['Eq(v.forward, v[t+1, x, y, z-1]+v[t, x, y, z]+v[t, x, y, z-1])'],
         [], ['x', 'y'])
    ])
    def test_iteration_parallelism_3d(self, exprs, atomic, parallel):
        """Tests detection of PARALLEL_* properties."""
        grid = Grid(shape=(10, 10, 10))
        time = grid.time_dim  # noqa
        t = grid.stepping_dim  # noqa
        x, y, z = grid.dimensions  # noqa

        p = Dimension(name='p')
        d = Dimension(name='d')
        rx = Dimension(name='rx')
        ry = Dimension(name='ry')
        rz = Dimension(name='rz')

        u = Function(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid, save=None)  # noqa

        cx = Function(name='coeff_x', dimensions=(p, rx), shape=(1, 2))  # noqa
        cy = Function(name='coeff_y', dimensions=(p, ry), shape=(1, 2))  # noqa
        cz = Function(name='coeff_z', dimensions=(p, rz), shape=(1, 2))  # noqa

        gp = Function(name='gridpoints', dimensions=(p, d), shape=(1, 3))  # noqa
        src = Function(name='src', dimensions=(p,), shape=(1,))  # noqa
        rcv = Function(name='rcv', dimensions=(time, p), shape=(100, 1), space_order=0)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        # Use 'openmp' here instead of 'advanced' to disable loop blocking
        op = Operator(exprs, dle='openmp')

        iters = FindNodes(Iteration).visit(op)
        assert all([i.is_ParallelAtomic for i in iters if i.dim.name in atomic])
        assert all([not i.is_ParallelAtomic for i in iters if i.dim.name not in atomic])
        assert all([i.is_Parallel for i in iters if i.dim.name in parallel])
        assert all([not i.is_Parallel for i in iters if i.dim.name not in parallel])

    @pytest.mark.parametrize('exprs,wrappable', [
        # Easy: wrappable
        (['Eq(u.forward, u + 1)'], True),
        # Easy: wrappable
        (['Eq(w.forward, w + 1)'], True),
        # Not wrappable, as we're accessing w's back in a subsequent equation
        (['Eq(w.forward, w + 1)', 'Eq(v.forward, w)'], False),
        # Wrappable, but need to touch multiple indices with different modulos
        (['Eq(w.forward, u + w + 1)'], True),
        # Wrappable as the back timeslot is accessed only once, even though
        # later equations are writing again to w.forward
        (['Eq(w.forward, w + 1)', 'Eq(w.forward, w.forward + 2)'], True),
        # Not wrappable as the front is written before the back timeslot could be read
        (['Eq(w.forward, w + 1)', 'Eq(u.forward, u + w + 2)'], False),
    ])
    def test_loop_wrapping(self, exprs, wrappable):
        """Tests detection of WRAPPABLE property."""
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid, time_order=4)  # noqa
        w = TimeFunction(name='w', grid=grid, time_order=4)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)

        iters = FindNodes(Iteration).visit(op)

        # Dependence analysis checks
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1
        time_iter = time_iter[0]
        if wrappable:
            assert time_iter.is_Wrappable
        assert all(not i.is_Wrappable for i in iters if i is not time_iter)


class TestEquationAlgorithms(object):

    @pytest.mark.parametrize('expr,expected', [
        ('Eq(a[time, p], b[time, c[p, 0]+r, c[p, 1]] * f[p, r])', '[time, p, r, d, x, y]')
    ])
    def test_dimension_sort(self, expr, expected):
        """
        Tests that ``dimension_sort()`` provides meaningful Dimension orderings.
        """
        grid = Grid(shape=(10, 10))
        p = Dimension('p')
        r = Dimension('r')
        d = Dimension('d')
        time = grid.time_dim  # noqa
        x, y = grid.dimensions

        a = Function(name='a', dimensions=(time, p), shape=(10, 1))  # noqa
        b = Function(name='b', dimensions=(time, x, y), shape=(10, 10, 10))  # noqa
        c = Function(name='c', shape=(1, 2),  # noqa
                     dimensions=(p, d), dtype=np.int32)
        f = Function(name='f', dimensions=(p, r), shape=(1, 1))  # noqa

        expr = eval(expr)

        assert list(dimension_sort(expr)) == eval(expected)
