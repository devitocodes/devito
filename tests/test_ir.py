import pytest
import numpy as np
from sympy import S

from conftest import EVAL, skipif  # noqa
from devito import (Eq, Inc, Grid, Constant, Function, TimeFunction, # noqa
                    Operator, Dimension, SubDimension, switchconfig)
from devito.ir.cgen import ccode
from devito.ir.equations import LoweredEq
from devito.ir.equations.algorithms import dimension_sort
from devito.ir.iet import Iteration, FindNodes
from devito.ir.support.basic import (IterationInstance, TimedAccess, Scope,
                                     Vector, AFFINE, REGULAR, IRREGULAR, mocksym0,
                                     mocksym1)
from devito.ir.support.space import (NullInterval, Interval, Forward, Backward,
                                     IntervalGroup, IterationSpace)
from devito.ir.support.guards import GuardOverflow
from devito.symbolics import DefFunction, FieldFromPointer
from devito.tools import prod
from devito.tools.data_structures import frozendict
from devito.types import Array, Bundle, CriticalRegion, Jump, Scalar, Symbol


class TestVectorHierarchy:

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
        return Array(name='fa', shape=(3,), dimensions=(x,)).indexed

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
        intervals_swap = [Interval(y, 0, 0), Interval(x, 0, 0)]
        fwd_ispace = IterationSpace(intervals, directions={x: Forward, y: Forward})
        fwd_ispace_swap = IterationSpace(intervals_swap,
                                         directions={x: Forward, y: Forward})
        mixed_ispace = IterationSpace(intervals, directions={x: Backward, y: Forward})
        tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, fwd_ispace)
        tcxy_r0 = TimedAccess(fc[x, y], 'R', 0, fwd_ispace)
        tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, fwd_ispace)
        tcx1y_r1 = TimedAccess(fc[x + 1, y], 'R', 1, fwd_ispace)
        rev_tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, mixed_ispace)
        rev_tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, mixed_ispace)
        tcyx_irr0 = TimedAccess(fc[y, x], 'R', 0, fwd_ispace)
        tcxx_irr1 = TimedAccess(fc[x, x], 'R', 0, fwd_ispace)
        tcxy_irr2 = TimedAccess(fc[x, y], 'R', 0, fwd_ispace_swap)
        return (tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1,
                tcyx_irr0, tcxx_irr1, tcxy_irr2)

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
        Test arithmetic operations involving objects of type IterationInstance.
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
        Test calculation of vector distance between objects of type IterationInstance.
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
        Test comparison of objects of type IterationInstance.
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

    def test_timed_access_regularity(self, ta_literal):
        """
        Test TimedAcces.{is_regular,is_irregular}
        """
        (tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1,
         tcyx_irr0, tcxx_irr1, tcxy_irr2) = ta_literal

        # Regulars
        assert tcxy_w0.is_regular and not tcxy_w0.is_irregular
        assert tcxy_r0.is_regular and not tcxy_r0.is_irregular
        assert tcx1y1_r1.is_regular and not tcx1y1_r1.is_irregular
        assert tcx1y_r1.is_regular and not tcx1y_r1.is_irregular
        assert rev_tcxy_w0.is_regular and not rev_tcxy_w0.is_irregular
        assert rev_tcx1y1_r1.is_regular and not rev_tcx1y1_r1.is_irregular

        # Irregulars
        assert tcyx_irr0.is_irregular and not tcyx_irr0.is_regular
        assert tcxx_irr1.is_irregular and not tcxx_irr1.is_regular
        assert tcxy_irr2.is_irregular and not tcxy_irr2.is_regular

    def test_timed_access_distance(self, x, y, ta_literal):
        """
        Test the comparison of objects of type TimedAccess.
        """
        (tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1,
         tcyx_irr0, tcxx_irr1, tcxy_irr2) = ta_literal

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

        # The distance must be infinity when the findices directions
        # are homogeneous, but one of the two TimedAccesses is irregular (in
        # this case, the aindices differ, as the irregular TimedAccess uses
        # `y` where `x` is expected)
        assert tcxy_w0.distance(tcyx_irr0) == (S.Infinity)
        assert tcx1y_r1.distance(tcyx_irr0) == (S.Infinity)
        assert tcxy_w0.distance(tcxx_irr1) == (0, S.Infinity)

        # The distance must be infinity when the aindices are compatible but
        # one of the TimedAccesses is irregular due to mismatching
        # findices-IterationSpace
        assert tcxy_w0.distance(tcxy_irr2) == (S.Infinity)

    def test_timed_access_cmp(self, ta_literal):
        """
        Test comparison of objects of type TimedAccess.
        """
        (tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1,
         tcyx_irr0, tcxx_irr1, tcxy_irr2) = ta_literal

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

        # Non-comparable due to different aindices
        try:
            tcxy_w0 > tcyx_irr0
            assert False
        except TypeError:
            assert True
        except:
            assert False

        # Non-comparable due to mismatching Intervals
        try:
            tcxy_w0 > tcyx_irr0
            assert False
        except TypeError:
            assert True
        except:
            assert False

        # Comparable even though the TimedAccess is irregular (reflexivity)
        assert tcyx_irr0 >= tcyx_irr0
        assert tcyx_irr0 == tcyx_irr0


class TestSpace:

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

    def test_intervals_switch(self, x, y):
        nullx = NullInterval(x)
        nully = NullInterval(y)

        assert nullx.switch(y) == nully

        ix = Interval(x, 2, -2)
        iy = Interval(y, 2, -2)

        assert ix.switch(y) == iy
        assert iy.switch(x) == ix
        assert ix.switch(y).switch(x) == ix

    def test_space_intersection(self, x, y):
        ig0 = IntervalGroup([Interval(x, 1, -1)])
        ig1 = IntervalGroup([Interval(x, 2, -2), Interval(y, 3, -3)])

        ig = IntervalGroup.generate('intersection', ig0, ig1)

        assert len(ig) == 1
        assert ig[0] == Interval(x, 2, -2)

        # Now the same but with IterationSpaces
        ispace0 = IterationSpace(ig0)
        ispace1 = IterationSpace(ig1)

        ispace = IterationSpace.intersection(ispace0, ispace1)

        assert len(ispace) == 1
        assert ispace.intervals == ig


class TestDependenceAnalysis:

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
        ('u[x*x+1,y,z]', (REGULAR, AFFINE, AFFINE)),
        ('u[x*y,y,z]', (IRREGULAR, AFFINE, AFFINE)),
        ('u[x + z,x + y,z*z]', (IRREGULAR, IRREGULAR, REGULAR)),
        ('u[x+1,u[2,2,2],z-1]', (AFFINE, IRREGULAR, AFFINE)),
        ('u[y,x,z]', (IRREGULAR, IRREGULAR, AFFINE)),
    ])
    def test_index_mode_detection(self, indexed, expected):
        """
        Test detection of IterationInstance access modes (AFFINE, REGULAR, IRREGULAR).

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
        ('Eq(ti0[x,y,z], ti0[fa[x],y,z])', 'all,carried,x,regular'),
        ('Eq(ti0[x,y,z], ti0[fa[x],y,fa[z]])', 'all,carried,x,regular'),
        ('Eq(ti0[x,y,z], ti0[fa[y],y,z])', 'all,carried,x,irregular'),
        ('Eq(ti0[x,fa[y],z], ti0[x,y,z])', 'all,carried,y,regular'),
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

        # Force unnatural flow, only to stress the compiler to see if it is
        # capable of detecting anti-dependences
        expr.ispace._directions = frozendict({i: Forward for i in expr.ispace.directions})

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

        # Force unnatural flow, only to stress the compiler to see if it is
        # capable of detecting anti-dependences
        for i in exprs:
            i.ispace._directions = frozendict({i: Forward for i in i.ispace.directions})

        scope = Scope(exprs)
        assert len(scope.d_all) == len(expected)

        for i in ['flow', 'anti', 'output']:
            for dep in getattr(scope, 'd_%s' % i):
                item = (dep.function.name, i, str(set(dep.cause)))
                assert item in expected
                expected.remove(item)

        # Sanity check: we did find all of the expected dependences
        assert len(expected) == 0

    def test_ffp(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        ffp = FieldFromPointer(f._C_field_data, f._C_symbol)

        exprs = [Eq(ffp, 1), Eq(g.indexify(), f.indexify())]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)
        assert len(scope.d_all) == 2
        assert len(scope.d_flow) == 1
        v = scope.d_flow.pop()
        assert v.function is f
        assert len(scope.d_anti) == 1
        v = scope.d_anti.pop()
        assert v.function is f

    def test_indexedbase_across_jump(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        class Foo(DefFunction, Jump):
            pass

        exprs = [Eq(f.indexed, 0),
                 Eq(h.indexed, Foo('foo', 3)),
                 Eq(g.indexed, f.indexed)]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)
        assert len(scope.d_all) == 3
        assert len(scope.d_flow) == 2
        assert len(scope.d_anti) == 1
        assert any(v.function is f for v in scope.d_flow)
        assert any(v.function is mocksym0 for v in scope.d_flow)

    def test_indirect_access(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        s0 = Symbol('s0', dtype=np.int32)
        s1 = Symbol('s1', dtype=np.int32)

        exprs = [Eq(s1, 1), Eq(f[s0, s1], 5)]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)
        assert len(scope.d_all) == len(scope.d_flow) == 1
        v = scope.d_flow.pop()
        assert v.function is s1

    def test_array_shared(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        a = Array(name='a', dimensions=(x, y), halo=((2, 2), (2, 2)), scope='shared')
        s = Symbol(name='s')

        exprs = [Eq(a[x, y], 1), Eq(s, a[x, y-2] + a[x, y+2])]
        exprs = [LoweredEq(i) for i in exprs]
        scope = Scope(exprs)
        # There *seems* to be a WAR here, but the fact that `a` is thread-shared
        # ensures that is not the case
        # There is instead a RAW -- no surprises here
        assert len(scope.d_all) == len(scope.d_flow) == 2
        assert scope.d_flow.pop().function is a

        # If the reads lexicographically precede the writes, then instead it really
        # is a WAR
        exprs = [Eq(s, a[x, y-2] + a[x, y+2]), Eq(a[x, y], 1)]
        exprs = [LoweredEq(i) for i in exprs]
        scope = Scope(exprs)
        assert len(scope.d_all) == len(scope.d_anti) == 2
        assert scope.d_anti.pop().function is a

    @pytest.mark.parametrize('eqns', [
        ['Eq(a0[4], 1)', 'Eq(s, a0[3])'],
        ['Eq(a1[3, 4], 1)', 'Eq(s, a1[3, 5])'],
        ['Eq(a1[x+1, 4], 1)', 'Eq(s, a1[x, 5])'],
    ])
    def test_nodep(self, eqns):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        a0 = Array(name='a', dimensions=(x,))  # noqa
        a1 = Array(name='a', dimensions=(x, y))  # noqa
        s = Symbol(name='s')  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(eqns)):
            eqns[i] = LoweredEq(eval(e))

        scope = Scope(eqns)
        assert len(scope.d_all) == 0

    @pytest.mark.parametrize('eqns', [
        ['Eq(a0[4], 1)', 'Eq(s, a0[4])'],
        ['Eq(a1[x+1, 4], 1)', 'Eq(s, a1[x, 4])'],
    ])
    def test_dep_nasty(self, eqns):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        a0 = Array(name='a', dimensions=(x,))  # noqa
        a1 = Array(name='a', dimensions=(x, y))  # noqa
        s = Symbol(name='s')  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(eqns)):
            eqns[i] = LoweredEq(eval(e))

        scope = Scope(eqns)
        assert len(scope.d_all) == 1

    def test_critical_region_v0(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)

        s0 = Symbol(name='s0')
        s1 = Symbol(name='s1')

        exprs = [Eq(s0, CriticalRegion(True)),
                 Eq(f.indexify(), 1),
                 Eq(s1, CriticalRegion(False))]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)

        # Mock depedencies so that the fences (CriticalRegions) don't float around
        assert len(scope.writes[mocksym0]) == 2
        assert len(scope.reads[mocksym0]) == 2
        assert len(scope.d_all) == 3

        # No other mock depedencies because there's no other place the Eq
        # within the critical sequence can float to
        assert len(scope.writes[mocksym1]) == 1
        assert mocksym1 not in scope.reads

    def test_critical_region_v1(self):
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        u = Function(name='u', grid=grid)

        s0 = Symbol(name='s0')
        s1 = Symbol(name='s1')

        exprs = [Eq(g.indexify(), 2),
                 Eq(h.indexify(), 2),
                 Eq(s0, CriticalRegion(True)),
                 Eq(f.indexify(), 1),
                 Eq(s1, CriticalRegion(False)),
                 Eq(u.indexify(), 3)]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)

        # Mock depedencies so that the fences (CriticalRegions) don't float around
        assert len(scope.writes[mocksym0]) == 2
        assert len(scope.reads[mocksym0]) == 4
        assert len([i for i in scope.d_all
                    if i.source.access is mocksym0
                    or i.sink.access is mocksym0]) == 7

        # More mock depedencies because Eq must not float outside of the critical
        # sequence
        assert len(scope.writes[mocksym1]) == 1
        assert len(scope.reads[mocksym1]) == 2
        assert len(scope.d_all) == 9

    def test_bundle_components(self):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        v = Function(name='v', grid=grid)
        w = Function(name='w', grid=grid)
        u0 = Function(name='u0', grid=grid)
        u1 = Function(name='u1', grid=grid)

        fg = Bundle(name='fg', components=(f, g))
        vw = Bundle(name='vw', components=(v, w))

        exprs = [Eq(fg.indexify(), 1),
                 Eq(u0.indexify(), fg[0, x, y] + 2),
                 Eq(vw[0, x, y], 3),
                 Eq(u1.indexify(), vw[1, x, y] + 4)]
        exprs = [LoweredEq(i) for i in exprs]

        scope = Scope(exprs)
        assert len(scope.d_all) == 1
        assert len(scope.d_flow) == 1
        dep, = scope.d_flow
        assert dep.function is f


class TestParallelismAnalysis:

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
         [], ['x', 'y']),
        # Test for issue #1902
        (['Eq(u[0, y], v)', 'Eq(w, u[0, y])'],
         [], ['y'])
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
        w = TimeFunction(name='w', grid=grid, save=None)  # noqa

        cx = Function(name='coeff_x', dimensions=(p, rx), shape=(1, 2))  # noqa
        cy = Function(name='coeff_y', dimensions=(p, ry), shape=(1, 2))  # noqa

        gp = Function(name='gridpoints', dimensions=(p, d), shape=(1, 2))  # noqa
        src = Function(name='src', dimensions=(p,), shape=(1,))  # noqa
        rcv = Function(name='rcv', dimensions=(time, p), shape=(100, 1), space_order=0)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, opt='openmp')

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
        (['Eq(s, 0, implicit_dims=[time, p])',
          'Inc(s, time*p*2, implicit_dims=[time, p])',
          'Eq(rcv, s)'],
         [], ['time', 'p']),
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

        s = Scalar(name='s')  # noqa

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
        op = Operator(exprs, opt='openmp')

        iters = FindNodes(Iteration).visit(op)
        assert all([i.is_ParallelAtomic for i in iters if i.dim.name in atomic])
        assert all([not i.is_ParallelAtomic for i in iters if i.dim.name not in atomic])
        assert all([i.is_Parallel for i in iters if i.dim.name in parallel])
        assert all([not i.is_Parallel for i in iters if i.dim.name not in parallel])


class TestEquationAlgorithms:

    @pytest.mark.parametrize('expr,expected', [
        ('Eq(a[time, p], b[time, c[p, 0]+r, c[p, 1]] * f[p, r])', '[time, p, r, d]')
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


class TestGuards:

    def test_guard_overflow(self):
        """
        A toy test showing how to create a Guard to prevent writes to buffers
        with not enough space to fit another snapshot.
        """
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)

        freespace = Scalar(name='freespace')
        size = prod(f.symbolic_shape)

        guard = GuardOverflow(freespace, size)

        assert ccode(guard) == 'freespace >= f_vec->size[0]*f_vec->size[1]'
