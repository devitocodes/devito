import pytest

from conftest import EVAL, time, x, y, z, skipif_backend  # noqa

import numpy as np

from devito import (Eq, Inc, Grid, Function, TimeFunction, # noqa
                    Operator, Dimension, configuration)
from devito.ir.equations import DummyEq, LoweredEq
from devito.ir.equations.algorithms import dimension_sort
from devito.ir.iet.nodes import Conditional, Expression, Iteration
from devito.ir.iet.visitors import FindNodes
from devito.ir.support.basic import IterationInstance, TimedAccess, Scope
from devito.ir.support.space import (NullInterval, Interval, IntervalGroup,
                                     Any, Forward, Backward)
from devito.ir.support.utils import detect_flow_directions
from devito.symbolics import indexify

pytestmark = pytest.mark.skipif(configuration['backend'] == 'yask' or
                                configuration['backend'] == 'ops',
                                reason="testing is currently restricted")


class TestVectorDistanceArithmetic(object):

    @pytest.fixture
    def ii_num(self, fa, fc):
        fa4 = IterationInstance(fa[4])
        fc00 = IterationInstance(fc[0, 0])
        fc11 = IterationInstance(fc[1, 1])
        fc23 = IterationInstance(fc[2, 3])
        return fa4, fc00, fc11, fc23

    @pytest.fixture
    def ii_literal(self, fa, fc):
        fax = IterationInstance(fa[x])
        fcxy = IterationInstance(fc[x, y])
        fcx1y = IterationInstance(fc[x + 1, y])
        return fax, fcxy, fcx1y

    @pytest.fixture
    def ta_literal(self, fc):
        fwd_directions = {x: Forward, y: Forward}
        mixed_directions = {x: Backward, y: Forward}
        tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, fwd_directions)
        tcxy_r0 = TimedAccess(fc[x, y], 'R', 0, fwd_directions)
        tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, fwd_directions)
        tcx1y_r1 = TimedAccess(fc[x + 1, y], 'R', 1, fwd_directions)
        rev_tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, mixed_directions)
        rev_tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, mixed_directions)
        return tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1

    def test_iteration_instance_arithmetic(self, dims, ii_num, ii_literal):
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

    def test_iteration_instance_distance(self, dims, ii_num, ii_literal):
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

        # Should faxl due non matching indices
        try:
            fcxy.distance(fax)
            assert False
        except TypeError:
            pass
        except:
            assert False

        # Distance up to provided dimension
        assert fcxy.distance(fcx1y, x) == (-1,)
        assert fcxy.distance(fcx1y, y) == (-1, 0)

    def test_timed_access_cmp(self, ta_literal):
        """
        Tests comparison of objects of type TimedAccess.
        """
        tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1 = ta_literal

        # Equality check
        assert tcxy_w0 == tcxy_w0
        assert (tcxy_w0 != tcxy_r0) is False
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

    def test_intervals_intersection(self):
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

        # All defined disjoint
        assert ix.intersection(ix2) == nullx
        assert ix.intersection(ix3) == nullx
        assert ix2.intersection(ix3) == nullx
        assert ix.intersection(iy) == nullx
        assert iy.intersection(ix) == nully

        ix4 = Interval(x, 1, 4)
        ix5 = Interval(x, -3, 0)

        # All defined overlapping
        assert ix.intersection(ix4) == Interval(x, 1, 2)
        assert ix.intersection(ix5) == Interval(x, -2, 0)

    def test_intervals_union(self):
        nullx = NullInterval(x)

        # All nulls
        assert nullx.union(nullx) == nullx

        ix = Interval(x, -2, 2)

        # Mixed nulls and defined on the same dimension
        assert nullx.union(ix) == ix
        assert ix.union(ix) == ix
        assert ix.union(nullx) == ix

        ix2 = Interval(x, 1, 4)
        ix3 = Interval(x, -3, 6)

        # All defined overlapping
        assert ix.union(ix2) == Interval(x, -2, 4)
        assert ix.union(ix3) == ix3
        assert ix2.union(ix3) == ix3

        ix4 = Interval(x, 4, 8)
        ix5 = Interval(x, -3, -3)
        ix6 = Interval(x, -10, -3)
        nully = NullInterval(y)
        iy = Interval(y, -2, 2)

        # Mixed disjoint (note: IntervalGroup input order is relevant)
        assert ix.union(ix4) == IntervalGroup([ix, ix4])
        assert ix.union(ix5) == Interval(x, -3, 2)
        assert ix6.union(ix) == IntervalGroup([ix6, ix])
        assert ix.union(nully) == IntervalGroup([ix, nully])
        assert ix.union(iy) == IntervalGroup([ix, iy])
        assert iy.union(ix) == IntervalGroup([iy, ix])

    def test_intervals_merge(self):
        nullx = NullInterval(x)

        # All nulls
        assert nullx.merge(nullx) == nullx

        ix = Interval(x, -2, 2)

        # Mixed nulls and defined on the same dimension
        assert nullx.merge(ix) == ix
        assert ix.merge(ix) == ix
        assert ix.merge(nullx) == ix

        ix2 = Interval(x, 1, 4)
        ix3 = Interval(x, -3, 6)

        # All defined overlapping
        assert ix.merge(ix2) == Interval(x, -2, 4)
        assert ix.merge(ix3) == ix3
        assert ix2.merge(ix3) == ix3

        ix4 = Interval(x, 0, 0)
        ix5 = Interval(x, -1, -1)
        ix6 = Interval(x, 9, 11)

        # Non-overlapping
        assert ix.merge(ix4) == Interval(x, -2, 2)
        assert ix.merge(ix5) == Interval(x, -2, 2)
        assert ix4.merge(ix5) == Interval(x, -1, 0)
        assert ix.merge(ix6) == Interval(x, -2, 11)
        assert ix6.merge(ix) == Interval(x, -2, 11)
        assert ix5.merge(ix6) == Interval(x, -1, 11)

    def test_intervals_subtract(self):
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


class TestDependenceAnalysis(object):

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
    def test_single_eq(self, expr, expected, ti0, ti1, fa):
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
        dep = deps[0]

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
        non_causes = [i for i in [x, y, z] if i is not cause]
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

    def test_flow_detection(self):
        """Test detection of information flow."""
        grid = Grid((10, 10))
        u2 = TimeFunction(name="u2", grid=grid, time_order=2)
        u1 = TimeFunction(name="u1", grid=grid, save=10, time_order=2)
        exprs = [LoweredEq(indexify(Eq(u1.forward, u1 + 2.0 - u1.backward))),
                 LoweredEq(indexify(Eq(u2.forward, u2 + 2*u2.backward - u1.dt2)))]
        mapper = detect_flow_directions(exprs)
        assert mapper.get(grid.stepping_dim) == {Forward}
        assert mapper.get(grid.time_dim) == {Any, Forward}
        assert all(mapper.get(i) == {Any} for i in grid.dimensions)


class TestIET(object):

    def test_nodes_conditional(self, fc):
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

        op = Operator(exprs, dle='speculative')

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
        Tests that ``dimension_sort()`` provides meaningful :class:`Dimension` orderings.
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
