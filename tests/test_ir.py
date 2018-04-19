import pytest

from conftest import EVAL, time, x, y, z, skipif_yask  # noqa

from devito import Eq, Grid, TimeFunction  # noqa
from devito.ir.equations import DummyEq, LoweredEq
from devito.ir.iet.nodes import Conditional, Expression
from devito.ir.support.basic import IterationInstance, TimedAccess, Scope
from devito.ir.support.space import (NullInterval, Interval, IntervalGroup,
                                     Any, Forward, Backward)
from devito.ir.support.utils import detect_flow_directions
from devito.symbolics import indexify


@pytest.fixture(scope="session")
def ii_num(fa, fc):
    fa4 = IterationInstance(fa[4])
    fc00 = IterationInstance(fc[0, 0])
    fc11 = IterationInstance(fc[1, 1])
    fc23 = IterationInstance(fc[2, 3])
    return fa4, fc00, fc11, fc23


@pytest.fixture(scope="session")
def ii_literal(fa, fc):
    fax = IterationInstance(fa[x])
    fcxy = IterationInstance(fc[x, y])
    fcx1y = IterationInstance(fc[x + 1, y])
    return fax, fcxy, fcx1y


@pytest.fixture(scope="session")
def ta_literal(fc):
    fwd_directions = {x: Forward, y: Forward}
    mixed_directions = {x: Backward, y: Forward}
    tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, fwd_directions)
    tcxy_r0 = TimedAccess(fc[x, y], 'R', 0, fwd_directions)
    tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, fwd_directions)
    tcx1y_r1 = TimedAccess(fc[x + 1, y], 'R', 1, fwd_directions)
    rev_tcxy_w0 = TimedAccess(fc[x, y], 'W', 0, mixed_directions)
    rev_tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1, mixed_directions)
    return tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1


@skipif_yask
def test_iteration_instance_arithmetic(dims, ii_num, ii_literal):
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


@skipif_yask
def test_iteration_instance_cmp(ii_num, ii_literal):
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


@skipif_yask
def test_iteration_instance_distance(dims, ii_num, ii_literal):
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


@skipif_yask
def test_timed_access_cmp(ta_literal):
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


@skipif_yask
@pytest.mark.parametrize('expr,expected', [
    ('Eq(ti0[x,y,z], ti1[x,y,z])', None),
    ('Eq(ti0[x,y,z], ti0[x,y,z])', 'flow,independent,None,direct'),
    ('Eq(ti0[x,y,z], ti0[x,y,z])', 'flow,inplace,None,direct'),
    ('Eq(ti0[x,y,z], ti0[x,y,z-1])', 'flow,carried,z,direct'),
    ('Eq(ti0[x,y,z], ti0[x-1,y,z-1])', 'flow,carried,x,direct'),
    ('Eq(ti0[x,y,z], ti0[x-1,y,z+1])', 'flow,carried,x,direct'),
    ('Eq(ti0[x,y,z], ti0[x+1,y+2,z])', 'anti,carried,x,direct'),
    ('Eq(ti0[x,y,z], ti0[x,y+2,z-3])', 'anti,carried,y,direct'),
    ('Eq(ti0[x,y,z], ti0[fa[x],y,z])', 'all,carried,x,indirect'),
    ('Eq(ti0[x,y,z], ti0[fa[x],y,fa[z]])', 'all,carried,x,indirect'),
    ('Eq(ti0[x,fa[y],z], ti0[x,y,z])', 'all,carried,y,indirect'),
    ('Eq(ti0[x,y,z], ti0[x-1,fa[y],z])', 'flow,carried,x,direct'),
])
def test_dependences_eq(expr, expected, ti0, ti1, fa):
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
        type, mode, cause, direct = expected.split(',')
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
    if cause == 'None':
        assert dep.cause is None
        return
    else:
        assert dep.cause.name == cause

    # Check mode restricted to the cause
    assert getattr(dep, 'is_%s' % mode)(dep.cause)
    non_causes = [i for i in [x, y, z] if i is not dep.cause]
    assert all(not getattr(dep, 'is_%s' % mode)(i) for i in non_causes)

    # Check if it's direct or indirect
    assert getattr(dep, 'is_%s' % direct)


@skipif_yask
@pytest.mark.parametrize('exprs,expected', [
    # Trivial flow dep
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti3[x,y,z], ti0[x,y,z])'],
     ['ti0,flow,None']),
    # One flow dep, one anti dep
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti1[x,y,z], ti0[x,y,z])'],
     ['ti0,flow,None', 'ti1,anti,None']),
    # One output dep, two identical flow deps
    (['Eq(ti3[x+1,y,z], ti1[x,y,z])',
      'Eq(ti3[x+1,y,z], ti3[x,y,z])'],
     ['ti3,output,None', 'ti3,flow,x', 'ti3,flow,x']),
    # One flow independent dep, two flow carried flow deps
    (['Eq(ti0[x,y,z], ti0[x,y,z])',
      'Eq(ti1[x,y,z], ti0[x,y-1,z])',
      'Eq(ti3[x,y,z], ti0[x-2,y,z])'],
     ['ti0,flow,None', 'ti0,flow,y', 'ti0,flow,x']),
    # An indirect dep, conservatively assumed flow and anti
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti3[x,y,z], ti0[fa[x],y,z])'],
     ['ti0,flow,x', 'ti0,anti,x']),
    # A direct anti dep "masking" the indirect dep in an inner dimension
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti3[x,y,z], ti0[x+1,fa[y],z])'],
     ['ti0,anti,x']),
    # Conservatively assume dependences due to "complex" affine index functions
    (['Eq(ti0[x,y,z], ti1[x,2*y,z])',
      'Eq(ti1[x,3*y,z], ti0[x+1,y,z])'],
     ['ti1,flow,y', 'ti1,anti,y', 'ti0,anti,x']),
    # Data indices don't match iteration indices, so conservatively assume
    # all sort of deps
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti3[x,y,z], ti0[y+1,y,y])'],
     ['ti0,flow,x', 'ti0,anti,x']),
    # Data indices don't match iteration indices, so conservatively assume
    # all sort of deps
    (['Eq(ti0[x,y,z], ti1[x,y,z])',
      'Eq(ti3[x,y,z], ti0[x,y,x])'],
     ['ti0,flow,z', 'ti0,anti,z']),
])
def test_dependences_scope(exprs, expected, ti0, ti1, ti3, fa):
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
            item = (dep.function.name, i, str(dep.cause))
            assert item in expected
            expected.remove(item)

    # Sanity check: we did find all of the expected dependences
    assert len(expected) == 0


@skipif_yask
def test_flow_detection():
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


@skipif_yask
def test_intervals_intersection():
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


@skipif_yask
def test_intervals_union():
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

    # Mixed disjoint (note: IntervalGroup input order is irrelevant)
    assert ix.union(ix4) == IntervalGroup([ix4, ix])
    assert ix.union(ix5) == Interval(x, -3, 2)
    assert ix6.union(ix) == IntervalGroup([ix, ix6])
    assert ix.union(nully) == IntervalGroup([ix, nully])
    assert ix.union(iy) == IntervalGroup([iy, ix])
    assert iy.union(ix) == IntervalGroup([iy, ix])


@skipif_yask
def test_intervals_merge():
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


@skipif_yask
def test_intervals_subtract():
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


@skipif_yask
def test_iet_conditional(fc):
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
