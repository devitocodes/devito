import pytest

from conftest import x, y

from devito.ir.support.basic import IterationInstance, TimedAccess


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
    tcxy_w0 = TimedAccess(fc[x, y], 'W', 0)
    tcxy_r0 = TimedAccess(fc[x, y], 'R', 0)
    tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1)
    tcx1y_r1 = TimedAccess(fc[x + 1, y], 'R', 1)
    x.reverse = True
    rev_tcxy_w0 = TimedAccess(fc[x, y], 'W', 0)
    rev_tcx1y1_r1 = TimedAccess(fc[x + 1, y + 1], 'R', 1)
    x.reverse = False
    return tcxy_w0, tcxy_r0, tcx1y1_r1, tcx1y_r1, rev_tcxy_w0, rev_tcx1y1_r1


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


def test_iteration_instance_cmp(ii_num, ii_literal):
    """
    Tests arithmetic operations involving objects of type IterationInstance.
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


def test_iteration_instance_distance(dims, ii_num, ii_literal):
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


def test_timed_access_cmp(ta_literal):
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
