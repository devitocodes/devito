import pytest
import numpy as np

from devito import (Constant, Grid, TimeFunction, SparseTimeFunction, Operator,
                    Eq, ConditionalDimension, SubDimension, SubDomain, configuration)
from devito.ir import FindSymbols, retrieve_iteration_tree
from devito.exceptions import InvalidOperator


def test_basic():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid)
    u1 = TimeFunction(name='u', grid=grid)

    eqn = Eq(u.forward, u + 1)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


@pytest.mark.parallel(mode=[(1, 'basic'), (1, 'diag2'), (1, 'full')])
def test_mpi():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dx2 + 1.)

    op0 = Operator(eqn)
    op1 = Operator(eqn, opt=('advanced', {'linearize': True}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)


def test_cire():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid, space_order=2)
    u1 = TimeFunction(name='u', grid=grid, space_order=2)

    eqn = Eq(u.forward, u.dy.dy + 1.)

    op0 = Operator(eqn, opt=('advanced', {'cire-mingain': 0}))
    op1 = Operator(eqn, opt=('advanced', {'linearize': True, 'cire-mingain': 0}))

    # Check generated code
    assert 'uL0' not in str(op0)
    assert 'uL0' in str(op1)

    op0.apply(time_M=10)
    op1.apply(time_M=10, u=u1)

    assert np.all(u.data == u1.data)
