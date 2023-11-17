import numpy as np
import pytest

from devito import Grid, TimeFunction, Eq, XDSLOperator, Operator, solve, norm

from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType


def test_xdsl_I():
    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_II():
    # Define a simple Devito Operator
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_III():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)

    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()

    # Check number of args
    assert len(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args) == 3
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[0].type, FieldType)  # noqa
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[1].type, FieldType)  # noqa
    assert isinstance(op._module.regions[0].blocks[0].ops.first.body.blocks[0]._args[2].type, LLVMPointerType)  # noqa

    ops = list(op._module.regions[0].blocks[0].ops.first.body.blocks[0].ops)
    assert type(ops[5] == Addi)
    assert type(ops[6] == For)

    scffor_ops = list(ops[6].regions[0].blocks[0].ops)

    assert isinstance(scffor_ops[0], LoadOp)
    assert isinstance(scffor_ops[1], ApplyOp)
    assert isinstance(scffor_ops[2], StoreOp)
    assert isinstance(scffor_ops[3], Yield)

    assert type(ops[7] == Call)
    assert type(ops[8] == StoreOp)
    assert type(ops[9] == Return)


def test_diffusion_2D():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = 1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=1, dt=0.1)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = 1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=1, dt=0.1)

    assert np.isclose(f.data, f2.data, rtol=1e-06).all()


@pytest.mark.parametrize('shape', [(11, 11), (31, 31), (51, 51), (101, 101)])
def test_diffusion_2D_II(shape):
    # Define a simple Devito Operator
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    sigma = .1
    dt = sigma * dx * dy
    nt = 10

    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=nt, dt=dt)

    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()


@pytest.mark.parametrize('shape', [(11, 11, 11), (31, 31, 31),
                         (51, 51, 51), (101, 101, 101)])
def test_diffusion_3D_II(shape):
    shape = (10, 10, 10)
    grid = Grid(shape=shape)
    rng = np.random.default_rng(123)
    dx = 2. / (shape[0] - 1)
    dy = 2. / (shape[1] - 1)
    dz = 2. / (shape[2] - 1)

    sigma = .1
    dt = sigma * dx * dy * dz
    nt = 50

    rng = np.random.default_rng(123)

    arr1 = rng.random(shape)

    # Devito setup
    f = TimeFunction(name='f', grid=grid, space_order=2)
    f.data[:] = arr1
    eqn = Eq(f.dt, 0.5 * f.laplace)
    op = Operator(Eq(f.forward, solve(eqn, f.forward)))
    op.apply(time_M=nt, dt=dt)

    # xDSL-Devito setup
    f2 = TimeFunction(name='f2', grid=grid, space_order=2)
    f2.data[:] = arr1
    eqn = Eq(f2.dt, 0.5 * f2.laplace)
    op = XDSLOperator(Eq(f2.forward, solve(eqn, f2.forward)))
    op.apply(time_M=50, dt=dt)

    max_error = np.max(np.abs(f.data - f2.data))
    assert np.isclose(max_error, 0.0, atol=1e-04)
    assert np.isclose(f.data, f2.data, rtol=1e-05).all()


@pytest.mark.parametrize('shape', [(11, 11, 11), (31, 31, 31),
                         (51, 51, 51), (101, 101, 101)])
@pytest.mark.parametrize('steps', [1, 3, 8, 40])
def test_unary(shape, steps):

    grid = Grid(shape=shape)

    u = TimeFunction(name='u', grid=grid)
    u.data[:, :] = 5
    eq = Eq(u.forward, u + 0.1)
    xop = XDSLOperator([eq])
    xop.apply(time_M=steps)
    xdsl_data = u.data_with_halo.copy()

    u.data[:, :] = 5
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data = u.data_with_halo.copy()

    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


@pytest.mark.parametrize('shape', [(101, 101, 101)])
@pytest.mark.parametrize('so', [2, 4, 8])
@pytest.mark.parametrize('to', [2])
@pytest.mark.parametrize('nt', [10, 20, 100])
def test_acoustic_3D(shape, so, to, nt):

    grid = Grid(shape=shape)
    dt = 0.0001

    # Define the wavefield with the size of the model and the time dimension
    u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

    pde = u.dt2 - u.laplace
    eq0 = solve(pde, u.forward)

    stencil = Eq(u.forward, eq0)
    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # Devito Operator
    op = Operator([stencil])
    op.apply(time=nt, dt=dt)
    devito_norm = norm(u)

    u.data[:, :, :] = 0
    u.data[:, 40:50, 40:50] = 1

    # XDSL Operator
    xdslop = XDSLOperator([stencil])
    xdslop.apply(time=nt, dt=dt)
    xdsl_norm = norm(u)

    assert np.isclose(devito_norm, xdsl_norm, rtol=1e-04).all()
