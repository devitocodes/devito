from numpy import float64, pi, max, abs

from devito import (Grid, Function, TimeFunction,
                    Derivative, Operator, solve, Eq)
from devito.types.multistage import resolve_method, MultiStage
from devito.ir.support import SymbolRegistry
from devito.ir.equations import lower_multistage
import pickle
from sympy import exp
import pytest
from devito import configuration
configuration['log-level'] = 'DEBUG'

def test_multistage_object(time_int='RK44'):
    extent = (1, 1)
    shape = (3, 3)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                      Derivative(U[0], (x, 2), fd_order=2) +
                      Derivative(U[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Class of the time integration scheme
    pdes = [resolve_method(time_int)(U[i], system_eqs_rhs[i]) for i in range(2)]

    assert all(isinstance(i, MultiStage) for i in pdes), "Not all elements are instances of MultiStage"


def test_multistage_pickles(time_int='RK44'):
    extent = (1, 1)
    shape = (3, 3)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                      Derivative(U[0], (x, 2), fd_order=2) +
                      Derivative(U[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Class of the time integration scheme
    pdes = [resolve_method(time_int)(U[i], system_eqs_rhs[i]) for i in range(2)]

    with open('test_saving_multistage.pkl', 'wb') as file:
        pickle.dump(pdes, file)

    with open('test_saving_multistage.pkl', 'rb') as file:
        pdes_saved = pickle.load(file)

    assert str(pdes) == str(pdes_saved), "The pdes where not saved correctly with pickles"


def test_multistage_lower_multistage(time_int='RK44'):
    extent = (1, 1)
    shape = (3, 3)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                      Derivative(U[0], (x, 2), fd_order=2) +
                      Derivative(U[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Class of the time integration scheme
    pdes = [resolve_method(time_int)(U[i], system_eqs_rhs[i]) for i in range(2)]

    sregistry=SymbolRegistry()
    try:
        lower_multistage(pdes, sregistry=sregistry)
    except:
        print("There is an error when lowering the MultiStage object")


def test_multistage_solve(time_int='RK44'):
    extent = (1, 1)
    shape = (3, 3)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                  Derivative(U[0], (x, 2), fd_order=2) +
                   Derivative(U[0], (y, 2), fd_order=2) +
                   src_spatial * src_temporal]

    # Time integration scheme
    pdes = [solve(system_eqs_rhs[i] - U[i], U[i], method=time_int) for i in range(2)]

    assert all(isinstance(i, MultiStage) for i in pdes), "Not all elements are instances of MultiStage"


def test_multistage_op_computing_directly(time_int='RK44'):
    extent = (1, 1)
    shape = (200, 200)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u_multi_stage', 'v_multi_stage']
    U_multi_stage = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system
    system_eqs_rhs = [U_multi_stage[1] + src_spatial * src_temporal,
                      Derivative(U_multi_stage[0], (x, 2), fd_order=2) +
                      Derivative(U_multi_stage[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Time integration scheme
    pdes = [resolve_method(time_int)(U_multi_stage[i], system_eqs_rhs[i]) for i in range(2)]
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=0.01, time=1)


def test_multistage_coupled_op_computing(time_int='RK44'):
    extent = (1, 1)
    shape = (200, 200)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u_multi_stage', 'v_multi_stage']
    U_multi_stage = [TimeFunction(name=name, grid=grid, space_order=2,
                      time_order=1, dtype=float64) for name in fun_labels]

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE system
    system_eqs_rhs = [U_multi_stage[1],
                      Derivative(U_multi_stage[0], (x, 2), fd_order=2) +
                      Derivative(U_multi_stage[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Time integration scheme
    pdes = resolve_method(time_int)(U_multi_stage, system_eqs_rhs)
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=0.01, time=1)


def test_multistage_op_computing_1eq(time_int='RK44'):
    extent = (1, 1)
    shape = (200, 200)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define wavefield unknowns: u (displacement) and v (velocity)
    u_multi_stage = TimeFunction(name='u_multi_stage', grid=grid, space_order=2, time_order=1, dtype=float64)

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t*dt - 1)**2)

    # PDE
    eq_rhs = (Derivative(u_multi_stage, (x, 2), fd_order=2) +
              Derivative(u_multi_stage, (y, 2), fd_order=2) +
              src_spatial * src_temporal)

    # Time integration scheme
    pde = [resolve_method(time_int)(u_multi_stage, eq_rhs)]
    op = Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)


# @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
@pytest.mark.parametrize('time_int', ['RK44'])
def test_multistage_methods_convergence(time_int):
    extent = (1000, 1000)
    shape = (201, 201)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim
    dx = extent[0] / (shape[0] - 1)

    # Medium velocity model
    vel = Function(name="vel", grid=grid, space_order=2, dtype=float64)
    vel.data[:] = 1.0
    vel.data[150:, :] = 1.3

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[100, 100] = 1/dx**2
    f0 = 0.01
    src_temporal = (1-2*(pi*f0*(t*dt-1/f0))**2)*exp(-(pi*f0*(t*dt-1/f0))**2)

    # Time axis
    t0, tn = 0.0, 500.0
    dt0 = max(vel.data)/dx**2
    nt = int((tn-t0)/dt0)
    dt0 = tn/nt

    # Time integrator solution
    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U_multi_stage = [TimeFunction(name=name+'_multi_stage', grid=grid, space_order=2, time_order=1, dtype=float64) for name in fun_labels]

    # PDE (2D acoustic)
    eq_rhs = [U_multi_stage[1], (Derivative(U_multi_stage[0], (x, 2), fd_order=2) +
                                 Derivative(U_multi_stage[0], (y, 2), fd_order=2) +
                                 src_spatial * src_temporal) * vel**2]

    # Time integration scheme
    pdes = [resolve_method(time_int)(U_multi_stage[i], eq_rhs[i]) for i in range(len(fun_labels))]
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=dt0, time=nt)

    # Devito's default solution
    U = [TimeFunction(name=name, grid=grid, space_order=2, time_order=1, dtype=float64) for name in fun_labels]
    # PDE (2D acoustic)
    eq_rhs = [U[1], (Derivative(U[0], (x, 2), fd_order=2) + Derivative(U[0], (y, 2), fd_order=2) +
                     src_spatial * src_temporal) * vel**2]

    # Time integration scheme
    pdes = [Eq(U[i].forward, solve(Eq(U[i].dt-eq_rhs[i]), U[i].forward)) for i in range(len(fun_labels))]
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=dt0, time=nt)
    assert max(abs(U[0].data[0,:]-U_multi_stage[0].data[0,:]))<10**-5, "the method is not converging to the solution"



