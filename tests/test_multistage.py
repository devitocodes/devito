from numpy import pi, float64, max, abs
from sympy import exp

from devito import (Grid, Function, TimeFunction,
                    Derivative, Operator, solve, Eq)
from devito.types.multistage import resolve_method
from devito.ir.support import SymbolRegistry
from devito.ir.equations import lower_multistage


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
    return [resolve_method(time_int)(U[i], system_eqs_rhs[i]) for i in range(2)]


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

    return lower_multistage(pdes, sregistry=sregistry)



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
    return [solve(system_eqs_rhs[i] - U[i], U[i], method=time_int) for i in range(2)]


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

    # Define wavefield unknowns: u (displacement) and v (velocity)
    fun_labels = ['u', 'v']
    U = [TimeFunction(name=name, grid=grid, space_order=2,
                                  time_order=1, dtype=float64) for name in fun_labels]
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                      Derivative(U[0], (x, 2), fd_order=2) +
                      Derivative(U[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Time integration scheme
    pdes = [Eq(U[i], system_eqs_rhs[i]) for i in range(2)]
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    return max(abs(U_multi_stage[0].data[0, :] - U[0].data[0, :]))


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

    # Solving now using Devito's standard time solver
    u = TimeFunction(name='u', grid=grid, space_order=2, time_order=1, dtype=float64)
    eq_rhs = (Derivative(u, (x, 2), fd_order=2) +
              Derivative(u, (y, 2), fd_order=2) +
              src_spatial * src_temporal)

    # Time integration scheme
    pde = Eq(u, eq_rhs)
    op = Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    return max(abs(u_multi_stage.data[0, :] - u.data[0, :]))
