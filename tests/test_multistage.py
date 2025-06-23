from numpy import pi, float64, max, abs
from sympy import exp

from devito import (Grid, Function, TimeFunction,
                    Derivative, Operator, solve, Eq)
from devito.types.multistage import resolve_method


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
    f0 = 0.01
    src_temporal = (1 - 2 * (pi * f0 * (t * dt - 1 / f0)) ** 2) * exp(-(pi * f0 * (t * dt - 1 / f0)) ** 2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                  Derivative(U[0], (x, 2), fd_order=2) +
                   Derivative(U[0], (y, 2), fd_order=2) +
                   src_spatial * src_temporal]

    # Time integration scheme
    return [solve(system_eqs_rhs[i] - U[i], U[i], method=time_int, eq_num=i) for i in range(2)]


def test_multistage_op_constructing_directly(time_int='RK44'):
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
    f0 = 0.01
    src_temporal = (1 - 2 * (pi * f0 * (t * dt - 1 / f0)) ** 2) * exp(-(pi * f0 * (t * dt - 1 / f0)) ** 2)

    # PDE system (2D acoustic)
    system_eqs_rhs = [U[1] + src_spatial * src_temporal,
                      Derivative(U[0], (x, 2), fd_order=2) +
                      Derivative(U[0], (y, 2), fd_order=2) +
                      src_spatial * src_temporal]

    # Time integration scheme

    pdes = [resolve_method(time_int)(U[i], system_eqs_rhs[i]) for i in range(2)]
    op = Operator(pdes, subs=grid.spacing_map)
    op(dt=0.01, time=1)


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
    f0 = 0.01
    src_temporal = (1 - 2 * (pi * f0 * (t * dt - 1 / f0)) ** 2) * exp(-(pi * f0 * (t * dt - 1 / f0)) ** 2)

    # PDE (2D heat eq.)
    eq_rhs = (Derivative(u_multi_stage, (x, 2), fd_order=2) + Derivative(u_multi_stage, (y, 2), fd_order=2) +
              src_spatial * src_temporal)

    # Time integration scheme
    pde = [MultiStage(eq_rhs, u_multi_stage, method=time_int)]
    op = Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    # Solving now using Devito's standard time solver
    u = TimeFunction(name='u', grid=grid, space_order=2, time_order=1, dtype=float64)
    eq_rhs = (Derivative(u, (x, 2), fd_order=2) + Derivative(u, (y, 2), fd_order=2) +
              src_spatial * src_temporal)

    # Time integration scheme
    pde = Eq(u, solve(eq_rhs - u, u))
    op = Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    return max(abs(u_multi_stage.data[0, :] - u.data[0, :]))

# test_multistage_op_constructing_directly()

# test_multistage_op_computing_directly()

def test_multistage_op_solve_computing(time_int='RK44'):
    extent = (1, 1)
    shape = (200, 200)
    origin = (0, 0)

    # Grid setup
    grid = Grid(origin=origin, extent=extent, shape=shape, dtype=float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim

    # Define unknown for the 'time_int' method: u (heat)
    u_time_int = TimeFunction(name='u', grid=grid, space_order=2, time_order=1, dtype=float64)

    # Source definition
    src_spatial = Function(name="src_spat", grid=grid, space_order=2, dtype=float64)
    src_spatial.data[1, 1] = 1
    f0 = 0.01
    src_temporal = (1 - 2 * (pi * f0 * (t * dt - 1 / f0)) ** 2) * exp(-(pi * f0 * (t * dt - 1 / f0)) ** 2)

    # PDE (2D heat eq.)
    eq_rhs = (Derivative(u_time_int, (x, 2), fd_order=2) + Derivative(u_time_int, (y, 2), fd_order=2) +
               src_spatial * src_temporal)

    # Time integration scheme
    pde = solve(eq_rhs - u_time_int, u_time_int, method=time_int)
    op=Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    # Solving now using Devito's standard time solver
    u = TimeFunction(name='u', grid=grid, space_order=2, time_order=1, dtype=float64)
    eq_rhs = (Derivative(u, (x, 2), fd_order=2) + Derivative(u, (y, 2), fd_order=2) +
              src_spatial * src_temporal)

    # Time integration scheme
    pde = Eq(u, solve(eq_rhs - u, u))
    op = Operator(pde, subs=grid.spacing_map)
    op(dt=0.01, time=1)

    return max(abs(u_time_int.data[0,:]-u.data[0,:]))

# test_multistage_op_solve_computing()