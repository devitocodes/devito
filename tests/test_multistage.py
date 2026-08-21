"""Tests for multistage time-stepping lowering, compilation, and accuracy."""

import pytest
import numpy as np
import sympy as sym

from devito import (Grid, Function, TimeFunction,
                    Derivative, Operator, Eq, configuration)
from devito.operations.solve import solve
import devito.timestepping.multistage as mstage
from devito.ir.support import SymbolRegistry
from devito.ir.equations import lower_timestepping


RK_LOW_ORDER = [mstage.RungeKutta32, mstage.RungeKutta44, mstage.RungeKutta97]

configuration['log-level'] = 'DEBUG'


def grid_parameters(extent=(10, 10), shape=(3, 3)):
    grid = Grid(extent=extent, shape=shape, dtype=np.float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim
    dx = extent[0] / (shape[0] - 1)
    # Return a more logical ordering: spatial dimensions and spacing first,
    # then temporal dimension and spacing.
    return grid, x, y, dx, t, dt


def time_parameters(tn, dx, scale=1, t0=0):
    dt0 = scale * dx**2
    nt = int((tn - t0) / dt0)
    dt0 = tn / nt

    return tn, dt0, nt


def _initial_condition_1d(x, x_0=51.0, delta=1.0):
    """Return the compactly supported 1D wave packet used in accuracy tests."""
    temp = np.zeros_like(x, dtype=np.float64)
    support = np.abs(x - x_0) < delta
    temp[support] = np.exp(-1 / ((x_0 - x[support])**2 - delta**2)**2)
    return temp


def _expected_1d_solution(x_num, vel):
    """Return the reference 1D solution used by the wave-equation tests."""
    return 0.5 * (_initial_condition_1d(x_num - 100 * vel)
                  + _initial_condition_1d(x_num + 100 * vel))


def _build_wave_setup(extent=(1, 1), shape=(200, 200), names=('u', 'v')):
    grid, x, y, dx, t, dt = grid_parameters(extent=extent, shape=shape)

    wavefields = [TimeFunction(name=name, grid=grid, space_order=2,
                               time_order=1, dtype=np.float64)
                  for name in names]

    src_spatial = Function(name='src_spat', grid=grid,
                           space_order=2, dtype=np.float64)
    src_spatial.data[1, 1] = 1
    src_temporal = (1 - 2 * (t * dt - 1)**2)

    return grid, x, y, dx, t, dt, wavefields, src_spatial, src_temporal


class TestAPI:

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_solve_low_order(self, time_int):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)
        
        # PDE system (2D acoustic)
        system_eqs_rhs = 2*u + 1.0

        pdes = [solve(system_eqs_rhs - u, u, method=time_int)]
        assert all(isinstance(i, mstage.MultiStage)
                   for i in pdes), "Not all elements are instances of MultiStage"

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_solve_high_order(self, degree, time_int=mstage.HORKE):
        grid = Grid(shape=1, dtype=np.float64)
               
        u = TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)
        
        # PDE system (2D acoustic)
        system_eqs_rhs = 2*u + 1.0

        # Time integration scheme
        pdes = [solve(system_eqs_rhs - u, u, method=time_int, degree=degree)]
        assert all(isinstance(i, mstage.MultiStage)
                    for i in pdes), "Not all elements are instances of MultiStage"


class TestLoweringLowOrder:

    # Low-order methods checks
    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_object_low_order_simple_eq(self, time_int):
        grid = Grid(shape=1, dtype=np.float64)

        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs)

        assert isinstance(
            pdes, mstage.MultiStage), "Not all elements are instances of MultiStage"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_object_low_order_pde_eq(self, time_int):
        grid, x, y, dx, t, dt, u, src_spatial, src_temporal = _build_wave_setup(
            shape=(3, 3), names=('u', 'v'))

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                            Derivative(u[0], (x, 2), fd_order=2)
                            + Derivative(u[0], (y, 2), fd_order=2)
                            + src_spatial * src_temporal]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs)

        assert isinstance(
            pdes, mstage.MultiStage), "Not all elements are instances of MultiStage"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_lower_multistage_low_order_simple_eq(self, time_int):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs)

        # Test the lowering process
        sregistry = SymbolRegistry()

        # Lower the multistage method - this should not raise an exception
        lowered_eqs = lower_timestepping(pdes, sregistry=sregistry)

        # Validate the lowered equations
        assert lowered_eqs is not None, "Lowering returned None"
        assert len(lowered_eqs) > 0, "Lowering returned empty list"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_lower_multistage_low_order_pde_eq(self, time_int):
        grid, x, y, dx, t, dt, u, src_spatial, src_temporal = _build_wave_setup(
            shape=(3, 3), names=('u', 'v'))

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                            Derivative(u[0], (x, 2), fd_order=2)
                            + Derivative(u[0], (y, 2), fd_order=2)
                            + src_spatial * src_temporal]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs)

        # Test the lowering process
        sregistry = SymbolRegistry()

        # Lower the multistage method - this should not raise an exception
        lowered_eqs = lower_timestepping(pdes, sregistry=sregistry)

        # Validate the lowered equations
        assert lowered_eqs is not None, "Lowering returned None"
        assert len(lowered_eqs) > 0, "Lowering returned empty list"


class TestLoweringHighOrder:

    # High-order methods checks
    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_object_high_order_simple_eq(self, degree, time_int=mstage.HORKE):
        grid = Grid(shape=1, dtype=np.float64)

        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs, degree=degree)

        assert isinstance(
            pdes, mstage.MultiStage), "Not all elements are instances of MultiStage"
    
    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_object_high_order_pde_eq(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u, src_spatial, src_temporal = _build_wave_setup(
            shape=(3, 3), names=('u', 'v'))

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                            Derivative(u[0], (x, 2), fd_order=2)
                            + Derivative(u[0], (y, 2), fd_order=2)]

        src = [[src_spatial, src_temporal, u[1]]]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs, source=src, degree=degree)

        assert isinstance(
            pdes, mstage.MultiStage), "Not all elements are instances of MultiStage"

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_lower_multistage_high_order_simple_eq(self, degree, time_int=mstage.HORKE):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs, degree=degree)

        # Test the lowering process
        sregistry = SymbolRegistry()

        # Lower the multistage method - this should not raise an exception
        lowered_eqs = lower_timestepping(pdes, sregistry=sregistry)

        # Validate the lowered equations
        assert lowered_eqs is not None, "Lowering returned None"
        assert len(lowered_eqs) > 0, "Lowering returned empty list"
               
    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_lower_multistage_high_order_pde_eq(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u, src_spatial, src_temporal = _build_wave_setup(
            shape=(3, 3), names=('u', 'v'))

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                            Derivative(u[0], (x, 2), fd_order=2)
                            + Derivative(u[0], (y, 2), fd_order=2)]

        src = [[src_spatial, src_temporal, u[1]]]

        # Class of the time integration scheme
        pdes = time_int(u, system_eqs_rhs, source=src, degree=degree)

        # Test the lowering process
        sregistry = SymbolRegistry()

        # Lower the multistage method - this should not raise an exception
        lowered_eqs = lower_timestepping(pdes, sregistry=sregistry)

        # Validate the lowered equations
        assert lowered_eqs is not None, "Lowering returned None"
        assert len(lowered_eqs) > 0, "Lowering returned empty list"


class TestCompilerLowOrder:

    # Low-order methods checks
    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_builds_low_order_simple_eq(self, time_int):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pde = time_int(u, system_eqs_rhs)

        op = Operator(pde, subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_builds_low_order_pde_eq(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage',))
        u = u_multi_stage[0]

        eq_rhs = (Derivative(u, (x, 2), fd_order=2)
                  + Derivative(u, (y, 2), fd_order=2)
                  + src_spatial * src_temporal)

        pde = time_int(u, eq_rhs)

        op = Operator([pde], subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_builds_low_order_pde_2eq_decoupled(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1] + src_spatial * src_temporal,
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        pdes = [time_int(u_multi_stage[i], system_eqs_rhs[i])
                for i in range(2)]

        op = Operator(pdes, subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_builds_low_order_pde_2eq_coupled(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1],
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        pdes = time_int(u_multi_stage, system_eqs_rhs)

        op = Operator(pdes, subs=grid.spacing_map)
        assert op.cfunction is not None


class TestCompilerHighOrder:
    # High-order methods checks
    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_builds_high_order_simple_eq(self, degree, time_int=mstage.HORKE):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        # Class of the time integration scheme
        pde = time_int(u, system_eqs_rhs, degree=degree)

        op = Operator(pde, subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_builds_high_order_pde_eq(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage',))
        u = u_multi_stage[0]

        eq_rhs = [(Derivative(u, (x, 2), fd_order=2)
                    + Derivative(u, (y, 2), fd_order=2))]

        src = [[src_spatial, src_temporal, u]]

        pde = time_int(u, eq_rhs, source=src, degree=degree)

        op = Operator([pde], subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_builds_high_order_pde_2eq_decoupled(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1] + src_spatial * src_temporal,
                            Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                            + Derivative(u_multi_stage[0], (y, 2), fd_order=2)]

        pdes = [time_int(u_multi_stage[i], system_eqs_rhs[i], degree=degree)
                for i in range(2)]

        op = Operator(pdes, subs=grid.spacing_map)
        assert op.cfunction is not None

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_builds_high_order_pde_2eq_coupled(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1],
                            Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                            + Derivative(u_multi_stage[0], (y, 2), fd_order=2)]

        src = [[src_spatial, src_temporal, u_multi_stage[1]]]

        pdes = time_int(u_multi_stage, system_eqs_rhs, source=src, degree=degree)

        op = Operator(pdes, subs=grid.spacing_map)
        assert op.cfunction is not None


class TestOperatorApplicationLowOrder:

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_low_order_simple_eq(self, time_int):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        initial_data = u[0].data.copy()

        # Time integration scheme - single equation MultiStage object
        pde = time_int(u, system_eqs_rhs)

        # Run the operator
        op = Operator([pde], subs=grid.spacing_map)  # Operator expects a list
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        assert not np.array_equal(
            u[0].data, initial_data), "Data should have changed"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_operator_low_order_pde_eq(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage',))

        eq_rhs = (Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                  + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                  + src_spatial * src_temporal)

        # Store initial data for comparison
        initial_data = u_multi_stage[0].data.copy()

        # Time integration scheme - single equation MultiStage object
        pde = time_int(u_multi_stage, eq_rhs)

        # Run the operator
        op = Operator([pde], subs=grid.spacing_map)  # Operator expects a list
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        assert not np.array_equal(
            u_multi_stage[0].data, initial_data), "Data should have changed"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_decoupled_low_order_equations(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1] + src_spatial * src_temporal,
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]

        # Time integration scheme - create separate MultiStage objects (decoupled)
        pdes = [time_int(u_multi_stage[i], system_eqs_rhs[i])
                for i in range(2)]

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_coupled_low_order_op_computing(self, time_int):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1],  # velocity equation: du/dt = v
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]  # displacement equation: dv/dt = ∇²u + source

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]

        # Time integration scheme - single coupled MultiStage object
        pdes = time_int(u_multi_stage, system_eqs_rhs)

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"


class TestOperatorApplicationHighOrder:
    
    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_high_order_simple_eq(self, degree, time_int=mstage.HORKE):
        grid = Grid(shape=1, dtype=np.float64)
        
        u = [TimeFunction(name='u', grid=grid, space_order=1, time_order=1, dtype=np.float64)]
        
        # PDE system (2D acoustic)
        system_eqs_rhs = [2*u[0] + 1.0]

        initial_data = u[0].data.copy()

        # Time integration scheme - single equation MultiStage object
        pde = time_int(u, system_eqs_rhs, degree=degree)

        # Run the operator
        op = Operator([pde], subs=grid.spacing_map)  # Operator expects a list
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        assert not np.array_equal(
            u[0].data, initial_data), "Data should have changed"

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_operator_high_order_pde_eq(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage',))
        u = u_multi_stage[0]
        
        eq_rhs = [(Derivative(u, (x, 2), fd_order=2)
                    + Derivative(u, (y, 2), fd_order=2))]

        src = [[src_spatial, src_temporal, u]]

        # Store initial data for comparison
        initial_data = u.data.copy()

        # Time integration scheme - single equation MultiStage object
        pde = time_int(u, eq_rhs, source=src, degree=degree)

        # Run the operator
        op = Operator([pde], subs=grid.spacing_map)  # Operator expects a list
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        assert not np.array_equal(
            u.data, initial_data), "Data should have changed"

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_decoupled_high_order_equations(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1]+src_spatial * src_temporal,
                            Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                            + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                            + src_spatial * src_temporal]

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]
        
        # Time integration scheme - create separate MultiStage objects (decoupled)
        pdes = [time_int(u_multi_stage[i], system_eqs_rhs[i], degree=degree)
                for i in range(2)]

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_coupled_high_order_op_computing(self, degree, time_int=mstage.HORKE):
        grid, x, y, dx, t, dt, u_multi_stage, src_spatial, src_temporal = _build_wave_setup(
            shape=(200, 200), names=('u_multi_stage', 'v_multi_stage'))

        system_eqs_rhs = [u_multi_stage[1],  # velocity equation: du/dt = v
                            Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                            + Derivative(u_multi_stage[0], (y, 2), fd_order=2)]  # displacement equation: dv/dt = ∇²u + source

        src = [[src_spatial, src_temporal, u_multi_stage[1]]]

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]

        # Time integration scheme - single coupled MultiStage object
        pdes = time_int(u_multi_stage, system_eqs_rhs, source=src, degree=degree)

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"


class TestAccuracyLowOrder:

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_convergence_low_order_ode_2eq(self, time_int):
        # Grid setup
        grid, x, y, dx, t, dt = grid_parameters(extent=(10, 10), shape=(3, 3))

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[:] = 1
        src_temporal = 2 * t * dt

        # Time axis
        tn, dt0, nt = time_parameters(3.0, dx, scale=1e-4)

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        u_multi_stage = [TimeFunction(name=name + '_multi_stage', grid=grid, space_order=2, time_order=1,
                                      dtype=np.float64) for name in ('u', 'v')]

        # PDE (2D acoustic)
        eq_rhs = [
            (-1.5 * u_multi_stage[0] + 0.5 * u_multi_stage[1]) * src_spatial * src_temporal,
            (-1.5 * u_multi_stage[1] + 0.5 * u_multi_stage[0]) * src_spatial * src_temporal]
        u_multi_stage[0].data[0, :] = 1

        # Time integration scheme
        pdes = time_int(u_multi_stage, eq_rhs)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # exact solution
        d = np.array([-1, -2])
        a = np.array([[1, 1], [1, -1]])
        exact_sol = np.dot(
            np.dot(a, np.diag(np.exp(d * tn**2))), np.linalg.inv(a))
        assert np.max(np.abs(exact_sol[0, 0] - u_multi_stage[0].data[0, :])
                      ) < 10 ** -5, "the method is not converging to the solution"

    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_convergence_low_order_wave_eq_1d(self, time_int, vel=1/2):
        grid = Grid(shape=20000, extent=102, dtype=np.float64)
        x = grid.dimensions[0]
        x_num = np.linspace(0, 102, 20000)

        dx = grid.spacing[0]
        tn, dt0, nt = time_parameters(100.0, dx, scale=1e+1)

        u_multi_stage = [TimeFunction(name=f"{name}_multi_stage", grid=grid, space_order=2, time_order=1,
                                      dtype=np.float64) for name in ('u', 'v')]
        u_multi_stage[0].data[0, :] = _initial_condition_1d(x_num)

        # PDE (1D acoustic)
        eq_rhs = [u_multi_stage[1],
                  Derivative(u_multi_stage[0], (x, 2), fd_order=2) * vel**2]

        # Time integration scheme
        pdes = time_int(u_multi_stage, eq_rhs)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        reference = _expected_1d_solution(x_num, vel)

        assert (np.linalg.norm(reference - u_multi_stage[0].data[0, :]) / np.linalg.norm(reference)) < 10**-1, "the method is not converging to the solution"
       
        
    @pytest.mark.parametrize('time_int', RK_LOW_ORDER)
    def test_convergence_low_order_wave_eq(self, time_int):
        # Grid setup
        grid, x, y, dx, t, dt = grid_parameters(
            extent=(1000, 1000), shape=(201, 201))

        # Medium velocity model
        vel = Function(name=f"vel",
                       grid=grid, space_order=2, dtype=np.float64)
        vel.data[:] = 1.0
        vel.data[150:, :] = 1.3

        # Source definition
        src_spatial = Function(
            name=f"src_spat", grid=grid, space_order=2, dtype=np.float64)
        src_spatial.data[100, 100] = 1 / dx**2
        f0 = 0.01
        src_temporal = (1 - 2 * (np.pi * f0 * (t * dt - 1 / f0))**2) * \
            sym.exp(-(np.pi * f0 * (t * dt - 1 / f0))**2)

        # Time axis
        tn, dt0, nt = time_parameters(500.0, dx, scale=np.max(vel.data)*1e-4)

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        u_multi_stage = [
            TimeFunction(name=f"{name}_multi_stage", grid=grid, space_order=2, time_order=1,
                         dtype=np.float64) for name in ('u', 'v')]

        # PDE (2D acoustic)
        eq_rhs = [u_multi_stage[1], (Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                                     + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                                     + src_spatial * src_temporal) * vel**2]

        # Time integration scheme
        pdes = time_int(u_multi_stage, eq_rhs)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # Devito's default solution
        u = [TimeFunction(name=f"{name}", grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in ('u', 'v')]

        # PDE (2D acoustic)
        eq_rhs = [u[1], (Derivative(u[0], (x, 2), fd_order=2) + Derivative(u[0], (y, 2), fd_order=2)
                         + src_spatial * src_temporal) * vel**2]

        # Time integration scheme
        pdes = [Eq(u[i].forward, solve(Eq(u[i].dt - eq_rhs[i]), u[i].forward))
                for i in range(2)]
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)
        assert (np.linalg.norm(u[0].data[0, :] - u_multi_stage[0].data[0, :]) / np.linalg.norm(
            u[0].data[0, :])) < 10**-1, "the method is not converging to the solution"


class TestAccuracyHighOrder:

    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_convergence_high_order_wave_eq_1d(self, degree, time_int=mstage.HORKE, vel = 1/2):
        grid = Grid(shape=20000,extent=102, dtype=np.float64)
        x = grid.dimensions[0]
        x_num = np.linspace(0, 102, 20000)

        dx = grid.spacing[0]
        tn, dt0, nt = time_parameters(100.0, dx, scale=1e+1)

        u_multi_stage = [TimeFunction(name=f"{name}_multi_stage", grid=grid, space_order=2, time_order=1,
                    dtype=np.float64) for name in ('u', 'v')]
        u_multi_stage[0].data[0, :] = _initial_condition_1d(x_num)

        # PDE (1D acoustic)
        eq_rhs = [u_multi_stage[1], 
                    Derivative(u_multi_stage[0], (x, 2), fd_order=2) * vel**2]
        

        # Time integration scheme
        pdes = time_int(u_multi_stage, eq_rhs, degree=degree)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        reference = _expected_1d_solution(x_num, vel)

        assert (np.linalg.norm(reference - u_multi_stage[0].data[0, :]) / np.linalg.norm(reference)) < 10**-1, "the method is not converging to the solution"       


    @pytest.mark.parametrize('degree', list(range(3, 8)))
    def test_convergence_high_order_wave_eq(self, degree, time_int=mstage.HORKE):
        # Grid setup
        grid, x, y, dx, t, dt = grid_parameters(
            extent=(1000, 1000), shape=(201, 201))

        # Medium velocity model
        vel = Function(name="vel", grid=grid, space_order=2, dtype=np.float64)
        vel.data[:] = 1.0
        vel.data[150:, :] = 1.3

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[100, 100] = 1 / dx**2
        f0 = 0.01
        src_temporal = (1 - 2 * (np.pi * f0 * (t - 1 / f0))**2) * sym.exp(-(np.pi * f0 * (t - 1 / f0))**2)

        # Time axis
        tn, dt0, nt = time_parameters(500.0, dx, scale=np.max(vel.data)*1e-4)

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        u_multi_stage = [TimeFunction(name=name + '_multi_stage', grid=grid, space_order=2, time_order=0,
                dtype=np.float64) for name in ('u_sol', 'v_sol')]
        
        # PDE (2D acoustic)
        eq_rhs = [u_multi_stage[1], (Derivative(u_multi_stage[0],(x,2), fd_order=2) + Derivative(
            u_multi_stage[0], (y,2), fd_order=2)) * vel**2]

        src = [[src_spatial * vel**2, src_temporal, u_multi_stage[1]]]

        # Time integration scheme
        pdes = time_int(u_multi_stage, eq_rhs, source=src, degree=degree)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # Devito's default solution
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in ('u_sol', 'v_sol')]
        
        # PDE (2D acoustic)
        src_temporal = (1 - 2 * (np.pi * f0 * (t * dt - 1 / f0))**2) * sym.exp(-(np.pi * f0 * (t * dt - 1 / f0))**2)
        eq_rhs = [u[1], (Derivative(u[0], (x, 2), fd_order=2)
                         + Derivative(u[0], (y, 2), fd_order=2)
                         + src_spatial * src_temporal) * vel**2]

        # Time integration scheme
        pdes = [Eq(u[i].forward, solve(Eq(u[i].dt - eq_rhs[i]), u[i].forward))
                for i in range(2)]
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        assert (np.linalg.norm(u[0].data[0, :] - u_multi_stage[0].data[0, :]) / np.linalg.norm(
            u[0].data[0, :])) < 10**-1, "the method is not converging to the solution"