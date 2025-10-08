import pytest
import numpy as np
import sympy as sym
import tempfile
import pickle
import os

from devito import (Grid, Function, TimeFunction,
                    Derivative, Operator, solve, Eq, configuration)
from devito.types.multistage import multistage_method, MultiStage
from devito.ir.support import SymbolRegistry
from devito.ir.equations import lower_multistage

configuration['log-level'] = 'DEBUG'


def grid_parameters(extent=(10, 10), shape=(3, 3)):
    grid = Grid(origin=(0, 0), extent=extent, shape=shape, dtype=np.float64)
    x, y = grid.dimensions
    dt = grid.stepping_dim.spacing
    t = grid.time_dim
    dx = extent[0] / (shape[0] - 1)
    return grid, x, y, dt, t, dx


def time_parameters(tn, dx, scale=1, t0=0):
    t0, tn = 0.0, tn
    dt0 = scale / dx**2
    nt = int((tn - t0) / dt0)
    dt0 = tn / nt
    return tn, dt0, nt


class Test_API:

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_pickles(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(extent=(1, 1), shape=(3, 3))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                          Derivative(u[0], (x, 2), fd_order=2)
                          + Derivative(u[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Class of the time integration scheme
        method = multistage_method(u, system_eqs_rhs, time_int)

        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            pickle.dump(method, tmpfile)
            filename = tmpfile.name

        with open(filename, 'rb') as file:
            method_saved = pickle.load(file)
            os.remove(filename)

        assert str(method) == str(
            method_saved), "Mismatch in PDE after pickling"

        op_orig = Operator(method)
        op_saved = Operator(method_saved)

        assert str(op_orig) == str(op_saved)

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_solve(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(extent=(1, 1), shape=(3, 3))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                          Derivative(u[0], (x, 2), fd_order=2)
                          + Derivative(u[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Time integration scheme
        pdes = [solve(system_eqs_rhs[i] - u[i], u[i], method=time_int)
                for i in range(2)]

        assert all(isinstance(i, MultiStage)
                   for i in pdes), "Not all elements are instances of MultiStage"


class Test_lowering:

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_object(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(extent=(1, 1), shape=(3, 3))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                          Derivative(u[0], (x, 2), fd_order=2)
                          + Derivative(u[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Class of the time integration scheme
        pdes = multistage_method(u, system_eqs_rhs, time_int)

        assert isinstance(
            pdes, MultiStage), "Not all elements are instances of MultiStage"

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_lower_multistage(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(extent=(1, 1), shape=(3, 3))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system (2D acoustic)
        system_eqs_rhs = [u[1] + src_spatial * src_temporal,
                          Derivative(u[0], (x, 2), fd_order=2)
                          + Derivative(u[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Class of the time integration scheme
        pdes = multistage_method(u, system_eqs_rhs, time_int)

        # Test the lowering process
        sregistry = SymbolRegistry()

        # Lower the multistage method - this should not raise an exception
        lowered_eqs = lower_multistage(pdes, sregistry=sregistry)

        # Validate the lowered equations
        assert lowered_eqs is not None, "Lowering returned None"
        assert len(lowered_eqs) > 0, "Lowering returned empty list"


class Test_RK:

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_single_equation_integration(self, time_int):
        """
        Test single equation time integration with MultiStage methods.

        This test verifies that time integration works correctly for the simplest case:
        a single PDE with a single unknown function. This represents the most basic
        MultiStage usage scenario (e.g., heat equation, simple wave equation).
        """

        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1, 1), shape=(200, 200))

        # Define single unknown function
        u_multi_stage = TimeFunction(name='u_multi_stage', grid=grid, space_order=2,
                                     time_order=1, dtype=np.float64)

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # Single PDE: du/dt = ∇²u + source (diffusion/wave equation)
        eq_rhs = (Derivative(u_multi_stage, (x, 2), fd_order=2)
                  + Derivative(u_multi_stage, (y, 2), fd_order=2)
                  + src_spatial * src_temporal)

        # Store initial data for comparison
        initial_data = u_multi_stage.data.copy()

        # Time integration scheme - single equation MultiStage object
        pde = multistage_method(u_multi_stage, eq_rhs, time_int)

        # Run the operator
        op = Operator([pde], subs=grid.spacing_map)  # Operator expects a list
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        assert not np.array_equal(
            u_multi_stage.data, initial_data), "Data should have changed"

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_decoupled_equations(self, time_int):
        """
        Test decoupled time integration where each equation gets its own MultiStage object.

        This test verifies that time integration works when creating separate MultiStage
        objects for each equation, as opposed to coupled integration where all equations
        are handled by a single MultiStage object.
        """
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1, 1), shape=(200, 200))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u_multi_stage', 'v_multi_stage']
        u_multi_stage = [TimeFunction(name=name, grid=grid, space_order=2, time_order=1, dtype=np.float64)
                          for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system - each equation independent for decoupled integration
        system_eqs_rhs = [u_multi_stage[1] + src_spatial * src_temporal,
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]

        # Time integration scheme - create separate MultiStage objects (decoupled)
        pdes = [multistage_method(u_multi_stage[i], system_eqs_rhs[i], time_int)
                for i in range(len(fun_labels))]

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_coupled_op_computing(self, time_int):
        """
        Test coupled time integration where all equations are handled by a single MultiStage object.

        This test verifies that time integration works correctly when multiple coupled equations
        are integrated together within a single MultiStage object, allowing for proper coupling
        between the equations during the time stepping process.
        """
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1, 1), shape=(200, 200))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u_multi_stage', 'v_multi_stage']
        u_multi_stage = [
            TimeFunction(
                name=name,
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[1, 1] = 1
        src_temporal = (1 - 2 * (t * dt - 1)**2)

        # PDE system - coupled acoustic wave equations
        system_eqs_rhs = [u_multi_stage[1],  # velocity equation: du/dt = v
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                          + src_spatial * src_temporal]  # displacement equation: dv/dt = ∇²u + source

        # Store initial data for comparison
        initial_data = [u.data.copy() for u in u_multi_stage]

        # Time integration scheme - single coupled MultiStage object
        pdes = multistage_method(u_multi_stage, system_eqs_rhs, time_int)

        # Run the operator
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.01, time=1)

        # Verify that computation actually occurred (data changed)
        for i, u in enumerate(u_multi_stage):
            assert not np.array_equal(
                u.data, initial_data[i]), f"Data should have changed for variable {i}"

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_low_order_convergence_ODE(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(extent=(10, 10), shape=(3, 3))

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[:] = 1
        src_temporal = 2 * t * dt

        # Time axis
        tn, dt0, nt = time_parameters(3.0, dx, scale=1e-2)

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u_multi_stage = [
            TimeFunction(
                name=name
                + '_multi_stage',
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # PDE (2D acoustic)
        eq_rhs = [
            (-1.5 * u_multi_stage[0] + 0.5 * u_multi_stage[1]) * src_spatial * src_temporal,
            (-1.5 * u_multi_stage[1] + 0.5 * u_multi_stage[0]) * src_spatial * src_temporal]
        u_multi_stage[0].data[0, :] = 1

        # Time integration scheme
        pdes = multistage_method(u_multi_stage, eq_rhs, time_int)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # exact solution
        d = np.array([-1, -2])
        a = np.array([[1, 1], [1, -1]])
        exact_sol = np.dot(
            np.dot(a, np.diag(np.exp(d * tn**2))), np.linalg.inv(a))
        assert np.max(np.abs(exact_sol[0, 0] - u_multi_stage[0].data[0, :])
                      ) < 10 ** -5, "the method is not converging to the solution"

    @pytest.mark.parametrize('time_int', ['RK44', 'RK32', 'RK97'])
    def test_low_order_convergence(self, time_int):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1000, 1000), shape=(201, 201))

        # Medium velocity model
        vel = Function(name=f"vel_{time_int}",
                       grid=grid, space_order=2, dtype=np.float64)
        vel.data[:] = 1.0
        vel.data[150:, :] = 1.3

        # Source definition
        src_spatial = Function(
            name=f"src_spat_{time_int}", grid=grid, space_order=2, dtype=np.float64)
        src_spatial.data[100, 100] = 1 / dx**2
        f0 = 0.01
        src_temporal = (1 - 2 * (np.pi * f0 * (t * dt - 1 / f0))**2) * \
            sym.exp(-(np.pi * f0 * (t * dt - 1 / f0))**2)

        # Time axis
        tn, dt0, nt = time_parameters(500.0, dx, scale=1e-1 * np.max(vel.data))

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u_multi_stage = [
            TimeFunction(
                name=f"{name}_multi_stage_{time_int}",
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # PDE (2D acoustic)
        eq_rhs = [u_multi_stage[1], (Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                                     + Derivative(u_multi_stage[0], (y, 2), fd_order=2)
                                     + src_spatial * src_temporal) * vel**2]

        # Time integration scheme
        pdes = multistage_method(u_multi_stage, eq_rhs, time_int)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # Devito's default solution
        u = [TimeFunction(name=f"{name}_{time_int}", grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]

        # PDE (2D acoustic)
        eq_rhs = [u[1], (Derivative(u[0], (x, 2), fd_order=2)
                         + Derivative(u[0], (y, 2), fd_order=2)
                         + src_spatial
                         * src_temporal)
                  * vel**2]

        # Time integration scheme
        pdes = [Eq(u[i].forward, solve(Eq(u[i].dt - eq_rhs[i]), u[i].forward))
                for i in range(len(fun_labels))]
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        assert (np.linalg.norm(u[0].data[0, :] - u_multi_stage[0].data[0, :]) / np.linalg.norm(
            u[0].data[0, :])) < 10**-1, "the method is not converging to the solution"


class Test_HORK:

    def test_trivial_coupled_op_computing_exp(self, time_int='HORK_EXP'):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1, 1), shape=(201, 201))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u_multi_stage', 'v_multi_stage']
        u_multi_stage = [
            TimeFunction(
                name=name,
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # PDE system
        system_eqs_rhs = [u_multi_stage[1],
                          u_multi_stage[0] + 1e-2 * u_multi_stage[1]]

        # Time integration scheme
        pdes = multistage_method(
            u_multi_stage, system_eqs_rhs, time_int, degree=4)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.001, time=2000)

    def test_coupled_op_computing_exp(self, time_int='HORK_EXP'):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
            extent=(1, 1), shape=(201, 201))

        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u_multi_stage', 'v_multi_stage']
        u_multi_stage = [
            TimeFunction(
                name=name,
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # Source definition
        src_spatial = Function(name="src_spat", grid=grid,
                               space_order=2, dtype=np.float64)
        src_spatial.data[100, 100] = 1
        src_temporal = sym.exp(- 100 * (t - 0.01) ** 2)
        # import matplotlib.pyplot as plt
        # import numpy as np
        # t=np.linspace(0,2000,1000)
        # plt.plot(t,np.exp(1 - 2 * (t - 1)**2))

        # PDE system
        system_eqs_rhs = [u_multi_stage[1],
                          Derivative(u_multi_stage[0], (x, 2), fd_order=2)
                          + Derivative(u_multi_stage[0], (y, 2), fd_order=2)]

        src = [[src_spatial, src_temporal, u_multi_stage[0]],
               [src_spatial, src_temporal * 10, u_multi_stage[0]],
               [src_spatial, src_temporal, u_multi_stage[1]]]

        # Time integration scheme
        pdes = multistage_method(
            u_multi_stage, system_eqs_rhs, time_int, degree=4, source=src)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=0.001, time=2000)

        # plt.imshow(u_multi_stage[0].data[0,:])
        # plt.colorbar()
        # plt.show()

    @pytest.mark.parametrize('degree', list(range(3, 11)))
    def test_HORK_EXP_convergence(self, degree):
        # Grid setup
        grid, x, y, dt, t, dx = grid_parameters(
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
        src_temporal = (1 - 2 * (np.pi * f0 * (t * dt - 1 / f0))**2) * \
            sym.exp(-(np.pi * f0 * (t * dt - 1 / f0))**2)

        # Time axis
        tn, dt0, nt = time_parameters(500.0, dx, scale=np.max(vel.data))

        # Time integrator solution
        # Define wavefield unknowns: u (displacement) and v (velocity)
        fun_labels = ['u', 'v']
        u_multi_stage = [
            TimeFunction(
                name=name
                + '_multi_stage',
                grid=grid,
                space_order=2,
                time_order=1,
                dtype=np.float64) for name in fun_labels]

        # PDE (2D acoustic)
        eq_rhs = [
            u_multi_stage[1],
            (Derivative(
                u_multi_stage[0],
                (x,
                 2),
                fd_order=2)
                + Derivative(
                u_multi_stage[0],
                (y,
                 2),
                fd_order=2))
            * vel**2]

        src = [[src_spatial * vel**2, src_temporal, u_multi_stage[1]]]

        # Time integration scheme
        pdes = multistage_method(
            u_multi_stage, eq_rhs, 'HORK_EXP', source=src, degree=degree)
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)

        # Devito's default solution
        u = [TimeFunction(name=name, grid=grid, space_order=2,
                          time_order=1, dtype=np.float64) for name in fun_labels]
        # PDE (2D acoustic)
        eq_rhs = [u[1], (Derivative(u[0], (x, 2), fd_order=2)
                         + Derivative(u[0], (y, 2), fd_order=2)
                         + src_spatial
                         * src_temporal)
                  * vel**2]

        # Time integration scheme
        pdes = [Eq(u[i].forward, solve(Eq(u[i].dt - eq_rhs[i]), u[i].forward))
                for i in range(len(fun_labels))]
        op = Operator(pdes, subs=grid.spacing_map)
        op(dt=dt0, time=nt)
        assert (np.linalg.norm(u[0].data[0, :] - u_multi_stage[0].data[0, :]) / np.linalg.norm(
            u[0].data[0, :])) < 10**-5, "the method is not converging to the solution"
