from devito import *
import os
import numpy as np
from examples.seismic.source import DGaussSource, TimeAxis
from devito.petsc import PETScSolve
from devito.petsc.initialize import PetscInitialize
configuration['compiler'] = 'custom'
os.environ['CC'] = 'mpicc'


# PETSc implementation of devito/examples/seismic/tutorials/05_staggered_acoustic.ipynb
# Test staggered grid implementation with PETSc

PetscInitialize()

extent = (2000., 2000.)
shape = (81, 81)

x = SpaceDimension(
    name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1), dtype=np.float64)
)
z = SpaceDimension(
    name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1), dtype=np.float64)
)

grid = Grid(extent=extent, shape=shape, dimensions=(x, z), dtype=np.float64)

# Timestep size
t0, tn = 0., 200.
dt = 1e2*(1. / np.sqrt(2.)) / 60.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = DGaussSource(name='src', grid=grid, f0=0.01, time_range=time_range, a=0.004)
src.coordinates.data[:] = [1000., 1000.]

# Now we create the velocity and pressure fields
# NOTE/TODO: PETSc does not yet fully support VectorTimeFunctions. Ideally,
# it should use the new "coupled" machinery
p2 = TimeFunction(name='p2', grid=grid, staggered=NODE, space_order=2, time_order=1)
vx2 = TimeFunction(name='vx2', grid=grid, staggered=(x,), space_order=2, time_order=1)
vz2 = TimeFunction(name='vz2', grid=grid, staggered=(z,), space_order=2, time_order=1)

t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 4.0
density = 1.

ro = 1/density
l2m = V_p*V_p*density

# The source injection term
src_p_2 = src.inject(field=p2.forward, expr=src)

# 2nd order acoustic according to fdelmoc
v_x_2 = Eq(vx2.dt, ro * p2.dx)
v_z_2 = Eq(vz2.dt, ro * p2.dz)

petsc_v_x_2 = PETScSolve(v_x_2, target=vx2.forward)
petsc_v_z_2 = PETScSolve(v_z_2, target=vz2.forward)

p_2 = Eq(p2.dt, l2m * (vx2.forward.dx + vz2.forward.dz))

petsc_p_2 = PETScSolve(p_2, target=p2.forward, solver_parameters={'ksp_rtol': 1e-7})

with switchconfig(language='petsc'):
    op_2 = Operator(petsc_v_x_2 + petsc_v_z_2 + petsc_p_2 + src_p_2, opt='noop')
    op_2(time=src.time_range.num-1, dt=dt)

norm_p2 = norm(p2)
assert np.isclose(norm_p2, .35098, atol=1e-4, rtol=0)


# 4th order acoustic according to fdelmoc
p4 = TimeFunction(name='p4', grid=grid, staggered=NODE, space_order=4)
vx4 = TimeFunction(name='vx4', grid=grid, staggered=(x,), space_order=4, time_order=1)
vz4 = TimeFunction(name='vz4', grid=grid, staggered=(z,), space_order=4, time_order=1)

src_p_4 = src.inject(field=p4.forward, expr=src)

v_x_4 = Eq(vx4.dt, ro * p4.dx)
v_z_4 = Eq(vz4.dt, ro * p4.dz)

petsc_v_x_4 = PETScSolve(v_x_4, target=vx4.forward)
petsc_v_z_4 = PETScSolve(v_z_4, target=vz4.forward)

p_4 = Eq(p4.dt, l2m * (vx4.forward.dx + vz4.forward.dz))

petsc_p_4 = PETScSolve(p_4, target=p4.forward, solver_parameters={'ksp_rtol': 1e-7})

with switchconfig(language='petsc'):
    op_4 = Operator(petsc_v_x_4 + petsc_v_z_4 + petsc_p_4 + src_p_4, opt='noop')
    op_4(time=src.time_range.num-1, dt=dt)

norm_p4 = norm(p4)
assert np.isclose(norm_p4, .33736, atol=1e-4, rtol=0)
