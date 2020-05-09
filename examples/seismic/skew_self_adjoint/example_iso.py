import numpy as np
<<<<<<< HEAD
from devito import info
from examples.seismic import RickerSource
from examples.seismic.skew_self_adjoint import (default_setup_iso,
                                                SsaIsoAcousticWaveSolver)

shape = (221, 211, 201)
=======
from devito import configuration
from examples.seismic import RickerSource
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

nx, ny, nz = 889, 889, 379
shape = (nx, ny, nz)
>>>>>>> 1da4f6b... completed notebook tutorials for SSA isotropic
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 500.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

<<<<<<< HEAD
b, v, time_axis, src_coords, rec_coords = default_setup_iso(npad, shape, dtype, tmax=tmax)

solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                  src_coords, rec_coords, time_axis, space_order=8)
=======
b, v, time_axis, src_coords, rec_coords = defaultSetupIso(npad, shape, dtype, tmax=tmax)

solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src_coords, rec_coords, time_axis, space_order=8)
>>>>>>> 1da4f6b... completed notebook tutorials for SSA isotropic

src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]

ns = src_coords.shape[0]
nr = rec_coords.shape[0]
<<<<<<< HEAD
info(time_axis)
info("ns, nr;         ", ns, nr)
info("grid.shape;     ", v.grid.shape)
info("b.shape;        ", b.grid.shape)
info("v.shape;        ", v.grid.shape)
info("grid.origin;    ", (v.grid.origin[0].data, v.grid.origin[1].data))
info("grid.spacing;   ", v.grid.spacing)

=======
print(time_axis)
print("ns, nr;         ", ns, nr)
print("grid.shape;     ", v.grid.shape)
print("b.shape;        ", b.grid.shape)
print("v.shape;        ", v.grid.shape)
print("grid.origin;    ", (v.grid.origin[0].data, v.grid.origin[1].data))
print("grid.spacing;   ", v.grid.spacing)

tol = 1.e-12
a = np.random.rand()
>>>>>>> 1da4f6b... completed notebook tutorials for SSA isotropic
rec, _, _ = solver.forward(src)
