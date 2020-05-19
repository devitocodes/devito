import numpy as np

from examples.seismic import setup_geometry, Model
from examples.seismic.skew_self_adjoint import (setup_w_over_q,
                                                SsaIsoAcousticWaveSolver)

shape = (221, 211, 201)
spacing = (10., 10., 10.)
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 500.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak
vp = 1.5
b = 1.0
so = 8

init_damp = lambda func, nbl: setup_w_over_q(func, omega, qmin, qmax, nbl, sigma=0)

model = Model(origin=(0., 0., 0.), shape=shape, vp=1.5, b=b,
              spacing=spacing, nbl=npad, space_order=so, bcs=init_damp)

geometry = setup_geometry(model, tmax)
solver = SsaIsoAcousticWaveSolver(model, geometry, space_order=so)

rec, _, _ = solver.forward()
