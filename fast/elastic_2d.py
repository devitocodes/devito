#!/usr/bin/env python
# coding: utf-8

# # Elastic wave equation implementation on a staggered grid
# This is a first attempt at implemenenting the elastic wave equation as described in:
#
# [1] Jean Virieux (1986). ”P-SV wave propagation in heterogeneous media:
# Velocity‐stress finite‐difference method.” GEOPHYSICS, 51(4), 889-901.
# https://doi.org/10.1190/1.1442147
#
# The current version actually attempts to mirror the FDELMODC implementation
# by Jan Thorbecke:
# [2] https://janth.home.xs4all.nl/Software/fdelmodcManual.pdf
#
# ## Explosive source
#
# We will first attempt to replicate the explosive source test case described in [1],
# Figure 4. We start by defining the source signature $g(t)$, the derivative of a
# Gaussian pulse, given by Eq 4:
#
# $$g(t) = -2 \alpha(t - t_0)e^{-\alpha(t-t_0)^2}$$

from devito import *
import argparse
from examples.seismic.source import WaveletSource, RickerSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

from sympy import init_printing
init_printing(use_latex='mathjax')


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(110, 110), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot3D")

parser.add_argument("-xdsl", "--xdsl", default=False, action='store_true')
args = parser.parse_args()

# Some variable declarations
nx, ny = args.shape
nt = args.nt
so = args.space_order
to = 1


# Initial grid: 1km x 1km, with spacing 100m
extent = (1500., 1500.)
shape = (nx, ny)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z))


class DGaussSource(WaveletSource):

    def wavelet(self, f0, t):
        a = 0.004
        return -2.*a*(t - 1/f0) * np.exp(-a * (t - 1/f0)**2)


# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., nt
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
src.coordinates.data[:] = [750., 750.]

# Plor source
# src.show()

# Now we create the velocity and pressure fields
v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

# Now let's try and create the staggered updates
t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 2.0
V_s = 1.0
density = 1.8

# The source injection term
src_xx = src.inject(field=tau.forward[0, 0], expr=src)
src_zz = src.inject(field=tau.forward[1, 1], expr=src)

# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1/density

mu = cs2*density
l = (cp2*density - 2*mu)

# First order elastic wave equation
pde_v = v.dt - ro * div(tau)
pde_tau = (tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) +
           grad(v.forward).transpose(inner=False)))

# Time update
u_v = Eq(v.forward, solve(pde_v, v.forward))
u_t = Eq(tau.forward, solve(pde_tau, tau.forward))

# This contains if conditions!!!
op = Operator([u_v] + [u_t] + src_xx + src_zz)
op(dt=dt)

# Up to here, let's only use Devito
# assert np.isclose(norm(v[0]), 0.6285093, atol=1e-4, rtol=0)

# This should NOT have conditions, we should use XDSL!
op = Operator([u_v] + [u_t])

op(dt=dt, time_M=100)


plot_image(v[0].data[0], vmin=-.5*1e-1, vmax=.5*1e-1, cmap="seismic")
plot_image(v[1].data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(tau[0, 0].data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(tau[1, 1].data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
plot_image(tau[0, 1].data[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")

print(norm(v[0]))

# assert np.isclose(norm(v[0]), 0.6285093, atol=1e-4, rtol=0)
