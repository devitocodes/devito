Solving the 2D Heat Equation using Devito
=========================================

Consider the 2D Heat Equation defined by

.. math::
   u_t = a\left(u_{xx}+u_{yy}\right)

This equation can be solved numerically using Finite Difference approximations.

First of all, we need to allocate the grid and set its initial condition::

   from devito import TimeData
   u = TimeData(name = 'u', shape = 'nx, ny', time_order = 1, space_order = 2)
   u.data[0, :] = ui[:]

TimeData is a Devito data object used to store and manage time-varying data.

We initialise our grid to be of size :obj:`nx * ny` for some :obj:`nx` 
and :obj:`ny`. :obj:`time_order` and :obj:`space_order` represent the discretization
order for time and space respectively. The initial configuration is given as a
:class:`numpy.array` in :obj:`u.data`.

The next step is to generate the stencil to be solved by a
:class:`devito.operator.Operator`::
    
   from devito import Operator
   from sympy import Eq, solve, symbols
   a, h, s = symbols("a h s")
   eqn = Eq(u.dt, a * (u.dx2 + u.dy2))
   stencil = solve(eqn, u.forward)[0]

The stencil is generated according to Devito conventions. It uses a sympy
equation to represent the 2D Heat equation and store it in :obj:`eqn`.
Devito makes easy to represent the equation by providing properties :obj:`dt`,
:obj:`dx2`, and :obj:`dx2` that represent the derivatives.

We then generate the stencil by solving :obj:`eqn` for :obj:`u.forward`, a
symbol for the time-forward state of the function.

We plug the stencil in an Operator, as shown, and define the values of the
thermal conductivity :obj:`a`, the spacing between cells :obj:`h` and the
temporal spacing :obj:`s`.::

   op = Operator(Eq(u.forward, stencil),
                 subs={h: spacing, s: dt, a: tc})


To execute the generated Operator, we simply call :samp:`op.apply(u=u,
t=timesteps)`. The results will then be found in :obj:`u.data[1, :]`.

For a complete example of this code, please see
`examples/diffusion/example_diffusion.py`. A more comprehensive set of
CFD tutorials based on the excellent `12 steps to Navier-Stokes`__
tutorial is currently under construction and will be published here soon.

.. _cfdtutorial: http://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/

__ cfdtutorial_
