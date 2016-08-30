Acoustic Wave Modelling using Devito
==================================================

Consider the acoustic wave equation given in 3D:

.. math::
   m(x,y,z)\frac{\partial^2 u}{\partial t^2} + \eta(x,y,z)\frac{\partial u}{\partial t}-\frac{\partial^2 u}{\partial x^2}-\frac{\partial^2 u}{\partial y^2}-\frac{\partial^2 u}{\partial z^2}= 0

   u(x,y,z,0) = 0

   \frac{\partial u(x,y,z,t)}{\partial t}|_{t=0} = 0

where :math:`\eta` represents damping, :math:`m` represents square slowness

In this tutorial, the origin, spacing and true velocity are stored in instance of :obj:`IGrid()` called :obj:`model`.
The gap between two consecutive timestep :obj:`dt` is :samp:`model.get_critical_dt()`.
All the seismic data and initial settings like the coordinate of source,
receiver coordinates are stored in an instance of :obj:`IShot()` called :obj:`data`::

  from containers import IGrid, IShot

  model = IGrid()
  dimensions = (50, 50, 50)
  model.shape = dimensions
  origin = (0., 0.)
  spacing = (20., 20.)

  # True velocity
  true_vp = np.ones(dimensions) + 2.0
  true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

  model.create_model(origin, spacing, true_vp)

  # Define seismic data.
  data = IShot()

  dt = model.get_critical_dt()

  # Omit the code that generate time_series, receiver_coords, location

  data.set_source(time_series, dt, location)
  data.set_receiver_pos(receiver_coords)
  data.set_shape(nt, 101)

First of all, we will set up the initial condition for damping::

  from devito.interfaces import DenseData
  self.damp = DenseData(name="damp", shape=self.model.get_shape_comp(),
                        dtype=self.dtype)
  # Initialize damp by calling the function that can precompute damping
  damp_boundary(self.damp.data)

DenseData is a devito data object used to store and manage spatially varying data.

:samp:`damp_boundary()` function initialises the damp on each grid point. The initial data are stored in :obj:`self.damp.data`.

Similarly, we will set up the initial condition and allocate the grid for m.
Initial value of m on each grid point is stored as a numpy array in :obj:`m.data[:]`.::

  m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
  m.data[:] = model.padm()

after that, we will initilisize u::

  u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                   time_order=time_order, space_order=spc_order, save=True,
                   dtype=damp.dtype)

TimeData is a devito data object used to store and manage time-varying data

We initialise our grid to be of size :samp:`model.get_shape_comp()` which is a 3-D tuple.
:obj:`time_dim` represents the size of the time dimension that dictates the leading dimension of the data buffer.
:obj:`time_order` and :obj:`space_order` represent the discretization order for time and space respectively.

The next step is to generate the stencil to be solved by a :obj:`devito.operator.Operator`

The stencil is generated according to Devito conventions. It uses a sympy
expression to represent the acoustic wave equation. Devito makes it easy to
represent the equation in a finite-difference form by providing properties :obj:`dt2`, :obj:`laplace`, :obj:`dt`.
We then generate the stencil by solving eqn for u.forward, a symbol for the time-forward state of the function.
::

  from devito import Operator
  from sympy import Eq, solve, symbols
  eqn = m*u.dt2-u.laplace+damp*u.dt
  stencil = solve(eqn, u.forward)[0]

We plug the stencil in an Operator, as shown, and define the values the spacing between cells :obj:`h` and the
temporal spacing :obj:`s`.::

  s, h = symbols('s h')
  subs = {s: model.get_critical_dt(), h: model.get_spacing()}
  super(AdjointOperator, self).__init__(nt, m.shape,
                                        stencils=Eq(v.backward, stencil),
                                        substitutions=subs,
                                        spc_border=spc_order/2,
                                        time_order=time_order,
                                        forward=False,
                                        dtype=m.dtype,
                                        **kwargs)


To execute the generated Operator, we simply call :samp:`op.apply()`. The results will then be found in :obj:`u.data`

For a complete example of this code, check file `acoustic_example.py` in the
`examples` folder.
