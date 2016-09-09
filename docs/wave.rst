Acoustic Wave Modelling using Devito
==================================================

Consider the acoustic wave equation given in 3D:

.. math::
   m(x,y,z)\frac{\partial^2 u}{\partial t^2} + \eta(x,y,z)\frac{\partial u}{\partial t}-\frac{\partial^2 u}{\partial x^2}-\frac{\partial^2 u}{\partial y^2}-\frac{\partial^2 u}{\partial z^2}= q

where Damp(:math:`\eta`) is dampening coefficient for absorbing boundary condition,
:math:`m=\frac{1}{v(x,y,z)^2}`, :math:`v` is the velocity
, :math:`u` is pressure field and :math:`q` is the pressure source term.

First of all, we will set up seismic datas,

In this tutorial, Model, an instance of :obj:`IGrid()` stores the origin,
the position in meters of the position of index (0,0,0), as (0,0,0) in 3D,
spacing, the size of the discrete grid, distance in meters between two consecutive grid point in each direction,
dimensions, the size of the model in number of grid points and true velocity.
The time stepping rate, dt is derived from :samp:`model.get_critical_dt()`.

Data, an instance of :obj:`IShot()` stores amplitudes of the source
at each time step, source coordinates and receiver coordinates.::

  from containers import IGrid, IShot

  model = IGrid()
  dimensions = (50, 50, 50)
  model.shape = dimensions
  origin = (0., 0., 0.)
  spacing = (20., 20., 20.)

  # True velocity
  true_vp = np.ones(dimensions) + 2.0
  true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

  model.create_model(origin, spacing, true_vp)

  # Define seismic data.
  data = IShot()

  f0 = .010
  dt = model.get_critical_dt()
  t0 = 0.0
  tn = 250.0
  nt = int(1+(tn-t0)/dt)
  h = model.get_spacing()


  # Set up the source as Ricker wavelet for f0
  def source(t, f0):
      r = (np.pi * f0 * (t - 1./f0))

      return (1-2.*r**2)*np.exp(-r**2)


  time_series = source(np.linspace(t0, tn, nt), f0)
  location = (origin[0] + dimensions[0] * spacing[0] * 0.5, 500,
                  origin[1] + 2 * spacing[1])
  data.set_source(time_series, dt, location)

  receiver_coords = np.zeros((101, 3))
  receiver_coords[:, 0] = np.linspace(50, 950, num=101)
  receiver_coords[:, 1] = 500
  receiver_coords[:, 2] = location[2]
  data.set_receiver_pos(receiver_coords)
  data.set_shape(nt, 101)

Then we will set up the dampening coefficient.
::

  from devito.interfaces import DenseData
  self.damp = DenseData(name="damp", shape=self.model.get_shape_comp(),
                        dtype=self.dtype)
  # Initialize damp by calling the function that can precompute damping
  damp_boundary(self.damp.data)

DenseData is a devito data object used to store and manage spatially varying data.

:samp:`damp_boundary()` function initialises the damp on each grid point.
The dampening values in data are stored in :obj:`self.damp.data`.

Then we will set up the source
::
  srccoord = np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :]
  self.src = SourceLike(name="src", npoint=1, nt=data.traces.shape[1],
                        dt=self.dt, h=self.model.get_spacing(),
                        coordinates=srccoord, ndim=len(self.damp.shape),
                        dtype=self.dtype, nbpml=nbpml)
  self.src.data[:] = data.get_source()[:, np.newaxis]

SourceLike is an object inheriting PointData, a devito data object for sparse point data
as a Function symbol.

We initialize the source to be of coordinates : :obj:`srccoord` and set its data to be :obj:`data.get_source()`.
Receivers are initialized by similar way
::
  rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                       coordinates=data.receiver_coords, ndim=len(damp.shape),
                       dtype=damp.dtype, nbpml=model.nbpml)

Then, We will set up the initial condition and allocate the grid for m.
Value of m on each grid point is stored as a numpy array in :obj:`m.data[:]`.
::
  m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
  m.data[:] = model.padm()

after that, we will initialize u
::
  u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                   time_order=time_order, space_order=spc_order, save=True,
                   dtype=damp.dtype)

TimeData is a devito data object used to store and manage time-varying as well as space-varying data

We initialize our grid to be of size :samp:`model.get_shape_comp()` which is a 3-D tuple.
:obj:`time_dim` represents the size of the time dimension that dictates
the leading dimension of the data buffer.
:obj:`time_order` and :obj:`space_order` represent the discretization order
for time and space respectively.

The next step is to generate the stencil to be solved by a :obj:`devito.operator.Operator`

The stencil is generated according to Devito conventions. It uses a sympy
expression to represent the acoustic wave equation. Devito makes it easy to
represent the equation in a finite-difference form by providing properties :obj:`dt2`, :obj:`laplace`, :obj:`dt`.
We then generate the stencil by solving eqn for :obj:`u.forward = u(t+dt,x,y,z)`,
a symbol for the time-forward state of the function.
::

  from devito import Operator
  from sympy import Eq, solve, symbols
  eqn = m*u.dt2-u.laplace+damp*u.dt
  stencil = solve(eqn, u.forward)[0]

We plug the stencil in an Operator, as shown, as well as the the values of the spacing
between cells :obj:`h`, the temporal spaces :obj:`s`, the number of timesteps :obj:`nt`, the :obj:`spc_border` which are
the ghost points at the edge of the domain to touse the same stencil everywhere and so on.
The :obj:`forward` argument represents propagation time direction. It's set to be True
if forwarding in time and False if backwarding in time.
::

  s, h = symbols('s h')
  subs = {s: model.get_critical_dt(), h: model.get_spacing()}
  super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              forward=True,
                                              dtype=m.dtype,
                                              **kwargs)

After that, we will insert source and receiver terms into the input parameters
to generate the C++ file that contains required inputs.
For the output, we will add receivers so that it will output :math:`u` on each receiver coordinate
on all time steps. :obj:`src.add(m, u)` and :obj:`red.read(u)` will generate C iteration codes over points
and they will be added into stencils in C++ file. ::

    self.input_params += [src, src.coordinates, rec, rec.coordinates]
    self.output_params += [rec]
    self.propagator.time_loop_stencils_a = src.add(m, u) + rec.read(u)
    self.propagator.add_devito_param(src)
    self.propagator.add_devito_param(src.coordinates)
    self.propagator.add_devito_param(rec)
    self.propagator.add_devito_param(rec.coordinates)

To execute the generated Operator, we simply call :samp:`apply()`. As mentioned,
it will output :math:`u` on each receiver coordinates and :math:`u`
on each grid points for all time steps. The results can be found in :obj:`u.data`.


For a complete example of this code, check file `acoustic_example.py` in the
`examples` folder.
