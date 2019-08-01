# coding: utf-8
from devito import TimeFunction, warning
from devito.tools import memoized_meth
from examples.seismic.tti.operators import ForwardOperator, particle_velocity_fields
from examples.seismic import Receiver


class AnisotropicWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.

    Notes
    -----
    space_order must be even and it is recommended to be a multiple of 4
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.geometry = geometry

        if space_order % 2 != 0:
            raise ValueError("space_order must be even but got %s" % space_order)

        if space_order % 4 != 0:
            warning("It is recommended for space_order to be a multiple of 4 " +
                    "but got %s" % space_order)

        self.space_order = space_order
        self.dt = self.model.critical_dt

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, kernel='centered', save=False):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order,
                               kernel=kernel, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, v=None, vp=None,
                epsilon=None, delta=None, theta=None, phi=None,
                save=False, kernel='centered', **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (SparseTimeFunction) and
            receivers (SparseTimeFunction) and their position.
        u : TimeFunction, optional
            The computed wavefield first component.
        v : TimeFunction, optional
            The computed wavefield second component.
        vp : Function or float, optional
            The time-constant velocity.
        epsilon : Function or float, optional
            The time-constant first Thomsen parameter.
        delta : Function or float, optional
            The time-constant second Thomsen parameter.
        theta : Function or float, optional
            The time-constant Dip angle (radians).
        phi : Function or float, optional
            The time-constant Azimuth angle (radians).
        save : int or Buffer
            Option to store the entire (unrolled) wavefield.
        kernel : str, optional
            Type of discretization, centered or shifted.

        Returns
        -------
        Receiver, wavefield and performance summary.
        """

        if kernel == 'staggered':
            time_order = 1
            dims = self.model.space_dimensions
            stagg_u = (-dims[-1])
            stagg_v = (-dims[0], -dims[1]) if self.model.grid.dim == 3 else (-dims[0])
        else:
            time_order = 2
            stagg_u = stagg_v = None
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefield if not provided
        if u is None:
            u = TimeFunction(name='u', grid=self.model.grid, staggered=stagg_u,
                             save=self.geometry.nt if save else None,
                             time_order=time_order, space_order=self.space_order)
        # Create the forward wavefield if not provided
        if v is None:
            v = TimeFunction(name='v', grid=self.model.grid, staggered=stagg_v,
                             save=self.geometry.nt if save else None,
                             time_order=time_order, space_order=self.space_order)

        if kernel == 'staggered':
            vx, vz, vy = particle_velocity_fields(self.model, self.space_order)
            kwargs["vx"] = vx
            kwargs["vz"] = vz
            if vy is not None:
                kwargs["vy"] = vy

        # Pick m from model unless explicitly provided
        kwargs.update(self.model.physical_params(vp=vp, epsilon=epsilon, delta=delta,
                                                 theta=theta, phi=phi))
        # Execute operator and return wavefield and receiver data
        op = self.op_fwd(kernel, save)
        summary = op.apply(src=src, rec=rec, u=u, v=v,
                           dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, v, summary
