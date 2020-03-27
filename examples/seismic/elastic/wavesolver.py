from devito.tools import memoized_meth
from devito import VectorTimeFunction, TensorTimeFunction

from examples.seismic import Receiver
from examples.seismic.elastic.operators import ForwardOperator


class ElasticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.geometry = geometry

        self.space_order = space_order
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec1=None, rec2=None, lam=None, mu=None, irho=None,
                v=None, tau=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec1 : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the pressure (tzz).
        rec2 : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the particle velocities.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        tau : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        lam : Function, optional
            The time-constant first Lame parameter lambda.
        mu : Function, optional
            The time-constant second Lame parameter mu.
        irho : Function, optional
            The time-constant inverse density (irho=1 for water).
        save : int or Buffer, optional
            Option to store the entire (unrolled) wavefield.

        Returns
        -------
        Rec1(tzz), Rec2(div(v)), particle velocities v, stress tensor tau and
        performance summary.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec1 = rec1 or Receiver(name='rec1', grid=self.model.grid,
                                time_range=self.geometry.time_axis,
                                coordinates=self.geometry.rec_positions)
        rec2 = rec2 or Receiver(name='rec2', grid=self.model.grid,
                                time_range=self.geometry.time_axis,
                                coordinates=self.geometry.rec_positions)

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        save_t = src.nt if save else None
        v = VectorTimeFunction(name='v', grid=self.model.grid, save=save_t,
                               space_order=self.space_order, time_order=1)
        tau = TensorTimeFunction(name='tau', grid=self.model.grid, save=save_t,
                                 space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})
        # Pick Lame parameters from model unless explicitly provided
        lam = lam or self.model.lam
        mu = mu or self.model.mu
        irho = irho or self.model.irho
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, lam=lam, mu=mu, irho=irho,
                                          rec2=rec2, dt=kwargs.pop('dt', self.dt),
                                          **kwargs)
        return rec1, rec2, v, tau, summary
