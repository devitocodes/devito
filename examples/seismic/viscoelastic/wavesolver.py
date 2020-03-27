from devito import VectorTimeFunction, TensorTimeFunction
from devito.tools import memoized_meth
from examples.seismic import Receiver
from examples.seismic.viscoelastic.operators import ForwardOperator


class ViscoelasticWaveSolver(object):
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

    Notes
    -----
    This is an experimental staggered grid viscoelastic modeling kernel.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.geometry = geometry

        self.space_order = space_order
        self.dt = self.model.critical_dt
        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec1=None, rec2=None, lam=None, qp=None, mu=None, qs=None,
                irho=None, v=None, tau=None, r=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (src : SparseTimeFunction) and
            receivers (rec1(txx) : SparseTimeFunction, rec2(tzz) : SparseTimeFunction)
            and their position.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        tau : TensorTimeFunction, optional
            The computed stress.
        r : TensorTimeFunction, optional
            The computed memory variable.
        lambda : Function, optional
            The time-constant first Lame parameter (rho * vp**2 - rho * vs **2).
        qp : Function, optional
            The P-wave quality factor (dimensionless).
        mu : Function, optional
            The Shear modulus (rho * vs*2).
        qs : Function, optional
            The S-wave quality factor (dimensionless).
        irho : Function, optional
            The time-constant inverse density (1/rho=1 for water).
        save : int or Buffer, optional
            Option to store the entire (unrolled) wavefield.

        Returns
        -------
        Rec1 (txx), Rec2 (tzz), particle velocities vx and vz, stress txx,
        tzz and txz, memory variables rxx, rzz, rxz and performance summary.
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

        # Create all the fields v, tau, r
        save_t = src.nt if save else None
        v = VectorTimeFunction(name="v", grid=self.model.grid, save=save_t,
                               time_order=1, space_order=self.space_order)
        # Stress:
        tau = TensorTimeFunction(name='t', grid=self.model.grid, save=save_t,
                                 space_order=self.space_order, time_order=1)
        # Memory variable:
        r = TensorTimeFunction(name='r', grid=self.model.grid, save=save_t,
                               space_order=self.space_order, time_order=1)

        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})
        kwargs.update({k.name: k for k in r})
        # Pick physical parameters from model unless explicitly provided
        lam = lam or self.model.lam
        qp = qp or self.model.qp
        mu = mu or self.model.mu
        qs = qs or self.model.qs
        irho = irho or self.model.irho
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, mu=mu, qp=qp, lam=lam,
                                          qs=qs, irho=irho, rec2=rec2,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec1, rec2, v, tau, summary
