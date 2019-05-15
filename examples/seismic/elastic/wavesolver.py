from devito.tools import memoized_meth
from examples.seismic import Receiver
from examples.seismic.elastic.operators import (ForwardOperator, stress_fields,
                                                particle_velocity_fields)


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

    Notes
    -----
    This is an experimental staggered grid elastic modeling kernel.
    Only 2D supported.
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

    def forward(self, src=None, rec1=None, rec2=None, vp=None, vs=None, rho=None,
                vx=None, vz=None, txx=None, tzz=None, txz=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        geometry : AcquisitionGeometry
            Geometry object that contains the source (src : SparseTimeFunction) and
            receivers (rec1(txx) : SparseTimeFunction, rec2(tzz) : SparseTimeFunction)
            and their position.
        vx : TimeFunction, optional
            The computed horizontal particle velocity.
        vz : TimeFunction, optional
            The computed vertical particle velocity.
        txx : TimeFunction, optional
            The computed horizontal stress.
        tzz : TimeFunction, optional
            The computed vertical stress.
        txz : TimeFunction, optional
            The computed diagonal stresss.
        vp : Function, optional
            The time-constant P-wave velocity (km/s).
        vs : Function, optional
            The time-constant S-wave velocity (km/s).
        rho : Function, optional
            The time-constant density (rho=1 for water).
        save : int or Buffer, optional
            Option to store the entire (unrolled) wavefield.

        Returns
        -------
        Rec1 (txx), Rec2 (tzz), particle velocities vx and vz, stress txx,
        tzz and txz and performance summary.
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
        vx, vy, vz = particle_velocity_fields(self.model, save_t, self.space_order)
        txx, tyy, tzz, txy, txz, tyz = stress_fields(self.model, save_t, self.space_order)
        kwargs['vx'] = vx
        kwargs['vz'] = vz
        kwargs['txx'] = txx
        kwargs['tzz'] = tzz
        kwargs['txz'] = txz
        if self.model.grid.dim == 3:
            kwargs['vy'] = vy
            kwargs['tyy'] = tyy
            kwargs['txy'] = txy
            kwargs['tyz'] = tyz
        # Pick m from model unless explicitly provided
        vp = vp or self.model.vp
        vs = vs or self.model.vs
        rho = rho or self.model.rho
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, vp=vp, vs=vs, rho=rho,
                                          rec2=rec2, dt=kwargs.pop('dt', self.dt),
                                          **kwargs)
        return rec1, rec2, vx, vz, txx, tzz, txz, summary
