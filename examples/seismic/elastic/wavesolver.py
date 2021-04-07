from devito.tools import memoized_meth
from devito import VectorTimeFunction, TensorTimeFunction

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
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry

        self.space_order = space_order
        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec1=None, rec2=None, v=None, tau=None,
                model=None, save=None, **kwargs):
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
        model : Model, optional
            Object containing the physical parameters.
        lam : Function, optional
            The time-constant first Lame parameter `rho * (vp**2 - 2 * vs **2)`.
        mu : Function, optional
            The Shear modulus `(rho * vs*2)`.
        b : Function, optional
            The time-constant inverse density (b=1 for water).
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Rec1(tzz), Rec2(div(v)), particle velocities v, stress tensor tau and
        performance summary.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec1 = rec1 or self.geometry.new_rec(name='rec1')
        rec2 = rec2 or self.geometry.new_rec(name='rec2')

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        save_t = src.nt if save else None
        v = v or VectorTimeFunction(name='v', grid=self.model.grid, save=save_t,
                                    space_order=self.space_order, time_order=1)
        tau = tau or TensorTimeFunction(name='tau', grid=self.model.grid, save=save_t,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})

        model = model or self.model
        # Pick Lame parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, rec2=rec2,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec1, rec2, v, tau, summary
