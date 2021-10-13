from devito import Function, TimeFunction, DevitoCheckpoint, CheckpointOperator
from devito.tools import memoized_meth
from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from pyrevolve import Revolver


class AcousticWaveSolver(object):
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
    kernel : str, optional
        Type of discretization, centered or shifted.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, kernel='OT2', space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="damp")
        self.geometry = geometry

        assert self.model.grid == geometry.grid

        self.space_order = space_order
        self.kernel = kernel

        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if self.kernel == 'OT4':
            return self.model.dtype(1.73 * self.model.critical_dt)
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=self.kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=self.kernel, space_order=self.space_order,
                                **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            kernel=self.kernel, space_order=self.space_order,
                            **self._kwargs)

    def forward(self, src=None, rec=None, u=None, model=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            Stores the computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u,
                                          dt=kwargs.pop('dt', self.dt), **kwargs)

        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, model=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        v: TimeFunction, optional
            The computed wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or self.geometry.new_src(name='srca', src_type=None)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(srca=srca, rec=rec, v=v,
                                      dt=kwargs.pop('dt', self.dt), **kwargs)
        return srca, v, summary

    def jacobian_adjoint(self, rec, u, src=None, v=None, grad=None, model=None,
                         checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        u : TimeFunction
            Full wavefield `u` (created with save=True).
        v : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.

        Returns
        -------
        Gradient field and performance summary.
        """
        dt = kwargs.pop('dt', self.dt)
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False),
                                         src=src or self.geometry.src,
                                         u=u, dt=dt, **kwargs)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          rec=rec, dt=dt, grad=grad, **kwargs)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad().apply(rec=rec, grad=grad, v=v, u=u, dt=dt,
                                           **kwargs)
        return grad, summary

    def jacobian(self, dmin, src=None, rec=None, u=None, U=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        u : TimeFunction, optional
            The forward wavefield.
        U : TimeFunction, optional
            The linearized wavefield.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        model = model or self.model
        # Pick vp from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                       dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, U, summary

    # Backward compatibility
    born = jacobian
    gradient = jacobian_adjoint
