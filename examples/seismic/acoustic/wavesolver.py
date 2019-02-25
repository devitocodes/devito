from devito import Function, TimeFunction, memoized_meth
from examples.seismic import PointSource, Receiver
from examples.seismic.acoustic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver


class AcousticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters
    geometry : AcquisitionGeometry
        Source and receivers geometry
    space_order : Int
        Order of the spatial stencil discretisation (default: 2)

    Note: space_order must always be an even number
    """
    def __init__(self, model, geometry, space_order=2, **kwargs):
        self.model = model
        self.geometry = geometry

        assert self.model == geometry.model

        self.space_order = space_order

        self._kwargs = kwargs

    def dt(self, kernel):
        dt = self.model.critical_dt
        if kernel == 'OT4':
            dt *= 1.73
        return dt

    @memoized_meth
    def op_fwd(self, save=None, kernel='OT2'):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               kernel=kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_adj(self, kernel='OT2'):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               kernel=kernel, space_order=self.space_order,
                               **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True, kernel='OT2'):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                kernel=kernel, space_order=self.space_order,
                                **self._kwargs)

    @memoized_meth
    def op_born(self, kernel='OT2'):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            kernel=kernel, space_order=self.space_order,
                            **self._kwargs)

    def forward(self, src=None, rec=None, u=None, m=None, save=None,
                kernel='OT2', **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or Array
            Symbol with time series data for the injected source term
        rec : SparseTimeFunction or Array
            Symbol to store interpolated receiver data (u+v)
        u : SparseTimeFunction (Optional)
            Symbol to store the computed wavefield
        m : Function or Array or Float (Optional)
            Symbol for the time-constant square slowness
        save: Bool
            Option to store the entire (unrolled) wavefield
        kernel: OT2 or OT4
            type of discretization
        **kwargs : Compiler options (dle, dse,autotuning,...)
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        dt = kwargs.pop('dt', self.dt(kernel))
        summary = self.op_fwd(save, kernel=kernel).apply(src=src, rec=rec, u=u, m=m,
                                                         dt=dt, **kwargs)
        return rec, u, summary

    def adjoint(self, rec, srca=None, v=None, m=None, kernel='OT2', **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or Array
            Symbol to store interpolated receiver data (u)
        srca : SparseTimeFunction or Array
            Symbol with time series data for the injected source term
        v : SparseTimeFunction (Optional)
            Symbol to store the computed wavefield
        m : Function or Array or Float (Optional)
            Symbol for the time-constant square slowness
        kernel: OT2 or OT4
            type of discretization
        **kwargs : Compiler options (dle, dse,autotuning,...)
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.geometry.time_axis,
                                   coordinates=self.geometry.src_positions)

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        dt = kwargs.pop('dt', self.dt(kernel))
        summary = self.op_adj(kernel=kernel).apply(srca=srca, rec=rec, v=v, m=m,
                                                   dt=dt, **kwargs)
        return srca, v, summary

    def gradient(self, rec, u, v=None, grad=None, m=None, checkpointing=False,
                 kernel='OT2', **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        recin : SparseTimeFunction or Array
            Adjoint source (at receiver locations)
        u : SparseTimeFunction (Optional)
            Symbol for the computed forward wavefield
        v : SparseTimeFunction (Optional)
            Symbol to store the computed adjoint wavefield
        m : Function or Array or Float (Optional)
            Symbol for the time-constant square slowness
        grad : Function or Array or Float (Optional)
            Symbol to store the gradient
        kernel: OT2 or OT4
            type of discretization
        **kwargs : Compiler options (dle, dse,autotuning,...)
        """
        dt = kwargs.pop('dt', self.dt(kernel))
        # Gradient symbol
        grad = grad or Function(name='grad', grid=self.model.grid)

        # Create the forward wavefield
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        if checkpointing:
            u = TimeFunction(name='u', grid=self.model.grid,
                             time_order=2, space_order=self.space_order)
            cp = DevitoCheckpoint([u])
            n_checkpoints = None
            wrap_fw = CheckpointOperator(self.op_fwd(save=False), src=self.geometry.src,
                                         u=u, m=m, dt=dt)
            wrap_rev = CheckpointOperator(self.op_grad(save=False), u=u, v=v,
                                          m=m, rec=rec, dt=dt, grad=grad)

            # Run forward
            wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
            wrp.apply_forward()
            summary = wrp.apply_reverse()
        else:
            summary = self.op_grad(kernel=kernel).apply(rec=rec, grad=grad, v=v, u=u,
                                                        m=m, dt=dt, **kwargs)
        return grad, summary

    def born(self, dmin, src=None, rec=None, u=None, U=None, m=None,
             kernel='OT2', **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or Array
            Symbol with time series data for the injected source term
        rec : SparseTimeFunction or Array
            Symbol to store interpolated receiver data (u+v)
        u : SparseTimeFunction (Optional)
            Symbol to store the computed wavefield
        u : SparseTimeFunction (Optional)
            Symbol to store the computed linearized (Born) wavefield
        m : Function or Array or Float (Optional)
            Symbol for the time-constant square slowness
        kernel: OT2 or OT4
            type of discretization
        **kwargs : Compiler options (dle, dse,autotuning,...)
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or Receiver(name='rec', grid=self.model.grid,
                              time_range=self.geometry.time_axis,
                              coordinates=self.geometry.rec_positions)

        # Create the forward wavefields u and U if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)
        U = U or TimeFunction(name='U', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick m from model unless explicitly provided
        m = m or self.model.m

        # Execute operator and return wavefield and receiver data
        dt = kwargs.pop('dt', self.dt(kernel))
        summary = self.op_born(kernel=kernel).apply(dm=dmin, u=u, U=U, src=src, rec=rec,
                                                    m=m, dt=dt, **kwargs)
        return rec, u, U, summary
