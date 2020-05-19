from devito import Function, TimeFunction
from devito.tools import memoized_meth
from examples.seismic.skew_self_adjoint.operators import IsoFwdOperator, IsoAdjOperator, \
    IsoJacobianFwdOperator, IsoJacobianAdjOperator


class SsaIsoAcousticWaveSolver(object):
    """
    Solver object for a scalar isotropic variable density visco- acoustic skew
    self adjoint wave equation that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem setup.

    Parameters
    ----------
    npad : int, required
        Number of points in the absorbing boundary.
        Typically set to 50.
    omega : float, required
        Center circular frequency for dissipation only attenuation.
    qmin : float, required
        Minimum Q value on the exterior of the absorbing boundary.
        Typically set to 0.1.
    qmax : float, required
        Maximum Q value in the interior of the model.
        Typically set to 100.0.
    b : Function, required
        Physical model with buoyancy (m^3/kg).
    v : Function, required
        Physical model with velocity (m/msec).
    src : SparseTimeFunction (PointSource)
        Source position and time signature.
    rec : SparseTimeFunction (PointSource)
        Receiver positions and time signature.
    time_axis : TimeAxis
        Defines temporal sampling.
    space_order: int, optional
        Order of the spatial stencil discretisation. Defaults to 8.
    """
    def __init__(self, model, geometry, space_order=8, **kwargs):
        self.model = model
        self.geometry = geometry

        assert self.model == geometry.model

        self.space_order = space_order

        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return IsoFwdOperator(self.model, save=save, geometry=self.geometry,
                              space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_adj(self, save=None):
        """Cached operator for adjoint runs"""
        return IsoAdjOperator(self.model, save=save, geometry=self.geometry,
                              space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_jadj(self, save=True):
        """Cached operator for gradient runs"""
        return IsoJacobianAdjOperator(self.model, save=save, geometry=self.geometry,
                                      space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_jac(self, save=None):
        """Cached operator for born runs"""
        return IsoJacobianFwdOperator(self.model, save=save, geometry=self.geometry,
                                      space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, b=None, vp=None, damp=None, u=None,
                save=None, **kwargs):
        """
        Forward modeling function that creates the necessary
        data objects for running a forward modeling operator.
        No required parameters.

        Parameters
        ----------
        src : SparseTimeFunction, required
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to new rec
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        damp : Function or float, optional, defaults to damp at construction
            The time-constant dissipation only attenuation w/Q field.
        u : Function or float, optional, defaults to new TimeFunction
            Stores the computed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Receiver time series data, TimeFunction wavefield u, and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefield if not provided
        u = u or TimeFunction(name='u', grid=self.model.grid,
                              save=self.geometry.nt if save else None,
                              time_order=2, space_order=self.space_order)

        # Pick input physical parameters
        kwargs.update(self.model.physical_params(vp=vp, damp=damp, b=b))
        kwargs.update({'dt': kwargs.pop('dt', self.dt)})

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec=rec, u=u, **kwargs)
        return rec, u, summary

    def adjoint(self, rec, src=None, b=None, v=None, damp=None, vp=None,
                save=None, **kwargs):
        """
        Adjoint modeling function that creates the necessary
        data objects for running a adjoint modeling operator.
        Required parameters: rec.

        Parameters
        ----------
        rec : SparseTimeFunction, required
            The interpolated receiver data to be injected.
        src : SparseTimeFunction, optional, defaults to new src
            Time series data for the adjoint source term.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        damp : Function or float, optional, defaults to damp at construction
            The time-constant dissipation only attenuation w/Q field.
        ua : Function or float, optional, defaults to new TimeFunction
            Stores the computed adjoint wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Adjoint source time series data, wavefield TimeFunction ua,
        and performance summary
        """
        # Create a new adjoint source and receiver symbol
        srca = src or self.geometry.src

        # Create the adjoint wavefield if not provided
        v = v or TimeFunction(name='v', grid=self.model.grid,
                              time_order=2, space_order=self.space_order)

        # Pick input physical parameters
        kwargs.update(self.model.physical_params(vp=vp, damp=damp, b=b))
        kwargs.update({'dt': kwargs.pop('dt', self.dt)})

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj(save).apply(srca=src, rec=rec, **kwargs)
        return srca, v, summary

    def jacobian_forward(self, dm, src, rec=None, b=None, vp=None, damp=None,
                         u0=None, du=None, save=None, **kwargs):
        """
        Linearized JacobianForward modeling function that creates the necessary
        data objects for running a Jacobian forward modeling operator.
        Required parameters: dm.

        Parameters
        ----------
        dm : Function or float, required
            The perturbation to the velocity model.
        src : SparseTimeFunction, required
            Time series data for the injected source term.
        rec : SparseTimeFunction, optional, defaults to new rec
            The interpolated receiver data.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        damp : Function or float, optional, defaults to damp at construction
            The time-constant dissipation only attenuation w/Q field.
        u0 : Function or float, optional, defaults to new TimeFunction
            Stores the computed background wavefield.
        du : Function or float, optional, defaults to new TimeFunction
            Stores the computed perturbed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Receiver time series data rec, TimeFunction background wavefield u0,
        TimeFunction perturbation wavefield du, and performance summary
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec = rec or self.geometry.rec

        # Create the forward wavefields u and U if not provided
        u0 = u0 or TimeFunction(name='u0', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)
        du = du or TimeFunction(name='du', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)

        # Pick input physical parameters
        kwargs.update(self.model.physical_params(vp=vp, damp=damp, b=b))
        kwargs.update({'dt': kwargs.pop('dt', self.dt)})

        # Execute operator and return wavefield and receiver data
        summary = self.op_jac(save).apply(dm=dm, u0=u0, du=du, src=src, rec=rec, **kwargs)
        return rec, u0, du, summary

    def jacobian_adjoint(self, rec, u0, b=None, vp=None, damp=None,
                         dm=None, du=None, save=None, **kwargs):
        """
        Linearized JacobianForward modeling function that creates the necessary
        data objects for running a Jacobian forward modeling operator.
        Required parameters: rec, u0.

        Parameters
        ----------
        rec : SparseTimeFunction, required
            The interpolated receiver data to be injected.
        u0 : Function or float, required, (created with save=True)
            Stores the computed background wavefield.
        b : Function or float, optional, defaults to b at construction
            The time-constant buoyancy.
        v : Function or float, optional, defaults to v at construction
            The time-constant velocity.
        damp : Function or float, optional, defaults to damp at construction
            The time-constant dissipation only attenuation w/Q field.
        dm : Function or float, optional, defaults to new Function
            The perturbation to the velocity model.
        du : Function or float, optional, defaults to new TimeFunction
            Stores the computed perturbed wavefield.
        save : int or Buffer, optional
            The entire (unrolled) wavefield.

        Returns
        ----------
        Function model perturbation dm, Receiver time series data rec,
        TimeFunction background wavefield u0, TimeFunction perturbation wavefield du,
        and performance summary
        """
        # Get model perturbation Function or create
        dm = dm or Function(name='dm', grid=self.v.grid, space_order=self.space_order)

        # Create the perturbation wavefield if not provided
        du = du or TimeFunction(name='du', grid=self.model.grid,
                                time_order=2, space_order=self.space_order)

        # Pick input physical parameters
        kwargs.update(self.model.physical_params(vp=vp, damp=damp, b=b))
        kwargs.update({'dt': kwargs.pop('dt', self.dt)})

        # Run operator
        summary = self.op_jadj().apply(rec=rec, dm=dm, du=du, u0=u0, **kwargs)
        return dm, u0, du, summary
