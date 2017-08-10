# coding: utf-8
from cached_property import cached_property

from devito import TimeData
from examples.seismic.tti.operators import ForwardOperator
from examples.seismic import Receiver


class AnisotropicWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    :param model: Physical model with domain parameters
    :param source: Sparse point symbol providing the injected wave
    :param receiver: Sparse point symbol describing an array of receivers
    :param time_order: Order of the time-stepping scheme (default: 2)
    :param space_order: Order of the spatial stencil discretisation (default: 4)

    Note: space_order must always be greater than time_order
    """
    def __init__(self, model, source, receiver,
                 time_order=2, space_order=2, **kwargs):
        self.model = model
        self.source = source
        self.receiver = receiver

        self.time_order = time_order
        self.space_order = space_order/2

        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        if self.time_order == 4:
            self.dt *= 1.73

        # Cache compiler options
        self._kwargs = kwargs

    @cached_property
    def op_fwd(self):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=False, source=self.source,
                               receiver=self.receiver, time_order=self.time_order,
                               space_order=self.space_order, **self._kwargs)

    @cached_property
    def op_fwd_save(self):
        """Cached operator for forward runs with unrolled wavefield"""
        return ForwardOperator(self.model, save=True, source=self.source,
                               receiver=self.receiver, time_order=self.time_order,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, v=None, m=None,
                epsilon=None, delta=None, theta=None, phi=None,
                save=False, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield

        :returns: Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        if src is None:
            src = self.source
        # Create a new receiver object to store the result
        if rec is None:
            rec = Receiver(name='rec', ntime=self.receiver.nt,
                           coordinates=self.receiver.coordinates.data)

        # Create the forward wavefield if not provided
        if u is None:
            u = TimeData(name='u', shape=self.model.shape_domain,
                         save=save, time_dim=self.source.nt,
                         time_order=self.time_order,
                         space_order=self.space_order,
                         dtype=self.model.dtype)
        # Create the forward wavefield if not provided
        if v is None:
            v = TimeData(name='v', shape=self.model.shape_domain,
                         save=save, time_dim=self.source.nt,
                         time_order=self.time_order,
                         space_order=self.space_order,
                         dtype=self.model.dtype)
        # Pick m from model unless explicitly provided
        if m is None:
            m = m or self.model.m
        if epsilon is None:
            epsilon = epsilon or self.model.epsilon
        if delta is None:
            delta = delta or self.model.delta
        if theta is None:
            theta = theta or self.model.theta
        if phi is None:
            phi = phi or self.model.phi

        # Execute operator and return wavefield and receiver data
        if save:
            op = self.op_fwd_save
        else:
            op = self.op_fwd

        if len(m.shape) == 2:
            summary = op.apply(src=src, rec=rec, u=u, v=v, m=m, epsilon=epsilon,
                               delta=delta, theta=theta, **kwargs)
        else:
            summary = op.apply(src=src, rec=rec, u=u, v=v, m=m, epsilon=epsilon,
                               delta=delta, theta=theta, phi=phi, **kwargs)
        return rec, u, v, summary
