# coding: utf-8
from devito import TimeFunction, memoized_meth
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
    def __init__(self, model, source, receiver, space_order=2, **kwargs):
        self.model = model
        self.source = source
        self.receiver = receiver

        self.space_order = space_order
        self.dt = self.model.critical_dt

        # Cache compiler options
        self._kwargs = kwargs

    @memoized_meth
    def op_fwd(self, kernel='shifted', save=False):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, source=self.source,
                               receiver=self.receiver,
                               space_order=self.space_order,
                               kernel=kernel, **self._kwargs)

    def forward(self, src=None, rec=None, u=None, v=None, m=None,
                epsilon=None, delta=None, theta=None, phi=None,
                save=False, kernel='centered', **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: Symbol with time series data for the injected source term
        :param rec: Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield
        :param kernel: type of discretization, centered or shifted

        :returns: Receiver, wavefield and performance summary
        """

        # Space order needs to be halved in the shifted case to have an
        # overall space_order discretization
        self.space_order = self.space_order // 2 if kernel == 'shifted' \
            else self.space_order
        # Source term is read-only, so re-use the default
        if src is None:
            src = self.source
        # Create a new receiver object to store the result
        if rec is None:
            rec = Receiver(name='rec', grid=self.model.grid,
                           time_range=self.receiver.time_range,
                           coordinates=self.receiver.coordinates.data)

        # Create the forward wavefield if not provided
        if u is None:
            u = TimeFunction(name='u', grid=self.model.grid,
                             save=self.source.nt if save else None,
                             time_order=2, space_order=self.space_order)
        # Create the forward wavefield if not provided
        if v is None:
            v = TimeFunction(name='v', grid=self.model.grid,
                             save=self.source.nt if save else None,
                             time_order=2, space_order=self.space_order)
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
        op = self.op_fwd(kernel, save)

        if len(m.shape) == 2:
            summary = op.apply(src=src, rec=rec, u=u, v=v, m=m, epsilon=epsilon,
                               delta=delta, theta=theta,
                               dt=kwargs.pop('dt', self.dt), **kwargs)
        else:
            summary = op.apply(src=src, rec=rec, u=u, v=v, m=m, epsilon=epsilon,
                               delta=delta, theta=theta, phi=phi,
                               dt=kwargs.pop('dt', self.dt), **kwargs)
        return rec, u, v, summary
