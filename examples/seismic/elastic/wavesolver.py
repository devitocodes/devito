from devito import TimeFunction, memoized_meth, error
from examples.seismic import Receiver
from examples.seismic.elastic.operators import ForwardOperator


class ElasticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    :param model: Physical model with domain parameters
    :param source: Sparse point symbol providing the injected wave
    :param receiver: Sparse point symbol describing an array of receivers
    :param space_order: Order of the spatial stencil discretisation (default: 4)

    Note: This is an experimental staggered grid elastic modeling kernel.
    Only 2D supported
    """
    def __init__(self, model, source, receiver, space_order=4, **kwargs):
        self.model = model
        self.source = source
        self.receiver = receiver

        self.space_order = space_order
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        # Cache compiler options
        self._kwargs = kwargs
        if model.grid.dim != 2:
            error("This is an experimental staggered grid elastic modeling kernel." +
                  "Only 2D supported")

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, source=self.source,
                               receiver=self.receiver,
                               space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, vp=None, vs=None, rho=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        :param src: (Optional) Symbol with time series data for the injected source term
        :param rec: (Optional) Symbol to store interpolated receiver data
        :param u: (Optional) Symbol to store the computed wavefield
        :param m: (Optional) Symbol for the time-constant square slowness
        :param save: Option to store the entire (unrolled) wavefield

        :returns: Receiver, wavefield and performance summary
        """
        # Source term is read-only, so re-use the default
        if src is None:
            src = self.source
        # Create a new receiver object to store the result
        rec1 = Receiver(name='rec1', grid=self.model.grid,
                        time_range=self.receiver.time_range,
                        coordinates=self.receiver.coordinates.data)
        rec2 = Receiver(name='rec2', grid=self.model.grid,
                        time_range=self.receiver.time_range,
                        coordinates=self.receiver.coordinates.data)

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        vx = TimeFunction(name='vx', grid=self.model.grid, staggered=(0, 1, 0),
                          save=src.nt if save else None,
                          time_order=2, space_order=self.space_order)
        vz = TimeFunction(name='vz', grid=self.model.grid, staggered=(0, 0, 1),
                          save=src.nt if save else None,
                          time_order=2, space_order=self.space_order)
        txx = TimeFunction(name='txx', grid=self.model.grid,
                           save=src.nt if save else None,
                           time_order=2, space_order=self.space_order)
        tzz = TimeFunction(name='tzz', grid=self.model.grid,
                           save=src.nt if save else None,
                           time_order=2, space_order=self.space_order)
        txz = TimeFunction(name='txz', grid=self.model.grid, staggered=(0, 1, 1),
                           save=src.nt if save else None,
                           time_order=2, space_order=self.space_order)
        # Pick m from model unless explicitly provided
        if vp is None:
            vp = vp or self.model.vp
        if vs is None:
            vs = vs or self.model.vs
        if rho is None:
            rho = rho or self.model.rho
        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec1=rec1, vx=vx, vz=vz, txx=txx,
                                          tzz=tzz, txz=txz, vp=vp, vs=vs, rho=rho,
                                          rec2=rec2, dt=kwargs.pop('dt', self.dt),
                                          **kwargs)
        return rec1, rec2, vx, vz, txx, tzz, txz, summary
