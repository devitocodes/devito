import numpy as np
from cached_property import cached_property

from devito.interfaces import DenseData, TimeData
from examples.acoustic.fwi_operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)
from examples.seismic import PointSource, Receiver


class AcousticWaveSolver(object):
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
        self.space_order = space_order

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

    @property
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=False, source=self.source,
                               receiver=self.receiver, time_order=self.time_order,
                               space_order=self.space_order, **self._kwargs)

    @property
    def op_grad(self):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=False, source=self.source,
                                receiver=self.receiver, time_order=self.time_order,
                                space_order=self.space_order, **self._kwargs)

    @property
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=False, source=self.source,
                            receiver=self.receiver, time_order=self.time_order,
                            space_order=self.space_order, **self._kwargs)

    def Forward(self, save=False, u_ini=None, **kwargs):
        """
        Forward modelling
        """
        # Create source and receiver symbols
        src = PointSource(name='src', data=self.source.data,
                          coordinates=self.source.coordinates.data)
        rec = Receiver(name='rec', ntime=self.receiver.nt,
                       coordinates=self.receiver.coordinates.data)

        # Create the forward wavefield
        u = TimeData(name='u', shape=self.model.shape_domain, save=save,
                     time_dim=self.source.nt, time_order=self.time_order,
                     space_order=self.space_order, dtype=self.model.dtype)
        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]

        # Execute operator and return wavefield and receiver data
        if save:
            summary = self.op_fwd_save.apply(src=src, rec=rec, u=u, **kwargs)
        else:
            summary = self.op_fwd.apply(src=src, rec=rec, u=u, **kwargs)
        return rec.data, u, summary

    def Adjoint(self, recin, u_ini=None, **kwargs):
        """
        Adjoint modelling
        """
        # Create a new adjoint source and receiver symbol
        srca = PointSource(name='srca', ntime=self.source.nt,
                           coordinates=self.source.coordinates.data)
        rec = Receiver(name='rec', data=recin,
                       coordinates=self.receiver.coordinates.data)

        # Create the adjoint wavefield
        v = TimeData(name='v', shape=self.model.shape_domain, save=False,
                     time_order=2, space_order=self.space_order,
                     dtype=self.model.dtype)

        summary = self.op_adj.apply(srca=srca, rec=rec, v=v, **kwargs)
        return srca.data, v, summary

    def Gradient(self, recin, u, **kwargs):
        """
        Gradient operator (adjoint of Linearized Born modelling, action of
        the Jacobian adjoint on an input data)
        """
        # Create receiver symbol
        rec = Receiver(name='rec', data=recin,
                       coordinates=self.receiver.coordinates.data)

        # Gradient symbol
        grad = DenseData(name='grad', shape=self.model.shape_domain,
                         dtype=self.model.dtype)

        # Create the forward wavefield
        v = TimeData(name='v', shape=self.model.shape_domain,
                     time_dim=self.source.nt, time_order=self.time_order,
                     space_order=self.space_order, dtype=self.model.dtype)

        summary = self.op_grad.apply(rec=rec, grad=grad, v=v, u=u, **kwargs)
        return grad.data, summary

    def Born(self, dmin, **kwargs):
        """
        Linearized Born modelling
        """
        # Create source and receiver symbols
        src = PointSource(name='src', data=self.source.data,
                          coordinates=self.source.coordinates.data)
        rec = Receiver(name='rec', ntime=self.receiver.nt,
                       coordinates=self.receiver.coordinates.data)

        # Create the forward wavefield
        u = TimeData(name='u', shape=self.model.shape_domain, save=False,
                     time_order=self.time_order, space_order=self.space_order,
                     dtype=self.model.dtype)
        U = TimeData(name='U', shape=self.model.shape_domain,
                     time_order=2, space_order=self.space_order,
                     dtype=self.model.dtype)
        if isinstance(dmin, np.ndarray):
            dm = DenseData(name='dm', shape=self.model.shape_domain,
                           dtype=self.model.dtype)
            dm.data[:] = self.model.pad(dmin)
        else:
            dm = dmin
        # Execute operator and return wavefield and receiver data
        summary = self.op_born.apply(u=u, U=U, src=src, rec=rec, dm=dm, **kwargs)
        return rec.data, u, U, summary
