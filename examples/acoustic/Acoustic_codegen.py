# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.interfaces import DenseData, TimeData
from examples.acoustic.fwi_operators import *
from examples.seismic import PointSource, Receiver


class Acoustic_cg(object):
    """
    Class to setup the problem for the Acoustic Wave.

    Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, t_order=2, s_order=2):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.source = source
        self.dtype = np.float32
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        self.dt = self.model.critical_dt
        if self.t_order == 4:
            self.dt *= 1.73

    def Forward(self, save=False, cache_blocking=None, u_ini=None, **kwargs):
        """
        Forward modelling
        """
        nt, nrec = self.data.shape

        # Create source and receiver symbol
        src = PointSource(name='src', data=self.source.traces,
                          coordinates=self.source.receiver_coords)
        rec = Receiver(name='rec', ntime=nt,
                       coordinates=self.data.receiver_coords)

        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order, save=save,
                     dtype=self.model.dtype)
        u.pad_time = save
        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]

        # Execute operator and return wavefield and receiver data
        fw = ForwardOperator(self.model, u, src, rec,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, u_ini=u_ini, **kwargs)

        summary = fw.apply(**kwargs)
        return rec.data, u, summary.gflopss, summary.oi, summary.timings

    def Adjoint(self, recin, u_ini=None, **kwargs):
        """
        Adjoint modelling
        """
        nt, nrec = self.data.shape

        # Create a new adjoint source and receiver symbol
        srca = PointSource(name='srca', ntime=nt,
                           coordinates=self.source.receiver_coords)
        rec = Receiver(name='rec', data=recin,
                       coordinates=self.data.receiver_coords)

        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)

        # Execute operator and return wavefield and receiver data
        adj = AdjointOperator(self.model, v, srca, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              **kwargs)

        summary = adj.apply(**kwargs)
        return srca.data, v, summary.gflopss, summary.oi, summary.timings

    def Gradient(self, recin, u, **kwargs):
        """
        Gradient operator (adjoint of Linearized Born modelling, action of
        the Jacobian adjoint on an input data)
        """
        nt, nrec = self.data.shape

        # Create receiver symbol
        rec = Receiver(name='rec', data=recin,
                       coordinates=self.data.receiver_coords)

        # Gradient symbol
        grad = DenseData(name="grad", shape=self.model.shape_domain,
                         dtype=self.model.dtype)

        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)

        # Execute operator and return wavefield and receiver data
        gradop = GradientOperator(self.model, v, grad, rec, u,
                                  time_order=self.t_order, spc_order=self.s_order,
                                  **kwargs)

        summary = gradop.apply(**kwargs)
        return grad.data, summary.gflopss, summary.oi, summary.timings

    def Born(self, dmin, **kwargs):
        """
        Linearized Born modelling
        """
        nt, nrec = self.data.shape

        # Create source and receiver symbols
        src = PointSource(name='src', data=self.source.traces,
                          coordinates=self.source.receiver_coords)
        rec = Receiver(name='rec', ntime=nt,
                       coordinates=self.data.receiver_coords)

        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)
        U = TimeData(name="U", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)
        if isinstance(dmin, np.ndarray):
            dm = DenseData(name="dm", shape=self.model.shape_domain,
                           dtype=self.model.dtype)
            dm.data[:] = self.model.pad(dmin)
        else:
            dm = dmin
        # Execute operator and return wavefield and receiver data
        born = BornOperator(self.model, u, U, src, rec, dm,
                            time_order=self.t_order, spc_order=self.s_order,
                            **kwargs)

        summary = born.apply(**kwargs)
        return rec.data, u, U, summary.gflopss, summary.oi, summary.timings
