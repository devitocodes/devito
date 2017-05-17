# coding: utf-8
from __future__ import print_function

from devito.interfaces import TimeData
from examples.tti.tti_operators import *
from examples.seismic import PointSource, Receiver


class TTI_cg:
    """ Class to setup the problem for the anisotropic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, t_order=2, s_order=2):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.source = source
        self.dt = model.critical_dt

    def Forward(self, save=False, u_ini=None, **kwargs):
        nt, nrec = self.data.shape

        # uses space_order/2 for the first derivatives to
        # have spc_order second derivatives for consistency
        # with the acoustic kernel
        u = TimeData(name="u", shape=self.model.shape_domain,
                     time_dim=nt, time_order=self.t_order,
                     space_order=self.s_order/2,
                     save=save, dtype=self.model.dtype)
        v = TimeData(name="v", shape=self.model.shape_domain,
                     time_dim=nt, time_order=self.t_order,
                     space_order=self.s_order/2,
                     save=save, dtype=self.model.dtype)

        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]
            v.data[0:3, :] = u_ini[:]

        # Create source and receiver symbol
        src = PointSource(name='src', data=self.source.traces,
                          coordinates=self.source.receiver_coords)
        rec = Receiver(name='rec', ntime=nt,
                       coordinates=self.data.receiver_coords)

        # Create forward operator
        fw = ForwardOperator(self.model, u, v, src, rec, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, **kwargs)

        summary = fw.apply(**kwargs)
        return rec.data, u.data, v.data, summary.gflopss, summary.oi, summary.timings
