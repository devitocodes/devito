# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.dimension import Dimension
from devito.interfaces import DenseData, TimeData
from examples.acoustic.fwi_operators import *
from examples.source_type import SourceLike


class Acoustic_cg(object):
    """
    Class to setup the problem for the Acoustic Wave.

    Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, nbpml=40, t_order=2, s_order=2,
                 auto_tuning=False, dse=True, dle='advanced', compiler=None):
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

    # Forward modelling
    def Forward(self, save=False, cache_blocking=None, auto_tuning=False,
                dse='advanced', dle='advanced', compiler=None, u_ini=None):
        """
        Forward modelling
        """
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create source symbol
        p_src = Dimension('p_src', size=nsrc)
        src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.source.receiver_coords)
        src.data[:] = self.source.traces[:]

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.data.receiver_coords)

        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order, save=save,
                     dtype=self.model.dtype)
        u.pad_time = save
        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]

        # Execute operator and return wavefield and receiver data
        fw = ForwardOperator(self.model, u, src, rec, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, dse=dse,
                             dle=dle, compiler=compiler, profile=True, u_ini=u_ini)

        summary = fw.apply(autotune=auto_tuning)
        return rec.data, u, summary.gflopss, summary.oi, summary.timings

    def Adjoint(self, recin, cache_blocking=None, auto_tuning=False,
                dse='advanced', dle='advanced', compiler=None, u_ini=None):
        """
        Adjoint modelling
        """
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create source symbol
        p_src = Dimension('p_src', size=nsrc)
        srca = SourceLike(name="srca", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                          dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                          coordinates=self.source.receiver_coords)

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.data.receiver_coords)
        rec.data[:] = recin[:]

        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)

        # Execute operator and return wavefield and receiver data
        adj = AdjointOperator(self.model, v, srca, rec, self.data,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking, dse=dse,
                              dle=dle, compiler=compiler, profile=True)

        summary = adj.apply(autotune=auto_tuning)
        return srca.data, v, summary.gflopss, summary.oi, summary.timings

    def Gradient(self, recin, u, cache_blocking=None, auto_tuning=False,
                 dse='advanced', dle='advanced', compiler=None):
        """
        Gradient operator (adjoint of Linearized Born modelling, action of
        the Jacobian adjoint on an input data)
        """
        nt, nrec = self.data.shape
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.data.receiver_coords)
        rec.data[:] = recin[:]

        # Gradient symbol
        grad = DenseData(name="grad", shape=self.model.shape_domain,
                         dtype=self.model.dtype)

        # Create the forward wavefield
        v = TimeData(name="v", shape=self.model.shape_domain, time_dim=nt,
                     time_order=2, space_order=self.s_order,
                     dtype=self.model.dtype)

        # Execute operator and return wavefield and receiver data
        gradop = GradientOperator(self.model, v, grad, rec, u, self.data,
                                  time_order=self.t_order, spc_order=self.s_order,
                                  cache_blocking=cache_blocking, dse=dse,
                                  dle=dle, compiler=compiler, profile=True)

        summary = gradop.apply(autotune=auto_tuning)
        return grad.data, summary.gflopss, summary.oi, summary.timings

    def Born(self, dmin, cache_blocking=None, auto_tuning=False,
             dse='advanced', dle='advanced', compiler=None):
        """
        Linearized Born modelling
        """
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.model.shape)
        h = self.model.get_spacing()
        dtype = self.model.dtype
        nbpml = self.model.nbpml

        # Create source symbol
        p_src = Dimension('p_src', size=nsrc)
        src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.source.receiver_coords)
        src.data[:] = self.source.traces[:]

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=self.dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
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
        born = BornOperator(self.model, u, U, src, rec, dm, self.data,
                            time_order=self.t_order, spc_order=self.s_order,
                            cache_blocking=cache_blocking, dse=dse,
                            dle=dle, compiler=compiler, profile=True)

        summary = born.apply(autotune=auto_tuning)
        return rec.data, u, U, summary.gflopss, summary.oi, summary.timings
