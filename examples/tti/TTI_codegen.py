# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.at_controller import AutoTuner
from devito.dimension import Dimension, time
from devito.interfaces import TimeData
from examples.tti.tti_operators import *
from examples.source_type import SourceLike


class TTI_cg:
    """ Class to setup the problem for the anisotropic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, t_order=2, s_order=2, nbpml=40):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.source = source
        self.dtype = np.float32
        self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml

        # Fill the dampening field with nbp points in the absorbing layer
        def damp_boundary(damp, nbp):
            h = self.model.get_spacing()
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            num_dim = len(damp.shape)
            for i in range(nbp):
                pos = np.abs((nbp-i+1)/float(nbp))
                val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
                if num_dim == 2:
                    damp[i, :] += val
                    damp[-(i + 1), :] += val
                    damp[:, i] += val
                    damp[:, -(i + 1)] += val
                else:
                    damp[i, :, :] += val
                    damp[-(i + 1), :, :] += val
                    damp[:, i, :] += val
                    damp[:, -(i + 1), :] += val
                    damp[:, :, i] += val
                    damp[:, :, -(i + 1)] += val

        self.damp = DenseData(name="damp", shape=self.model.shape_pml,
                              dtype=self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(self.damp.data, nbpml)

    def Forward(self, save=False, dse='advanced', dle='advanced', auto_tuning=False,
                cache_blocking=None, compiler=None, u_ini=None, legacy=True):
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.damp.shape)
        dt = self.dt
        h = self.model.get_spacing()
        dtype = self.damp.dtype
        nbpml = self.model.nbpml

        # uses space_order/2 for the first derivatives to
        # have spc_order second derivatives for consistency
        # with the acoustic kernel
        u = TimeData(name="u", shape=self.model.shape_pml,
                     time_dim=nt, time_order=self.t_order,
                     space_order=self.s_order/2,
                     save=save, dtype=dtype)
        v = TimeData(name="v", shape=self.model.shape_pml,
                     time_dim=nt, time_order=self.t_order,
                     space_order=self.s_order/2,
                     save=save, dtype=dtype)

        u.pad_time = save
        v.pad_time = save

        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]
            v.data[0:3, :] = u_ini[:]

        # Create source symbol
        p_src = Dimension('p_src', size=nsrc)
        src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                         dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.source.receiver_coords)
        src.data[:] = .5 * self.source.traces[:]

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.data.receiver_coords)

        # Create forward operator
        fw = ForwardOperator(self.model, u, v, src, rec, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             profile=True, save=save, cache_blocking=cache_blocking,
                             dse=dse, dle=dle, compiler=compiler, legacy=legacy)

        if auto_tuning and legacy:
            # uses space_order/2 for the first derivatives to
            # have spc_order second derivatives for consistency
            # with the acoustic kernel
            u = TimeData(name="u", shape=self.model.shape_pml,
                         time_dim=nt, time_order=self.t_order,
                         space_order=self.s_order/2,
                         save=save, dtype=dtype)
            v = TimeData(name="v", shape=self.model.shape_pml,
                         time_dim=nt, time_order=self.t_order,
                         space_order=self.s_order/2,
                         save=save, dtype=dtype)

            u.pad_time = save
            v.pad_time = save

            # Create source symbol
            src_new = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                                 dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                                 coordinates=self.source.receiver_coords)
            src_new.data[:] = .5 * self.source.traces[:]

            # Create receiver symbol
            rec_new = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                                 dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                                 coordinates=self.data.receiver_coords)

            # Ceate tuning operator
            fw_new = ForwardOperator(self.model, u, v, src_new, rec_new, self.damp,
                                     self.data, time_order=self.t_order,
                                     spc_order=self.s_order, profile=True, save=save,
                                     dse=dse, compiler=compiler)

            at = AutoTuner(fw_new)
            at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)
            fw.propagator.cache_blocking = at.block_size

        if legacy:
            fw.apply()
            return (rec.data, u.data, v.data,
                    fw.propagator.gflopss, fw.propagator.oi, fw.propagator.timings)
        else:
            summary = fw.apply(autotune=auto_tuning)
            return rec.data, u.data, v.data, summary.gflopss, summary.oi, summary.timings
