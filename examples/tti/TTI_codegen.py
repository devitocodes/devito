# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.at_controller import AutoTuner
from examples.source_type import SourceLike
from examples.tti.tti_operators import *


class TTI_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source=None, t_order=2, s_order=2, nbpml=40,
                 save=False):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.dtype = np.float32
        self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)
        self.data.reinterpolate(self.dt)

        if source is not None:
            self.source = source.read()
            self.source.reinterpolate(self.dt)
            source_time = self.source.traces[0, :]
            while len(source_time) < self.data.nsamples:
                source_time = np.append(source_time, [0.0])
            self.data.set_source(source_time, self.dt, self.data.source_coords)

        def damp_boundary(damp):
            h = self.model.get_spacing()
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.model.nbpml
            num_dim = len(damp.shape)
            for i in range(nbpml):
                pos = np.abs((nbpml-i)/float(nbpml))
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

        self.damp = DenseData(name="damp", shape=self.model.get_shape_comp(),
                              dtype=self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(self.damp.data)
        srccoord = np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :]
        self.src = SourceLike(name="src", npoint=1, nt=data.shape[1],
                              dt=self.dt, h=self.model.get_spacing(),
                              coordinates=srccoord, ndim=len(self.damp.shape),
                              dtype=self.dtype, nbpml=nbpml)
        self.src.data[:] = data.get_source()[:, np.newaxis]

    def Forward(self, save=False, cse=True, auto_tuning=False,
                cache_blocking=None, compiler=None):
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             profile=True, save=False, cache_blocking=cache_blocking,
                             cse=cse, compiler=compiler)

        if auto_tuning:
            fw_new = ForwardOperator(self.model, self.src, self.damp, self.data,
                                     time_order=self.t_order, spc_order=self.s_order,
                                     profile=True, save=False, cse=cse, compiler=compiler)

            at = AutoTuner(fw_new)
            at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)
            fw.propagator.cache_blocking = at.block_size

        u, v, rec = fw.apply()
        return (rec.data, u.data, v.data,
                fw.propagator.gflopss, fw.propagator.oi, fw.propagator.timings)

    def Adjoint(self, rec, cache_blocking=None):
        adj = AdjointOperator(self.model, self.damp, self.data, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking)
        srca = adj.apply()[0]
        return srca.data
