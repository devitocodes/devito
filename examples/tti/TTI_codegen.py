# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.at_controller import AutoTuner
from examples.tti.tti_operators import *


class TTI_cg:
    """ Class to setup the problem for the anisotropic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, t_order=2, s_order=2, nbpml=40):
        self.model = model
        self.t_order = t_order
        self.s_order = int(s_order/2)
        self.data = data
        self.src = source
        self.dtype = np.float32
        self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)

        def damp_boundary(damp, nbp):
            h = self.model.get_spacing()
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            num_dim = len(damp.shape)
            for i in range(nbp):
                pos = np.abs((nbp-i)/float(nbp))
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
        damp_boundary(self.damp.data, nbpml)

        if len(self.damp.shape) == 2 and self.src.receiver_coords.shape[1] == 3:
            self.src.receiver_coords = np.delete(self.src.receiver_coords, 1, 1)
        if len(self.damp.shape) == 2 and self.data.receiver_coords.shape[1] == 3:
            self.data.receiver_coords = np.delete(self.data.receiver_coords, 1, 1)

    def Forward(self, save=False, dse='advanced', auto_tuning=False,
                cache_blocking=None, compiler=None):
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             profile=True, save=save, cache_blocking=cache_blocking,
                             dse=dse, compiler=compiler)

        if auto_tuning:
            fw_new = ForwardOperator(self.model, self.src, self.damp, self.data,
                                     time_order=self.t_order, spc_order=self.s_order,
                                     profile=True, save=save, dse=dse, compiler=compiler)

            at = AutoTuner(fw_new)
            at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)
            fw.propagator.cache_blocking = at.block_size

        u, v, rec = fw.apply()
        return (rec.data, u, v,
                fw.propagator.gflopss, fw.propagator.oi, fw.propagator.timings)
