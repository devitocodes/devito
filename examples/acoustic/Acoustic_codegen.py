# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.at_controller import AutoTuner
from examples.acoustic.fwi_operators import *


class Acoustic_cg(object):
    """
    Class to setup the problem for the Acoustic Wave.

    Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, nbpml=40, t_order=2, s_order=2,
                 auto_tuning=False, dse=True, compiler=None):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.src = source
        self.dtype = np.float32
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if t_order == 4:
            self.dt = 1.73 * model.get_critical_dt()
        else:
            self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)

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

        self.damp = DenseData(name="damp", shape=self.model.get_shape_comp(),
                              dtype=self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(self.damp.data, nbpml)

        if len(self.damp.shape) == 2 and self.src.receiver_coords.shape[1] == 3:
            self.src.receiver_coords = np.delete(self.src.receiver_coords, 1, 1)
        if len(self.damp.shape) == 2 and self.data.receiver_coords.shape[1] == 3:
            self.data.receiver_coords = np.delete(self.data.receiver_coords, 1, 1)

        if auto_tuning:  # auto tuning with dummy forward operator
            fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                                 time_order=self.t_order, spc_order=self.s_order,
                                 profile=True, save=False, dse=dse, compiler=compiler)
            self.at = AutoTuner(fw)
            self.at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)

    def Forward(self, save=False, cache_blocking=None,
                auto_tuning=False, dse='advanced', compiler=None, u_ini=None):

        if auto_tuning:
            cache_blocking = self.at.block_size
        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, dse=dse,
                             compiler=compiler, profile=True, u_ini=u_ini)

        if isinstance(fw, StencilKernel):
            fw.apply()
            return None, None, None, None, None
        else:
            u, rec = fw.apply()
            return (rec.data, u, fw.propagator.gflopss,
                    fw.propagator.oi, fw.propagator.timings)

    def Adjoint(self, rec, cache_blocking=None):
        adj = AdjointOperator(self.model, self.damp, self.data, self.src, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking)
        v = adj.apply()[0]
        return v.data

    def Gradient(self, rec, u, cache_blocking=None):
        grad_op = GradientOperator(self.model, self.damp, self.data, rec, u,
                                   time_order=self.t_order, spc_order=self.s_order,
                                   cache_blocking=cache_blocking)
        grad = grad_op.apply()[0]
        return grad.data

    def Born(self, dm, cache_blocking=None):
        born_op = BornOperator(self.model, self.src, self.damp, self.data, dm,
                               time_order=self.t_order, spc_order=self.s_order,
                               cache_blocking=cache_blocking)
        rec = born_op.apply()[0]
        return rec.data

    def run(self):
        print('Starting forward')
        rec, u = self.Forward()

        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2

        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)

        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]
