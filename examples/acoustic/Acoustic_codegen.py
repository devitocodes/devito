# coding: utf-8
from __future__ import print_function

import numpy as np

from devito.at_controller import AutoTuner
from devito.dimension import Dimension, time
from examples.acoustic.fwi_operators import *


class Acoustic_cg(object):
    """
    Class to setup the problem for the Acoustic Wave.

    Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, source, nbpml=40, t_order=2, s_order=2,
                 auto_tuning=False, dse=True, dle='advanced', compiler=None,
                 legacy=True):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.source = source
        self.dtype = np.float32
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if t_order == 4:
            self.dt = 1.73 * model.get_critical_dt()
        else:
            self.dt = model.get_critical_dt()

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

        if auto_tuning and legacy:  # auto tuning with dummy forward operator
            nt, nrec = self.data.shape
            nsrc = self.source.shape[1]
            ndim = len(self.damp.shape)
            h = self.model.get_spacing()
            dtype = self.damp.dtype
            nbpml = self.model.nbpml
            dt = self.model.get_critical_dt()
            if self.t_order == 4:
                dt *= 1.73

            # Create source symbol
            p_src = Dimension('p_src', size=nsrc)
            src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                             dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                             coordinates=self.source.receiver_coords)
            src.data[:] = self.source.traces[:]

            # Create receiver symbol
            p_rec = Dimension('p_rec', size=nrec)
            rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                             dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                             coordinates=self.data.receiver_coords)

            # Create the forward wavefield
            u = TimeData(name="u", shape=self.model.shape_pml, time_dim=nt,
                         time_order=2, space_order=self.s_order, save=False,
                         dtype=self.damp.dtype)
            fw = ForwardOperator(self.model, u, src, rec, self.damp, self.data,
                                 time_order=self.t_order, spc_order=self.s_order,
                                 profile=True, save=False, dse=dse, compiler=compiler)
            self.at = AutoTuner(fw)
            self.at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)

    def Forward(self, save=False, cache_blocking=None, auto_tuning=False,
                dse='advanced', dle='advanced', compiler=None, u_ini=None, legacy=True):
        nt, nrec = self.data.shape
        nsrc = self.source.shape[1]
        ndim = len(self.damp.shape)
        h = self.model.get_spacing()
        dtype = self.damp.dtype
        nbpml = self.model.nbpml
        dt = self.model.get_critical_dt()
        if self.t_order == 4:
            dt *= 1.73

        # Create source symbol
        p_src = Dimension('p_src', size=nsrc)
        src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                         dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.source.receiver_coords)
        src.data[:] = self.source.traces[:]

        # Create receiver symbol
        p_rec = Dimension('p_rec', size=nrec)
        rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                         dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                         coordinates=self.data.receiver_coords)

        if legacy and auto_tuning:
            cache_blocking = self.at.block_size

        # Create the forward wavefield
        u = TimeData(name="u", shape=self.model.shape_pml, time_dim=nt,
                     time_order=2, space_order=self.s_order, save=save,
                     dtype=self.damp.dtype)
        u.pad_time = save
        if u_ini is not None:
            u.data[0:3, :] = u_ini[:]

        # Execute operator and return wavefield and receiver data
        fw = ForwardOperator(self.model, u, src, rec, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, dse=dse,
                             dle=dle, compiler=compiler, profile=True, u_ini=u_ini,
                             legacy=legacy)

        if legacy:
            fw.apply()
            return (rec.data, u, fw.propagator.gflopss,
                    fw.propagator.oi, fw.propagator.timings)
        else:
            summary = fw.apply(autotune=auto_tuning)
            return rec.data, u, summary.gflopss, summary.oi, summary.timings

    def Adjoint(self, rec, cache_blocking=None):
        adj = AdjointOperator(self.model, self.damp, self.data, self.source, rec,
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
        born_op = BornOperator(self.model, self.source, self.damp, self.data, dm,
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
