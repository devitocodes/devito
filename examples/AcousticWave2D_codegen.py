# coding: utf-8
from __future__ import print_function
import numpy as np
from examples.fwi_operators import *
from devito.interfaces import DenseData


class AcousticWave2D_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, dm_initializer=None, source=None, nbpml=40, t_order=2, s_order=2,
                 cache_blocking=False, auto_tune=False):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.auto_tune = auto_tune
        self.cache_blocking = cache_blocking
        self.data = data
        self.dtype = np.float64
        self.dt = model.get_critical_dt()
        self.h = model.get_spacing()
        self.nbpml = nbpml
        dimensions = self.model.get_shape()
        pad_list = []
        for dim_index in range(len(dimensions)):
            pad_list.append((nbpml, nbpml))
        self.model.vp = np.pad(self.model.vp, tuple(pad_list), 'edge')
        self.data.reinterpolate(self.dt)
        self.nrec, self.nt = self.data.traces.shape
        self.model.set_origin(nbpml)
        self.dm_initializer = dm_initializer
        if source is not None:
            self.source = source.read()
            self.source.reinterpolate(self.dt)
            source_time = self.source.traces[0, :]
            while len(source_time) < self.data.nsamples:
                source_time = np.append(source_time, [0.0])
            self.data.set_source(source_time, self.dt, self.data.source_coords)

        def damp_boundary(damp):
            h = self.h
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.nbpml
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

        self.m = DenseData(name="m", shape=self.model.vp.shape, dtype=self.dtype)
        self.m.data[:] = self.model.vp**(-2)
        self.damp = DenseData(name="damp", shape=self.model.vp.shape, dtype=self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(damp.data)
        self.damp = damp
        src = SourceLike("src", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], len(dimensions), self.dtype, nbpml)
        self.src = src
        rec = SourceLike("rec", self.nrec, self.nt, self.dt, self.h, self.data.receiver_coords, len(dimensions), self.dtype, nbpml)
        src.data[:] = self.data.get_source()[:, np.newaxis]
        self.rec = rec
        u = TimeData("u", m.shape, src.nt, time_order=t_order, save=True, dtype=m.dtype)
        self.u = u
        srca = SourceLike("srca", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], len(dimensions), self.dtype, nbpml)
        self.srca = srca
        dm = DenseData("dm", self.model.vp.shape, self.dtype)
        dm.initializer = self.dm_initializer
        self.dm = dm

    def Forward(self):
        fw = ForwardOperator(self.m, self.src, self.damp, self.rec, self.u, time_order=self.t_order, profile=True,
                             spc_order=self.s_order, auto_tune=self.auto_tune, cache_blocking=self.cache_blocking)
        fw.apply()
        return (self.rec.data, self.u.data)

    def Adjoint(self, rec):
        adj = AdjointOperator(self.m, self.rec, self.damp, self.srca, time_order=self.t_order, spc_order=self.s_order)
        v = adj.apply()[0]
        return (self.srca.data, v)

    def Gradient(self, rec, u):
        grad_op = GradientOperator(self.u, self.m, self.rec, self.damp, time_order=self.t_order, spc_order=self.s_order)
        dt = self.dt
        grad = grad_op.apply()[0]
        return (dt**-2)*grad

    def Born(self):
        born_op = BornOperator(self.dm, self.m, self.src, self.damp, self.rec, time_order=self.t_order, spc_order=self.s_order)
        born_op.apply()
        return self.rec.data

    def run(self):
        print('Starting forward')
        rec, u = self.Forward()
        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2
        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)
        return (f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml])
