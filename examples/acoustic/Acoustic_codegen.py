# coding: utf-8
from __future__ import print_function

from examples.acoustic.fwi_operators import *
from devito.at_controller import AutoTuner
from examples.source_type import SourceLike


class Acoustic_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(
            self, model, data, source=None, nbpml=40,
            auto_tuning=False, t_order=2, s_order=2):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.data = data
        self.dtype = np.float32
        # Time step can be \sqrt{3}=1.73 bigger with 4th order
        if t_order == 4:
            self.dt = 1.73 * model.get_critical_dt()
        else:
            self.dt = model.get_critical_dt()
        self.model.nbpml = nbpml
        self.model.set_origin(nbpml)
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
        if len(self.damp.shape) == 2 and srccoord.shape[1] == 3:
            srccoord = np.delete(srccoord, 1, 1)
        if len(self.damp.shape) == 2 and self.data.receiver_coords.shape[1] == 3:
            self.data.receiver_coords = np.delete(self.data.receiver_coords, 1, 1)
        self.src = SourceLike(name="src", npoint=1, nt=data.shape[1],
                              dt=self.dt, h=self.model.get_spacing(),
                              coordinates=srccoord, ndim=len(self.damp.shape),
                              dtype=self.dtype, nbpml=nbpml)
        self.src.data[:] = data.get_source()[:, np.newaxis]

        if auto_tuning:  # auto tuning with dummy forward operator
            fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                                 time_order=self.t_order, spc_order=self.s_order,
                                 save=False, profile=True)
            self.at = AutoTuner(fw)
            self.at.auto_tune_blocks(self.s_order + 1, self.s_order * 4 + 2)

    def Forward(self, save=False, cache_blocking=None,
                auto_tuning=False, cse=True, compiler=None):
        if auto_tuning:
            cache_blocking = self.at.block_size

        fw = ForwardOperator(self.model, self.src, self.damp, self.data,
                             time_order=self.t_order, spc_order=self.s_order,
                             save=save, cache_blocking=cache_blocking, cse=cse,
                             compiler=compiler, profile=True)

        u, rec = fw.apply()
        return rec.data, u, fw.propagator.gflops, fw.propagator.oi, fw.propagator.timings

    def Adjoint(self, rec, cache_blocking=None):
        adj = AdjointOperator(self.model, self.damp, self.data, rec,
                              time_order=self.t_order, spc_order=self.s_order,
                              cache_blocking=cache_blocking)
        v = adj.apply()[0]
        return v.data

    def Gradient(self, rec, u, cache_blocking=None):
        grad_op = GradientOperator(self.model, self.damp, self.data, rec, u,
                                   time_order=self.t_order, spc_order=self.s_order,
                                   cache_blocking=self.at.block_size, profile=True)
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
