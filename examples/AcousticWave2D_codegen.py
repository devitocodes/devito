# coding: utf-8
from __future__ import print_function
from sympy import Matrix
import numpy as np
from examples.fwi_operators import *
from devito.interfaces import DenseData, PointData


class AcousticWave2D_cg:

    def __init__(self, model, data, dm_initializer, source=None, nbpml=40, t_order=2, s_order=2):
        self.model = model
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
            if len(source_time) < self.data.nsamples:
                source_time = np.append(source_time, [0.0])
            self.data.set_source(source_time, self.dt, self.data.source_coords)

        def damp_boundary(damp):
            h = self.h
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40 * h)
            nbpml = self.nbpml
            for i in range(nbpml):
                pos = np.abs((nbpml-i)/nbpml)
                val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
                damp[i, :] += val
                damp[-i, :] += val
                damp[:, i] += val
                damp[:, -i] += val

        m = DenseData("m", self.model.vp.shape, self.dtype)
        m.data[:] = self.model.vp**(-2)
        self.m = m
        damp = DenseData("damp", self.model.vp.shape, self.dtype)
        damp.initializer = damp_boundary
        self.damp = damp
        src = SourceLike("src", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], self.dtype)
        self.src = src
        rec = SourceLike("rec", self.nrec, self.nt, self.dt, self.h, self.data.receiver_coords, self.dtype)
        src.data[:] = self.data.get_source()[:, np.newaxis]
        self.rec = rec
        u = TimeData("u", m.shape, src.nt, time_order=t_order, save=True, dtype=m.dtype)
        self.u = u
        srca = SourceLike("srca", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], self.dtype)
        self.srca = srca
        dm = DenseData("dm", self.model.vp.shape, self.dtype)
        dm.initializer = self.dm_initializer
        self.dm = dm
        self._forward_stencil = ForwardOperator(m, src, damp, rec, u, time_order=t_order, spc_order=s_order)
        self._adjoint_stencil = AdjointOperator(m, rec, damp, srca, time_order=t_order, spc_order=s_order)
        self._gradient_stencil = GradientOperator(u, m, rec, damp, time_order=t_order, spc_order=s_order)
        self._born_stencil = BornOperator(dm, m, src, damp, rec, time_order=t_order, spc_order=s_order)

    def Forward(self):
        self._forward_stencil.apply()
        return (self.rec.data, self.u.data)

    def Adjoint(self, rec):
        v = self._adjoint_stencil.apply()[0]
        return (self.srca.data, v)

    def Gradient(self, rec, u):
        dt = self.dt
        grad = self._gradient_stencil.apply()[0]
        return (dt**-2)*grad

    def Born(self):
        self._born_stencil.apply()
        return self.rec.data

    def run(self):
        print('Starting forward')
        rec, u = self.Forward()
        res = rec - np.transpose(self.data.traces)
        f = 0.5*np.linalg.norm(res)**2
        print('Residual is ', f, 'starting gradient')
        g = self.Gradient(res, u)
        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, name, npoint, nt, dt, h, data, dtype):
        self.orig_data = data
        self.dt = dt
        self.h = h
        super(SourceLike, self).__init__(name, npoint, nt, dtype)
        x1, z1, x2, z2 = symbols('x1, z1, x2, z2')
        A = Matrix([[1, x1, z1, x1*z1],
                    [1, x1, z2, x1*z2],
                    [1, x2, z1, x2*z1],
                    [1, x2, z2, x2*z2]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)

        # Form expression for interpolant weights on reference cell.
        self.rs = symbols('rx, rz')
        rx, rz = self.rs
        p = Matrix([[1],
                    [rx],
                    [rz],
                    [rx*rz]])

        self.bs = A.inv().T.dot(p)

    def point2grid(self, x, z):
        # In: s - Magnitude of the source
        #     x, z - Position of the source
        # Returns: (i, k) - Grid coordinate at top left of grid cell.
        #          (s11, s12, s21, s22) - source values at coordinates
        #          (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        i = int(x/self.h)
        k = int(z/self.h)

        x = x - i*self.h
        z = z - k*self.h

        s11 = b11.subs(((rx, x), (rz, z))).evalf()
        s12 = b12.subs(((rx, x), (rz, z))).evalf()
        s21 = b21.subs(((rx, x), (rz, z))).evalf()
        s22 = b22.subs(((rx, x), (rz, z))).evalf()
        return (i, k), (s11, s12, s21, s22)

    # Interpolate onto receiver point.
    def grid2point(self, u, x, z):
        rx, rz = self.rs
        b11, b12, b21, b22 = self.bs
        i = int(x/self.h)
        j = int(z/self.h)

        x = x - i*self.h
        z = z - j*self.h

        return (b11.subs(((rx, x), (rz, z))) * u[t, i, j] +
                b12.subs(((rx, x), (rz, z))) * u[t, i, j+1] +
                b21.subs(((rx, x), (rz, z))) * u[t, i+1, j] +
                b22.subs(((rx, x), (rz, z))) * u[t, i+1, j+1])

    def read(self, u):
        eqs = []
        for i in range(self.npoints):
            eqs.append(Eq(self[t, i], self.grid2point(u, self.orig_data[i, 0],
                                                      self.orig_data[i, 2])))
        return eqs

    def add(self, m, u):
        assignments = []
        dt = self.dt
        for j in range(self.npoints):
            add = self.point2grid(self.orig_data[j, 0],
                                  self.orig_data[j, 2])
            (i, k) = add[0]
            assignments.append(Eq(u[t, i, k], u[t, i, k]+self[t, j]*dt*dt/m[i, k]*add[1][0]))
            assignments.append(Eq(u[t, i, k+1], u[t, i, k+1]+self[t, j]*dt*dt/m[i, k]*add[1][1]))
            assignments.append(Eq(u[t, i+1, k], u[t, i+1, k]+self[t, j]*dt*dt/m[i, k]*add[1][2]))
            assignments.append(Eq(u[t, i+1, k+1], u[t, i+1, k+1]+self[t, j]*dt*dt/m[i, k]*add[1][3]))
        filtered = [x for x in assignments if isinstance(x, Eq)]
        return filtered
