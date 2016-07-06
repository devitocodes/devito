# coding: utf-8
from __future__ import print_function
from sympy import Matrix, symbols
import numpy as np
from examples.fwi_operators import *
from devito.interfaces import DenseData, PointData


class AcousticWave2D_cg:
    """ Class to setup the problem for the Acoustic Wave
        Note: s_order must always be greater than t_order
    """
    def __init__(self, model, data, dm_initializer=None, source=None, nbpml=40, t_order=2, s_order=2,
                 auto_tune=False):
        self.model = model
        self.t_order = t_order
        self.s_order = s_order
        self.auto_tune = auto_tune
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

        m = DenseData("m", self.model.vp.shape, self.dtype)
        m.data[:] = self.model.vp**(-2)
        self.m = m
        damp = DenseData("damp", self.model.vp.shape, self.dtype)
        # Initialize damp by calling the function that can precompute damping
        damp_boundary(damp.data)
        self.damp = damp
        src = SourceLike("src", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], len(dimensions), self.dtype, nbpml)
        self.src = src
        rec = SourceLike("rec", self.nrec, self.nt, self.dt, self.h, self.data.receiver_coords, len(dimensions), self.dtype, nbpml)
        src.data[:] = self.data.get_source()[:, np.newaxis]
        self.rec = rec
        u = TimeData("u", m.shape, src.nt, time_order=t_order, save=False, dtype=m.dtype)
        self.u = u
        srca = SourceLike("srca", 1, self.nt, self.dt, self.h, np.array(self.data.source_coords, dtype=self.dtype)[np.newaxis, :], len(dimensions), self.dtype, nbpml)
        self.srca = srca
        dm = DenseData("dm", self.model.vp.shape, self.dtype)
        dm.initializer = self.dm_initializer
        self.dm = dm

    def Forward(self):
        fw = ForwardOperator(self.m, self.src, self.damp, self.rec, self.u, time_order=self.t_order,
                             spc_order=self.s_order, cache_blocking=True, profile=True, auto_tune=self.auto_tune)
        fw.apply()
        return self.rec.data, self.u.data

    def Adjoint(self, rec):
        adj = AdjointOperator(self.m, self.rec, self.damp, self.srca, time_order=self.t_order, spc_order=self.s_order)
        v = adj.apply()[0]
        return self.srca.data, v

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
        return f, g[self.nbpml:-self.nbpml, self.nbpml:-self.nbpml]


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, name, npoint, nt, dt, h, data, ndim, dtype, nbpml):
        self.orig_data = data
        self.dt = dt
        self.h = h
        self.ndim = ndim
        self.nbpml = nbpml
        super(SourceLike, self).__init__(name, npoint, nt, dtype)
        x1, y1, z1, x2, y2, z2 = symbols('x1, y1, z1, x2, y2, z2')
        if ndim == 2:
            A = Matrix([[1, x1, z1, x1*z1],
                        [1, x1, z2, x1*z2],
                        [1, x2, z1, x2*z1],
                        [1, x2, z2, x2*z2]])
            self.increments = (0, 0), (0, 1), (1, 0), (1, 1)
            self.rs = symbols('rx, rz')
            rx, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [rz],
                        [rx*rz]])
        else:
            A = Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                        [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                        [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                        [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                        [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                        [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                        [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                        [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])
            self.increments = (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)
            self.rs = symbols('rx, ry, rz')
            rx, ry, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [ry],
                        [rz],
                        [rx*ry],
                        [rx*rz],
                        [ry*rz],
                        [rx*ry*rz]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (y1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (y2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)
        self.bs = A.inv().T.dot(p)

    def point2grid(self, pt_coords):
        # In: s - Magnitude of the source
        #     x, z - Position of the source
        # Returns: (i, k) - Grid coordinate at top left of grid cell.
        #          (s11, s12, s21, s22) - source values at coordinates
        #          (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        if self.ndim == 2:
            rx, rz = self.rs
        else:
            rx, ry, rz = self.rs
        x, y, z = pt_coords
        i = int(x/self.h)
        k = int(z/self.h)
        coords = (i + self.nbpml, k + self.nbpml)
        subs = []
        x = x - i*self.h
        subs.append((rx, x))
        if self.ndim == 3:
            j = int(y/self.h)
            y = y - j*self.h
            subs.append((ry, y))
            coords = (i + self.nbpml, j + self.nbpml, k + self.nbpml)
        z = z - k*self.h
        subs.append((rz, z))
        s = [b.subs(subs).evalf() for b in self.bs]
        return coords, tuple(s)

    # Interpolate onto receiver point.
    def grid2point(self, u, pt_coords):
        if self.ndim == 2:
            rx, rz = self.rs
        else:
            rx, ry, rz = self.rs
        x, y, z = pt_coords
        i = int(x/self.h)
        k = int(z/self.h)

        x = x - i*self.h
        z = z - k*self.h

        subs = []
        subs.append((rx, x))

        if self.ndim == 3:
            j = int(y/self.h)
            y = y - j*self.h
            subs.append((ry, y))
        subs.append((rz, z))
        if self.ndim == 2:
            return sum([b.subs(subs) * u[t, i+inc[0]+self.nbpml, k+inc[1]+self.nbpml] for inc, b in zip(self.increments, self.bs)])
        else:
            return sum([b.subs(subs) * u[t, i+inc[0]+self.nbpml, j+inc[1]+self.nbpml, k+inc[2]+self.nbpml] for inc, b in zip(self.increments, self.bs)])

    def read(self, u):
        eqs = []
        for i in range(self.npoints):
            eqs.append(Eq(self[t, i], self.grid2point(u, self.orig_data[i, :])))
        return eqs

    def add(self, m, u):
        assignments = []
        dt = self.dt
        for j in range(self.npoints):
            add = self.point2grid(self.orig_data[j, :])
            coords = add[0]
            s = add[1]
            assignments += [Eq(u[tuple([t] + [coords[i] + inc[i] for i in range(self.ndim)])],
                               u[tuple([t] + [coords[i] + inc[i] for i in range(self.ndim)])] + self[t, j]*dt*dt/m[coords]*w) for w, inc in zip(s, self.increments)]
        filtered = [x for x in assignments if isinstance(x, Eq)]
        return filtered
