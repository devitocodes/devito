from scipy.special import hankel2
import numpy as np
import pytest
from devito import Grid, Function, Eq, SpaceDimension, Constant, Operator, info
from examples.seismic import RickerSource, Receiver, TimeAxis
from examples.seismic.skew_self_adjoint import (default_setup_iso,
                                                SsaIsoAcousticWaveSolver)

# Defaults in global scope
npad = 10
fpeak = 0.010
qmin = 0.1
qmax = 500.0
tmax = 500.0
shapes = [(71, 61), (71, 61, 51)]
dtypes = [np.float64, ]
space_orders = [8, ]


class TestWavesolver(object):

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_linearity_forward_F(self, shape, dtype, so):
        """
        Test the linearity of the forward modeling operator by verifying:
            a F(s) = F(a s)
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)
        src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1,
                           time_range=time_axis)
        src.coordinates.data[:] = src_coords[:]
        a = -1 + 2 * np.random.rand()
        rec1, _, _ = solver.forward(src)
        src.data[:] *= a
        rec2, _, _ = solver.forward(src)
        rec1.data[:] *= a

        # Check receiver wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(rec2.data**2))
        diff = (rec1.data - rec2.data) / rms2
        info("linearity forward F %s (so=%d) rms 1,2,diff; "
             "%+16.10e %+16.10e %+16.10e" %
             (shape, so, np.sqrt(np.mean(rec1.data**2)), np.sqrt(np.mean(rec2.data**2)),
              np.sqrt(np.mean(diff**2))))
        tol = 1.e-12
        assert np.allclose(diff, 0.0, atol=tol)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_linearity_adjoint_F(self, shape, dtype, so):
        """
        Test the linearity of the adjoint modeling operator by verifying:
            a F^T(r) = F^T(a r)
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)
        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src_coords[:]
        rec, _, _ = solver.forward(src0)
        a = -1 + 2 * np.random.rand()
        src1, _, _ = solver.adjoint(rec)
        rec.data[:] = a * rec.data[:]
        src2, _, _ = solver.adjoint(rec)
        src1.data[:] *= a

        # Check adjoint source wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(src2.data**2))
        diff = (src1.data - src2.data) / rms2
        info("linearity adjoint F %s (so=%d) rms 1,2,diff; "
             "%+16.10e %+16.10e %+16.10e" %
             (shape, so, np.sqrt(np.mean(src1.data**2)), np.sqrt(np.mean(src2.data**2)),
              np.sqrt(np.mean(diff**2))))
        tol = 1.e-12
        assert np.allclose(diff, 0.0, atol=tol)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_adjoint_F(self, shape, dtype, so):
        """
        Test the forward modeling operator by verifying for random s, r:
            r . F(s) = F^T(r) . s
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)
        src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src1.coordinates.data[:] = src_coords[:]

        rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                        coordinates=rec_coords)
        rec2, _, _ = solver.forward(src1)
        # flip sign of receiver data for adjoint to make it interesting
        rec1.data[:] = rec2.data[:]
        src2, _, _ = solver.adjoint(rec1)
        sum_s = np.dot(src1.data.reshape(-1), src2.data.reshape(-1))
        sum_r = np.dot(rec1.data.reshape(-1), rec2.data.reshape(-1))
        diff = (sum_s - sum_r) / (sum_s + sum_r)
        info("adjoint F %s (so=%d) sum_s, sum_r, diff; %+16.10e %+16.10e %+16.10e" %
             (shape, so, sum_s, sum_r, diff))
        assert np.isclose(diff, 0., atol=1.e-12)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_linearization_F(self, shape, dtype, so):
        """
        Test the linearization of the forward modeling operator by verifying
        for sequence of h decreasing that the error in the linearization E is
        of second order.

            E = 0.5 || F(m + h   dm) - F(m) - h   J(dm) ||^2

        This is done by fitting a 1st order polynomial to the norms
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)
        src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1,
                           time_range=time_axis)
        src.coordinates.data[:] = src_coords[:]

        # Create Functions for models and perturbation
        m0 = Function(name='m0', grid=v.grid, space_order=so)
        mm = Function(name='mm', grid=v.grid, space_order=so)
        dm = Function(name='dm', grid=v.grid, space_order=so)

        # Background model
        m0.data[:] = 1.5

        # Model perturbation, box of random values centered on middle of model
        dm.data[:] = 0
        size = 5
        ns = 2 * size + 1
        if len(shape) == 2:
            nx2, nz2 = shape[0]//2, shape[1]//2
            dm.data[nx2-size:nx2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns)
        else:
            nx2, ny2, nz2 = shape[0]//2, shape[1]//2, shape[2]//2
            nx, ny, nz = shape
            dm.data[nx2-size:nx2+size, ny2-size:ny2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns, ns)

        # Compute F(m + dm)
        rec0, u0, summary0 = solver.forward(src, v=m0)

        # Compute J(dm)
        rec1, u1, du, summary1 = solver.jacobian_forward(dm, src=src, v=m0)

        # Linearization test via polyfit (see devito/tests/test_gradient.py)
        # Solve F(m + h dm) for sequence of decreasing h
        dh = np.sqrt(2.0)
        h = 0.1
        nstep = 7
        scale = np.empty(nstep)
        norm1 = np.empty(nstep)
        norm2 = np.empty(nstep)
        for kstep in range(nstep):
            h = h / dh
            mm.data[:] = m0.data + h * dm.data
            rec2, _, _ = solver.forward(src, v=mm)
            scale[kstep] = h
            norm1[kstep] = 0.5 * np.linalg.norm(rec2.data - rec0.data)**2
            norm2[kstep] = 0.5 * np.linalg.norm(rec2.data - rec0.data - h * rec1.data)**2

        # Fit 1st order polynomials to the error sequences
        #   Assert the 1st order error has slope dh^2
        #   Assert the 2nd order error has slope dh^4
        p1 = np.polyfit(np.log10(scale), np.log10(norm1), 1)
        p2 = np.polyfit(np.log10(scale), np.log10(norm2), 1)
        info("linearization F %s (so=%d) 1st (%.1f) = %.4f, 2nd (%.1f) = %.4f" %
             (shape, so, dh**2, p1[0], dh**4, p2[0]))

        # we only really care the 2nd order err is valid, not so much the 1st order error
        assert np.isclose(p1[0], dh**2, rtol=0.25)
        assert np.isclose(p2[0], dh**4, rtol=0.10)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_linearity_forward_J(self, shape, dtype, so):
        """
        Test the linearity of the forward Jacobian of the forward modeling operator
        by verifying
            a J(dm) = J(a dm)
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        nt = time_axis.num

        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)

        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src_coords[:]

        m0 = Function(name='m0', grid=v.grid, space_order=so)
        m1 = Function(name='m1', grid=v.grid, space_order=so)
        m0.data[:] = 1.5

        # Model perturbation, box of random values centered on middle of model
        m1.data[:] = 0
        size = 5
        ns = 2 * size + 1
        if len(shape) == 2:
            nx2, nz2 = shape[0]//2, shape[1]//2
            m1.data[nx2-size:nx2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns)
        else:
            nx2, ny2, nz2 = shape[0]//2, shape[1]//2, shape[2]//2
            nx, ny, nz = shape
            m1.data[nx2-size:nx2+size, ny2-size:ny2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns, ns)

        a = np.random.rand()
        rec1, _, _, _ = solver.jacobian_forward(m1, src0, v=m0, save=nt)
        rec1.data[:] = a * rec1.data[:]
        m1.data[:] = a * m1.data[:]
        rec2, _, _, _ = solver.jacobian_forward(m1, src0, v=m0, save=nt)

        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(rec2.data**2))
        diff = (rec1.data - rec2.data) / rms2
        info("linearity forward J %s (so=%d) rms 1,2,diff; "
             "%+16.10e %+16.10e %+16.10e" %
             (shape, so, np.sqrt(np.mean(rec1.data**2)), np.sqrt(np.mean(rec2.data**2)),
              np.sqrt(np.mean(diff**2))))
        tol = 1.e-12
        assert np.allclose(diff, 0.0, atol=tol)

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_linearity_adjoint_J(self, shape, dtype, so):
        """
        Test the linearity of the adjoint Jacobian of the forward modeling operator
        by verifying
            a J^T(dr) = J^T(a dr)
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        nt = time_axis.num

        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)

        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src_coords[:]

        m0 = Function(name='m0', grid=v.grid, space_order=so)
        m1 = Function(name='m1', grid=v.grid, space_order=so)
        m0.data[:] = 1.5

        # Model perturbation, box of random values centered on middle of model
        m1.data[:] = 0
        size = 5
        ns = 2 * size + 1
        if len(shape) == 2:
            nx2, nz2 = shape[0]//2, shape[1]//2
            m1.data[nx2-size:nx2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns)
        else:
            nx2, ny2, nz2 = shape[0]//2, shape[1]//2, shape[2]//2
            nx, ny, nz = shape
            m1.data[nx2-size:nx2+size, ny2-size:ny2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns, ns)

        a = np.random.rand()
        rec0, u0, _ = solver.forward(src0, v=m0, save=nt)
        dm1, _, _, _ = solver.jacobian_adjoint(rec0, u0, v=m0, save=nt)
        dm1.data[:] = a * dm1.data[:]
        rec0.data[:] = a * rec0.data[:]
        dm2, _, _, _ = solver.jacobian_adjoint(rec0, u0, v=m0, save=nt)

        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(dm2.data**2))
        diff = (dm1.data - dm2.data) / rms2
        info("linearity adjoint J %s (so=%d) rms 1,2,diff; "
             "%+16.10e %+16.10e %+16.10e" %
             (shape, so, np.sqrt(np.mean(dm1.data**2)), np.sqrt(np.mean(dm2.data**2)),
              np.sqrt(np.mean(diff**2))))

    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_adjoint_J(self, shape, dtype, so):
        """
        Test the Jacobian of the forward modeling operator by verifying for
        'random' dm, dr:
            dr . J(dm) = J^T(dr) . dm
        """
        np.random.seed(0)
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            default_setup_iso(npad, shape, dtype, tmax=tmax)
        nt = time_axis.num

        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=so)

        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src_coords[:]

        m0 = Function(name='m0', grid=v.grid, space_order=so)
        dm1 = Function(name='dm1', grid=v.grid, space_order=so)
        m0.data[:] = 1.5

        # Model perturbation, box of random values centered on middle of model
        dm1.data[:] = 0
        size = 5
        ns = 2 * size + 1
        if len(shape) == 2:
            nx2, nz2 = shape[0]//2, shape[1]//2
            dm1.data[nx2-size:nx2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns)
        else:
            nx2, ny2, nz2 = shape[0]//2, shape[1]//2, shape[2]//2
            nx, ny, nz = shape
            dm1.data[nx2-size:nx2+size, ny2-size:ny2+size, nz2-size:nz2+size] = \
                -1 + 2 * np.random.rand(ns, ns, ns)

        # Data perturbation
        rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                        coordinates=rec_coords)
        nt, nr = rec1.data.shape
        rec1.data[:] = np.random.rand(nt, nr)

        # Nonlinear modeling
        rec0, u0, _ = solver.forward(src0, v=m0, save=nt)

        # Linearized modeling
        rec2, _, _, _ = solver.jacobian_forward(dm1, src0, v=m0, save=nt)
        dm2, _, _, _ = solver.jacobian_adjoint(rec1, u0, v=m0, save=nt)

        sum_m = np.dot(dm1.data.reshape(-1), dm2.data.reshape(-1))
        sum_d = np.dot(rec1.data.reshape(-1), rec2.data.reshape(-1))
        diff = (sum_m - sum_d) / (sum_m + sum_d)
        info("adjoint J %s (so=%d) sum_m, sum_d, diff; %16.10e %+16.10e %+16.10e" %
             (shape, so, sum_m, sum_d, diff))
        assert np.isclose(diff, 0., atol=1.e-11)

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_derivative_skew_symmetry(self, dtype, so):
        """
        We ensure that the first derivatives constructed with calls like
            f.dx(x0=x+0.5*x.spacing)
        Are skew (anti) symmetric. See the notebook ssa_01_iso_implementation.ipynb
        for more details.
        """
        np.random.seed(0)
        n = 101
        d = 1.0
        shape = (n, )
        origin = (0., )
        extent = (d * (n-1), )

        # Initialize Devito grid and Functions for input(f1,g1) and output(f2,g2)
        grid1d = Grid(shape=shape, extent=extent, origin=origin, dtype=dtype)
        x = grid1d.dimensions[0]
        f1 = Function(name='f1', grid=grid1d, space_order=8)
        f2 = Function(name='f2', grid=grid1d, space_order=8)
        g1 = Function(name='g1', grid=grid1d, space_order=8)
        g2 = Function(name='g2', grid=grid1d, space_order=8)

        # Fill f1 and g1 with random values in [-1,+1]
        f1.data[:] = -1 + 2 * np.random.rand(n,)
        g1.data[:] = -1 + 2 * np.random.rand(n,)

        # Equation defining: [f2 = forward 1/2 cell shift derivative applied to f1]
        equation_f2 = Eq(f2, f1.dx(x0=x+0.5*x.spacing))

        # Equation defining: [g2 = backward 1/2 cell shift derivative applied to g1]
        equation_g2 = Eq(g2, g1.dx(x0=x-0.5*x.spacing))

        # Define an Operator to implement these equations and execute
        op = Operator([equation_f2, equation_g2])
        op()

        # Compute the dot products and the relative error
        f1g2 = np.dot(f1.data, g2.data)
        g1f2 = np.dot(g1.data, f2.data)
        diff = (f1g2 + g1f2) / (f1g2 - g1f2)

        info("skew symmetry (so=%d) -- f1g2, g1f2, diff; %+16.10e %+16.10e %+16.10e" %
             (so, f1g2, g1f2, diff))
        assert np.isclose(diff, 0., atol=1.e-12)

    @pytest.mark.parametrize('dtype', dtypes)
    @pytest.mark.parametrize('so', space_orders)
    def test_analytic_comparison_2d(self, dtype, so):
        """
        Wnsure that the farfield response from the propagator matches analytic reponse
        in a wholespace.
        """
        # Setup time / frequency
        nt = 1001
        dt = 0.1
        tmin = 0.0
        tmax = dt * (nt - 1)
        fpeak = 0.090
        t0w = 1.0 / fpeak
        omega = 2.0 * np.pi * fpeak
        time_axis = TimeAxis(start=tmin, stop=tmax, step=dt)

        # Model
        space_order = 8
        npad = 50
        dx, dz = 0.5, 0.5
        nx, nz = 801 + 2 * npad, 801 + 2 * npad
        shape = (nx, nz)
        extent = (dx * (nx - 1), dz * (nz - 1))
        origin = (0.0 - dx * npad, 0.0 - dz * npad)
        dtype = np.float64
        qmin = 0.1
        qmax = 100000
        v0 = 1.5
        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=dx))
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=dz))
        grid = Grid(extent=extent, shape=shape, origin=origin,
                    dimensions=(x, z), dtype=dtype)
        b = Function(name='b', grid=grid, space_order=space_order)
        v = Function(name='v', grid=grid, space_order=space_order)
        b.data[:] = 1.0
        v.data[:] = v0

        # Source and reciver coordinates
        src_coords = np.empty((1, 2), dtype=dtype)
        rec_coords = np.empty((1, 2), dtype=dtype)
        src_coords[:, 0] = origin[0] + extent[0] / 2
        src_coords[:, 1] = origin[1] + extent[1] / 2
        rec_coords[:, 0] = origin[0] + extent[0] / 2 + 60
        rec_coords[:, 1] = origin[1] + extent[1] / 2 + 60

        # Solver setup
        solver = SsaIsoAcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                          src_coords, rec_coords, time_axis,
                                          space_order=space_order)

        # Source function
        src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1,
                           time_range=time_axis, t0w=t0w)
        src.coordinates.data[:] = src_coords[:]

        # Numerical solution
        recNum, uNum, _ = solver.forward(src)

        # Analytic response
        def analytic_response():
            """
            Computes analytic solution of 2D acoustic wave-equation with Ricker wavelet
            peak frequency fpeak, temporal padding 20x per the accuracy notebook:
            examples/seismic/acoustic/accuracy.ipynb
                u(r,t) = 1/(2 pi) sum[ -i pi H_0^2(k,r) q(w) e^{i w t} dw
                where:
                    r = sqrt{(x_s - x_r)^2 + (z_s - z_r)^2}
                    w = 2 pi f
                    q(w) = Fourier transform of Ricker source wavelet
                    H_0^2(k,r) Hankel function of the second kind
                    k = w/v (wavenumber)
            """
            sx, sz = src_coords[0, :]
            rx, rz = rec_coords[0, :]
            ntpad = 20 * (nt - 1) + 1
            tmaxpad = dt * (ntpad - 1)
            time_axis_pad = TimeAxis(start=tmin, stop=tmaxpad, step=dt)
            srcpad = RickerSource(name='srcpad', grid=v.grid, f0=fpeak, npoint=1,
                                  time_range=time_axis_pad, t0w=t0w)
            nf = int(ntpad / 2 + 1)
            df = 1.0 / tmaxpad
            faxis = df * np.arange(nf)

            # Take the Fourier transform of the source time-function
            R = np.fft.fft(srcpad.wavelet[:])
            R = R[0:nf]
            nf = len(R)

            # Compute the Hankel function and multiply by the source spectrum
            U_a = np.zeros((nf), dtype=complex)
            for a in range(1, nf - 1):
                w = 2 * np.pi * faxis[a]
                r = np.sqrt((rx - sx)**2 + (rz - sz)**2)
                U_a[a] = -1j * np.pi * hankel2(0.0, w * r / v0) * R[a]

            # Do inverse fft on 0:dt:T and you have analytical solution
            U_t = 1.0/(2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], ntpad))

            # Note that the analytic solution is scaled by dx^2 to convert to pressure
            return (np.real(U_t) * (dx**2)), srcpad

        uAnaPad, srcpad = analytic_response()
        uAna = uAnaPad[0:nt]

        # Compute RMS and difference
        diff = (recNum.data - uAna)
        nrms = np.max(np.abs(recNum.data))
        arms = np.max(np.abs(uAna))
        drms = np.max(np.abs(diff))

        info("Maximum absolute numerical,analytic,diff; %+12.6e %+12.6e %+12.6e" %
             (nrms, arms, drms))

        # This isnt a very strict tolerance ...
        tol = 0.1
        assert np.allclose(diff, 0.0, atol=tol)
