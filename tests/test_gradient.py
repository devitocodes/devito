import numpy as np
import pytest
from numpy import linalg

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


class TestGradient(object):
    @pytest.fixture(params=[(70, 80)])
    def acoustic(self, request, time_order, space_order):
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0]*len(dimensions))
        spacing = tuple([10]*len(dimensions))

        # velocity models
        def smooth10(vel):
            out = np.zeros(dimensions)
            out[:] = vel[:]
            for a in range(5, dimensions[-1]-6):
                if len(dimensions) == 2:
                    out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
                else:
                    out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10
            return out

        # True velocity
        true_vp = np.ones(dimensions) + .5
        if len(dimensions) == 2:
            true_vp[:, int(dimensions[1] / 2):] = 2
        else:
            true_vp[:, :, int(dimensions[2] / 2):] = 2
        # Smooth velocity
        initial_vp = smooth10(true_vp)
        dm = true_vp**-2 - initial_vp**-2
        model = IGrid(origin, spacing, true_vp)
        model0 = IGrid(origin, spacing, initial_vp)
        # Define seismic data.
        data = IShot()
        src = IShot()
        f0 = .010
        if time_order == 4:
            dt = 1.73 * model.get_critical_dt()
        else:
            dt = model.get_critical_dt()
        t0 = 0.0
        tn = 750.0
        nt = int(1+(tn-t0)/dt)
        # Set up the source as Ricker wavelet for f0

        def source(t, f0):
            r = (np.pi * f0 * (t - 1./f0))
            return (1-2.*r**2)*np.exp(-r**2)

        # Source geometry
        time_series = np.zeros((nt, 1))
        time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

        location = np.zeros((1, 3))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + 2 * spacing[1]
        if len(dimensions) == 3:
            location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
            location[0, 2] = origin[1] + 2 * spacing[2]
        src.set_receiver_pos(location)
        src.set_shape(nt, 1)
        src.set_traces(time_series)

        receiver_coords = np.zeros((101, 3))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            dimensions[0] * spacing[0], num=101)
        receiver_coords[:, 1] = location[0, 1]
        if len(dimensions) == 3:
            receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
            receiver_coords[:, 2] = location[0, 2]
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 101)
        # Adjoint test
        wave_true = Acoustic_cg(model, data, src, t_order=time_order,
                                s_order=space_order, nbpml=40)
        wave_0 = Acoustic_cg(model0, data, src, t_order=time_order,
                             s_order=space_order, nbpml=40)
        return wave_true, wave_0, dm, initial_vp

    @pytest.fixture(params=[2])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[4])
    def space_order(self, request):
        return request.param

    def test_grad(self, acoustic):
        rec = acoustic[0].Forward()[0]
        rec0, u0, _, _, _ = acoustic[1].Forward(save=True)
        F0 = .5*linalg.norm(rec0 - rec)**2
        gradient = acoustic[1].Gradient(rec0 - rec, u0)
        # Actual Gradient test
        G = np.dot(gradient.reshape(-1), acoustic[1].model.pad(acoustic[2]).reshape(-1))
        # FWI Gradient test
        H = [0.5, 0.25, .125, 0.0625, 0.0312, 0.015625, 0.0078125]
        error1 = np.zeros(7)
        error2 = np.zeros(7)
        for i in range(0, 7):
            acoustic[1].model.set_vp(np.sqrt((acoustic[3]**-2 + H[i] *
                                              acoustic[2])**(-1)))
            d = acoustic[1].Forward()[0]
            error1[i] = np.absolute(.5*linalg.norm(d - rec)**2 - F0)
            error2[i] = np.absolute(.5*linalg.norm(d - rec)**2 - F0 - H[i] * G)

        hh = np.zeros(7)
        for i in range(0, 7):
            hh[i] = H[i] * H[i]

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        print(p1)
        print(p2)
        assert np.isclose(p1[0], 1.0, rtol=0.1)
        assert np.isclose(p2[0], 2.0, rtol=0.1)


if __name__ == "__main__":
    t = TestGradient()
    request = type('', (), {})()
    request.param = (60, 70)
    ac = t.acoustic(request, 2, 4)
    t.test_grad(ac)
