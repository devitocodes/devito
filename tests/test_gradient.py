from examples.Acoustic_codegen import Acoustic_cg
import numpy as np
from numpy import linalg
from examples.containers import IGrid, IShot
import pytest


class Test_Gradient(object):
    @pytest.fixture(params=[(70, 80), (60, 70, 80)])
    def Acoustic(self, request, time_order, space_order):
        model = IGrid()
        model0 = IGrid()
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0]*len(dimensions))
        spacing = tuple([15]*len(dimensions))

        # velociity models
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
            true_vp[:, int(dimensions[1] / 3):] = 2.5
        else:
            true_vp[:, :, int(dimensions[2] / 3):] = 2.5
        # Smooth velocity
        initial_vp = smooth10(true_vp)
        dm = true_vp**-2 - initial_vp**-2
        nbpml = 10
        if len(dimensions) == 2:
            pad = ((nbpml, nbpml), (nbpml, nbpml))
        else:
            pad = ((nbpml, nbpml), (nbpml, nbpml), (nbpml, nbpml))
        dm_pad = np.pad(dm, pad, 'edge')
        initial_vp_pad = np.pad(initial_vp, pad, 'edge')
        model.create_model(origin, spacing, true_vp)
        model0.create_model(origin, spacing, initial_vp)
        # Define seismic data.
        data = IShot()
        data0 = IShot()
        f0 = .007
        dt = model.get_critical_dt()
        t0 = 0.0
        tn = 500.0
        nt = int(1+(tn-t0)/dt)
        # Set up the source as Ricker wavelet for f0

        def source(t, f0):
            r = (np.pi * f0 * (t - 1./f0))
            return (1-2.*r**2)*np.exp(-r**2)

        time_series = source(np.linspace(t0, tn, nt), f0)
        location = (origin[0] + dimensions[0] * spacing[0] * 0.5,
                    origin[-1] + 2 * spacing[-1])
        if len(dimensions) == 3:
            location = (location[0], origin[1] + dimensions[1] * spacing[1] * 0.5, location[1])
        data.set_source(time_series, dt, location)
        data0.set_source(time_series, dt, location)
        receiver_coords = np.zeros((50, len(dimensions)))
        receiver_coords[:, 0] = np.linspace(50, origin[0] + dimensions[0]*spacing[0] - 50, num=50)
        receiver_coords[:, -1] = location[-1]
        if len(dimensions) == 3:
            receiver_coords[:, 1] = location[1]
        data.set_receiver_pos(receiver_coords)
        data0.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 50)
        data0.set_shape(nt, 50)
        # Adjoint test
        wave_true = Acoustic_cg(model, data, None, None, t_order=time_order, s_order=space_order, nbpml=10)
        wave_0 = Acoustic_cg(model0, data0, None, None, t_order=time_order, s_order=space_order, nbpml=10)
        return wave_true, wave_0, dm_pad, initial_vp_pad

    @pytest.fixture(params=[2])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[2, 4, 6, 8, 10])
    def space_order(self, request):
        return request.param

    def test_Grad(self, Acoustic):
        rec = Acoustic[0].Forward()[0]
        rec0, u0 = Acoustic[1].Forward()
        F0 = .5*linalg.norm(rec0 - rec)**2
        gradient = Acoustic[1].Gradient(rec0 - rec, u0)
        # Actual Gradient test
        G = np.dot(gradient.reshape(-1), Acoustic[2].reshape(-1))
        # FWI Gradient test
        # print(F0, G)
        H = [1, 0.1, 0.01, .001, 0.0001, 0.00001, 0.000001]
        error1 = np.zeros((7))
        error2 = np.zeros((7))
        for i in range(0, 7):
            Acoustic[1].m.data[:] = (Acoustic[3]**-2 + H[i] * Acoustic[2])
            d = Acoustic[1].Forward()[0]
            error1[i] = np.absolute(.5*linalg.norm(d - rec)**2 - F0)
            error2[i] = np.absolute(.5*linalg.norm(d - rec)**2 - F0 - H[i] * G)
            # print(F0,.5*linalg.norm(d - rec)**2, error1[i], H[i] *G, error2[i])
            # print('For h = ', H[i], '\nFirst order errors is : ', error1[i],
            #       '\nSecond order errors is ', error2[i])

        hh = np.zeros((7))
        for i in range(0, 7):
            hh[i] = H[i] * H[i]

        # Test slope of the  tests
        p1 = np.polyfit(np.log10(H), np.log10(error1), 1)
        p2 = np.polyfit(np.log10(H), np.log10(error2), 1)
        print(p1)
        print(p2)
        assert np.isclose(p1[0], 1.0, rtol=0.05)
        assert np.isclose(p2[0], 2.0, rtol=0.05)

if __name__ == "__main__":
    t = Test_Gradient()
    request = type('', (), {})()
    request.param = (60, 70, 80)
    ac = t.Acoustic(request, 2, 2)
    t.test_Grad(ac)
