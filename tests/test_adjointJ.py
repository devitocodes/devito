from examples.Acoustic_codegen import Acoustic_cg
import numpy as np
from numpy import linalg
from examples.containers import IGrid, IShot
import pytest


class Test_AdjointJ(object):
    @pytest.fixture(params=[(60, 70), (50, 60, 70)])
    def Acoustic(self, request, time_order, space_order):
        model = IGrid()
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0]*len(dimensions))
        spacing = tuple([15]*len(dimensions))

        # True velocity
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
        model.create_model(origin, spacing, initial_vp)

        # Define seismic data.
        data = IShot()

        f0 = .010
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
        receiver_coords = np.zeros((50, len(dimensions)))
        receiver_coords[:, 0] = np.linspace(50, origin[0] + dimensions[0]*spacing[0] - 50, num=50)
        receiver_coords[:, -1] = location[-1]
        if len(dimensions) == 3:
            receiver_coords[:, -1] = location[1]
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 50)
        # Adjoint test
        wave_true = Acoustic_cg(model, data, None, None, t_order=time_order, s_order=space_order, nbpml=10)
        return wave_true, dm_pad

    @pytest.fixture(params=[2])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[4, 6, 8, 10])
    def space_order(self, request):
        return request.param

    def test_adjointJ(self, Acoustic):
        rec0, u0 = Acoustic[0].Forward()
        Im2 = Acoustic[0].Gradient(rec0, u0)
        du1 = Acoustic[0].Born(Acoustic[1])
        # Actual adjoint test
        print(linalg.norm(rec0), linalg.norm(du1))
        term1 = np.dot(rec0.reshape(-1), du1.reshape(-1))
        term2 = np.dot(Im2.reshape(-1), Acoustic[1].reshape(-1))
        print(term1, term2, term1 - term2, term1 / term2)
        assert np.isclose(term1 / term2, 1.0, atol=0.001)

if __name__ == "__main__":
    t = Test_AdjointJ()
    request = type('', (), {})()
    request.param = (60, 70)
    ac = t.Acoustic(request, 2, 2)
    t.test_adjointJ(ac)
