import numpy as np
import pytest
from numpy import linalg

from examples.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


class Test_AdjointA(object):
    @pytest.fixture(params=[(60, 70), (50, 60, 70)])
    def Acoustic(self, request, time_order, space_order):
        model = IGrid()
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0]*len(dimensions))
        spacing = tuple([15]*len(dimensions))

        # True velocity
        true_vp = np.ones(dimensions) + 2.0
        if len(dimensions) == 2:
            true_vp[:, int(dimensions[0] / 2):dimensions[0]] = 4.5
        else:
            true_vp[:, :, int(dimensions[0] / 2):dimensions[0]] = 4.5
        model.create_model(origin, spacing, true_vp)
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
        wave_true = Acoustic_cg(model, data, t_order=time_order, s_order=space_order, nbpml=10)
        return wave_true

    @pytest.fixture(params=[2])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[2, 4, 6, 8, 10])
    def space_order(self, request):
        return request.param

    @pytest.fixture
    def forward(self, Acoustic):
        rec, u = Acoustic.Forward()
        return rec

    def test_adjoint(self, Acoustic, forward):
        rec = forward
        print(linalg.norm(rec))
        srca = Acoustic.Adjoint(rec)
        print(linalg.norm(srca))
        nt = srca.shape[0]
        # Actual adjoint test
        term1 = 0
        for ti in range(0, nt):
            term1 = term1 + srca[ti] * Acoustic.data.get_source(ti)
        term2 = linalg.norm(rec)**2
        print(term1, term2, term1 - term2, term1 / term2)
        assert np.isclose(term1 / term2, 1.0, atol=0.001)

if __name__ == "__main__":
    t = Test_AdjointA()
    request = type('', (), {})()
    request.param = (60, 70, 80)
    ac = t.Acoustic(request, 2, 12)
    fw = t.forward(ac)
    t.test_adjoint(ac, fw)
