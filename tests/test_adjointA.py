import numpy as np
import pytest
from numpy import linalg

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot


class TestAdjointA(object):
    @pytest.fixture(params=[(60, 70), (60, 70, 80)])
    def acoustic(self, request, time_order, space_order):
        dimensions = request.param
        # dimensions are (x,z) and (x, y, z)
        origin = tuple([0.0]*len(dimensions))
        spacing = tuple([15.0]*len(dimensions))

        # True velocity
        true_vp = np.ones(dimensions) + .5
        if len(dimensions) == 2:
            true_vp[:, int(dimensions[0] / 3):dimensions[0]] = 2.5
        else:
            true_vp[:, :, int(dimensions[0] / 3):dimensions[0]] = 2.5
        model = IGrid(origin, spacing, true_vp)
        # Define seismic data.
        data = IShot()
        src = IShot()

        f0 = .010
        if time_order == 4:
            dt = 1.73 * model.get_critical_dt()
        else:
            dt = model.get_critical_dt()
        t0 = 0.0
        tn = 500.0
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
        receiver_coords[:, 0] = np.linspace(50, 950, num=101)
        receiver_coords[:, 1] = location[0, 1]
        if len(dimensions) == 3:
            receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
            receiver_coords[:, 2] = location[0, 2]
        data.set_receiver_pos(receiver_coords)
        data.set_shape(nt, 101)

        # Adjoint test
        wave_true = Acoustic_cg(model, data, src, t_order=time_order, s_order=space_order,
                                nbpml=10)
        return wave_true

    @pytest.fixture(params=[2, 4])
    def time_order(self, request):
        return request.param

    @pytest.fixture(params=[4, 8, 10])
    def space_order(self, request):
        return request.param

    @pytest.fixture
    def forward(self, acoustic):
        rec, u, _, _, _ = acoustic.Forward(save=False)
        return rec

    def test_adjoint(self, acoustic, forward):
        rec = forward
        srca = acoustic.Adjoint(rec)
        # Actual adjoint test
        term1 = np.dot(srca.reshape(-1), acoustic.src.traces)
        term2 = linalg.norm(rec)**2
        print(term1, term2, term1 - term2, term1 / term2)
        assert np.isclose(term1 / term2, 1.0, atol=0.001)


if __name__ == "__main__":
    t = TestAdjointA()
    request = type('', (), {})()
    request.param = (60, 70)
    ac = t.acoustic(request, 2, 4)
    fw = t.forward(ac)
    t.test_adjoint(ac, fw)
