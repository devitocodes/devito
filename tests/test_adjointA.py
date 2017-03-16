import numpy as np
import pytest
from numpy import linalg

from devito.logger import error

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IShot
from examples.seismic import Model


@pytest.mark.parametrize('space_order', [4, 8, 12])
@pytest.mark.parametrize('time_order', [2, 4])
@pytest.mark.parametrize('dimensions', [(60, 70), (60, 70, 80)])
def test_acoustic(dimensions, time_order, space_order):
    nbpml = 10

    if len(dimensions) == 2:
        # Dimensions in 2D are (x, z)
        origin = (0., 0.)
        spacing = (15., 15.)

        # True velocity
        true_vp = np.ones(dimensions) + .5
        true_vp[:, int(dimensions[0] / 3):dimensions[0]] = 2.5

        # Source location
        location = np.zeros((1, 2))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + 2 * spacing[1]

        # Receiver coordinates
        receiver_coords = np.zeros((dimensions[0], 2))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            (dimensions[0]-1) * spacing[0], num=dimensions[0])
        receiver_coords[:, 1] = location[0, 1]

    elif len(dimensions) == 3:
        # Dimensions in 3D are (x, y, z)
        origin = (0., 0., 0.)
        spacing = (15., 15., 15.)

        # True velocity
        true_vp = np.ones(dimensions) + .5
        true_vp[:, :, int(dimensions[0] / 3):dimensions[0]] = 2.5

        # Source location
        location = np.zeros((1, 3))
        location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
        location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        location[0, 2] = origin[1] + 2 * spacing[2]

        # Receiver coordinates
        receiver_coords = np.zeros((dimensions[0], 3))
        receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                            (dimensions[0] - 1) * spacing[0], num=dimensions[0])
        receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
        receiver_coords[:, 2] = location[0, 2]

    else:
        error("Unknown dimension size. `dimensions` parameter"
              "must be a tuple of either size 2 or 3.")

    # Define seismic data
    model = Model(origin, spacing, true_vp, nbpml=nbpml)
    data = IShot()
    src = IShot()

    f0 = .010
    dt = model.critical_dt
    if time_order == 4:
        dt *= 1.73
    t0 = 0.0
    tn = 500.0
    nt = int(1+(tn-t0)/dt)

    # Set up the source as Ricker wavelet for f0
    def source(t, f0):
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    # Source geometry
    time_series = np.zeros((nt, 1), dtype=np.float32)
    time_series[:, 0] = source(np.linspace(t0, tn, nt), f0)

    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, dimensions[0])

    # Adjoint test
    acoustic = Acoustic_cg(model, data, src, t_order=time_order,
                           s_order=space_order, nbpml=nbpml)
    rec, _, _, _, _ = acoustic.Forward(save=False, legacy=False)
    srca, _, _, _, _ = acoustic.Adjoint(rec, legacy=False)
    # Actual adjoint test
    term1 = np.dot(srca.reshape(-1), time_series)
    term2 = linalg.norm(rec) ** 2
    print(np.finfo(np.float32).eps)
    print(term1, term2, ("%12.12f") % (term1 - term2), term1 / term2)
    assert np.isclose(term1 / term2, 1.0, atol=0.001)


if __name__ == "__main__":
    test_acoustic(dimensions=(60, 70), time_order=2, space_order=4)
