import numpy as np

from examples.seismic import RickerSource, demo_model


def test_resample():

    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbpml = 10

    f0 = 0.01
    t0 = 0.0
    tn = 250.0
    nt = 10000

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    src_a = RickerSource(name='src_a', grid=model.grid, f0=f0,
                         time=np.linspace(t0, tn, nt, endpoint=False))
    src_b = RickerSource(name='src_b', grid=model.grid, f0=f0,
                         time=np.linspace(t0, tn, nt*13, endpoint=False))

    src_c = src_b.resample(dt=(tn-t0)/nt)

    assert src_a.data.shape == src_c.data.shape

    res = src_a.data - src_c.data

    assert np.linalg.norm(res) < 0.005


if __name__ == "__main__":
    test_resample()
