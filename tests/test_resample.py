import numpy as np
from examples.seismic import TimeAxis, RickerSource, demo_model


def test_resample():

    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbpml = 10

    f0 = 0.01
    t0 = 0.0
    tn = 500

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    time_range = TimeAxis(start=t0, stop=tn, step=model.critical_dt)
    src_a = RickerSource(name='src_a', grid=model.grid, f0=f0, time_range=time_range)

    time_range_f = TimeAxis(start=t0, step=time_range.step/8, stop=time_range.stop)
    src_b = RickerSource(name='src_b', grid=model.grid, f0=f0, time_range=time_range_f)

    src_c = src_b.resample(dt=src_a._time_range.step)

    assert src_a.data.size == src_c.data.size

    assert np.isclose(src_a.data, src_c.data).any()
    assert np.isclose(np.linalg.norm(src_a.data - src_c.data), 0,
                      rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    test_resample()
