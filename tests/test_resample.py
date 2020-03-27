import numpy as np

from conftest import skipif
from examples.seismic import TimeAxis, RickerSource, demo_model

pytestmark = skipif(['yask', 'ops'])


def test_resample():

    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbl = 10

    f0 = 0.01
    t0 = 0.0
    tn = 500

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbl=nbl)

    time_range = TimeAxis(start=t0, stop=tn, step=model.critical_dt)
    src_a = RickerSource(name='src_a', grid=model.grid, f0=f0, time_range=time_range)

    time_range_f = TimeAxis(start=t0, step=time_range.step/(10*np.sqrt(2)),
                            stop=time_range.stop)
    src_b = RickerSource(name='src_b', grid=model.grid, f0=f0, time_range=time_range_f)

    # Test resampling specifying dt.
    src_c = src_b.resample(dt=src_a._time_range.step)

    end = min(src_a.data.shape[0], src_c.data.shape[0])

    assert np.allclose(src_a.data[:end], src_c.data[:end])
    assert np.allclose(src_a.data[:end], src_c.data[:end])

    # Text resampling based on num
    src_d = RickerSource(name='src_d', grid=model.grid, f0=f0,
                         time_range=TimeAxis(start=time_range_f.start,
                                             stop=time_range_f.stop,
                                             num=src_a._time_range.num))
    src_e = src_b.resample(num=src_d._time_range.num)

    assert np.isclose(src_d._time_range.step, src_e._time_range.step)
    assert np.isclose(src_d._time_range.stop, src_e._time_range.stop)
    assert src_d._time_range.num == src_e._time_range.num
    assert np.allclose(src_d.data, src_e.data)
    assert np.allclose(src_d.data, src_e.data)


if __name__ == "__main__":
    test_resample()
