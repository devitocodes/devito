import numpy as np
from examples.seismic import demo_model
from examples.seismic.source import TimeAxis, RickerSource
from devito.dimension import TimeDimension, SteppingDimension
from devito.base import Constant

import cloudpickle as pickle


def test_pickle():

    shape = (50, 50, 50)
    spacing = [10. for _ in shape]
    nbpml = 10

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    # Test Model pickling
    pkl_model = pickle.dumps(model)
    new_model = pickle.loads(pkl_model)
    assert np.isclose(np.linalg.norm(model.vp-new_model.vp), 0)

    f0 = .010
    dt = model.critical_dt
    t0 = 0.0
    tn = 350.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Test TimeAxis pickling
    pkl_time_range = pickle.dumps(time_range)
    new_time_range = pickle.loads(pkl_time_range)
    assert np.isclose(np.linalg.norm(time_range.time_values),
                      np.linalg.norm(new_time_range.time_values))

    # Test Class Constant pickling
    pkl_origin = pickle.dumps(model.grid.origin)
    new_origin = pickle.loads(pkl_origin)

    for a, b in zip(model.grid.origin, new_origin):
        assert a.compare(b) == 0

    # Test Class TimeDimension pickling
    time_dim = TimeDimension(name='time', spacing=Constant(name='dt', dtype=np.float32))
    pkl_time_dim = pickle.dumps(time_dim)
    new_time_dim = pickle.loads(pkl_time_dim)
    assert time_dim.spacing._value == new_time_dim.spacing._value

    # Test Class SteppingDimension
    stepping_dim = SteppingDimension(name='t', parent=time_dim)
    pkl_stepping_dim = pickle.dumps(stepping_dim)
    new_stepping_dim = pickle.loads(pkl_stepping_dim)
    assert stepping_dim.is_Time == new_stepping_dim.is_Time

    # Test Grid pickling
    pkl_grid = pickle.dumps(model.grid)
    new_grid = pickle.loads(pkl_grid)
    assert model.grid.shape == new_grid.shape

    assert model.grid.extent == new_grid.extent
    assert model.grid.shape == new_grid.shape
    for a, b in zip(model.grid.dimensions, new_grid.dimensions):
        assert a.compare(b) == 0

    ricker = RickerSource(name='src', grid=model.grid, f0=f0, time_range=time_range)

    pkl_ricker = pickle.dumps(ricker)
    new_ricker = pickle.loads(pkl_ricker)
    assert np.isclose(np.linalg.norm(ricker.data.flatten()),
                      np.linalg.norm(new_ricker.data.flatten()))


if __name__ == "__main__":
    test_pickle()
