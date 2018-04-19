import numpy as np
from examples.seismic import demo_model
from examples.seismic.source import TimeAxis
import cloudpickle as pickle


def test_pickle():

    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbpml = 10

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=1., vp_bottom=2.,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    pkl_model = pickle.dumps(model)
    new_model = pickle.loads(pkl_model)

    assert np.isclose(np.linalg.norm(model.vp-new_model.vp), 0)

    time = TimeAxis(start=0, stop=1, num=10)
    pkl_time = pickle.dumps(time)
    new_time = pickle.loads(pkl_time)

    assert np.isclose(np.linalg.norm(time.time_values),
                      np.linalg.norm(new_time.time_values))


if __name__ == "__main__":
    test_pickle()
