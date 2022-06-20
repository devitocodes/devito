from examples.seismic import demo_model
import numpy as np
import pytest


@pytest.mark.parametrize('shape', [(51, 51), (16, 16, 16)])
def test_model_updt(shape):

    # Layered model (tti)
    tti_model = demo_model('layers-tti', shape=shape, spacing=[20. for _ in shape],
                           space_order=4, nbl=10)
    tpl1_set = set(tti_model.physical_parameters)

    # Layered model (isotropic)
    iso_model = demo_model('layers-isotropic', shape=shape, spacing=[20. for _ in shape],
                           space_order=4, nbl=10)
    tpl2_set = set(iso_model.physical_parameters)

    # Physical parameters in either set but not in the intersection.
    diff_phys_par = tuple(tpl1_set ^ tpl2_set)

    # Convert iso model in tti model
    slices = tuple(slice(tti_model.nbl, -tti_model.nbl) for _ in range(tti_model.dim))
    for i in diff_phys_par:
        iso_model.update(i, getattr(tti_model, i).data[slices])
        assert np.array_equal(getattr(iso_model, i).data, getattr(tti_model, i).data)
