import numpy as np
import pytest
from numpy import linalg

from devito import time
from examples.seismic.acoustic.acoustic_example import setup
from examples.seismic import PointSource


@pytest.mark.parametrize('space_order', [4, 8, 12])
@pytest.mark.parametrize('time_order', [2, 4])
@pytest.mark.parametrize('dimensions', [(60, 70), (60, 70, 80)])
@pytest.mark.parametrize('fix_dim', [True, False])
def test_acoustic(dimensions, time_order, space_order, fix_dim):
    solver = setup(dimensions=dimensions, time_order=time_order,
                   space_order=space_order, nbpml=30, tn=750.)
    srca = PointSource(name='srca', ntime=solver.source.nt,
                       coordinates=solver.source.coordinates.data)

    if fix_dim:
        time.size = solver.source.nt
    else:
        time.size = None
    # Run forward and adjoint operators
    rec, _, _ = solver.forward(save=False)
    solver.adjoint(rec=rec, srca=srca)

    # Actual adjoint test
    term1 = np.dot(srca.data.reshape(-1), solver.source.data)
    term2 = linalg.norm(rec.data) ** 2
    print(term1, term2, ("%12.12f") % (term1 - term2), term1 / term2)
    assert np.isclose(term1 / term2, 1.0, atol=0.001)


if __name__ == "__main__":
    test_acoustic(dimensions=(60, 70), space_order=4, time_order=2, fix_dim=False)
