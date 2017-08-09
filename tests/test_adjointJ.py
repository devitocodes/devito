import numpy as np
import pytest
from numpy import linalg

from devito.logger import info
from examples.seismic.acoustic.acoustic_example import acoustic_setup as setup


@pytest.mark.parametrize('space_order', [4, 8, 12])
@pytest.mark.parametrize('dimensions', [(60, 70), (40, 50, 30)])
def test_acousticJ(dimensions, space_order):
    spacing = tuple(15. for _ in dimensions)
    solver = setup(dimensions=dimensions, spacing=spacing,
                   space_order=space_order, nbpml=10+space_order/2)
    initial_vp = np.ones(solver.model.shape_domain) + .5
    m0 = np.float32(initial_vp**-2)
    dm = np.float32(solver.model.m.data - m0)

    # Compute the full wavefield
    _, u0, _ = solver.forward(save=True, m=m0)

    du, _, _, _ = solver.born(dm, m=m0)
    im, _ = solver.gradient(du, u0, m=m0)

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    term1 = np.dot(im.data.reshape(-1), dm.reshape(-1))
    term2 = linalg.norm(du.data)**2
    info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %12.12f, ratio: %f'
         % (term1, term2, term1 - term2, term1 / term2))
    assert np.isclose(term1 / term2, 1.0, atol=0.001)


if __name__ == "__main__":
    test_acousticJ(dimensions=(60, 70), space_order=4)
