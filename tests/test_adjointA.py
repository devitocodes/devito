import numpy as np
import pytest
from numpy import linalg

from devito import time
from devito.logger import info
from examples.seismic.acoustic.acoustic_example import acoustic_setup as setup
from examples.seismic.acoustic.constant_example import acoustic_setup as setup_c
from examples.seismic import PointSource


@pytest.mark.parametrize('space_order', [4, 8, 12])
@pytest.mark.parametrize('time_order', [2, 4])
@pytest.mark.parametrize('dimensions', [(60, 70), (60, 70, 80)])
@pytest.mark.parametrize('fix_dim', [True, False])
def test_acoustic(dimensions, time_order, space_order, fix_dim):
    spacing = tuple(15. for _ in dimensions)
    solver = setup(dimensions=dimensions, spacing=spacing,
                   time_order=time_order, space_order=space_order,
                   nbpml=10+space_order/2)
    srca = PointSource(name='srca', ntime=solver.source.nt,
                       coordinates=solver.source.coordinates.data)

    # Set fixed ("baked-in") time dimension if requested
    time.size = solver.source.nt if fix_dim else None

    # Run forward and adjoint operators
    rec, _, _ = solver.forward(save=False)
    solver.adjoint(rec=rec, srca=srca)

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    term1 = np.dot(srca.data.reshape(-1), solver.source.data)
    term2 = linalg.norm(rec.data) ** 2
    info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %12.12f, ratio: %f'
         % (term1, term2, term1 - term2, term1 / term2))
    assert np.isclose(term1 / term2, 1.0, atol=0.001)


@pytest.mark.parametrize('dimensions', [(60, 70), (60, 70, 80)])
@pytest.mark.parametrize('fix_dim', [True, False])
def test_acoustic_constant(dimensions, fix_dim):
    solver = setup_c(dimensions=dimensions, time_order=2,
                     space_order=8, nbpml=14)
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
