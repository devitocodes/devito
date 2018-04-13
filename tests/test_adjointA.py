import numpy as np
import pytest
from numpy import linalg
from conftest import skipif_yask

from devito.logger import info
from examples.seismic import demo_model, TimeAxis, RickerSource, Receiver
from examples.seismic.acoustic import AcousticWaveSolver


presets = {
    'constant': {'preset': 'constant-isotropic'},
    'layers': {'preset': 'layers-isotropic', 'ratio': 3},
}


@skipif_yask
@pytest.mark.parametrize('mkey, shape, kernel, space_order, nbpml', [
    # 2D tests with varying time and space orders
    ('layers', (60, 70), 'OT2', 4, 10), ('layers', (60, 70), 'OT2', 8, 10),
    ('layers', (60, 70), 'OT2', 12, 10), ('layers', (60, 70), 'OT4', 4, 10),
    ('layers', (60, 70), 'OT4', 8, 10), ('layers', (60, 70), 'OT4', 12, 10),
    # 3D tests with varying time and space orders
    ('layers', (60, 70, 80), 'OT2', 4, 10), ('layers', (60, 70, 80), 'OT2', 8, 10),
    ('layers', (60, 70, 80), 'OT2', 12, 10), ('layers', (60, 70, 80), 'OT4', 4, 10),
    ('layers', (60, 70, 80), 'OT4', 8, 10), ('layers', (60, 70, 80), 'OT4', 12, 10),
    # Constant model in 2D and 3D
    ('constant', (60, 70), 'OT2', 8, 14), ('constant', (60, 70, 80), 'OT2', 8, 14),
])
def test_acoustic(mkey, shape, kernel, space_order, nbpml):
    t0 = 0.0  # Start time
    tn = 500.  # Final time
    nrec = 130  # Number of receivers

    # Create model from preset
    model = demo_model(spacing=[15. for _ in shape], dtype=np.float64,
                       space_order=space_order, shape=shape, nbpml=nbpml,
                       **(presets[mkey]))

    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if kernel == 'OT4' else 1.0)
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 30.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec,
                                kernel=kernel, space_order=space_order)

    # Create adjoint receiver symbol
    srca = Receiver(name='srca', grid=model.grid, time_range=solver.source.time_range,
                    coordinates=solver.source.coordinates.data)

    # Run forward and adjoint operators
    rec, _, _ = solver.forward(save=False)
    solver.adjoint(rec=rec, srca=srca)

    # Adjoint test: Verify <Ax,y> matches  <x, A^Ty> closely
    term1 = np.dot(srca.data.reshape(-1), solver.source.data)
    term2 = linalg.norm(rec.data) ** 2
    info('<Ax,y>: %f, <x, A^Ty>: %f, difference: %12.12f, ratio: %f'
         % (term1, term2, (term1 - term2)/term1, term1 / term2))
    assert np.isclose((term1 - term2)/term1, 0., rtol=1.e-10)
