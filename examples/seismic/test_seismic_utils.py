import pytest
import numpy as np

from devito import norm
from examples.seismic import Model, setup_geometry, AcquisitionGeometry


def not_bcs(bc):
    return ("mask", 1) if bc == "damp" else ("damp", 0)


@pytest.mark.parametrize('nbl, bcs', [
    (20, ("mask", 1)), (0, ("mask", 1)),
    (20, ("damp", 0)), (0, ("damp", 0))

])
def test_damp(nbl, bcs):
    shape = (21, 21)
    vp = np.ones(shape)
    model = Model((0, 0), (10, 10), shape, 4, vp, nbl=nbl, bcs=bcs[0])

    try:
        center = model.damp.data[tuple(s // 2 for s in model.damp.shape)]
    except AttributeError:
        center = model.damp

    assert all([s == s0 + 2 * nbl for s, s0 in zip(model.vp.shape, shape)])
    assert center == bcs[1]

    switch_bcs = not_bcs(bcs[0])
    model._initialize_bcs(bcs=switch_bcs[0])
    try:
        center = model.damp.data[tuple(s // 2 for s in model.damp.shape)]
    except AttributeError:
        center = model.damp
    assert center == switch_bcs[1]


@pytest.mark.parametrize('shape', [(41,), (21, 21), (11, 11, 11)])
def test_default_geom(shape):
    vp = np.ones(shape)
    o = tuple([0]*len(shape))
    d = tuple([10]*len(shape))
    model = Model(o, d, shape, 4, vp, nbl=20, dt=1)
    assert model.critical_dt == 1

    geometry = setup_geometry(model, 250)
    nrec = shape[0] * (shape[1] if len(shape) > 2 else 1)
    assert geometry.grid == model.grid
    assert geometry.nrec == nrec
    assert geometry.nsrc == 1
    assert geometry.src_type == "Ricker"

    assert geometry.rec.shape == (251, nrec)
    assert norm(geometry.rec) == 0
    assert geometry.src.shape == (251, 1)
    assert norm(geometry.new_src(src_type=None)) == 0

    rec2 = geometry.rec.resample(num=501)
    assert rec2.shape == (501, nrec)
    assert rec2.grid == model.grid

    assert geometry.new_rec(name="bonjour").name == "bonjour"
    assert geometry.new_src(name="bonjour").name == "bonjour"


@pytest.mark.parametrize('shape', [(41,), (21, 21), (11, 11, 11)])
def test_geom(shape):
    vp = np.ones(shape)
    o = tuple([0]*len(shape))
    d = tuple([10]*len(shape))
    model = Model(o, d, shape, 4, vp, nbl=20, dt=1)
    assert model.critical_dt == 1

    nrec = 31
    nsrc = 4
    rec_coordinates = np.ones((nrec, len(shape)))
    src_coordinates = np.ones((nsrc, len(shape)))
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=250)
    assert geometry.grid == model.grid
    assert geometry.nrec == nrec
    assert geometry.nsrc == nsrc
    assert geometry.src_type is None

    assert geometry.rec.shape == (251, nrec)
    assert norm(geometry.rec) == 0
    assert geometry.src.shape == (251, nsrc)
    assert norm(geometry.new_src(src_type=None)) == 0
    assert norm(geometry.src) == 0

    rec2 = geometry.rec.resample(num=501)
    assert rec2.shape == (501, nrec)
    assert rec2.grid == model.grid

    assert geometry.new_rec(name="bonjour").name == "bonjour"
    assert geometry.new_src(name="bonjour").name == "bonjour"
