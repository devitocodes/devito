from examples.checkpointing.checkpointing_example import CheckpointingExample
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic.acoustic.acoustic_example import acoustic_setup
from pyrevolve import Revolver
import numpy as np
from conftest import skipif_yask
import pytest


@skipif_yask
@pytest.mark.parametrize('space_order', [4])
@pytest.mark.parametrize('time_order', [2])
@pytest.mark.parametrize('shape', [(70, 80), (50, 50, 50)])
def test_forward_with_breaks(shape, time_order, space_order):
    spacing = tuple([15.0 for _ in shape])
    tn = 500.
    example = CheckpointingExample(shape, spacing, tn, time_order, space_order)
    m0, dm = example.initial_estimate()

    cp = DevitoCheckpoint([example.forward_field])
    wrap_fw = CheckpointOperator(example.forward_operator, {'u': example.forward_field,
                                                            'rec': example.rec, 'm': m0,
                                                            'src': example.src,
                                                            'dt': example.dt})
    wrap_rev = CheckpointOperator(example.gradient_operator, {'u': example.forward_field,
                                                              'v': example.adjoint_field,
                                                              'm': m0,
                                                              'rec': example.rec_g,
                                                              'grad': example.grad,
                                                              'dt': example.dt})
    wrp = Revolver(cp, wrap_fw, wrap_rev, None, example.nt-time_order)
    example.forward_operator.apply(u=example.forward_field, rec=example.rec, m=m0,
                                   src=example.src, dt=example.dt)
    u_temp = np.copy(example.forward_field.data)
    rec_temp = np.copy(example.rec.data)
    example.forward_field.data[:] = 0
    wrp.apply_forward()
    assert(np.allclose(u_temp, example.forward_field.data))
    assert(np.allclose(rec_temp, example.rec.data))


@skipif_yask
def test_acoustic_save_and_nosave(shape=(50, 50), spacing=(15.0, 15.0), tn=500.,
                                  time_order=2, space_order=4, nbpml=10):
    solver = acoustic_setup(shape=shape, spacing=spacing, nbpml=nbpml, tn=tn,
                            space_order=space_order, time_order=time_order)
    rec, u, summary = solver.forward(save=True)
    last_time_step = solver.source.nt-1
    field_last_time_step = np.copy(u.data[last_time_step, :, :])
    rec_bk = np.copy(rec.data)
    rec, u, summary = solver.forward(save=False)
    last_time_step = (last_time_step) % (time_order + 1)
    assert(np.allclose(u.data[last_time_step, :, :], field_last_time_step))
    assert(np.allclose(rec.data, rec_bk))


@skipif_yask
@pytest.mark.parametrize('space_order', [4])
@pytest.mark.parametrize('time_order', [2])
@pytest.mark.parametrize('shape', [(70, 80), (50, 50, 50)])
def test_checkpointed_gradient_test(shape, time_order, space_order):
    spacing = tuple([15.0 for _ in shape])
    tn = 500.
    example = CheckpointingExample(shape, spacing, tn, time_order, space_order)
    m0, dm = example.initial_estimate()
    gradient, rec_data = example.gradient(m0)
    example.verify(m0, gradient, rec_data, dm)
