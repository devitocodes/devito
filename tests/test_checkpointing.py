from examples.checkpointing.checkpointing_example import setup as cp_setup
from examples.checkpointing.checkpoint import DevitoCheckpoint, DevitoOperator
from examples.seismic.acoustic.acoustic_example import acoustic_setup
from pyrevolve import Revolver
import numpy as np


def test_forward_with_breaks():
    time_order = 2
    fw, gradop, u, rec_s, m0, src, rec_g, v, grad, rec_t, dm, nt, dt = \
      cp_setup((150, 150), 750.0, (15.0, 15.0), time_order, 4, 40)
    cp = DevitoCheckpoint([u])
    wrap_fw = DevitoOperator(fw, {'u': u, 'rec': rec_s, 'm': m0, 'src': src, 'dt': dt},
                             {'t_start': 't_s', 't_end': 't_e'})
    wrap_rev = DevitoOperator(gradop, {'u': u, 'v': v, 'm': m0, 'rec': rec_g,
                                       'grad': grad, 'dt': dt},
                              {'t_start': 't_s', 't_end': 't_e'})
    wrp = Revolver(cp, wrap_fw, wrap_rev, None, nt-time_order)
    fw.apply(u=u, rec=rec_s, m=m0, src=src, dt=dt)
    u_temp = np.copy(u.data)
    rec_temp = np.copy(rec_s.data)
    u.data[:] = 0
    wrp.apply_forward()
    assert(np.allclose(u_temp, u.data))
    assert(np.allclose(rec_temp, rec_s.data))


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
