from examples.checkpointing.checkpointing_example import CheckpointingExample
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic.acoustic.acoustic_example import acoustic_setup
from pyrevolve import Revolver
import numpy as np
from conftest import skipif_yask
import pytest
from devito import Grid, TimeFunction, Operator, Backward, Function
from sympy import Eq


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
    wrap_fw = CheckpointOperator(example.forward_operator, u=example.forward_field,
                                 rec=example.rec, m=m0, src=example.src, dt=example.dt)
    wrap_rev = CheckpointOperator(example.gradient_operator, u=example.forward_field,
                                  v=example.adjoint_field, m=m0, rec=example.rec_g,
                                  grad=example.grad, dt=example.dt)
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


@skipif_yask
def test_index_alignment(const):
    nt = 10
    grid = Grid(shape=(3, 5))
    time = grid.time_dim
    t = grid.stepping_dim
    x, y = grid.dimensions
    order_of_eqn = 1
    modulo_factor = order_of_eqn + 1
    last_time_step_u = nt - order_of_eqn
    u = TimeFunction(name='u', grid=grid, save=True, time_dim=nt)
    # Increment one in the forward pass 0 -> 1 -> 2 -> 3
    fwd_eqn = Eq(u.indexed[time+1, x, y], u.indexed[time, x, y] + 1.*const)
    fwd_op = Operator(fwd_eqn)
    fwd_op(time=last_time_step_u, constant=1)
    last_time_step_v = (last_time_step_u) % modulo_factor
    # Last time step should be equal to the number of timesteps we ran
    assert(np.allclose(u.data[last_time_step_u, :, :], nt - order_of_eqn))
    v = TimeFunction(name='v', grid=grid, save=False)
    v.data[last_time_step_v, :, :] = u.data[last_time_step_u, :, :]
    # Decrement one in the reverse pass 3 -> 2 -> 1 -> 0
    adj_eqn = Eq(v.indexed[t-1, x, y], v.indexed[t, x, y] - 1.*const)
    adj_op = Operator(adj_eqn, time_axis=Backward)
    adj_op(t=(nt - order_of_eqn), constant=1)
    # Last time step should be back to 0
    assert(np.allclose(v.data[0, :, :], 0))

    # Reset v to run the backward again
    v.data[last_time_step_v, :, :] = u.data[last_time_step_u, :, :]
    prod = Function(name="prod", grid=grid)
    # Multiply u and v and add them
    # = 3*3 + 2*2 + 1*1 + 0*0
    prod_eqn = Eq(prod, prod + u * v)
    comb_op = Operator([adj_eqn, prod_eqn], time_axis=Backward)
    comb_op(time=nt-order_of_eqn, constant=1)
    final_value = sum([n**2 for n in range(nt)])
    # Final value should be sum of squares of first nt natural numbers
    assert(np.allclose(prod.data, final_value))

    # Now reset to repeat all the above tests with checkpointing
    prod.data[:] = 0
    v.data[last_time_step_v, :, :] = u.data[last_time_step_u, :, :]
    # Checkpointed version doesn't require to save u
    u_nosave = TimeFunction(name='u_n', grid=grid)
    # change equations to use new symbols
    fwd_eqn_2 = Eq(u_nosave.indexed[t+1, x, y], u_nosave.indexed[t, x, y] + 1.*const)
    fwd_op_2 = Operator(fwd_eqn_2)
    cp = DevitoCheckpoint([u_nosave])
    wrap_fw = CheckpointOperator(fwd_op_2, time=nt, constant=1)

    prod_eqn_2 = Eq(prod, prod + u_nosave * v)
    comb_op_2 = Operator([adj_eqn, prod_eqn_2], time_axis=Backward)
    wrap_rev = CheckpointOperator(comb_op_2, time=nt-order_of_eqn, constant=1)
    wrp = Revolver(cp, wrap_fw, wrap_rev, None, nt-order_of_eqn)
    wrp.apply_forward()
    assert(np.allclose(u_nosave.data[last_time_step_v, :, :], nt - order_of_eqn))
    wrp.apply_reverse()
    assert(np.allclose(v.data[0, :, :], 0))
    assert(np.allclose(prod.data, final_value))
