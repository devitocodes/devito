from functools import reduce

import pytest
from pyrevolve import Revolver
import numpy as np

from devito import Grid, TimeFunction, Operator, Function, Eq, switchconfig, Constant
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic.acoustic.acoustic_example import acoustic_setup


@switchconfig(log_level='WARNING')
def test_segmented_incremment():
    """
    Test for segmented operator execution of a one-sided first order
    function (increment). The corresponding set of stencil offsets in
    the time domain is (1, 0)
    """
    grid = Grid(shape=(5, 5))
    x, y = grid.dimensions
    t = grid.stepping_dim
    f = TimeFunction(name='f', grid=grid, time_order=1)
    fi = f.indexed
    op = Operator(Eq(fi[t, x, y], fi[t-1, x, y] + 1.))

    # Reference solution with a single invocation, up to timestep 21 (included)
    # IOW, run for 20 timesteps in total (time_m=1 is implicit)
    f_ref = TimeFunction(name='f', grid=grid, time_order=1)
    op(f=f_ref, time=21)
    assert (f_ref.data[20] == 20.).all()
    assert (f_ref.data[21] == 21.).all()

    # Now run with 5 invocations of 4 timesteps each (again, 20 timesteps in total)
    nsteps = 4
    for i in range(5):
        op(f=f, time_m=1+i*nsteps, time_M=1+(i+1)*nsteps)
    assert (f.data[20] == 20.).all()
    assert (f.data[21] == 21.).all()


@switchconfig(log_level='WARNING')
def test_segmented_fibonacci():
    """
    Test for segmented operator execution of a one-sided second order
    function (fibonacci). The corresponding set of stencil offsets in
    the time domain is (2, 0)
    """
    # Reference Fibonacci solution from:
    # https://stackoverflow.com/questions/4935957/fibonacci-numbers-with-an-one-liner-in-python-3
    fib = lambda n: reduce(lambda x, n: [x[1], x[0] + x[1]], range(n), [0, 1])[0]

    grid = Grid(shape=(5, 5))
    x, y = grid.dimensions
    t = grid.stepping_dim
    f = TimeFunction(name='f', grid=grid, time_order=2)
    fi = f.indexed
    op = Operator(Eq(fi[t, x, y], fi[t-1, x, y] + fi[t-2, x, y]))

    # Reference solution with a single invocation, up to timestep=12 (included)
    # =========================================================================
    # Developer note: the i-th Fibonacci number resides at logical index i-1
    f_ref = TimeFunction(name='f', grid=grid, time_order=2)
    f_ref.data[:] = 1.
    op(f=f_ref, time=12)
    assert (f_ref.data[11] == fib(12)).all()
    assert (f_ref.data[12] == fib(13)).all()

    # Now run with 2 invocations of 5 timesteps each
    nsteps = 5
    f.data[:] = 1.
    for i in range(2):
        op(f=f, time_m=2+i*nsteps, time_M=2+(i+1)*nsteps)
    assert (f.data[11] == fib(12)).all()
    assert (f.data[12] == fib(13)).all()


@switchconfig(log_level='WARNING')
def test_segmented_averaging():
    """
    Test for segmented operator execution of a two-sided, second order
    function (averaging) in space. The corresponding set of stencil
    offsets in the x domain are (1, 1).
    """
    grid = Grid(shape=(20, 20))
    x, y = grid.dimensions
    t = grid.stepping_dim
    f = TimeFunction(name='f', grid=grid)
    fi = f.indexed
    op = Operator(Eq(f, f.backward + (fi[t-1, x+1, y] + fi[t-1, x-1, y]) / 2.))

    # We add the average to the point itself, so the grid "interior"
    # (domain) is updated only.
    f_ref = TimeFunction(name='f', grid=grid)
    f_ref.data_with_halo[:] = 1.
    op(f=f_ref, time=1)
    assert (f_ref.data[1, :] == 2.).all()
    assert (f_ref.data_with_halo[1, 0] == 1.).all()
    assert (f_ref.data_with_halo[1, -1] == 1.).all()

    # Now we sweep the x direction in 4 segmented steps of 5 iterations each
    nsteps = 5
    f.data_with_halo[:] = 1.
    for i in range(4):
        op(f=f, time=1, x_m=i*nsteps, x_M=(i+1)*nsteps-1)
    assert (f_ref.data[1, :] == 2.).all()
    assert (f_ref.data_with_halo[1, 0] == 1.).all()
    assert (f_ref.data_with_halo[1, -1] == 1.).all()


@switchconfig(log_level='WARNING')
@pytest.mark.parametrize('space_order', [4])
@pytest.mark.parametrize('kernel', ['OT2'])
@pytest.mark.parametrize('shape', [(70, 80), (50, 50, 50)])
def test_forward_with_breaks(shape, kernel, space_order):
    """ Test running forward in one go and "with breaks"
    and ensure they produce the same result
    """
    spacing = tuple([15.0 for _ in shape])
    tn = 500.
    time_order = 2

    solver = acoustic_setup(shape=shape, spacing=spacing, tn=tn,
                            space_order=space_order, kernel=kernel)

    grid = solver.model.grid

    rec = solver.geometry.rec

    dt = solver.model.critical_dt

    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)
    cp = DevitoCheckpoint([u])
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), rec=rec,
                                 src=solver.geometry.src, u=u, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, dt=dt, rec=rec)

    wrp = Revolver(cp, wrap_fw, wrap_rev, None, rec._time_range.num-time_order)
    rec1, u1, summary = solver.forward()

    wrp.apply_forward()
    assert(np.allclose(u1.data, u.data))
    assert(np.allclose(rec1.data, rec.data))


@switchconfig(log_level='WARNING')
def test_acoustic_save_and_nosave(shape=(50, 50), spacing=(15.0, 15.0), tn=500.,
                                  time_order=2, space_order=4, nbl=10):
    """ Run the acoustic example with and without save=True. Make sure the result is the
    same
    """
    solver = acoustic_setup(shape=shape, spacing=spacing, nbl=nbl, dtype=np.float64,
                            space_order=space_order, time_order=time_order, tn=tn)
    rec, u, summary = solver.forward(save=True)
    last_time_step = solver.geometry.nt-1
    field_last_time_step = np.copy(u.data[last_time_step, :, :])
    rec_bk = np.copy(rec.data)
    rec, u, summary = solver.forward(save=False)
    last_time_step = (last_time_step) % (time_order + 1)
    assert(np.allclose(u.data[last_time_step, :, :], field_last_time_step))
    assert(np.allclose(rec.data, rec_bk))


def test_index_alignment():
    """ A much simpler test meant to ensure that the forward and reverse indices are
    correctly aligned (i.e. u * v , where u is the forward field and v the reverse field
    corresponds to the correct timesteps in u and v). The forward operator does u = u + 1
    which means that the field a will be equal to nt (0 -> 1 -> 2 -> 3), the number of
    timesteps this operator is run for. The field at the last time step of the forward is
    used to initialise the field v for the reverse pass. The reverse operator does
    v = v - 1, which means that if the reverse operator is run for the same number of
    timesteps as the forward operator, v should be 0 at the last time step
    (3 -> 2 -> 1 -> 0). There is also a grad = grad + u * v accumulator in the reverse
    operator. If the alignment is correct, u and v should have the same value at every
    time step:
    0 -> 1 -> 2 -> 3 u
    0 <- 1 <- 2 <- 3 v
    and hence grad = 0*0 + 1*1 + 2*2 + 3*3 = sum(n^2) where n -> [0, nt]
    If the test fails, the resulting number can tell you how the fields are misaligned
    """
    const = Constant(name="constant")
    n = 4
    grid = Grid(shape=(2, 2))
    order_of_eqn = 1
    modulo_factor = order_of_eqn + 1
    nt = n - order_of_eqn
    u = TimeFunction(name='u', grid=grid, save=n)
    # Increment one in the forward pass 0 -> 1 -> 2 -> 3
    fwd_op = Operator(Eq(u.forward, u + 1.*const))

    # Invocation 1
    fwd_op(time=nt-1, constant=1)
    last_time_step_v = nt % modulo_factor
    # Last time step should be equal to the number of timesteps we ran
    assert(np.allclose(u.data[nt, :, :], nt))

    v = TimeFunction(name='v', grid=grid, save=None)
    v.data[last_time_step_v, :, :] = u.data[nt, :, :]
    # Decrement one in the reverse pass 3 -> 2 -> 1 -> 0
    adj_eqn = Eq(v, v.forward - 1.*const)
    adj_op = Operator(adj_eqn)

    # Invocation 2
    adj_op(time=nt-1, constant=1)
    # Last time step should be back to 0
    assert(np.allclose(v.data[0, :, :], 0))

    # Reset v to run the backward again
    v.data[last_time_step_v, :, :] = u.data[nt, :, :]
    prod = Function(name="prod", grid=grid)
    # Multiply u and v and add them
    # = 3*3 + 2*2 + 1*1 + 0*0
    prod_eqn = Eq(prod, prod + u * v)
    comb_op = Operator([adj_eqn, prod_eqn])

    # Invocation 3
    comb_op(time=nt-1, constant=1)
    final_value = sum([n**2 for n in range(nt)])
    # Final value should be sum of squares of first nt natural numbers
    assert(np.allclose(prod.data, final_value))

    # Now reset to repeat all the above tests with checkpointing
    prod.data[:] = 0
    v.data[last_time_step_v, :, :] = u.data[nt, :, :]
    # Checkpointed version doesn't require to save u
    u_nosave = TimeFunction(name='u_n', grid=grid)
    # change equations to use new symbols
    fwd_eqn_2 = Eq(u_nosave.forward, u_nosave + 1.*const)
    fwd_op_2 = Operator(fwd_eqn_2)
    cp = DevitoCheckpoint([u_nosave])
    wrap_fw = CheckpointOperator(fwd_op_2, constant=1)

    prod_eqn_2 = Eq(prod, prod + u_nosave * v)
    comb_op_2 = Operator([adj_eqn, prod_eqn_2])
    wrap_rev = CheckpointOperator(comb_op_2, constant=1)
    wrp = Revolver(cp, wrap_fw, wrap_rev, None, nt)

    # Invocation 4
    wrp.apply_forward()
    assert(np.allclose(u_nosave.data[last_time_step_v, :, :], nt))

    # Invocation 5
    wrp.apply_reverse()
    assert(np.allclose(v.data[0, :, :], 0))
    assert(np.allclose(prod.data, final_value))
