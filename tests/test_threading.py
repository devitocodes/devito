from concurrent.futures import ThreadPoolExecutor
from devito import Operator, TimeFunction, Grid, Eq
from devito.logger import info
import numpy as np
from threading import current_thread


def test_concurrent_executing_operators():
    rng = np.random.default_rng()

    # build a simple operator and force it to compile
    grid = Grid(shape=(50, 50, 50))
    u = TimeFunction(name='u', grid=grid)
    op = Operator(Eq(u.forward, u + 1))

    # this forces the compile
    op.cfunction

    def do_run(op):
        # choose a new size
        shape = (rng.integers(20, 22), 30, rng.integers(20, 22))

        # make concurrent executions put a different value in the array
        # so we can be sure they aren't sharing an object even though the
        # name is reused
        val = current_thread().ident % 100000

        grid_private = Grid(shape=shape)
        u_private = TimeFunction(name='u', grid=grid_private)
        u_private.data[:] = val

        op(u=u_private, time_m=1, time_M=100)
        assert np.all(u_private.data[1, :, :, :] == val + 100)

    info("First running serially to demonstrate it works")
    do_run(op)

    info("Now creating thread pool")
    tpe = ThreadPoolExecutor(max_workers=16)

    info("Running operator in threadpool")
    futures = []
    for i in range(1000):
        futures.append(tpe.submit(do_run, op))

    # Get results - exceptions will be raised here if there are any
    for f in futures:
        f.result()
