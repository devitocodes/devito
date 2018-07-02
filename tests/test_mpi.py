from mpi4py import MPI

import pytest

from devito import Grid, Function, Distributor


@pytest.mark.parallel(nprocs=2)
def test_hello_mpi():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print("Hello, World! I am rank %d of %d on %s" % (rank, size, name), flush=True)


@pytest.mark.parallel(nprocs=2)
def test_basic_partitioning():
    grid = Grid(shape=(10, 10, 10))  # Gonna use a default distributor underneath
    f = Function(name='f', grid=grid)
    from IPython import embed; embed()
