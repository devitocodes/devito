from mpi4py import MPI

import pytest

@pytest.mark.parallel(nprocs=2)
def test_hello_mpi():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print("Hello, World! I am rank %d of %d on %s" % (rank, size, name), flush=True)
