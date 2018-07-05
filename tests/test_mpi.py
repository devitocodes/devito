from mpi4py import MPI

import pytest

from devito import Grid, Function


@pytest.mark.parallel(nprocs=2)
def test_hello_mpi():
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    print("Hello, World! I am rank %d of %d on %s" % (rank, size, name), flush=True)


@pytest.mark.parallel(nprocs=[2, 4])
def test_basic_partitioning():
    grid = Grid(shape=(15, 15))  # Gonna use a default distributor underneath
    f = Function(name='f', grid=grid)

    distributor = grid._distributor
    expected = {  # nprocs -> [(rank0 shape), (rank1 shape), ...]
        2: [(8, 15), (7, 15)],
        4: [(8, 8), (8, 7), (7, 8), (7, 7)]
    }
    assert f.shape == expected[distributor.nprocs][distributor.rank]


#@pytest.mark.parallel(nprocs=[9])
def test_neighborhood_2d():
    grid = Grid(shape=(3, 3))  # Gonna use a default distributor underneath

    distributor = grid._distributor
    expected = {  # coords -> [neighbour0 coords, neighbour1 coords, ...]
        (1, 1): {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    }
    print (distributor.neighbours)
    from IPython import embed; embed()


@pytest.mark.parallel(nprocs=[2, 4])
def test_halo_exchange_1d():
    pass


@pytest.mark.parallel(nprocs=[2, 4])
def test_halo_exchange_2d():
    grid = Grid(shape=(12, 12))  # Gonna use a default distributor underneath
    f = Function(name='f', grid=grid)


if __name__ == "__main__":
    test_neighborhood_2d()
