import pytest
from devito import Operator, Grid, Eq, Function
from mpi4py import MPI


@pytest.mark.parallel(nprocs=2)
def test2():
    grid = Grid(shape=(30, 30, 30), comm=MPI.COMM_WORLD)

    f = Function(name='f', grid=grid,
                 dimensions=(grid.dimensions[2],),
                 shape=(None,),
                 space_order=0)

    # These should work right?
    import numpy as np
    print(f.local_indices)
    print(f.data.shape)
    print(f.shape)
    f.data[0:30] = 1.0
    f.data[0:5] = 0
    f.data[-5:] = np.arange(5)


@pytest.mark.parallel(nprocs=2)
def test3():
    grid = Grid(shape=(30, 30, 30), comm=MPI.COMM_WORLD)

    f = Function(name='f', grid=grid,
                 space_order=0)

    g = Function(name='g', grid=grid,
                 dimensions=(grid.dimensions[2],),
                 space_order=0)

    print("shapes %s and %s" % (str(f.shape), str(g.shape)))


@pytest.mark.parallel(nprocs=2)
def test4():
    grid1 = Grid(shape=(30, 30, 30))
    f1 = Function(name='f', grid=grid1,
                  space_order=4)

    op = Operator(Eq(f1, f1.dx2))

    grid2 = Grid(shape=(40, 40, 40))
    f2 = Function(name='f', grid=grid2,
                  space_order=4)

    op.apply(f=f2)


# Pickling of an uncompiled operator fails.
# Don't care about this, just accidentally discovered it when writing test6
def test5():
    grid = Grid(shape=(30, 30))
    f = Function(name='f', grid=grid,
                 space_order=4)

    op = Operator(Eq(f, f.dx2))

    import cloudpickle
    import pickle
    s = cloudpickle.dumps(op)
    pickle._loads(s)


# This one throws very different errors in the case of 1proc and 2proc
# 1proc: AttributeError: type object 'PyCSimpleType' has no attribute '__mul__'
# 2proc: size mismatch in __shape_setup__
@pytest.mark.parallel(nprocs=[1, 2])
def test6():
    grid = Grid(shape=(30, 30))
    f = Function(name='f', grid=grid,
                 space_order=4)

    op = Operator(Eq(f, f.dx2))

    # compile it
    op.cfunction

    import cloudpickle
    import pickle
    s = cloudpickle.dumps(op)
    pickle._loads(s)
