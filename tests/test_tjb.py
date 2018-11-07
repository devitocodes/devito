import pytest
from devito import Operator, Grid, Eq, Function, configuration
from mpi4py import MPI


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


if __name__ == "__main__":
    configuration['mpi'] = True
    test6()
