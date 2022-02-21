import os

from devito import Eq, Grid, Operator, TimeFunction
from devito.types import Timer


def test_basic():
    grid = Grid(shape=(4, 4))

    f = TimeFunction(name='f', grid=grid)

    eq = Eq(f.forward, f + 1.)

    name = "foo"
    op = Operator(eq, name=name)

    # Trigger the generation of a .c and a .h files
    ccode, hcode = op.cinterface(force=True)

    dirname = op._compiler.get_jit_dir()
    assert os.path.isfile(os.path.join(dirname, "%s.c" % name))
    assert os.path.isfile(os.path.join(dirname, "%s.h" % name))

    ccode = str(ccode)
    hcode = str(hcode)

    assert 'include "%s.h"' % name in ccode

    # The public `struct dataobj` only appears in the header file
    assert str(f._C_typedecl) not in ccode
    assert str(f._C_typedecl) in hcode

    # Same with `struct profiler`
    timers = op.parameters[-1]
    assert isinstance(timers, Timer)
    assert str(timers._C_typedecl) not in ccode
    assert str(timers._C_typedecl) in hcode
