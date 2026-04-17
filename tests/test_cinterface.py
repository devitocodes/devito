import os
import sys
from contextlib import suppress

from devito import Eq, Grid, Operator, TimeFunction, configuration
from devito.types import Timer


def test_basic():
    grid = Grid(shape=(4, 4))

    f = TimeFunction(name='f', grid=grid)

    eq = Eq(f.forward, f + 1.)

    name = "foo"
    op = Operator(eq, name=name)

    # Trigger the generation of a .c and a .h files
    cpp = configuration['compiler']._cpp or 'CXX' in configuration['language']
    ccode, hcode = op.cinterface(force=True)
    ec, eh = ('cpp', 'h') if cpp else ('c', 'h')

    dirname = op._compiler.get_jit_dir()
    assert os.path.isfile(os.path.join(dirname, f"{name}.{ec}"))
    assert os.path.isfile(os.path.join(dirname, f"{name}.{eh}"))

    ccode = str(ccode)
    hcode = str(hcode)

    assert f'include "{name}.h"' in ccode

    # The public `struct dataobj` only appears in the header file
    assert 'struct dataobj\n{' not in ccode
    assert 'struct dataobj\n{' in hcode

    # Same with `struct profiler`
    timers = op.parameters[-1]
    assert isinstance(timers, Timer)
    assert 'struct profiler\n{' not in ccode
    assert 'struct profiler\n{' in hcode


def test_load_without_source():
    grid = Grid(shape=(4, 4))

    f = TimeFunction(name='f', grid=grid)

    eq = Eq(f.forward, f + 1.)

    name = "foo"
    op = Operator(eq, name=name)

    # Make sure the so file is not present
    dirname = op._compiler.get_jit_dir()
    soname = op._soname
    ext = '.dylib' if sys.platform == "darwin" else '.so'
    with suppress(FileNotFoundError):
        os.remove(f"{os.path.join(dirname, soname)}{ext}")

    # Trigger compilation
    recompiled, src_file = op._compiler.jit_compile(op._soname, str(op))
    assert recompiled

    os.remove(src_file)
    # Trigger compilation again. It should skip the C part and directly load the .so file
    recompiled, src_file = op._compiler.jit_compile(op._soname, str(op))
    assert not recompiled
    assert not os.path.isfile(src_file)
