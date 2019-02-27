import os
from subprocess import check_call

import numpy as np
import pytest

from sympy import cos, Symbol  # noqa

from devito import (Grid, TimeDimension, SteppingDimension, SpaceDimension, # noqa
                    Constant, Function, TimeFunction, Eq, configuration, SparseFunction, # noqa
                    SparseTimeFunction)  # noqa
from devito.compiler import sniff_mpi_distro
from devito.ir.iet import Iteration
from devito.tools import as_tuple
from devito.types import Scalar, Array

try:
    from mpi4py import MPI  # noqa
except ImportError:
    MPI = None


def skipif(items, whole_module=False):
    assert isinstance(whole_module, bool)
    items = as_tuple(items)
    if whole_module is True:
        skipper = lambda i, j: pytest.skip(j, allow_module_level=True)
    else:
        skipper = lambda i, j: pytest.mark.skipif(i, reason=j)
    # Sanity check
    accepted = set(configuration._accepted['backend'])
    accepted.update({'no%s' % i for i in configuration._accepted['backend']})
    accepted.update({'nompi'})
    unknown = sorted(set(items) - accepted)
    if unknown:
        raise ValueError("Illegal skipif argument(s) `%s`" % unknown)
    for i in items:
        # Skip if no MPI
        if i == 'nompi':
            if MPI is None:
                return skipper(True, "mpi4py/MPI not installed")
            continue
        # Skip if an unsupported backend
        if i == configuration['backend']:
            return skipper(True, "`%s` backend unsupported" % i)
        try:
            _, noi = i.split('no')
            if noi != configuration['backend']:
                return skipper(True, "`%s` backend unsupported" % i)
        except ValueError:
            pass
    return skipper(False, "")


# Testing dimensions for space and time
grid = Grid(shape=(3, 3, 3))
time = grid.time_dim
t = grid.stepping_dim
x, y, z = grid.dimensions


def scalar(name):
    return Scalar(name=name)


def array(name, shape, dimensions, scope='heap'):
    return Array(name=name, shape=shape, dimensions=dimensions, scope=scope)


def constant(name):
    return Constant(name=name)


def function(name, shape, dimensions):
    return Function(name=name, shape=shape, dimensions=dimensions)


def timefunction(name, space_order=1):
    return TimeFunction(name=name, grid=grid, space_order=space_order)


def unit_box(name='a', shape=(11, 11), grid=None):
    """Create a field with value 0. to 1. in each dimension"""
    grid = grid or Grid(shape=shape)
    a = Function(name=name, grid=grid)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[:] = np.meshgrid(*dims)[1]
    return a


def unit_box_time(name='a', shape=(11, 11)):
    """Create a field with value 0. to 1. in each dimension"""
    grid = Grid(shape=shape)
    a = TimeFunction(name=name, grid=grid, time_order=1)
    dims = tuple([np.linspace(0., 1., d) for d in shape])
    a.data[0, :] = np.meshgrid(*dims)[1]
    a.data[1, :] = np.meshgrid(*dims)[1]
    return a


def points(grid, ranges, npoints, name='points'):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseFunction(name=name, grid=grid, npoint=npoints)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


def time_points(grid, ranges, npoints, name='points', nt=10):
    """Create a set of sparse points from a set of coordinate
    ranges for each spatial dimension.
    """
    points = SparseTimeFunction(name=name, grid=grid, npoint=npoints, nt=nt)
    for i, r in enumerate(ranges):
        points.coordinates.data[:, i] = np.linspace(r[0], r[1], npoints)
    return points


@pytest.fixture(scope="session")
def dims():
    return {'i': SpaceDimension(name='i'),
            'j': SpaceDimension(name='j'),
            'k': SpaceDimension(name='k'),
            'l': SpaceDimension(name='l'),
            's': SpaceDimension(name='s'),
            'q': SpaceDimension(name='q')}


@pytest.fixture(scope="session")
def iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['s'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['q'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['l'], (0, 6, 1)),
            lambda ex: Iteration(ex, x, (0, 5, 1)),
            lambda ex: Iteration(ex, y, (0, 7, 1)),
            lambda ex: Iteration(ex, z, (0, 7, 1))]


@pytest.fixture(scope="session", autouse=True)
def t0(dims):
    return scalar('t0').indexify()


@pytest.fixture(scope="session", autouse=True)
def t1(dims):
    return scalar('t1').indexify()


@pytest.fixture(scope="session", autouse=True)
def t2(dims):
    return scalar('t2').indexify()


@pytest.fixture(scope="session", autouse=True)
def t3(dims):
    return scalar('t3').indexify()


@pytest.fixture(scope="session", autouse=True)
def a(dims):
    return array('a', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def a_dense(dims):
    return function('a_dense', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def const():
    return constant('constant').indexify()


@pytest.fixture(scope="session", autouse=True)
def b(dims):
    return array('b', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def b_dense(dims):
    return function('b_dense', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def c(dims):
    return array('c', (3, 5), (dims['i'], dims['j'])).indexify()


@pytest.fixture(scope="session", autouse=True)
def c_stack(dims):
    return array('c_stack', (3, 5), (dims['i'], dims['j']), 'stack').indexify()


@pytest.fixture(scope="session", autouse=True)
def d(dims):
    return array('d', (3, 5, 7), (dims['i'], dims['j'], dims['k'])).indexify()


@pytest.fixture(scope="session", autouse=True)
def e(dims):
    dimensions = [dims['k'], dims['s'], dims['q'], dims['i'], dims['j']]
    return array('e', (7, 4, 4, 3, 5), dimensions).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti0(dims):
    return array('ti0', (3, 5, 7), (x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti1(dims):
    return array('ti1', (3, 5, 7), (x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti2(dims):
    return array('ti2', (3, 5), (x, y)).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti3(dims):
    return array('ti3', (3, 5, 7), (x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def tu(dims):
    return timefunction('tu', space_order=4).indexify()


@pytest.fixture(scope="session", autouse=True)
def tv(dims):
    return timefunction('tv', space_order=4).indexify()


@pytest.fixture(scope="session", autouse=True)
def tw(dims):
    return timefunction('tw', space_order=4).indexify()


@pytest.fixture(scope="session", autouse=True)
def fa(dims):
    return array('fa', (3,), (x,)).indexed


@pytest.fixture(scope="session", autouse=True)
def fb(dims):
    return array('fb', (3,), (x,)).indexed


@pytest.fixture(scope="session", autouse=True)
def fc(dims):
    return array('fc', (3, 5), (x, y)).indexed


@pytest.fixture(scope="session", autouse=True)
def fd(dims):
    return array('fd', (3, 5), (x, y)).indexed


@pytest.fixture(scope="session", autouse=True)
def fe(dims):
    return array('fe', (3, 5, 3), (x, y, z)).indexed


def EVAL(exprs, *args):
    """
    Convert strings into SymPy objects.

    Required to work around this 'won't fix' Python3 issue: ::

        http://stackoverflow.com/questions/29336616/eval-scope-in-python-2-vs-3
    """
    # Cannot use list comprehension because of the issue linked in the docstring
    scope = {}
    for i in args:
        try:
            scope[i.base.function.name] = i
            for j in i.base.function.indices:
                scope[j.name] = j
        except AttributeError:
            scope[i.label.name] = i
            for j in i.function.indices:
                scope[j.name] = j
    processed = []
    for i in as_tuple(exprs):
        processed.append(eval(i, globals(), scope))
    return processed[0] if isinstance(exprs, str) else processed


def parallel(item):
    """
    Run a test in parallel. Readapted from:

        ``https://github.com/firedrakeproject/firedrake/blob/master/tests/conftest.py``
    """
    mpi_exec = 'mpiexec'
    mpi_distro = sniff_mpi_distro(mpi_exec)

    marker = item.get_closest_marker("parallel")
    mode = as_tuple(marker.kwargs.get("mode", 2))
    for m in mode:
        # Parse the `mode`
        if isinstance(m, int):
            nprocs = m
            scheme = 'basic'
            restrain = False
        else:
            if len(m) == 2:
                nprocs, scheme = m
                restrain = False
            elif len(m) == 3:
                nprocs, scheme, restrain = m
            else:
                raise ValueError("Can't run test: unexpected mode `%s`" % m)

        if restrain and os.environ.get('MPI_RESTRAIN', False):
            # A computationally expensive test that would take too long to
            # run on the current machine
            return

        # Only spew tracebacks on rank 0.
        # Run xfailing tests to ensure that errors are reported to calling process
        if item.cls is not None:
            testname = "%s::%s::%s" % (item.fspath, item.cls.__name__, item.name)
        else:
            testname = "%s::%s" % (item.fspath, item.name)
        args = ["-n", "1", "python", "-m", "pytest", "--runxfail", "-s",
                "-q", testname]
        if nprocs > 1:
            args.extend([":", "-n", "%d" % (nprocs - 1), "python", "-m", "pytest",
                         "--runxfail", "--tb=no", "-q", testname])
        # OpenMPI requires an explicit flag for oversubscription. We need it as some
        # of the MPI tests will spawn lots of processes
        if mpi_distro == 'OpenMPI':
            call = [mpi_exec, '--oversubscribe'] + args
        else:
            call = [mpi_exec] + args

        # Tell the MPI ranks that they are running a parallel test
        os.environ['DEVITO_MPI'] = scheme
        try:
            check_call(call)
        finally:
            os.environ['DEVITO_MPI'] = '0'


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(mode): mark test to run in parallel"
    )


def pytest_runtest_setup(item):
    partest = os.environ.get('DEVITO_MPI', 0)
    try:
        partest = int(partest)
    except ValueError:
        pass
    if item.get_closest_marker("parallel") and not partest:
        # Blow away function arg in "master" process, to ensure
        # this test isn't run on only one process
        dummy_test = lambda *args, **kwargs: True
        if item.cls is not None:
            attr = item.originalname or item.name
            setattr(item.cls, attr, dummy_test)
        else:
            item.obj = dummy_test


def pytest_runtest_call(item):
    partest = os.environ.get('DEVITO_MPI', 0)
    try:
        partest = int(partest)
    except ValueError:
        pass
    if item.get_closest_marker("parallel") and not partest:
        # Spawn parallel processes to run test
        parallel(item)
