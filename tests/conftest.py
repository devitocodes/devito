import os
from subprocess import check_call

import pytest
import sys

from sympy import cos, Symbol  # noqa

from devito import (Grid, TimeDimension, SteppingDimension, SpaceDimension, # noqa
                    Constant, Function, TimeFunction, Eq, configuration, SparseFunction, # noqa
                    SparseTimeFunction)  # noqa
from devito.archinfo import Device
from devito.compiler import sniff_mpi_distro
from devito.tools import as_tuple

try:
    from mpi4py import MPI  # noqa
except ImportError:
    MPI = None


def skipif(items, whole_module=False):
    assert isinstance(whole_module, bool)
    items = as_tuple(items)
    # Sanity check
    accepted = set(configuration._accepted['backend'])
    # TODO: to be refined in the near future, once we start adding support for
    # multiple GPU languages (openmp, openacc, cuda, ...)
    accepted.add('device')
    accepted.update({'no%s' % i for i in configuration._accepted['backend']})
    accepted.update({'nompi', 'nodevice'})
    unknown = sorted(set(items) - accepted)
    if unknown:
        raise ValueError("Illegal skipif argument(s) `%s`" % unknown)
    skipit = False
    for i in items:
        # Skip if no MPI
        if i == 'nompi':
            if MPI is None:
                skipit = "mpi4py/MPI not installed"
                break
            continue
        # Skip if an unsupported backend
        if i == configuration['backend']:
            skipit = "`%s` backend unsupported" % i
            break
        try:
            _, noi = i.split('no')
            if noi in configuration._accepted['backend']:
                if noi != configuration['backend']:
                    skipit = "`%s` backend unsupported" % i
                    break
        except ValueError:
            pass
        # Skip if won't run on GPUs
        if i == 'device' and isinstance(configuration['platform'], Device):
            skipit = "device `%s` unsupported" % configuration['platform'].name
            break
        # Skip if must run GPUs but not currently on a GPU
        if i == 'nodevice' and not isinstance(configuration['platform'], Device):
            skipit = ("must run on device, but currently on `%s`" %
                      configuration['platform'].name)
            break
    if skipit is False:
        return pytest.mark.skipif(False, reason='')
    else:
        if whole_module:
            return pytest.skip(skipit, allow_module_level=True)
        else:
            return pytest.mark.skip(skipit)


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
            continue

        pyversion = sys.executable
        # Only spew tracebacks on rank 0.
        # Run xfailing tests to ensure that errors are reported to calling process
        if item.cls is not None:
            testname = "%s::%s::%s" % (item.fspath, item.cls.__name__, item.name)
        else:
            testname = "%s::%s" % (item.fspath, item.name)
        args = ["-n", "1", pyversion, "-m", "pytest", "--runxfail", "-s",
                "-q", testname]
        if nprocs > 1:
            args.extend([":", "-n", "%d" % (nprocs - 1), pyversion, "-m", "pytest",
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
