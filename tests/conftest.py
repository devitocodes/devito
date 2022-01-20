import os
from subprocess import check_call

import pytest
import sys

from sympy import Symbol  # noqa

from devito import (Grid, TimeDimension, SteppingDimension, SpaceDimension, # noqa
                    Constant, Function, TimeFunction, Eq, configuration, SparseFunction, # noqa
                    SparseTimeFunction, cos)  # noqa
from devito.finite_differences.differentiable import EvalDerivative
from devito.arch import Device, sniff_mpi_distro
from devito.arch.compiler import compiler_registry
from devito.ir.iet import retrieve_iteration_tree, FindNodes, Iteration, ParallelBlock
from devito.tools import as_tuple

try:
    from mpi4py import MPI  # noqa
except ImportError:
    MPI = None


def skipif(items, whole_module=False):
    assert isinstance(whole_module, bool)
    items = as_tuple(items)
    # Sanity check
    accepted = set()
    accepted.update({'device', 'device-C', 'device-openmp', 'device-openacc',
                     'device-aomp'})
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
        # Skip if won't run on GPUs
        if i == 'device' and isinstance(configuration['platform'], Device):
            skipit = "device `%s` unsupported" % configuration['platform'].name
            break
        # Skip if won't run on a specific GPU backend
        langs = configuration._accepted['language']
        if any(i == 'device-%s' % l and configuration['language'] == l for l in langs)\
                and isinstance(configuration['platform'], Device):
            skipit = "language `%s` for device unsupported" % configuration['language']
            break
        if any(i == 'device-%s' % k and isinstance(configuration['compiler'], v)
               for k, v in compiler_registry.items()) and\
                isinstance(configuration['platform'], Device):
            skipit = "compiler `%s` for device unsupported" % configuration['compiler']
            break
        # Skip if must run on GPUs but not currently on a GPU
        if i in ('nodevice', 'nodevice-omp', 'nodevice-acc') and\
                not isinstance(configuration['platform'], Device):
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
        else:
            if len(m) == 2:
                nprocs, scheme = m
            else:
                raise ValueError("Can't run test: unexpected mode `%s`" % m)

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


# A list of optimization options/pipelines to be used in testing
# regarding spatial and/or temporal blocking.
opts_tiling = ['advanced',
               ('advanced', {'skewing': True}),
               ('advanced', {'skewing': True, 'blockinner': True})]


# Utilities for retrocompatibility


def _R(expr):
    """
    Originally Devito searched for sum-of-products in the Eq's, while now
    it searches for Derivatives (or, to be more precise, EvalDerivative).
    However, far too many tests were written with artificial sum-of-products
    as input (rather than actual FD derivative expressions), so here we "fake"
    such expressions as derivatives.
    """
    if any(a.has(EvalDerivative) for a in expr.args):
        base = expr
    else:
        base = {i.function for i in expr.free_symbols if i.function.is_TimeFunction}
        assert len(base) == 1
        base = base.pop()
    return EvalDerivative(*expr.args, base=base)


# Utilities for testing tree structure


def assert_structure(operator, exp_trees=None, exp_iters=None):
    """
    Utility function that helps to check loop structure of IETs. Retrieves trees from an
    Operator and check that the blocking structure is as expected.

    Examples
    --------
    To check that an Iteration tree has the following structure:

    .. code-block:: python

        for time
            for x
                for y
            for f
                for y

    we call:

    .. code-block:: python

        assert_structure(op, ['t,x,y', 't,f,y'], 't,x,y,f,y')`

    Notes
    -----
    `time` is mapped to `t`
    """
    mapper = {'time': 't'}

    if exp_trees is not None:
        trees = retrieve_iteration_tree(operator)
        exp_trees = [i.replace(',', '') for i in exp_trees]  # 't,x,y' -> 'txy'
        tree_struc = (["".join(mapper.get(i.dim.name, i.dim.name) for i in j)
                       for j in trees])  # Flatten every tree's dims as a string
        assert tree_struc == exp_trees

    if exp_iters is not None:
        iters = FindNodes(Iteration).visit(operator)
        exp_iters = exp_iters.replace(',', '')  # 't,x,y' -> 'txy'
        iter_struc = "".join(mapper.get(i.dim.name, i.dim.name) for i in iters)
        assert iter_struc == exp_iters


def assert_blocking(operator, exp_nests):
    """
    Utility function that helps to check existence of blocked nests. The structure of the
    operator is not used.

    Examples
    --------
    For the following structure:

    .. code-block:: python

        for t
            for x0_blk0
                for x
            for x1_blk0
                for x

    we call:

    .. code-block:: python

       bns, pbs = assert_blocking(op, {'x0_blk0', 'x1_blk0'})

    to assert the existence of 'x0_blk0', 'x1_blk0' and then the function returns a
    dictionary with the blocking Iterations that start a blocking subtree:

    ['x0_blk0': Iteration x0_blk0..., 'x1_blk0': Iteration x1_blk0...]

    and the ParallelBlock that encapsulates the above blocking subtree

    ['x0_blk0': ParallelBlock encapsulating Iteration x0_blk0,
     'x1_blk0': ParallelBlock encapsulating Iteration x1_blk0]
    """
    bns = {}
    pbs = {}
    trees = retrieve_iteration_tree(operator)
    for tree in trees:
        iterations = [i for i in tree if i.dim.is_Incr]  # Collect Incr dimensions
        if iterations:
            # If Incr dimensions exist map the first one to its name in the dict
            bns[iterations[0].dim.name] = iterations[0]
            try:
                parallel_blocks = FindNodes(ParallelBlock).visit(tree)
                pbs[iterations[0].dim.name] = parallel_blocks[0]
            except IndexError:
                pbs[iterations[0].dim.name] = tree[0]

    # Return if no Incr dimensions, ensuring that no Incr expected
    if not bns and not exp_nests:
        return {}, {}

    # Assert Incr dimensions found as expected
    assert bns.keys() == exp_nests

    return bns, pbs


# A list of optimization options/pipelines to be used in testing
# regarding GPU spatial and/or temporal blocking.
opts_device_tiling = [('advanced', {'blocklevels': 1}),
                      ('advanced', {'blocklevels': 1, 'skewing': True}),
                      ('advanced',
                       {'blocklevels': 1, 'skewing': True, 'blockinner': True})]
