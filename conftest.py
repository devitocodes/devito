import os
import sys
from subprocess import check_call

import pytest
from sympy import Add
from sympy.printing import sstr

from devito import Eq, configuration, Revolver  # noqa
from devito.checkpointing import NoopRevolver
from devito.finite_differences.differentiable import EvalDerivative
from devito.arch import Cpu64, Device, sniff_mpi_distro, Arm, get_advisor_path
from devito.arch.compiler import (compiler_registry, IntelCompiler, OneapiCompiler,
                                  NvidiaCompiler)
from devito.ir.iet import (FindNodes, FindSymbols, Iteration, ParallelBlock,
                           retrieve_iteration_tree)
from devito.tools import as_tuple

try:
    from mpi4py import MPI  # noqa
except ImportError:
    MPI = None


def pytest_collectstart(collector):
    if collector.fspath and collector.fspath.ext == '.ipynb':
        collector.skip_compare += ('text/latex', 'stderr')


def skipif(items, whole_module=False):
    assert isinstance(whole_module, bool)
    items = as_tuple(items)
    # Sanity check
    accepted = set()
    accepted.update({'device', 'device-C', 'device-openmp', 'device-openacc',
                     'device-aomp', 'cpu64-icc', 'cpu64-icx', 'cpu64-nvc',
                     'noadvisor', 'cpu64-arm', 'cpu64-icpx', 'chkpnt'})
    accepted.update({'nodevice'})
    unknown = sorted(set(items) - accepted)
    if unknown:
        raise ValueError("Illegal skipif argument(s) `%s`" % unknown)
    skipit = False
    for i in items:
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
        # Skip if it won't run with nvc on CPU backend
        if i == 'cpu64-nvc' and \
           isinstance(configuration['compiler'], NvidiaCompiler) and \
           isinstance(configuration['platform'], Cpu64):
            skipit = "`nvc+cpu64` won't work with this test"
            break
        # Skip if it won't run with IntelCompiler
        if i == 'cpu64-icc' and \
           isinstance(configuration['compiler'], IntelCompiler) and \
           isinstance(configuration['platform'], Cpu64):
            skipit = "`icc+cpu64` won't work with this test"
            break
        # Skip if it won't run with OneAPICompiler
        if i == 'cpu64-icx' and \
           isinstance(configuration['compiler'], OneapiCompiler) and \
           isinstance(configuration['platform'], Cpu64):
            skipit = "`icx+cpu64` won't work with this test"
            break
        # Skip if icx or advisor are not available
        if i == 'noadvisor' and \
            (not isinstance(configuration['compiler'], IntelCompiler) or
             not get_advisor_path()):
            skipit = "Only `icx+advisor` should be tested here"
            break
        # Skip if it won't run on Arm
        if i == 'cpu64-arm' and isinstance(configuration['platform'], Arm):
            skipit = "Arm doesn't support x86-specific instructions"
            break
        # Skip if pyrevolve not installed
        if i == 'chkpnt' and Revolver is NoopRevolver:
            skipit = "pyrevolve not installed"
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


def get_testname(item):
    if item.cls is not None:
        return "%s::%s::%s" % (item.fspath, item.cls.__name__, item.name)
    else:
        return "%s::%s" % (item.fspath, item.name)


def set_run_reset(env_vars, call):
    old_env_vars = {k: os.environ.get(k, None) for k in env_vars}

    os.environ.update(env_vars)
    os.environ['DEVITO_PYTEST_FLAG'] = '1'

    try:
        check_call(call)
        return True
    except:
        return False
    finally:
        os.environ['DEVITO_PYTEST_FLAG'] = '0'
        for k, v in old_env_vars.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def parallel(item, m):
    """
    Run a test in parallel. Readapted from:

        ``https://github.com/firedrakeproject/firedrake/blob/master/tests/conftest.py``
    """
    mpi_exec = 'mpiexec'
    mpi_distro = sniff_mpi_distro(mpi_exec)

    # Parse the `mode`
    if isinstance(m, int):
        nprocs = m
        scheme = 'basic'
    else:
        if len(m) == 2:
            nprocs, scheme = m
        else:
            raise ValueError("Can't run test: unexpected mode `%s`" % m)

    env_vars = {'DEVITO_MPI': scheme}

    pyversion = sys.executable
    testname = get_testname(item)
    # Only spew tracebacks on rank 0.
    # Run xfailing tests to ensure that errors are reported to calling process
    args = ["-n", "1", pyversion, "-m", "pytest", "-s", "--runxfail", "-qq", testname]
    if nprocs > 1:
        args.extend([":", "-n", "%d" % (nprocs - 1), pyversion, "-m", "pytest",
                     "-s", "--runxfail", "--tb=no", "-qq", "--no-summary", testname])
    # OpenMPI requires an explicit flag for oversubscription. We need it as some
    # of the MPI tests will spawn lots of processes
    if mpi_distro == 'OpenMPI':
        call = [mpi_exec, '--oversubscribe', '--timeout', '300'] + args
    else:
        call = [mpi_exec] + args

    return set_run_reset(env_vars, call)


def decoupler(item, m):
    """
    Run a test in decoupled mode.
    """
    mpi_exec = 'mpiexec'
    assert sniff_mpi_distro(mpi_exec) != 'unknown', "Decoupled tests require MPI"

    env_vars = {'DEVITO_DECOUPLER': '1'}
    if isinstance(m, int):
        env_vars['DEVITO_DECOUPLER_WORKERS'] = str(m)

    testname = get_testname(item)

    pyversion = sys.executable
    args = ["-n", "1", pyversion, "-m", "pytest", "-s", "--runxfail", testname]
    call = [mpi_exec] + args

    return set_run_reset(env_vars, call)


def pytest_configure(config):
    """Register an additional marker."""
    config.addinivalue_line(
        "markers",
        "parallel(mode): mark test to run in parallel"
    )
    config.addinivalue_line(
        "markers",
        "decoupler(mode): mark test to run in decoupled mode",
    )


def pytest_generate_tests(metafunc):
    # Process custom parallel marker as a parametrize to avoid
    # running a single test for all modes
    if 'mode' in metafunc.fixturenames:
        markers = metafunc.definition.iter_markers()
        for marker in markers:
            if marker.name in ('parallel', 'decoupler'):
                mode = list(as_tuple(marker.kwargs.get('mode', 2)))
                metafunc.parametrize("mode", mode)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item):
    inside_pytest_marker = os.environ.get('DEVITO_PYTEST_FLAG', 0)
    try:
        inside_pytest_marker = int(inside_pytest_marker)
    except ValueError:
        pass

    if inside_pytest_marker:
        outcome = yield

    elif item.get_closest_marker("parallel"):
        # Spawn parallel processes to run test

        outcome = parallel(item, item.funcargs['mode'])
        if outcome:
            pytest.skip(f"{item} success in parallel")
        else:
            pytest.fail(f"{item} failed in parallel")

    elif item.get_closest_marker("decoupler"):
        outcome = decoupler(item, item.funcargs.get('mode'))
        if outcome:
            pytest.skip(f"{item} success in decoupled mode")
        else:
            pytest.fail(f"{item} failed in decoupled mode")

    else:
        outcome = yield


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    inside_pytest_marker = os.environ.get('DEVITO_PYTEST_FLAG', 0)
    try:
        inside_pytest_marker = int(inside_pytest_marker)
    except ValueError:
        pass
    if inside_pytest_marker:
        return

    if item.get_closest_marker("parallel") or \
       item.get_closest_marker("decoupler"):
        if call.when == 'call' and result.outcome == 'skipped':
            result.outcome = 'passed'


def pytest_make_parametrize_id(config, val, argname):
    """
    Prevents pytest from making obscure parameter names (param0, param1, ...)
    and default to sympy.sstr(val) instead for better log readability.
    """
    # First see if it has a name
    if hasattr(val, '__name__'):
        return val.__name__
    try:
        return sstr(val)
    except Exception:
        return None  # Fall back to default behavior


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


# More utilities for testing


def get_params(op, *names):
    ret = []
    for i in names:
        for p in op.parameters:
            if i == p.name:
                ret.append(p)
    return tuple(ret)


def get_arrays(iet):
    return [i for i in FindSymbols().visit(iet)
            if i.is_Array and i._mem_heap]


def check_array(array, exp_halo, exp_shape, rotate=False):
    assert len(array.dimensions) == len(exp_halo)

    shape = []
    for i in array.symbolic_shape:
        if i.is_Number or i.is_Symbol:
            shape.append(i)
        else:
            assert i.is_Add
            shape.append(Add(*i.args))

    if rotate:
        exp_shape = (sum(exp_halo[0]) + 1,) + tuple(exp_shape[1:])
        exp_halo = ((0, 0),) + tuple(exp_halo[1:])

    assert tuple(array.halo) == exp_halo
    assert tuple(shape) == tuple(exp_shape)
