import numpy as np
import pytest

from devito import Eq, Grid, Operator, TimeFunction
from devito.ir.iet import (
    Call, EntryFunction, FindNodes, Iteration, KernelLaunch, MacroStatement, TimedList
)
from devito.mpi.distributed import MPICommObject
from devito.passes.iet.engine import Graph
from devito.passes.iet.errors import check_launch
from devito.types import Dimension, Symbol


@pytest.mark.parametrize('language', ['C', 'openmp'])
def test_operator_timer_macros(language):
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)
    op = Operator(Eq(u.forward, u + 1), language=language)

    timers = FindNodes(TimedList).visit(op)
    assert timers
    for timer in timers:
        assert isinstance(timer.header[0], MacroStatement)
        assert isinstance(timer.footer[0], MacroStatement)
        assert f'START({timer.name});' in str(op)
        assert f'STOP({timer.name},{timer.timer.name});' in str(op)

    summary = op.apply(time_M=1)
    assert np.all(u.data[0] == 2)
    assert np.all(u.data[1] == 1)
    assert summary


def test_operator_denormal_macros():
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)
    op = Operator(Eq(u.forward, u + 1), platform='intel64', language='C')

    macros = FindNodes(MacroStatement).visit(op)
    names = [macro.name for macro in macros]
    assert '_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)' in names
    assert '_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)' in names
    code = str(op)
    assert '_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);' in code
    assert '_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);' in code


@pytest.mark.parametrize('mpi, with_comm', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('errctl', [False, 'basic'])
@pytest.mark.parametrize('inside_loop', [False, True])
def test_kernel_launch_macros(errctl, inside_loop, mpi, with_comm):
    launch = KernelLaunch('kernel', Symbol(name='blocks'), Symbol(name='threads'))
    body = Iteration(launch, Dimension(name='t'), (0, 1, 1)) if inside_loop else launch
    if with_comm:
        pytest.importorskip('mpi4py')
        comm = MPICommObject()
        body = [Call('MPI_Barrier', comm), body]
        parameters = (comm,)
    else:
        parameters = ()
    graph = Graph(EntryFunction('entry', body, 'int', parameters))
    langbb = {'peek-error': lambda: Call('peek_error'), 'error-none': 'SUCCESS'}

    check_launch(graph, options={'errctl': errctl, 'mpi': mpi}, langbb=langbb)

    macros = FindNodes(MacroStatement).visit(graph.root)
    if errctl and inside_loop:
        name = 'CHECK_LAUNCH_RETURN' if with_comm else 'CHECK_LAUNCH'
        assert [macro.name for macro in macros] == [name]
        code = str(graph.root)
        assert f'{name};' in code
        assert code.index('kernel<<<') < code.index(f'{name};')
        assert ('CHECK_LAUNCH', 'if (peek_error() != SUCCESS) {break;}') in graph.headers
    else:
        assert not macros
        assert not graph.headers
