from __future__ import absolute_import

import pytest

from sympy import cos, Symbol  # noqa

from devito import (Dimension, Eq, TimeDimension, SteppingDimension, SpaceDimension,  # noqa
                    Constant, Function, TimeFunction, Grid, configuration)  # noqa
from devito.types import Scalar, Array
from devito.ir.iet import Iteration
from devito.tools import as_tuple


skipif_yask = pytest.mark.skipif(configuration['backend'] == 'yask',
                                 reason="YASK testing is currently restricted")


# Testing dimensions for space and time
grid = Grid(shape=(3, 3, 3))
time = grid.time_dim
t = grid.stepping_dim
x, y, z = grid.dimensions


def scalar(name):
    return Scalar(name=name)


def array(name, shape, dimensions, onstack=False):
    return Array(name=name, shape=shape, dimensions=dimensions,
                 onstack=onstack, onheap=(not onstack))


def constant(name):
    return Constant(name=name)


def function(name, shape, dimensions):
    return Function(name=name, shape=shape, dimensions=dimensions)


def timefunction(name, space_order=1):
    return TimeFunction(name=name, grid=grid, space_order=space_order)


@pytest.fixture(scope="session")
def dims():
    return {'i': Dimension(name='i'),
            'j': Dimension(name='j'),
            'k': Dimension(name='k'),
            'l': Dimension(name='l'),
            's': Dimension(name='s'),
            'q': Dimension(name='q')}


@pytest.fixture(scope="session")
def iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['s'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['q'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['l'], (0, 6, 1)),
            lambda ex: Iteration(ex, x, (0, 5, 1)),
            lambda ex: Iteration(ex, y, (0, 7, 1))]


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
    return array('c_stack', (3, 5), (dims['i'], dims['j']), True).indexify()


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
