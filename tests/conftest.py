from __future__ import absolute_import

import pytest

from devito.dimension import Dimension
from devito.interfaces import DenseData, ScalarFunction, TensorFunction
from devito.nodes import Iteration


def scalarfunction(name):
    return ScalarFunction(name=name)


def tensorfunction(name, shape, dimensions, onstack=False):
    return TensorFunction(name=name, shape=shape, dimensions=dimensions, onstack=onstack)


def densedata(name, shape, dimensions):
    return DenseData(name=name, shape=shape, dimensions=dimensions)


@pytest.fixture(scope="session")
def dims():
    return {'i': Dimension(name='i', size=3),
            'j': Dimension(name='j', size=5),
            'k': Dimension(name='k', size=7),
            's': Dimension(name='s', size=4),
            'p': Dimension(name='p', size=4)}


@pytest.fixture(scope="session")
def iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['s'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['p'], (0, 4, 1))]


@pytest.fixture(scope="session", autouse=True)
def t0(dims):
    return scalarfunction('t0')


@pytest.fixture(scope="session", autouse=True)
def t1(dims):
    return scalarfunction('t1')


@pytest.fixture(scope="session", autouse=True)
def a(dims):
    return tensorfunction('a', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def a_dense(dims):
    return densedata('a_dense', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def b(dims):
    return tensorfunction('b', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def b_dense(dims):
    return densedata('b_dense', (3,), (dims['i'],)).indexify()


@pytest.fixture(scope="session", autouse=True)
def c(dims):
    return tensorfunction('c', (3, 5), (dims['i'], dims['j'])).indexify()


@pytest.fixture(scope="session", autouse=True)
def c_stack(dims):
    return tensorfunction('c_stack', (3, 5), (dims['i'], dims['j']), True).indexify()


@pytest.fixture(scope="session", autouse=True)
def d(dims):
    return tensorfunction('d', (3, 5, 7), (dims['i'], dims['j'], dims['k'])).indexify()


@pytest.fixture(scope="session", autouse=True)
def e(dims):
    dimensions = [dims['k'], dims['s'], dims['p'], dims['i'], dims['j']]
    return tensorfunction('e', (7, 4, 4, 3, 5), dimensions).indexify()
