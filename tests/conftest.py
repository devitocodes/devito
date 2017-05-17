from __future__ import absolute_import

import pytest

from sympy import Eq, cos  # noqa

from devito.dimension import Dimension, t, x, y, z
from devito.interfaces import DenseData, ScalarFunction, TensorFunction
from devito.nodes import Iteration
from devito.tools import as_tuple


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
            'l': Dimension(name='l', size=6),
            's': Dimension(name='s', size=4),
            'q': Dimension(name='q', size=4)}


@pytest.fixture(scope="session")
def iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['s'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['q'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['l'], (0, 6, 1))]


@pytest.fixture(scope="session", autouse=True)
def t0(dims):
    return scalarfunction('t0').indexify()


@pytest.fixture(scope="session", autouse=True)
def t1(dims):
    return scalarfunction('t1').indexify()


@pytest.fixture(scope="session", autouse=True)
def t2(dims):
    return scalarfunction('t2').indexify()


@pytest.fixture(scope="session", autouse=True)
def t3(dims):
    return scalarfunction('t3').indexify()


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
    dimensions = [dims['k'], dims['s'], dims['q'], dims['i'], dims['j']]
    return tensorfunction('e', (7, 4, 4, 3, 5), dimensions).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti0(dims):
    return tensorfunction('ti0', (3, 5, 7), (x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti1(dims):
    return tensorfunction('ti1', (3, 5, 7), (x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def ti2(dims):
    return tensorfunction('ti2', (3, 5), (x, y)).indexify()


@pytest.fixture(scope="session", autouse=True)
def tu(dims):
    return tensorfunction('tu', (10, 3, 5, 7), (t, x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def tv(dims):
    return tensorfunction('tv', (10, 3, 5, 7), (t, x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def tw(dims):
    return tensorfunction('tw', (10, 3, 5, 7), (t, x, y, z)).indexify()


@pytest.fixture(scope="session", autouse=True)
def fa(dims):
    return tensorfunction('fa', (3,), (x,)).indexed


@pytest.fixture(scope="session", autouse=True)
def fb(dims):
    return tensorfunction('fb', (3,), (x,)).indexed


@pytest.fixture(scope="session", autouse=True)
def fc(dims):
    return tensorfunction('fc', (3, 5), (x, y)).indexed


@pytest.fixture(scope="session", autouse=True)
def fd(dims):
    return tensorfunction('fd', (3, 5), (x, y)).indexed


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
        except AttributeError:
            scope[i.label.name] = i
    processed = []
    for i in as_tuple(exprs):
        processed.append(eval(i, globals(), scope))
    return processed[0] if isinstance(exprs, str) else processed
