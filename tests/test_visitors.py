import pytest

import cgen as c
from sympy import Eq, Symbol  # noqa

from devito.dimension import Dimension
from devito.interfaces import DenseData
from devito.nodes import Block, Expression, Iteration
from devito.visitors import FindSections, IsPerfectIteration, Transformer


def _symbol(name, dimensions):
    return DenseData(name=name, dimensions=dimensions).indexify()


@pytest.fixture(scope="session")
def exprs():
    i = Dimension(name='i', size=3)
    a = _symbol(name='a', dimensions=(i,))
    b = _symbol(name='b', dimensions=(i,))
    return [Expression(Eq(a, a + b + 5.)),
            Expression(Eq(a, b - a)),
            Expression(Eq(a, 4 * (b * a))),
            Expression(Eq(a, (6. / b) + (8. * a)))]


@pytest.fixture(scope="session")
def iters():
    return [lambda ex: Iteration(ex, Dimension('i', size=3), (0, 3, 1)),
            lambda ex: Iteration(ex, Dimension('j', size=5), (0, 5, 1)),
            lambda ex: Iteration(ex, Dimension('k', size=7), (0, 7, 1)),
            lambda ex: Iteration(ex, Dimension('s', size=4), (0, 4, 1)),
            lambda ex: Iteration(ex, Dimension('p', size=4), (0, 4, 1))]


@pytest.fixture(scope="session")
def block1(exprs, iters):
    # Perfect loop nest:
    # for i
    #   for j
    #     for k
    #       expr0
    return iters[0](iters[1](iters[2](exprs[0])))


@pytest.fixture(scope="session")
def block2(exprs, iters):
    # Non-perfect simple loop nest:
    # for i
    #   expr0
    #   for j
    #     for k
    #       expr1
    return iters[0]([exprs[0], iters[1](iters[2](exprs[1]))])


@pytest.fixture(scope="session")
def block3(exprs, iters):
    # Non-perfect non-trivial loop nest:
    # for i
    #   for s
    #     expr0
    #   for j
    #     for k
    #       expr1
    #       expr2
    #   for p
    #     expr3
    return iters[0]([iters[3](exprs[0]),
                     iters[1](iters[2]([exprs[1], exprs[2]])),
                     iters[4](exprs[3])])


def test_find_sections(block1, block2, block3, exprs):
    finder = FindSections()

    sections = finder.visit(block1)
    assert len(sections) == 1

    sections = finder.visit(block2)
    assert len(sections) == 2
    found = sections.values()
    assert len(found[0]) == 1
    assert found[0][0].stencil == exprs[0].stencil
    assert len(found[1]) == 1
    assert found[1][0].stencil == exprs[1].stencil

    sections = finder.visit(block3)
    assert len(sections) == 3
    found = sections.values()
    assert len(found[0]) == 1
    assert found[0][0].stencil == exprs[0].stencil
    assert len(found[1]) == 2
    assert found[1][0].stencil == exprs[1].stencil
    assert found[1][1].stencil == exprs[2].stencil
    assert len(found[2]) == 1
    assert found[2][0].stencil == exprs[3].stencil


def test_is_perfect_iteration(block1, block2, block3):
    checker = IsPerfectIteration()

    assert checker.visit(block1) == True
    assert checker.visit(block1._children()[0]) == True
    assert checker.visit(block1._children()[0]._children()[0]) == True

    assert checker.visit(block2) == False
    assert checker.visit(block2._children()[1]) == True
    assert checker.visit(block2._children()[1]._children()[0]) == True

    assert checker.visit(block3) == False
    assert checker.visit(block3._children()[0]) == True
    assert checker.visit(block3._children()[1]) == True
    assert checker.visit(block3._children()[2]) == True


def test_transformer(block1, block2, block3):
    line1 = '// This is the opening comment'
    line2 = '// This is the closing comment'
    line3 = '// Adding a simple line'
    line4 = '// Replaced expression'
    wrapper = lambda n: Block(c.Line(line1), n, c.Line(line2))
    adder = lambda n: Block(c.Line(line3), n)
    replacer = Block(c.Line(line4))

    finder = FindSections()
    sections = finder.visit(block1)
    exprs = sections.values()[0]
    transformer = Transformer({exprs[0]: wrapper(exprs[0])})

    newblock = transformer.visit(block1)
    assert """
        // This is the opening comment
        a[i0] = a[i0] + b[i0] + 5.0F;
        // This is the closing comment
""" in str(newblock.ccode)

    # TODO: add block2 and block3
    #transformer = Transformer({exprs[0]: wrapper(exprs[0]),
    #                           exprs[1]: adder(exprs[1]),
    #                           exprs[3]: replacer})
