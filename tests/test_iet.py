import pytest
import cgen

from devito import (Eq, Grid, Function, TimeFunction, Operator, Dimension,
                    switchconfig)
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, Expression, List, CGen, FindSymbols,
                           filter_iterations, make_efunc, retrieve_iteration_tree)
from devito.tools import as_tuple
from devito.types import Array, Symbol


@pytest.fixture
def grid():
    return Grid((3, 3, 3))

@pytest.fixture
def fc(grid):
    return Array(name='fc', dimensions=(grid.dimensions[0], grid.dimensions[1]),
                 shape=(3, 5)).indexed

def test_conditional(fc, grid):
    x, y, _ = grid.dimensions
    then_body = Expression(DummyEq(fc[x, y], fc[x, y] + 1))
    else_body = Expression(DummyEq(fc[x, y], fc[x, y] + 2))
    conditional = Conditional(x < 3, then_body, else_body)
    assert str(conditional) == """\
if (x < 3)
{
  fc[x][y] = fc[x][y] + 1;
}
else
{
  fc[x][y] = fc[x][y] + 2;
}"""


@pytest.mark.parametrize("exprs,nfuncs,ntimeiters,nests", [
    (('Eq(v[t+1,x,y], v[t,x,y] + 1)',), (1,), (2,), ('xy',)),
    (('Eq(v[t,x,y], v[t,x-1,y] + 1)', 'Eq(v[t,x,y], v[t,x+1,y] + u[x,y])'),
     (1, 2), (1, 1), ('xy', 'xy'))
])
@switchconfig(openmp=False)
def test_make_efuncs(exprs, nfuncs, ntimeiters, nests):
    """Test construction of ElementalFunctions."""
    exprs = list(as_tuple(exprs))

    grid = Grid(shape=(10, 10))
    t = grid.stepping_dim  # noqa
    x, y = grid.dimensions  # noqa

    u = Function(name='u', grid=grid)  # noqa
    v = TimeFunction(name='v', grid=grid)  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)

    op = Operator(exprs)

    # We create one ElementalFunction for each Iteration nest over space dimensions
    efuncs = []
    for n, tree in enumerate(retrieve_iteration_tree(op)):
        root = filter_iterations(tree, key=lambda i: i.dim.is_Space)[0]
        efuncs.append(make_efunc('f%d' % n, root))

    assert len(efuncs) == len(nfuncs) == len(ntimeiters) == len(nests)

    for efunc, nf, nt, nest in zip(efuncs, nfuncs, ntimeiters, nests):
        # Check the `efunc` parameters
        assert all(i in efunc.parameters for i in (x.symbolic_min, x.symbolic_max))
        assert all(i in efunc.parameters for i in (y.symbolic_min, y.symbolic_max))
        functions = FindSymbols().visit(efunc)
        assert len(functions) == nf
        assert all(i in efunc.parameters for i in functions)
        timeiters = [i for i in FindSymbols('free-symbols').visit(efunc)
                     if isinstance(i, Dimension) and i.is_Time]
        assert len(timeiters) == nt
        assert all(i in efunc.parameters for i in timeiters)
        assert len(efunc.parameters) == 4 + len(functions) + len(timeiters)

        # Check the loop nest structure
        trees = retrieve_iteration_tree(efunc)
        assert len(trees) == 1
        tree = trees[0]
        assert all(i.dim.name == j for i, j in zip(tree, nest))

        assert efunc.make_call()


def test_nested_calls_cgen():
    call = Call('foo', [
        Call('bar', [])
    ])

    code = CGen().visit(call)

    assert str(code) == 'foo(bar());'


@pytest.mark.parametrize('mode,expected', [
    ('free-symbols', '["f", "x"]'),
    ('symbolics', '["f"]')
])
def test_find_symbols_nested(mode, expected):
    grid = Grid(shape=(4, 4, 4))
    call = Call('foo', [
        Call('bar', [
            Symbol(name='x'),
            Call('baz', [Function(name='f', grid=grid)])
        ])
    ])

    found = FindSymbols(mode).visit(call)

    assert [f.name for f in found] == eval(expected)


def test_list_denesting():
    l0 = List(header=cgen.Line('a'), body=List(header=cgen.Line('b')))
    l1 = l0._rebuild(body=List(header=cgen.Line('c')))
    assert len(l0.body) == 0
    assert len(l1.body) == 0
    assert str(l1) == "a\nb\nc"

    l2 = l1._rebuild(l1.body)
    assert len(l2.body) == 0
    assert str(l2) == str(l1)

    l3 = l2._rebuild(l2.body, **l2.args_frozen)
    assert len(l3.body) == 0
    assert str(l3) == str(l2)
