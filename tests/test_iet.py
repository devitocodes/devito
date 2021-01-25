import pytest

from ctypes import c_void_p
import cgen
import sympy

from devito import (Eq, Grid, Function, TimeFunction, Operator, Dimension,  # noqa
                    switchconfig)
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, Expression, Iteration, List, Lambda,
                           LocalExpression, ElementalFunction, CGen, FindSymbols,
                           filter_iterations, make_efunc, retrieve_iteration_tree)
from devito.symbolics import Byref, FieldFromComposite, InlineIf
from devito.tools import as_tuple
from devito.types import Array, LocalObject, Symbol


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
    ('free-symbols', '["f_vec", "x"]'),
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


def test_make_cpp_parfor():
    """
    Test construction of a CPP parallel for. This excites the IET construction
    machinery in several ways, in particular by using Lambda nodes (to generate
    C++ lambda functions) and nested Calls.
    """

    class STDVectorThreads(LocalObject):
        dtype = type('std::vector<std::thread>', (c_void_p,), {})

        def __init__(self):
            self.name = 'threads'

    class STDThread(LocalObject):
        dtype = type('std::thread&', (c_void_p,), {})

        def __init__(self, name):
            self.name = name

    class FunctionType(LocalObject):
        dtype = type('FuncType&&', (c_void_p,), {})

        def __init__(self, name):
            self.name = name

    # Basic symbols
    nthreads = Symbol(name='nthreads', is_const=True)
    threshold = Symbol(name='threshold', is_const=True)
    last = Symbol(name='last', is_const=True)
    first = Symbol(name='first', is_const=True)
    portion = Symbol(name='portion', is_const=True)

    # Composite symbols
    threads = STDVectorThreads()

    # Iteration helper symbols
    begin = Symbol(name='begin')
    l = Symbol(name='l')
    end = Symbol(name='end')

    # Functions
    stdmax = sympy.Function('std::max')

    # Construct the parallel-for body
    func = FunctionType('func')
    i = Dimension(name='i')
    threadobj = Call('std::thread', Lambda(
        Iteration(Call(func.name, i), i, (begin, end-1, 1)),
        ['=', Byref(func.name)],
    ))
    threadpush = Call(FieldFromComposite('push_back', threads), threadobj)
    it = Dimension(name='it')
    iteration = Iteration([
        LocalExpression(DummyEq(begin, it)),
        LocalExpression(DummyEq(l, it + portion)),
        LocalExpression(DummyEq(end, InlineIf(l > last, last, l))),
        threadpush
    ], it, (first, last, portion))
    thread = STDThread('x')
    waitcall = Call('std::for_each', [
        Call(FieldFromComposite('begin', threads)),
        Call(FieldFromComposite('end', threads)),
        Lambda(Call(FieldFromComposite('join', thread.name)), [], [thread])
    ])
    body = [
        LocalExpression(DummyEq(threshold, 1)),
        LocalExpression(DummyEq(portion, stdmax(threshold, (last - first) / nthreads))),
        Call(FieldFromComposite('reserve', threads), nthreads),
        iteration,
        waitcall
    ]

    parfor = ElementalFunction('parallel_for', body, 'void',
                               [first, last, func, nthreads])

    assert str(parfor) == """\
void parallel_for(const int first, const int last, FuncType&& func, const int nthreads)
{
  int threshold = 1;
  int portion = std::max(threshold, (-first + last)/nthreads);
  threads.reserve(nthreads);
  for (int it = first; it <= last; it += portion)
  {
    int begin = it;
    int l = it + portion;
    int end = (l > last) ? last : l;
    threads.push_back(std::thread([=, &func]()
    {
      for (int i = begin; i <= end - 1; i += 1)
      {
        func(i);
      }
    }));
  }
  std::for_each(threads.begin(),threads.end(),[](std::thread& x)
  {
    x.join();
  });
}"""
