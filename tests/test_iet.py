import pytest

from ctypes import c_void_p
import cgen
import numpy as np
import sympy

from devito import (Eq, Grid, Function, TimeFunction, Operator, Dimension,  # noqa
                    switchconfig)
from devito.ir.iet import (Call, Callable, Conditional, DeviceCall, DummyExpr,
                           Iteration, List, KernelLaunch, Lambda, ElementalFunction,
                           CGen, FindSymbols, filter_iterations, make_efunc,
                           retrieve_iteration_tree, Transformer)
from devito.ir import SymbolRegistry
from devito.passes.iet.engine import Graph
from devito.passes.iet.languages.C import CDataManager
from devito.symbolics import (Byref, FieldFromComposite, InlineIf, Macro, Class,
                              FLOAT)
from devito.tools import CustomDtype, as_tuple, dtype_to_ctype
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
    then_body = DummyExpr(fc[x, y], fc[x, y] + 1)
    else_body = DummyExpr(fc[x, y], fc[x, y] + 2)
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
@switchconfig(language='C')
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
        timeiters = [i for i in FindSymbols('basics').visit(efunc)
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
    ('basics', '["x"]'),
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


def test_lambda():
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions

    u = Function(name='u', grid=grid)

    e0 = DummyExpr(u.indexed, 1)
    e1 = DummyExpr(u.indexed, 2)

    body = List(body=[e0, e1])
    lam = Lambda(body, ['='], [u.indexed], attributes=['my_attr'])

    assert str(lam) == """\
[=](float *restrict u) [[my_attr]]
{
  u = 1;
  u = 2;
}"""


def test_make_cpp_parfor():
    """
    Test construction of a C++ parallel for. This excites the IET construction
    machinery in several ways, in particular by using Lambda nodes (to generate
    C++ lambda functions) and nested Calls.
    """

    class STDVectorThreads(LocalObject):

        dtype = type('std::vector<std::thread>', (c_void_p,), {})

        def __init__(self):
            super().__init__('threads')

    class STDThread(LocalObject):

        dtype = type('std::thread&', (c_void_p,), {})

    class FunctionType(LocalObject):

        dtype = type('FuncType&&', (c_void_p,), {})

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
        DummyExpr(begin, it, init=True),
        DummyExpr(l, it + portion, init=True),
        DummyExpr(end, InlineIf(l > last, last, l), init=True),
        threadpush
    ], it, (first, last, portion))
    thread = STDThread('x')
    waitcall = Call('std::for_each', [
        Call(FieldFromComposite('begin', threads)),
        Call(FieldFromComposite('end', threads)),
        Lambda(Call(FieldFromComposite('join', thread.name)), [], [thread])
    ])
    body = [
        DummyExpr(threshold, 1, init=True),
        DummyExpr(portion, stdmax(threshold, (last - first) / nthreads), init=True),
        Call(FieldFromComposite('reserve', threads), nthreads),
        iteration,
        waitcall
    ]

    parfor = ElementalFunction('parallel_for', body,
                               parameters=[first, last, func, nthreads])

    assert str(parfor) == """\
static \
void parallel_for(const int first, const int last, FuncType&& func, const int nthreads)
{
  const int threshold = 1;
  const int portion = std::max(threshold, (-first + last)/nthreads);
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


def test_make_cuda_stream():

    class CudaStream(LocalObject):

        dtype = type('cudaStream_t', (c_void_p,), {})

        @property
        def _C_init(self):
            return Call('cudaStreamCreate', Byref(self))

        @property
        def _C_free(self):
            return Call('cudaStreamDestroy', self)

    stream = CudaStream('stream')

    iet = Call('foo', stream)
    iet = ElementalFunction('foo', iet, parameters=())
    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert str(iet) == """\
static void foo()
{
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  foo(stream);

  cudaStreamDestroy(stream);
}"""


def test_cpp_local_object():
    """
    Test C++ support for LocalObjects.
    """

    class MyObject(LocalObject):
        dtype = CustomDtype('dummy')

    # Locally-scoped objects are declared in the function body
    lo0 = MyObject('obj0')

    # Globally-scoped objects must not be declared in the function body
    lo1 = MyObject('obj1', is_global=True)

    # A LocalObject using both a template and a modifier
    class SpecialObject(LocalObject):
        dtype = CustomDtype('bar', template=('int', 'float'), modifier='&')

    lo2 = SpecialObject('obj2')

    # A LocalObject instantiated and subsequently assigned a value
    lo3 = MyObject('obj3', initvalue=Macro('meh'))

    # A LocalObject instantiated calling its 2-args constructor and subsequently
    # assigned a value
    lo4 = MyObject('obj4', cargs=(1, 2), initvalue=Macro('meh'))

    # A LocalObject with generic sympy exprs used as constructor args
    expr = sympy.Function('ceil')(FLOAT(Symbol(name='s'))**-1)
    lo5 = MyObject('obj5', cargs=(expr,), initvalue=Macro('meh'))

    # A LocalObject with class-level initvalue and numeric dtype
    class SpecialObject2(LocalObject):
        dtype = dtype_to_ctype(np.float32)
        default_initvalue = Macro('meh')

    lo6 = SpecialObject2('obj6')

    iet = Call('foo', [lo0, lo1, lo2, lo3, lo4, lo5, lo6])
    iet = ElementalFunction('foo', iet, parameters=())

    dm = CDataManager(sregistry=None)
    iet = CDataManager.place_definitions.__wrapped__(dm, iet)[0]

    assert 'dummy obj0;' in str(iet)
    assert 'dummy obj1;' not in str(iet)
    assert 'bar<int,float>& obj2;' in str(iet)
    assert 'dummy obj3 = meh;' in str(iet)
    assert 'dummy obj4(1,2) = meh;' in str(iet)
    assert 'dummy obj5(ceil(1.0F/(float)s)) = meh;' in str(iet)
    assert 'float obj6 = meh;' in str(iet)


def test_call_indexed():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)

    foo = Callable('foo', DummyExpr(u, 1), 'void', parameters=[u, u.indexed])
    call = Call(foo.name, [u, u.indexed])

    assert str(call) == "foo(u_vec,u);"
    assert str(foo) == """\
void foo(struct dataobj *restrict u_vec, float *restrict u)
{
  u(x, y) = 1;
}"""


def test_call_retobj_indexed():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)
    v = Function(name='v', grid=grid)

    call = Call('foo', [u], retobj=v.indexify())

    assert str(call) == "v[x][y] = foo(u_vec);"

    assert not call.defines


def test_call_lambda_transform():
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions

    u = Function(name='u', grid=grid)

    e0 = DummyExpr(x, 1)
    e1 = DummyExpr(y, 1)

    body = List(body=[e0, e1])
    call = Call('foo', [u, Lambda(body)])

    subs = {e0: DummyExpr(x, 2), e1: DummyExpr(y, 2)}

    assert str(Transformer(subs).visit(call)) == """\
foo(u_vec,[]()
{
  x = 2;
  y = 2;
});"""


def test_null_init():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)

    expr = DummyExpr(u.indexed, Macro('NULL'), init=True)

    assert str(expr) == "float * u = NULL;"
    assert expr.defines == (u.indexed,)


def test_templates_callable():
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions

    u = Function(name='u', grid=grid)

    foo = Callable('foo', DummyExpr(u, 1), 'void', parameters=[u],
                   templates=[x, y])

    assert str(foo) == """\
template <int x, int y>
void foo(struct dataobj *restrict u_vec)
{
  u(x, y) = 1;
}"""


@pytest.mark.parametrize('cls', [Call, DeviceCall])
def test_templates_call(cls):
    grid = Grid(shape=(10, 10))
    x, y = grid.dimensions

    u = Function(name='u', grid=grid)

    foo = cls('foo', u, templates=[Class('a'), Class('b')])

    assert str(foo) == "foo<class a, class b>(u_vec);"


def test_kernel_launch():
    grid = Grid(shape=(10, 10))

    u = Function(name='u', grid=grid)
    s = Symbol(name='s')

    class Dim3(LocalObject):
        dtype = type('dim3', (c_void_p,), {})

    kl = KernelLaunch('mykernel', Dim3('mygrid'), Dim3('myblock'),
                      arguments=(u.indexed,), templates=[s])

    assert str(kl) == 'mykernel<s><<<mygrid,myblock,0>>>(d_u);'


def test_codegen_quality0():
    grid = Grid(shape=(4, 4, 4))
    _, y, z = grid.dimensions

    a = Array(name='a', dimensions=grid.dimensions)

    expr = DummyExpr(a.indexed, 1)
    foo = Callable('foo', expr, 'void',
                   parameters=[a, y.symbolic_size, z.symbolic_size])

    # Emulate what the compiler would do
    graph = Graph(foo)

    CDataManager(sregistry=SymbolRegistry()).process(graph)

    foo1 = graph.root

    assert len(foo.parameters) == 3
    assert len(foo1.parameters) == 1
    assert foo1.parameters[0] is a
