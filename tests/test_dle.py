import pytest
from sympy import Eq

from devito.dimension import Dimension
from devito.dle import transform
from devito.interfaces import DenseData
from devito.nodes import Expression, Function, Iteration, List
from devito.visitors import ResolveIterationVariable, SubstituteExpression


def _symbol(name, dimensions):
    return DenseData(name=name, dimensions=dimensions)


@pytest.fixture(scope="session")
def dims():
    return {'i': Dimension(name='i', size=3),
            'j': Dimension(name='j', size=5),
            'k': Dimension(name='k', size=7),
            's': Dimension(name='s', size=4),
            'p': Dimension(name='p', size=4)}


@pytest.fixture(scope="session")
def symbols(dims):
    a = _symbol(name='a', dimensions=(dims['i'],))
    b = _symbol(name='b', dimensions=(dims['i'],))
    c = _symbol(name='c', dimensions=(dims['i'], dims['j']))
    d = _symbol(name='d', dimensions=(dims['i'], dims['j'], dims['k']))
    return [a, b, c, d]


@pytest.fixture(scope="session")
def exprs(symbols):
    a, b, c, d = [i.indexify() for i in symbols]
    return [Expression(Eq(a, a + b + 5.)),
            Expression(Eq(a, b*d - a*c)),
            Expression(Eq(a, a + b*b + 3)),
            Expression(Eq(a, a*b*d*c)),
            Expression(Eq(a, 4 * ((b + d) * (a + c)))),
            Expression(Eq(a, (6. / b) + (8. * a)))]


@pytest.fixture(scope="session")
def iters(dims):
    return [lambda ex: Iteration(ex, dims['i'], (0, 3, 1)),
            lambda ex: Iteration(ex, dims['j'], (0, 5, 1)),
            lambda ex: Iteration(ex, dims['k'], (0, 7, 1)),
            lambda ex: Iteration(ex, dims['s'], (0, 4, 1)),
            lambda ex: Iteration(ex, dims['p'], (0, 4, 1))]


@pytest.fixture(scope="session")
def simple_function(symbols, exprs, iters):
    # void foo(a, b)
    #   for i
    #     for j
    #       for k
    #         expr0
    #         expr1
    body = iters[0](iters[1](iters[2]([exprs[0], exprs[1]])))
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


@pytest.fixture(scope="session")
def complex_function(symbols, exprs, iters):
    # void foo(a, b, c, d)
    #   for i
    #     for s
    #       expr0
    #     for j
    #       for k
    #         expr1
    #         expr2
    #     for p
    #       expr3
    body = iters[0]([iters[3](exprs[2]),
                     iters[1](iters[2]([exprs[3], exprs[4]])),
                     iters[4](exprs[5])])
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


def test_create_elemental_functions_simple(simple_function):
    handle = transform(simple_function, mode=('basic',))
    block = List(body=handle.nodes + handle.elemental_functions)
    assert str(block.ccode) == \
        ("""void foo(float *a_vec, float *b_vec, float *c_vec, float *d_vec)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  float (*c)[5] = (float (*)[5]) c_vec;
  float (*d)[5][7] = (float (*)[5][7]) d_vec;
  for (int i0 = 0 + 0; i0 < 3 - 0; i0 += 1)
  {
    for (int j0 = 0 + 0; j0 < 5 - 0; j0 += 1)
    {
      f_0_0(a_vec,b_vec,c_vec,d_vec,i0,j0);
    }
  }
}
void f_0_0(float *a_vec, float *b_vec, float *c_vec, float *d_vec,"""
         """ const int i0, const int j0)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  float (*c)[5] = (float (*)[5]) c_vec;
  float (*d)[5][7] = (float (*)[5][7]) d_vec;
  for (int k0 = 0 + 0; k0 < 7 - 0; k0 += 1)
  {
    a[i0] = a[i0] + b[i0] + 5.0F;
    a[i0] = -a[i0]*c[i0][j0] + b[i0]*d[i0][j0][k0];
  }
}""")


def test_create_elemental_functions_complex(complex_function):
    handle = transform(complex_function, mode=('basic',))
    block = List(body=handle.nodes + handle.elemental_functions)
    assert str(block.ccode) == \
        ("""void foo(float *a_vec, float *b_vec, float *c_vec, float *d_vec)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  float (*c)[5] = (float (*)[5]) c_vec;
  float (*d)[5][7] = (float (*)[5][7]) d_vec;
  for (int i1 = 0 + 0; i1 < 3 - 0; i1 += 1)
  {
    f_0_0(a_vec,b_vec,i1);
    for (int j1 = 0 + 0; j1 < 5 - 0; j1 += 1)
    {
      f_0_1(a_vec,b_vec,c_vec,d_vec,i1,j1);
    }
    f_0_2(a_vec,b_vec,i1);
  }
}
void f_0_0(float *a_vec, float *b_vec, const int i1)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  for (int s0 = 0 + 0; s0 < 4 - 0; s0 += 1)
  {
    a[i1] = a[i1] + pow(b[i1], 2) + 3;
  }
}
void f_0_1(float *a_vec, float *b_vec, float *c_vec, float *d_vec,"""
         """ const int i1, const int j1)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  float (*c)[5] = (float (*)[5]) c_vec;
  float (*d)[5][7] = (float (*)[5][7]) d_vec;
  for (int k1 = 0 + 0; k1 < 7 - 0; k1 += 1)
  {
    a[i1] = a[i1]*b[i1]*c[i1][j1]*d[i1][j1][k1];
    a[i1] = 4*(a[i1] + c[i1][j1])*(b[i1] + d[i1][j1][k1]);
  }
}
void f_0_2(float *a_vec, float *b_vec, const int i1)
{
  float (*a) = (float (*)) a_vec;
  float (*b) = (float (*)) b_vec;
  for (int p0 = 0 + 0; p0 < 4 - 0; p0 += 1)
  {
    a[i1] = 8.0F*a[i1] + 6.0F/b[i1];
  }
}""")
