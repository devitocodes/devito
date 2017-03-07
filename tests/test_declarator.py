from __future__ import absolute_import

import pytest
from sympy import Eq

from devito.stencilkernel import StencilKernel


@pytest.fixture(scope="session")
def exprs(symbols):
    a, b, c, d = [i.indexify() for i in symbols]
    return [Eq(a, a + b + 5.),
            Eq(a, b*d - a*c),
            Eq(a, a + b*b + 3),
            Eq(a, a*b*d*c),
            Eq(a, 4 * ((b + d) * (a + c))),
            Eq(a, (6. / b) + (8. * a))]


def test_heap_1D_stencil(a, b):
    operator = StencilKernel(Eq(a, a + b + 5.), dse='noop', dle='noop')
    assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)


def test_heap_perfect_2D_stencil(a, c):
    operator = StencilKernel([Eq(a, c), Eq(c, c*a)], dse='noop', dle='noop')
    assert """\
  float (*a);
  float (*c)[5];
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  posix_memalign((void**)&c, 64, sizeof(float[3][5]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      a[i] = c[i][j];
      c[i][j] = a[i]*c[i][j];
    }
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)


def test_heap_nonperfect_2D_stencil(a, c):
    operator = StencilKernel([Eq(a, 0.), Eq(c, c*a)], dse='noop', dle='noop')
    assert """\
  float (*a);
  float (*c)[5];
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  posix_memalign((void**)&c, 64, sizeof(float[3][5]));
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = 0.0F;
    struct timeval start_loop_j_0, end_loop_j_0;
    gettimeofday(&start_loop_j_0, NULL);
    for (int j = 0; j < 5; j += 1)
    {
      c[i][j] = a[i]*c[i][j];
    }
    gettimeofday(&end_loop_j_0, NULL);
    timings->loop_j_0 += (double)(end_loop_j_0.tv_sec-start_loop_j_0.tv_sec)\
+(double)(end_loop_j_0.tv_usec-start_loop_j_0.tv_usec)/1000000;
  }
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)


def test_stack_scalar_temporaries(a, t0, t1):
    operator = StencilKernel([Eq(t0, 1.), Eq(t1, 2.), Eq(a, t0*t1*3.)],
                             dse='noop', dle='noop')
    assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    float t0 = 1.00000000000000F;
    float t1 = 2.00000000000000F;
    a[i] = 3.0F*t0*t1;
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)


def test_stack_vector_temporaries(c_stack, e):
    operator = StencilKernel([Eq(c_stack, e*1.)],
                             dse='noop', dle='noop')
    assert """\
  struct timeval start_loop_k_0, end_loop_k_0;
  gettimeofday(&start_loop_k_0, NULL);
  for (int k = 0; k < 7; k += 1)
  {
    for (int s = 0; s < 4; s += 1)
    {
      for (int p = 0; p < 4; p += 1)
      {
        double c_stack[3][5] __attribute__((aligned(64)));
        for (int i = 0; i < 3; i += 1)
        {
          for (int j = 0; j < 5; j += 1)
          {
            c_stack[i][j] = 1.0F*e[k][s][p][i][j];
          }
        }
      }
    }
  }
  gettimeofday(&end_loop_k_0, NULL);
  timings->loop_k_0 += (double)(end_loop_k_0.tv_sec-start_loop_k_0.tv_sec)\
+(double)(end_loop_k_0.tv_usec-start_loop_k_0.tv_usec)/1000000;
  return 0;""" in str(operator.ccode)
