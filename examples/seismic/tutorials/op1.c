#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct dataobj
{
  void *restrict data;
  int * size;
  int * npsize;
  int * dsize;
  int * hsize;
  int * hofs;
  int * oofs;
} ;

struct profiler
{
  double section0;
  double section1;
  double section2;
} ;


int Kernel(struct dataobj *restrict damp_vec, const float dt, struct dataobj *restrict m_vec, const float o_x, const float o_y, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, struct profiler * timers)
{
  float (*restrict damp)[damp_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[damp_vec->size[1]]) damp_vec->data;
  float (*restrict m)[m_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[m_vec->size[1]]) m_vec->data;
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]]) u_vec->data;
  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  for (int time = time_m, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3); time <= time_M; time += 1, t0 = (time)%(3), t1 = (time + 1)%(3), t2 = (time + 2)%(3))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    for (int x = x_m; x <= x_M; x += 1)
    {
      #pragma omp simd
      for (int y = y_m; y <= y_M; y += 1)
      {
        float r0 = 1.0e+4F*dt*m[x + 2][y + 2] + 5.0e+3F*(dt*dt)*damp[x + 1][y + 1];
        u[t1][x + 2][y + 2] = 2.0e+4F*dt*m[x + 2][y + 2]*u[t0][x + 2][y + 2]/r0 - 1.0e+4F*dt*m[x + 2][y + 2]*u[t2][x + 2][y + 2]/r0 + 1.0e+2F*((dt*dt*dt)*u[t0][x + 1][y + 2]/r0 + (dt*dt*dt)*u[t0][x + 2][y + 1]/r0 + (dt*dt*dt)*u[t0][x + 2][y + 3]/r0 + (dt*dt*dt)*u[t0][x + 3][y + 2]/r0) + 5.0e+3F*(dt*dt)*damp[x + 1][y + 1]*u[t2][x + 2][y + 2]/r0 - 4.0e+2F*dt*dt*dt*u[t0][x + 2][y + 2]/r0;
      }
    }
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float r1 = (int)(floor(-1.0e-1F*o_x + 1.0e-1F*src_coords[p_src][0]));
      int ii_src_0 = r1 + 10;
      float r2 = (int)(floor(-1.0e-1F*o_y + 1.0e-1F*src_coords[p_src][1]));
      int ii_src_1 = r2 + 10;
      int ii_src_2 = r2 + 11;
      int ii_src_3 = r1 + 11;
      float px = (float)(-o_x - 1.0e+1F*r1 + src_coords[p_src][0]);
      float py = (float)(-o_y - 1.0e+1F*r2 + src_coords[p_src][1]);
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1)
      {
        int r3 = ii_src_0 + 2;
        int r4 = ii_src_1 + 2;
        float r5 = 2.819041F*(1.0e-2F*px*py - 1.0e-1F*px - 1.0e-1F*py + 1)*src[time][p_src]/m[r3][r4];
        u[t1][r3][r4] += r5;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= y_M + 1)
      {
        int r6 = ii_src_0 + 2;
        int r7 = ii_src_2 + 2;
        float r8 = 2.819041F*(-1.0e-2F*px*py + 1.0e-1F*py)*src[time][p_src]/m[r6][r7];
        u[t1][r6][r7] += r8;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        int r9 = ii_src_3 + 2;
        int r10 = ii_src_1 + 2;
        float r11 = 2.819041F*(-1.0e-2F*px*py + 1.0e-1F*px)*src[time][p_src]/m[r9][r10];
        u[t1][r9][r10] += r11;
      }
      if (ii_src_2 >= y_m - 1 && ii_src_3 >= x_m - 1 && ii_src_2 <= y_M + 1 && ii_src_3 <= x_M + 1)
      {
        int r12 = ii_src_3 + 2;
        int r13 = ii_src_2 + 2;
        float r14 = 2.81904108401397e-2F*px*py*src[time][p_src]/m[r12][r13];
        u[t1][r12][r13] += r14;
      }
    }
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float r15 = (int)(floor(-1.0e-1F*o_x + 1.0e-1F*rec_coords[p_rec][0]));
      int ii_rec_0 = r15 + 10;
      float r16 = (int)(floor(-1.0e-1F*o_y + 1.0e-1F*rec_coords[p_rec][1]));
      int ii_rec_1 = r16 + 10;
      int ii_rec_2 = r16 + 11;
      int ii_rec_3 = r15 + 11;
      float px = (float)(-o_x - 1.0e+1F*r15 + rec_coords[p_rec][0]);
      float py = (float)(-o_y - 1.0e+1F*r16 + rec_coords[p_rec][1]);
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1)
      {
        int r17 = ii_rec_0 + 2;
        int r18 = ii_rec_1 + 2;
        sum += (1.0e-2F*px*py - 1.0e-1F*px - 1.0e-1F*py + 1)*u[t0][r17][r18];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= y_M + 1)
      {
        int r19 = ii_rec_0 + 2;
        int r20 = ii_rec_2 + 2;
        sum += (-1.0e-2F*px*py + 1.0e-1F*py)*u[t0][r19][r20];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        int r21 = ii_rec_3 + 2;
        int r22 = ii_rec_1 + 2;
        sum += (-1.0e-2F*px*py + 1.0e-1F*px)*u[t0][r21][r22];
      }
      if (ii_rec_2 >= y_m - 1 && ii_rec_3 >= x_m - 1 && ii_rec_2 <= y_M + 1 && ii_rec_3 <= x_M + 1)
      {
        int r23 = ii_rec_3 + 2;
        int r24 = ii_rec_2 + 2;
        sum += 1.0e-2F*px*py*u[t0][r23][r24];
      }
      rec[time][p_rec] = sum;
    }
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }
  return 0;
}

