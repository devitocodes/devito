#define _POSIX_C_SOURCE 200809L
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define START_TIMER(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP_TIMER(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "xmmintrin.h"
#include "pmmintrin.h"
#include "omp.h"

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
} ;


int Kernel(const float dt, const float h_x, const float h_y, const float h_z, struct dataobj *restrict u_vec, const int time0_blk0_size, const int time_M, const int time_m, const int x0_blk0_size, const int x0_blk1_size, const int x_M, const int x_m, const int y0_blk0_size, const int y0_blk1_size, const int y_M, const int y_m, const int z_M, const int z_m, const int nthreads, struct profiler * timers)
{
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  int xb = 32;
  int yb = 32;
  int xb2 = 8;
  int yb2 = 8;
  float r0 = 1.0F/dt;
  float r1 = 1.0F/(h_x*h_x);
  float r2 = 1.0F/(h_y*h_y);
  float r3 = 1.0F/(h_z*h_z);
  for (int time0_blk0 = time_m; time0_blk0 <= time_M; time0_blk0 += 258)
  {
    /* Begin section0 */
    START_TIMER(section0)
   for (int x0_blk0 = x_m; x0_blk0 <= time_M - time_m + x_M; x0_blk0 += xb)
    {
      for (int y0_blk0 = y_m; y0_blk0 <= time_M - time_m + y_M; y0_blk0 += xb)
      {
        for (int time = time0_blk0, t0 = (time)%(2), t1 = (time + 1)%(2); time <= MIN(time0_blk0 + 258 - 1, time_M); time += 1, t0 = (time)%(2), t1 = (time + 1)%(2))
        {
	 #pragma omp parallel num_threads(nthreads)
         {
          #pragma omp for collapse(2) schedule(dynamic,1)
            for (int x0_blk1 = MAX(x0_blk0, time + x_m); x0_blk1 <= MIN(x0_blk0 + xb - 1, time + x_M); x0_blk1 += xb2)
            {
              for (int y0_blk1 = MAX(y0_blk0, time + y_m); y0_blk1 <= MIN(y0_blk0 + yb - 1, time + y_M); y0_blk1 += xb2)
              {
                for (int x = x0_blk1; x <= MIN(x0_blk1+xb2 ,MIN(x0_blk0 + xb - 1, time + x_M)); x += 1)
                {
                  for (int y = y0_blk1; y <= MIN(y0_blk1+yb2 , MIN(y0_blk0 + xb - 1, time + y_M)); y += 1)
                  {
                    #pragma omp simd aligned(u:32)
                    for (int z = z_m; z <= z_M; z += 1)
                    {
                      float r4 = -2.0F*u[t0][-time + x + 2][-time + y + 2][z + 2];
                      u[t1][-time + x + 2][-time + y + 2][z + 2] = dt*(r0*u[t0][-time + x + 2][-time + y + 2][z + 2] + r1*r4 + r1*u[t0][-time + x + 1][-time + y + 2][z + 2] + r1*u[t0][-time + x + 3][-time + y + 2][z + 2] + r2*r4 + r2*u[t0][-time + x + 2][-time + y + 1][z + 2] + r2*u[t0][-time + x + 2][-time + y + 3][z + 2] + r3*r4 + r3*u[t0][-time + x + 2][-time + y + 2][z + 1] + r3*u[t0][-time + x + 2][-time + y + 2][z + 3]);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    STOP_TIMER(section0,timers)
    /* End section0 */
  }

  return 0;
}
/* Backdoor edit at Wed Aug  4 18:07:30 2021*/ 
/* Backdoor edit at Wed Aug  4 18:07:37 2021*/ 
/* Backdoor edit at Wed Aug  4 18:58:23 2021*/ 