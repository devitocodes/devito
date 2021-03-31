#define _POSIX_C_SOURCE 200809L
#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "openacc.h"

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

extern "C" int Forward(const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, const int z0_blk0_size, struct profiler * timers);


int Forward(const float dt, const float o_x, const float o_y, const float o_z, struct dataobj *restrict rec_vec, struct dataobj *restrict rec_coords_vec, struct dataobj *restrict src_vec, struct dataobj *restrict src_coords_vec, struct dataobj *restrict u_vec, struct dataobj *restrict vp_vec, const int x_M, const int x_m, const int y_M, const int y_m, const int z_M, const int z_m, const int p_rec_M, const int p_rec_m, const int p_src_M, const int p_src_m, const int time_M, const int time_m, const int x0_blk0_size, const int y0_blk0_size, const int z0_blk0_size, struct profiler * timers)
{
  float (*restrict rec)[rec_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_vec->size[1]]) rec_vec->data;
  float (*restrict rec_coords)[rec_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[rec_coords_vec->size[1]]) rec_coords_vec->data;
  float (*restrict src)[src_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_vec->size[1]]) src_vec->data;
  float (*restrict src_coords)[src_coords_vec->size[1]] __attribute__ ((aligned (64))) = (float (*)[src_coords_vec->size[1]]) src_coords_vec->data;
  float (*restrict u)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]] __attribute__ ((aligned (64))) = (float (*)[u_vec->size[1]][u_vec->size[2]][u_vec->size[3]]) u_vec->data;
  float (*restrict vp)[vp_vec->size[1]][vp_vec->size[2]] __attribute__ ((aligned (64))) = (float (*)[vp_vec->size[1]][vp_vec->size[2]]) vp_vec->data;

  #pragma acc enter data copyin(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc enter data copyin(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc enter data copyin(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]])
  #pragma acc enter data copyin(src[0:src_vec->size[0]][0:src_vec->size[1]])
  #pragma acc enter data copyin(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])
  #pragma acc enter data copyin(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])

  const float r14 = dt*dt;
  const float r15 = 1.0F/(dt*dt); 

  const int x_size = x_M - x_m + 1;
  const int y_size = y_M - y_m + 1;
  const int z_size = z_M - z_m + 1;
  float (*r49)[y_size + 8 + 8][z_size + 8 + 8] = (float (*)[y_size + 8 + 8][z_size + 8 + 8]) acc_malloc(sizeof(float[(x_size + 8 + 8)*(y_size + 8 + 8)*(z_size + 8 + 8)]));
  float (*r50)[y_size + 8 + 8][z_size + 8 + 8] = (float (*)[y_size + 8 + 8][z_size + 8 + 8]) acc_malloc(sizeof(float[(x_size + 8 + 8)*(y_size + 8 + 8)*(z_size + 8 + 8)]));
  #pragma acc parallel loop collapse(3) present(vp) deviceptr(r49,r50)
  for (int x = x_m - 8; x <= x_M + 8; x += 1)
  {
    for (int y = y_m - 8; y <= y_M + 8; y += 1)
    {
      for (int z = z_m - 8; z <= z_M + 8; z += 1)
      {
        r49[x + 8][y + 8][z + 8] = vp[x + 8][y + 8][z + 8]*vp[x + 8][y + 8][z + 8];
        r50[x + 8][y + 8][z + 8] = 1.0F/r49[x + 8][y + 8][z + 8];
      }
    }
  }

  float *coeffs = (float*) malloc(sizeof(float[6]));
  coeffs[0] = -3.79629639e-2F;
  coeffs[1] = 7.11111128e-3F;
  coeffs[2] = -8.8888891e-4F;
  coeffs[3] = 1.12874782e-4F;
  coeffs[4] = -7.93650813e-6F;
  coeffs[5] = -2.0F;
  #pragma acc enter data copyin(coeffs[0:6])

  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2), t2 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2), t2 = (time + 1)%(2))
  {
    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    #pragma acc parallel num_gangs(65535) vector_length(128) present(u,vp,coeffs) deviceptr(r49,r50)
    {
      #pragma acc loop collapse(3)
      for (int x0_blk0 = x_m; x0_blk0 <= x_M; x0_blk0 += x0_blk0_size)
      {
        for (int y0_blk0 = y_m; y0_blk0 <= y_M; y0_blk0 += y0_blk0_size)
        {
          for (int z0_blk0 = z_m; z0_blk0 <= z_M; z0_blk0 += z0_blk0_size)
          {
            #pragma acc loop collapse(3)
            for (int x = x0_blk0; x <= x0_blk0 + x0_blk0_size - 1; x += 1)
            {
              for (int y = y0_blk0; y <= y0_blk0 + y0_blk0_size - 1; y += 1)
              {
                for (int z = z0_blk0; z <= z0_blk0 + z0_blk0_size - 1; z += 1)
                {
                  float laplacian = 0.0F;
                  #pragma acc loop seq
                  for (int i = 1; i <= 4; i++)
                  {
                    laplacian += coeffs[i]*(u[t0][x + 8 - i][y + 8][z + 8] + u[t0][x + 8 + i][y + 8][z + 8]);
                    laplacian += coeffs[i]*(u[t0][x + 8][y + 8 - i][z + 8] + u[t0][x + 8][y + 8 + i][z + 8]);
                    laplacian += coeffs[i]*(u[t0][x + 8][y + 8][z + 8 - i] + u[t0][x + 8][y + 8][z + 8 + i]);
                  }

                  u[t2][x + 8][y + 8][z + 8] = r49[x + 8][y + 8][z + 8]*r14*(laplacian + coeffs[0]*u[t0][x + 8][y + 8][z + 8] + (-(coeffs[5]*u[t0][x + 8][y + 8][z + 8] + u[t1][x + 8][y + 8][z + 8])*r15)*r50[x + 8][y + 8][z + 8]);
                }
              }
            }
          }
        }
      }
    }
    /* End section0 */
    gettimeofday(&end_section0, NULL);
    timers->section0 += (double)(end_section0.tv_sec-start_section0.tv_sec)+(double)(end_section0.tv_usec-start_section0.tv_usec)/1000000;
    struct timeval start_section1, end_section1;
    gettimeofday(&start_section1, NULL);
    /* Begin section1 */
    #pragma acc parallel loop collapse(1) present(src,src_coords,u,vp)
    for (int p_src = p_src_m; p_src <= p_src_M; p_src += 1)
    {
      float posx = -o_x + src_coords[p_src][0];
      float posy = -o_y + src_coords[p_src][1];
      float posz = -o_z + src_coords[p_src][2];
      int ii_src_0 = (int)(floor(6.66667e-2*posx));
      int ii_src_1 = (int)(floor(6.66667e-2*posy));
      int ii_src_2 = (int)(floor(6.66667e-2*posz));
      int ii_src_3 = (int)(floor(6.66667e-2*posz)) + 1;
      int ii_src_4 = (int)(floor(6.66667e-2*posy)) + 1;
      int ii_src_5 = (int)(floor(6.66667e-2*posx)) + 1;
      float px = (float)(posx - 1.5e+1F*(int)(floor(6.66667e-2F*posx)));
      float py = (float)(posy - 1.5e+1F*(int)(floor(6.66667e-2F*posy)));
      float pz = (float)(posz - 1.5e+1F*(int)(floor(6.66667e-2F*posz)));
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1)
      {
        float r0 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py + 4.44445e-3F*px*pz - 6.66667e-2F*px + 4.44445e-3F*py*pz - 6.66667e-2F*py - 6.66667e-2F*pz + 1)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 8][ii_src_1 + 8][ii_src_2 + 8] += r0;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_0 <= x_M + 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1)
      {
        float r1 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*pz - 4.44445e-3F*py*pz + 6.66667e-2F*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 8][ii_src_1 + 8][ii_src_3 + 8] += r1;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r2 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*py*pz + 6.66667e-2F*py)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 8][ii_src_4 + 8][ii_src_2 + 8] += r2;
      }
      if (ii_src_0 >= x_m - 1 && ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_0 <= x_M + 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1)
      {
        float r3 = (dt*dt)*(vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*py*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_0 + 8][ii_src_4 + 8][ii_src_3 + 8] += r3;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_2 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_2 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r4 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8])*(2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*px*pz + 6.66667e-2F*px)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 8][ii_src_1 + 8][ii_src_2 + 8] += r4;
      }
      if (ii_src_1 >= y_m - 1 && ii_src_3 >= z_m - 1 && ii_src_5 >= x_m - 1 && ii_src_1 <= y_M + 1 && ii_src_3 <= z_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r5 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*pz)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 8][ii_src_1 + 8][ii_src_3 + 8] += r5;
      }
      if (ii_src_2 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_2 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r6 = (dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8])*(-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py)*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 8][ii_src_4 + 8][ii_src_2 + 8] += r6;
      }
      if (ii_src_3 >= z_m - 1 && ii_src_4 >= y_m - 1 && ii_src_5 >= x_m - 1 && ii_src_3 <= z_M + 1 && ii_src_4 <= y_M + 1 && ii_src_5 <= x_M + 1)
      {
        float r7 = 2.96296e-4F*px*py*pz*(dt*dt)*(vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8]*vp[ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8])*src[time][p_src];
        #pragma acc atomic update
        u[t2][ii_src_5 + 8][ii_src_4 + 8][ii_src_3 + 8] += r7;
      }
    }
    /* End section1 */
    gettimeofday(&end_section1, NULL);
    timers->section1 += (double)(end_section1.tv_sec-start_section1.tv_sec)+(double)(end_section1.tv_usec-start_section1.tv_usec)/1000000;
    struct timeval start_section2, end_section2;
    gettimeofday(&start_section2, NULL);
    /* Begin section2 */
    #pragma acc parallel loop collapse(1) present(rec,rec_coords,u)
    for (int p_rec = p_rec_m; p_rec <= p_rec_M; p_rec += 1)
    {
      float posx = -o_x + rec_coords[p_rec][0];
      float posy = -o_y + rec_coords[p_rec][1];
      float posz = -o_z + rec_coords[p_rec][2];
      int ii_rec_0 = (int)(floor(6.66667e-2*posx));
      int ii_rec_1 = (int)(floor(6.66667e-2*posy));
      int ii_rec_2 = (int)(floor(6.66667e-2*posz));
      int ii_rec_3 = (int)(floor(6.66667e-2*posz)) + 1;
      int ii_rec_4 = (int)(floor(6.66667e-2*posy)) + 1;
      int ii_rec_5 = (int)(floor(6.66667e-2*posx)) + 1;
      float px = (float)(posx - 1.5e+1F*(int)(floor(6.66667e-2F*posx)));
      float py = (float)(posy - 1.5e+1F*(int)(floor(6.66667e-2F*posy)));
      float pz = (float)(posz - 1.5e+1F*(int)(floor(6.66667e-2F*posz)));
      float sum = 0.0F;
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py + 4.44445e-3F*px*pz - 6.66667e-2F*px + 4.44445e-3F*py*pz - 6.66667e-2F*py - 6.66667e-2F*pz + 1)*u[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_2 + 8];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*pz - 4.44445e-3F*py*pz + 6.66667e-2F*pz)*u[t0][ii_rec_0 + 8][ii_rec_1 + 8][ii_rec_3 + 8];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*py*pz + 6.66667e-2F*py)*u[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_2 + 8];
      }
      if (ii_rec_0 >= x_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_0 <= x_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*py*pz)*u[t0][ii_rec_0 + 8][ii_rec_4 + 8][ii_rec_3 + 8];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_2 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_2 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (2.96296e-4F*px*py*pz - 4.44445e-3F*px*py - 4.44445e-3F*px*pz + 6.66667e-2F*px)*u[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_2 + 8];
      }
      if (ii_rec_1 >= y_m - 1 && ii_rec_3 >= z_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_1 <= y_M + 1 && ii_rec_3 <= z_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*pz)*u[t0][ii_rec_5 + 8][ii_rec_1 + 8][ii_rec_3 + 8];
      }
      if (ii_rec_2 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_2 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += (-2.96296e-4F*px*py*pz + 4.44445e-3F*px*py)*u[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_2 + 8];
      }
      if (ii_rec_3 >= z_m - 1 && ii_rec_4 >= y_m - 1 && ii_rec_5 >= x_m - 1 && ii_rec_3 <= z_M + 1 && ii_rec_4 <= y_M + 1 && ii_rec_5 <= x_M + 1)
      {
        sum += 2.96296e-4F*px*py*pz*u[t0][ii_rec_5 + 8][ii_rec_4 + 8][ii_rec_3 + 8];
      }
      rec[time][p_rec] = sum;
    }
    /* End section2 */
    gettimeofday(&end_section2, NULL);
    timers->section2 += (double)(end_section2.tv_sec-start_section2.tv_sec)+(double)(end_section2.tv_usec-start_section2.tv_usec)/1000000;
  }

  acc_free(r49);
  acc_free(r50);

  #pragma acc exit data copyout(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc exit data delete(rec[0:rec_vec->size[0]][0:rec_vec->size[1]])
  #pragma acc exit data copyout(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc exit data delete(u[0:u_vec->size[0]][0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])
  #pragma acc exit data delete(rec_coords[0:rec_coords_vec->size[0]][0:rec_coords_vec->size[1]])
  #pragma acc exit data delete(src[0:src_vec->size[0]][0:src_vec->size[1]])
  #pragma acc exit data delete(src_coords[0:src_coords_vec->size[0]][0:src_coords_vec->size[1]])
  #pragma acc exit data delete(vp[0:vp_vec->size[0]][0:vp_vec->size[1]][0:vp_vec->size[2]])

  #pragma acc exit data delete(coeffs[0:6])
  free(coeffs);

  return 0;
}
/* Backdoor edit at Mon Dec 21 18:45:42 2020*/ 
/* Backdoor edit at Mon Dec 21 19:54:43 2020*/ 
/* Backdoor edit at Mon Dec 21 19:55:32 2020*/ 
/* Backdoor edit at Mon Dec 21 19:56:07 2020*/ 
/* Backdoor edit at Thu Jan  7 23:45:33 2021*/ 
/* Backdoor edit at Thu Jan  7 23:48:06 2021*/ 
/* Backdoor edit at Fri Jan  8 00:07:39 2021*/ 
