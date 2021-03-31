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

  const int nx = u_vec->size[1];
  const int ny = u_vec->size[2];
  const int nz = u_vec->size[3];
  float *restrict uf = (float*) u_vec->data;

  for (int time = time_m, t0 = (time)%(2), t1 = (time + 1)%(2), t2 = (time + 1)%(2); time <= time_M; time += 1, t0 = (time)%(2), t1 = (time + 1)%(2), t2 = (time + 1)%(2))
  {

    struct timeval start_section0, end_section0;
    gettimeofday(&start_section0, NULL);
    /* Begin section0 */
    #pragma acc parallel num_gangs(65535) vector_length(128) present(u,uf,vp) deviceptr(r49,r50)
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
                  const int A0 = t0*nx*ny*nz;
                  const int A1 = ny*nz;
                  const int A2 = t1*nx*ny*nz;
                  const int A3 = t2*nx*ny*nz;
                 
                  const int K0 = A0 + A1*4 + nz*8 + 8;
                  const int K1 = A0 + A1*8 + nz*4 + 8;
                  const int K2 = A0 + A1*8 + nz*8 + 4;
                  const int K3 = A0 + A1*8 + nz*8 + 12;
                  const int K4 = A0 + A1*8 + nz*12 + 8;
                  const int K5 = A0 + A1*12 + nz*8 + 8;
                 
                  const int K6 = A0 + A1*5 + nz*8 + 8;
                  const int K7 = A0 + A1*8 + nz*5 + 8;
                  const int K8 = A0 + A1*8 + nz*8 + 5;
                  const int K9 = A0 + A1*8 + nz*8 + 11;
                  const int K10 = A0 + A1*8 + nz*11 + 8;
                  const int K11 = A0 + A1*11 + nz*8 + 8;
                 
                  const int K12 = A0 + A1*6 + nz*8 + 8;
                  const int K13 = A0 + A1*8 + nz*6 + 8;
                  const int K14 = A0 + A1*8 + nz*8 + 6;
                  const int K15 = A0 + A1*8 + nz*8 + 10;
                  const int K16 = A0 + A1*8 + nz*10 + 8;
                  const int K17 = A0 + A1*10 + nz*8 + 8;
                 
                  const int K18 = A0 + A1*7 + nz*8 + 8;
                  const int K19 = A0 + A1*8 + nz*7 + 8;
                  const int K20 = A0 + A1*8 + nz*8 + 7;
                  const int K21 = A0 + A1*8 + nz*8 + 9;
                  const int K22 = A0 + A1*8 + nz*9 + 8;
                  const int K23 = A0 + A1*9 + nz*8 + 8;
                 
                  const int K24 = A0 + A1*8 + nz*8 + 8;
                  const int K25 = A2 + A1*8 + nz*8 + 8;
                 
                  const int K26 = A3 + A1*8 + nz*8 + 8;
                 
                  const float coeffs[6] = {-7.93650813e-6F, 1.12874782e-4F, -8.8888891e-4F, 7.11111128e-3F, 3.79629639e-2F, -2.0F};

                  const int base = A1*x + nz*y + z;

                  float sum = 0.0;
                  sum += coeffs[0]*(uf[base + K0] + uf[base + K1] + uf[base + K2] + uf[base + K3] + uf[base + K4] + uf[base + K5]);

                  sum += coeffs[1]*(uf[base + K6] + uf[base + K7] + uf[base + K8] + uf[base + K9] + uf[base + K10] + uf[base + K11]);

                  sum += coeffs[2]*(uf[base + K12] + uf[base + K13] + uf[base + K14] + uf[base + K15] + uf[base + K16] + uf[base + K17]);

                  sum += coeffs[3]*(uf[base + K18] + uf[base + K19] + uf[base + K20] + uf[base + K21] + uf[base + K22] + uf[base + K23]);

                  const float u28 = coeffs[5]*uf[base + K24];
                  const float u29 = uf[base + K25];
                  const float u30 = u28 + u29;

                  const float u31 = uf[base + K24];

                  const float u32 = sum - coeffs[4]*u31;

                  uf[base + K26] = r49[x + 8][y + 8][z + 8]*r14*(u32 + (-(u30)*r15)*r50[x + 8][y + 8][z + 8]);
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
  return 0;
}
/* Backdoor edit at Mon Dec 21 18:45:42 2020*/ 
/* Backdoor edit at Mon Dec 21 19:54:43 2020*/ 
/* Backdoor edit at Mon Dec 21 19:55:32 2020*/ 
/* Backdoor edit at Mon Dec 21 19:56:07 2020*/ 
/* Backdoor edit at Mon Dec 21 20:05:22 2020*/ 
/* Backdoor edit at Mon Dec 21 20:06:32 2020*/ 
/* Backdoor edit at Mon Dec 21 23:20:18 2020*/ 
/* Backdoor edit at Tue Dec 22 01:00:41 2020*/ 
/* Backdoor edit at Thu Jan  7 01:24:15 2021*/ 
/* Backdoor edit at Thu Jan  7 17:43:49 2021*/ 
/* Backdoor edit at Thu Jan  7 17:54:18 2021*/ 
