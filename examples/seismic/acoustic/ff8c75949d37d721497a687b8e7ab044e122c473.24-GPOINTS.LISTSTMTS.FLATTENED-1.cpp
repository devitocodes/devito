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
    const int A0 = t0*nx*ny*nz;

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
                  //const float u0 = u[t0][x + 4][y + 8][z + 8];
                  //const float u1 = u[t0][x + 8][y + 4][z + 8];
                  //const float u2 = u[t0][x + 8][y + 8][z + 4];
                  //const float u3 = u[t0][x + 8][y + 8][z + 12];
                  //const float u4 = u[t0][x + 8][y + 12][z + 8];
                  //const float u5 = u[t0][x + 12][y + 8][z + 8];
                  const float u0 = uf[A0 + (x + 4)*ny*nz + (y + 8)*nz + z + 8];
                  const float u1 = uf[A0 + (x + 8)*ny*nz + (y + 4)*nz + z + 8];
                  const float u2 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 4];
                  const float u3 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 12];
                  const float u4 = uf[A0 + (x + 8)*ny*nz + (y + 12)*nz + z + 8];
                  const float u5 = uf[A0 + (x + 12)*ny*nz + (y + 8)*nz + z + 8];
                  const float u6 = u0 + u1 + u2 + u3 + u4 + u5;

                  //const float u7 = u[t0][x + 5][y + 8][z + 8];
                  //const float u8 = u[t0][x + 8][y + 5][z + 8];
                  //const float u9 = u[t0][x + 8][y + 8][z + 5];
                  //const float u10 = u[t0][x + 8][y + 8][z + 11];
                  //const float u11 = u[t0][x + 8][y + 11][z + 8];
                  //const float u12 = u[t0][x + 11][y + 8][z + 8];
                  const float u7 = uf[A0 + (x + 5)*ny*nz + (y + 8)*nz + z + 8];
                  const float u8 = uf[A0 + (x + 8)*ny*nz + (y + 5)*nz + z + 8];
                  const float u9 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 5];
                  const float u10 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 11];
                  const float u11 = uf[A0 + (x + 8)*ny*nz + (y + 11)*nz + z + 8];
                  const float u12 = uf[A0 + (x + 11)*ny*nz + (y + 8)*nz + z + 8];
                  const float u13 = u7 + u8 + u9 + u10 + u11 + u12;

                  //const float u14 = u[t0][x + 6][y + 8][z + 8];
                  //const float u15 = u[t0][x + 8][y + 6][z + 8];
                  //const float u16 = u[t0][x + 8][y + 8][z + 6];
                  //const float u17 = u[t0][x + 8][y + 8][z + 10];
                  //const float u18 = u[t0][x + 8][y + 10][z + 8];
                  //const float u19 = u[t0][x + 10][y + 8][z + 8];
                  const float u14 = uf[A0 + (x + 6)*ny*nz + (y + 8)*nz + z + 8];
                  const float u15 = uf[A0 + (x + 8)*ny*nz + (y + 6)*nz + z + 8];
                  const float u16 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 6];
                  const float u17 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 10];
                  const float u18 = uf[A0 + (x + 8)*ny*nz + (y + 10)*nz + z + 8];
                  const float u19 = uf[A0 + (x + 10)*ny*nz + (y + 8)*nz + z + 8];
                  const float u20 = u14 + u15 + u16 + u17 + u18 + u19;

                  //const float u21 = u[t0][x + 7][y + 8][z + 8];
                  //const float u22 = u[t0][x + 8][y + 7][z + 8];
                  //const float u23 = u[t0][x + 8][y + 8][z + 7];
                  //const float u24 = u[t0][x + 8][y + 8][z + 9];
                  //const float u25 = u[t0][x + 8][y + 9][z + 8];
                  //const float u26 = u[t0][x + 9][y + 8][z + 8];
                  const float u21 = uf[A0 + (x + 7)*ny*nz + (y + 8)*nz + z + 8];
                  const float u22 = uf[A0 + (x + 8)*ny*nz + (y + 7)*nz + z + 8];
                  const float u23 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 7];
                  const float u24 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 9];
                  const float u25 = uf[A0 + (x + 8)*ny*nz + (y + 9)*nz + z + 8];
                  const float u26 = uf[A0 + (x + 9)*ny*nz + (y + 8)*nz + z + 8];
                  const float u27 = u21 + u22 + u23 + u24 + u25 + u26;

                  //const float u28 = -2.0F*u[t0][x + 8][y + 8][z + 8];
                  //const float u29 = u[t1][x + 8][y + 8][z + 8];
                  const float u28 = -2.0F*uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 8];
                  const float u29 = uf[t1*nx*ny*nz + (x + 8)*ny*nz + (y + 8)*nz + z + 8];
                  const float u30 = u28 + u29;

                  const float u31 = uf[A0 + (x + 8)*ny*nz + (y + 8)*nz + z + 8];

                  const float u32 = -7.93650813e-6F*u6 + 1.12874782e-4F*u13 - 8.8888891e-4F*u20 + 7.11111128e-3F*u27 - 3.79629639e-2F*u31;

                  u[t2][x + 8][y + 8][z + 8] = r49[x + 8][y + 8][z + 8]*r14*(u32 + (-(u30)*r15)*r50[x + 8][y + 8][z + 8]);
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
/* Backdoor edit at Tue Dec 22 00:30:07 2020*/ 
