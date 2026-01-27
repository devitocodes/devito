/* Devito generated code for Operator `Kernel` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscsnes.h"
#include "petscdmda.h"
#include "petscsection.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct UserCtx0
{
  PetscScalar h_x;
  PetscScalar h_y;
  PetscInt x_M;
  PetscInt x_ltkn0;
  PetscInt x_ltkn1;
  PetscInt x_m;
  PetscInt x_rtkn0;
  PetscInt x_rtkn2;
  PetscInt y_M;
  PetscInt y_ltkn1;
  PetscInt y_ltkn2;
  PetscInt y_m;
  PetscInt y_rtkn0;
  PetscInt y_rtkn2;
  struct dataobj * bc_vec;
  struct dataobj * f_vec;
  PetscInt ix_max1;
  PetscInt ix_min1;
} ;

struct dataobj
{
  void * data;
  PetscInt * size;
  unsigned long nbytes;
  unsigned long * npsize;
  unsigned long * dsize;
  PetscInt * hsize;
  PetscInt * hofs;
  PetscInt * oofs;
  void * dmap;
} ;

struct petscprofiler0
{
  KSPConvergedReason reason0;
  PetscInt kspits0;
  KSPNormType kspnormtype0;
  PetscScalar rtol0;
  PetscScalar atol0;
  PetscScalar divtol0;
  PetscInt max_it0;
  KSPType ksptype0;
  PetscInt snesits0;
} ;

struct profiler
{
  PetscScalar section0;
} ;

PetscErrorCode CountBCs0(DM dm0, PetscInt * numBCPtr0);
PetscErrorCode SetPointBCs0(DM dm0, PetscInt numBC0);
PetscErrorCode SetPetscOptions0();
PetscErrorCode MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode FormRHS0(DM dm0, Vec B);
PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc);
PetscErrorCode ClearPetscOptions0();
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * bc_vec, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt ix_max1, const PetscInt ix_min1, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2);

int Kernel(struct dataobj * u_vec, struct petscprofiler0 * petscinfo0, struct dataobj * bc_vec, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt ix_max1, const PetscInt ix_min1, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2, struct profiler * timers)
{
  Mat J0;
  PetscScalar atol0;
  Vec bglobal0;
  DM da0;
  PetscScalar divtol0;
  PetscSection gsection0;
  KSP ksp0;
  PetscInt kspits0;
  KSPNormType kspnormtype0;
  KSPType ksptype0;
  PetscInt localsize0;
  PetscSection lsection0;
  PetscInt max_it0;
  PetscInt numBC0 = 0;
  KSPConvergedReason reason0;
  PetscScalar rtol0;
  PetscSF sf0;
  PetscMPIInt size;
  SNES snes0;
  PetscInt snesits0;
  Vec xglobal0;
  Vec xlocal0;

  struct UserCtx0 ctx0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,17,17,1,1,1,2,NULL,NULL,&da0));
  PetscCall(PetscOptionsSetValue(NULL,"-da_use_section",NULL));
  PetscCall(DMSetFromOptions(da0));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(PopulateUserContext0(&ctx0,bc_vec,f_vec,h_x,h_y,ix_max1,ix_min1,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2,y_M,y_ltkn1,y_ltkn2,y_m,y_rtkn0,y_rtkn2));
  PetscCall(DMSetApplicationContext(da0,&ctx0));
  PetscCall(CountBCs0(da0,&numBC0));
  PetscCall(SetPointBCs0(da0,numBC0));
  PetscCall(DMGetLocalSection(da0,&lsection0));
  PetscCall(PetscSectionView(lsection0, NULL));
  PetscCall(DMGetPointSF(da0,&sf0));
  PetscCall(PetscSectionCreateGlobalSection(lsection0,sf0,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,&gsection0));
  PetscCall(DMSetGlobalSection(da0,gsection0));
  PetscCall(DMCreateSectionSF(da0,lsection0,gsection0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes0));
  PetscCall(SNESSetOptionsPrefix(snes0,"poisson_2d_"));
  PetscCall(SetPetscOptions0());
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&J0));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da0,&xglobal0));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,441,PETSC_DECIDE,u_vec->data,&xlocal0));
  PetscCall(VecGetSize(xlocal0,&localsize0));
  PetscCall(DMCreateGlobalVector(da0,&bglobal0));
  PetscCall(SNESGetKSP(snes0,&ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))MatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,FormFunction0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  // PetscCall(PopulateUserContext0(&ctx0,bc_vec,f_vec,h_x,h_y,ix_max1,ix_min1,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2,y_M,y_ltkn1,y_ltkn2,y_m,y_rtkn0,y_rtkn2));
  PetscCall(MatSetDM(J0,da0));
  // PetscCall(DMSetApplicationContext(da0,&ctx0));

  START(section0)
  PetscCall(FormRHS0(da0,bglobal0));
  PetscCall(FormInitialGuess0(da0,xlocal0));
  PetscCall(DMLocalToGlobal(da0,xlocal0,INSERT_VALUES,xglobal0));
  PetscCall(SNESSolve(snes0,bglobal0,xglobal0));
  PetscCall(DMGlobalToLocal(da0,xglobal0,INSERT_VALUES,xlocal0));

  PetscCall(KSPGetConvergedReason(ksp0,&reason0));
  petscinfo0->reason0 = reason0;
  PetscCall(KSPGetIterationNumber(ksp0,&kspits0));
  petscinfo0->kspits0 = kspits0;
  PetscCall(KSPGetNormType(ksp0,&kspnormtype0));
  petscinfo0->kspnormtype0 = kspnormtype0;
  PetscCall(KSPGetTolerances(ksp0,&rtol0,&atol0,&divtol0,&max_it0));
  petscinfo0->rtol0 = rtol0;
  petscinfo0->atol0 = atol0;
  petscinfo0->divtol0 = divtol0;
  petscinfo0->max_it0 = max_it0;
  PetscCall(KSPGetType(ksp0,&ksptype0));
  petscinfo0->ksptype0 = ksptype0;
  PetscCall(SNESGetIterationNumber(snes0,&snesits0));
  petscinfo0->snesits0 = snesits0;
  STOP(section0,timers)
  PetscCall(ClearPetscOptions0());

  PetscCall(VecDestroy(&bglobal0));
  PetscCall(VecDestroy(&xglobal0));
  PetscCall(VecDestroy(&xlocal0));
  PetscCall(MatDestroy(&J0));
  PetscCall(SNESDestroy(&snes0));
  PetscCall(PetscSectionDestroy(&gsection0));
  PetscCall(DMDestroy(&da0));

  return 0;
}

PetscErrorCode CountBCs0(DM dm0, PetscInt * numBCPtr0)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DMView(dm0, PETSC_VIEWER_STDOUT_WORLD);
  PetscCall(DMGetApplicationContext(dm0,&ctx0));

  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);


  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      (*numBCPtr0)++;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      (*numBCPtr0)++;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      (*numBCPtr0)++;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      (*numBCPtr0)++;
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode SetPointBCs0(DM dm0, PetscInt numBC0)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  PetscInt k_iter = 0;
  DMDALocalInfo info;
  IS       bcPointsIS;

  PetscInt * bcPointsArr0;

  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * bc_vec = ctx0->bc_vec;

  PetscScalar (* bc)[bc_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[bc_vec->size[1]]) bc_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  printf("numBC0 = %d\n", numBC0);

  PetscCall(PetscMalloc1(numBC0, &bcPointsArr0));

  // NOTE TODO: the loops were wrong in 2D so fix it

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      bcPointsArr0[k_iter++] = (ix+2)*21 + (iy+2);
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      bcPointsArr0[k_iter++] = (ix+2)*21 + (iy+2);
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      bcPointsArr0[k_iter++] = (ix+2)*21 + (y+2);
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      bcPointsArr0[k_iter++] = (ix+2)*21 + (y+2);
    }
  }
  // create an IS of boundary points
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm0), numBC0, bcPointsArr0, PETSC_OWN_POINTER, &bcPointsIS));
  // view the IS
  // PetscCall(ISView(bcPointsIS, PETSC_VIEWER_STDOUT_WORLD));
  IS bcPoints[1] = {bcPointsIS};
  PetscCall(DMDASetPointBC(dm0, 1, bcPoints, NULL));

  PetscCall(ISDestroy(&bcPointsIS));
  PetscFunctionReturn(0);
}

PetscErrorCode SetPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_ksp_type","cg"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_ksp_rtol","1e-12"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-poisson_2d_ksp_max_it","10000"));

  PetscFunctionReturn(0);
}

PetscErrorCode MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0;
  PetscCall(MatGetDM(J,&dm0));
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * x_u_vec;
  PetscScalar * y_u_vec;

  PetscCall(VecSet(Y,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;
  PetscScalar (* y_u)[info.gxm] = (PetscScalar (*)[info.gxm]) y_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_u[ix + 2][iy + 2];
      x_u[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      y_u[ix + 2][y + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_u[ix + 2][y + 2];
      x_u[ix + 2][y + 2] = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      y_u[ix + 2][y + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_u[ix + 2][y + 2];
      x_u[ix + 2][y + 2] = 0.0;
    }
  }

  PetscScalar r0 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r1 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      y_u[ix + 2][iy + 2] = (2.0*(r0*x_u[ix + 2][iy + 2] + r1*x_u[ix + 2][iy + 2]) - (r0*x_u[ix + 1][iy + 2] + r0*x_u[ix + 3][iy + 2] + r1*x_u[ix + 2][iy + 1] + r1*x_u[ix + 2][iy + 3]))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormFunction0(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  PetscScalar * f_u_vec;
  PetscScalar * x_u_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&floc));
  PetscCall(VecGetArray(floc,&f_u_vec));
  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * bc_vec = ctx0->bc_vec;

  PetscScalar (* bc)[bc_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[bc_vec->size[1]]) bc_vec->data;
  PetscScalar (* f_u)[info.gxm] = (PetscScalar (*)[info.gxm]) f_u_vec;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*(-bc[ix + 2][iy + 2] + x_u[ix + 2][iy + 2])*ctx0->h_x*ctx0->h_y;
      x_u[ix + 2][iy + 2] = bc[ix + 2][iy + 2];
    }
    #pragma omp simd aligned(bc:32)
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*(-bc[ix + 2][iy + 2] + x_u[ix + 2][iy + 2])*ctx0->h_x*ctx0->h_y;
      x_u[ix + 2][iy + 2] = bc[ix + 2][iy + 2];
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      f_u[ix + 2][y + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*(-bc[ix + 2][y + 2] + x_u[ix + 2][y + 2])*ctx0->h_x*ctx0->h_y;
      x_u[ix + 2][y + 2] = bc[ix + 2][y + 2];
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      f_u[ix + 2][y + 2] = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*(-bc[ix + 2][y + 2] + x_u[ix + 2][y + 2])*ctx0->h_x*ctx0->h_y;
      x_u[ix + 2][y + 2] = bc[ix + 2][y + 2];
    }
  }

  PetscScalar r2 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r3 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      f_u[ix + 2][iy + 2] = (2.0*(r2*x_u[ix + 2][iy + 2] + r3*x_u[ix + 2][iy + 2]) - (r2*x_u[ix + 1][iy + 2] + r2*x_u[ix + 3][iy + 2] + r3*x_u[ix + 2][iy + 1] + r3*x_u[ix + 2][iy + 3]))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_u_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode FormRHS0(DM dm0, Vec B)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec blocal0;
  DMDALocalInfo info;

  PetscScalar * b_u_vec;

  PetscCall(DMGetLocalVector(dm0,&blocal0));
  PetscCall(DMGlobalToLocalBegin(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(DMGlobalToLocalEnd(dm0,B,INSERT_VALUES,blocal0));
  PetscCall(VecGetArray(blocal0,&b_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * f_vec = ctx0->f_vec;

  PetscScalar (* b_u)[info.gxm] = (PetscScalar (*)[info.gxm]) b_u_vec;
  PetscScalar (* f)[f_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f_vec->size[1]]) f_vec->data;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      b_u[ix + 2][iy + 2] = 0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      b_u[ix + 2][iy + 2] = 0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      b_u[ix + 2][y + 2] = 0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      b_u[ix + 2][y + 2] = 0;
    }
  }
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd aligned(f:32)
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      b_u[ix + 2][iy + 2] = ctx0->h_x*ctx0->h_y*f[ix + 2][iy + 2];
    }
  }
  PetscCall(DMLocalToGlobalBegin(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(DMLocalToGlobalEnd(dm0,blocal0,INSERT_VALUES,B));
  PetscCall(VecRestoreArray(blocal0,&b_u_vec));
  PetscCall(DMRestoreLocalVector(dm0,&blocal0));

  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialGuess0(DM dm0, Vec xloc)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;

  PetscScalar * x_u_vec;

  PetscCall(VecGetArray(xloc,&x_u_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * bc_vec = ctx0->bc_vec;

  PetscScalar (* bc)[bc_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[bc_vec->size[1]]) bc_vec->data;
  PetscScalar (* x_u)[info.gxm] = (PetscScalar (*)[info.gxm]) x_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      x_u[ix + 2][iy + 2] = bc[ix + 2][iy + 2];
    }
    #pragma omp simd aligned(bc:32)
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      x_u[ix + 2][iy + 2] = bc[ix + 2][iy + 2];
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      x_u[ix + 2][y + 2] = bc[ix + 2][y + 2];
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd aligned(bc:32)
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      x_u[ix + 2][y + 2] = bc[ix + 2][y + 2];
    }
  }
  PetscCall(VecRestoreArray(xloc,&x_u_vec));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-poisson_2d_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * bc_vec, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt ix_max1, const PetscInt ix_min1, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2)
{
  PetscFunctionBeginUser;

  ctx0->h_x = h_x;
  ctx0->h_y = h_y;
  ctx0->x_M = x_M;
  ctx0->x_ltkn0 = x_ltkn0;
  ctx0->x_ltkn1 = x_ltkn1;
  ctx0->x_m = x_m;
  ctx0->x_rtkn0 = x_rtkn0;
  ctx0->x_rtkn2 = x_rtkn2;
  ctx0->y_M = y_M;
  ctx0->y_ltkn1 = y_ltkn1;
  ctx0->y_ltkn2 = y_ltkn2;
  ctx0->y_m = y_m;
  ctx0->y_rtkn0 = y_rtkn0;
  ctx0->y_rtkn2 = y_rtkn2;
  ctx0->bc_vec = bc_vec;
  ctx0->f_vec = f_vec;
  ctx0->ix_max1 = ix_max1;
  ctx0->ix_min1 = ix_min1;

  PetscFunctionReturn(0);
}

