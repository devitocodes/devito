/* Devito generated code for Operator `Kernel` */

#define _POSIX_C_SOURCE 200809L
#define START(S) struct timeval start_ ## S , end_ ## S ; gettimeofday(&start_ ## S , NULL);
#define STOP(S,T) gettimeofday(&end_ ## S, NULL); T->S += (double)(end_ ## S .tv_sec-start_ ## S.tv_sec)+(double)(end_ ## S .tv_usec-start_ ## S .tv_usec)/1000000;

#include "stdlib.h"
#include "math.h"
#include "sys/time.h"
#include "petscsnes.h"
#include "petscdmda.h"
#include "petscsection.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

struct Field0
{
  PetscScalar v;
  PetscScalar u;
} ;

struct JacobianCtx
{
  DM * subdms;
  IS * fields;
  Mat * submats;
} ;

struct SubMatrixCtx
{
  IS * rows;
  IS * cols;
} ;

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
  struct dataobj * f_vec;
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
  char ksptype0[64];
  PetscInt snesits0;
} ;

struct profiler
{
  PetscScalar section0;
} ;

PetscErrorCode CountBCs0(DM dm0, PetscInt * numBCPtr0);
PetscErrorCode CountBCs1(DM dm0, PetscInt * numBCPtr1);
PetscErrorCode SetPointBCs(DM dm0, PetscInt numBC0, PetscInt numBC1);
PetscErrorCode SetPetscOptions0();
PetscErrorCode WholeMatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode WholeFormFunc0(SNES snes, Vec X, Vec F, void* dummy);
PetscErrorCode ClearPetscOptions0();
PetscErrorCode DestroySubMatrixCtx0(Mat J);
PetscErrorCode J00_MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode J10_MatMult0(Mat J, Vec X, Vec Y);
PetscErrorCode MatCreateSubMatrices0(Mat J, PetscInt nfields, IS * irow, IS * icol, MatReuse scall, Mat * * submats);
PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2);
PetscErrorCode PopulateMatContext(struct JacobianCtx * jctx, DM * subdms, IS * fields);

int Kernel(struct dataobj * u_vec, struct dataobj * v_vec, struct petscprofiler0 * petscinfo0, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2, struct profiler * timers)
{
  Mat J0;
  PetscScalar atol0;
  DM da0;
  PetscScalar divtol0;
  KSP ksp0;
  PetscInt kspits0;
  KSPNormType kspnormtype0;
  KSPType ksptype0;
  PetscInt localsize0;
  PetscInt max_it0;
  PetscInt nfields0;
  KSPConvergedReason reason0;
  PetscScalar rtol0;
  VecScatter scatteru0;
  VecScatter scatterv0;
  PetscMPIInt size;
  SNES snes0;
  PetscInt snesits0;
  Vec xglobal0;
  Vec xglobalu0;
  Vec xglobalv0;
  Vec xlocal0;
  Vec xlocalu0;
  Vec xlocalv0;

  struct UserCtx0 ctx0;
  IS * fields0;
  struct JacobianCtx jctx0;
  DM * subdms0;
  PetscSection gsection0;
  PetscSection lsection0;
  PetscInt numBC0 = 0;
  PetscInt numBC1 = 0;
  PetscSF sf0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_GHOSTED,DM_BOUNDARY_GHOSTED,DMDA_STENCIL_BOX,9,9,1,1,2,2,NULL,NULL,&da0));
  PetscCall(PetscOptionsSetValue(NULL,"-da_use_section",NULL));
  PetscCall(DMSetFromOptions(da0));
  PetscCall(DMSetUp(da0));
  PetscCall(DMSetMatType(da0,MATSHELL));
  PetscCall(PopulateUserContext0(&ctx0,f_vec,h_x,h_y,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2,y_M,y_ltkn1,y_ltkn2,y_m,y_rtkn0,y_rtkn2));
  PetscCall(DMSetApplicationContext(da0,&ctx0));
  PetscCall(CountBCs0(da0,&numBC0));
  PetscCall(CountBCs1(da0,&numBC1));
  PetscCall(SetPointBCs(da0,numBC0,numBC1));
  PetscCall(DMGetLocalSection(da0,&lsection0));
  PetscCall(DMGetPointSF(da0,&sf0));
  PetscCall(PetscSectionCreateGlobalSection(lsection0,sf0,PETSC_TRUE,PETSC_FALSE,PETSC_FALSE,&gsection0));
  PetscCall(DMSetGlobalSection(da0,gsection0));
  PetscCall(DMCreateSectionSF(da0,lsection0,gsection0));
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes0));
  PetscCall(SNESSetOptionsPrefix(snes0,"biharm_"));
  PetscCall(SetPetscOptions0());
  PetscCall(SNESSetDM(snes0,da0));
  PetscCall(DMCreateMatrix(da0,&J0));
  PetscCall(SNESSetJacobian(snes0,J0,J0,MatMFFDComputeJacobian,NULL));
  PetscCall(DMCreateGlobalVector(da0,&xglobal0));
  PetscCall(DMCreateLocalVector(da0,&xlocal0));
  PetscCall(VecGetSize(xlocal0,&localsize0));
  PetscCall(SNESGetKSP(snes0,&ksp0));
  PetscCall(MatShellSetOperation(J0,MATOP_MULT,(void (*)(void))WholeMatMult0));
  PetscCall(SNESSetFunction(snes0,NULL,WholeFormFunc0,(void*)(da0)));
  PetscCall(SNESSetFromOptions(snes0));
  PetscCall(PopulateUserContext0(&ctx0,f_vec,h_x,h_y,x_M,x_ltkn0,x_ltkn1,x_m,x_rtkn0,x_rtkn2,y_M,y_ltkn1,y_ltkn2,y_m,y_rtkn0,y_rtkn2));
  PetscCall(MatSetDM(J0,da0));
  PetscCall(DMCreateFieldDecomposition(da0,&nfields0,NULL,&fields0,&subdms0));
  PetscCall(MatShellSetOperation(J0,MATOP_CREATE_SUBMATRICES,(void (*)(void))MatCreateSubMatrices0));
  PetscCall(PopulateMatContext(&jctx0,subdms0,fields0));
  PetscCall(MatShellSetContext(J0,&jctx0));
  PetscCall(MatCreateSubMatrices(J0,nfields0,fields0,fields0,MAT_INITIAL_MATRIX,&jctx0.submats));
  DM dav0 = subdms0[0];
  DM dau0 = subdms0[1];
  PetscCall(DMCreateGlobalVector(dav0,&xglobalv0));
  PetscCall(DMCreateGlobalVector(dau0,&xglobalu0));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,169,PETSC_DECIDE,v_vec->data,&xlocalv0));
  PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,169,PETSC_DECIDE,u_vec->data,&xlocalu0));

  START(section0)
  PetscCall(DMLocalToGlobal(dav0,xlocalv0,INSERT_VALUES,xglobalv0));
  PetscCall(VecScatterCreate(xglobal0,fields0[0],xglobalv0,NULL,&scatterv0));
  PetscCall(VecScatterBegin(scatterv0,xglobalv0,xglobal0,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatterv0,xglobalv0,xglobal0,INSERT_VALUES,SCATTER_REVERSE));

  PetscCall(DMLocalToGlobal(dau0,xlocalu0,INSERT_VALUES,xglobalu0));
  PetscCall(VecScatterCreate(xglobal0,fields0[1],xglobalu0,NULL,&scatteru0));
  PetscCall(VecScatterBegin(scatteru0,xglobalu0,xglobal0,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(scatteru0,xglobalu0,xglobal0,INSERT_VALUES,SCATTER_REVERSE));

  PetscCall(SNESSolve(snes0,NULL,xglobal0));
  PetscCall(VecScatterBegin(scatterv0,xglobal0,xglobalv0,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatterv0,xglobal0,xglobalv0,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(DMGlobalToLocal(dav0,xglobalv0,INSERT_VALUES,xlocalv0));
  PetscCall(VecScatterBegin(scatteru0,xglobal0,xglobalu0,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatteru0,xglobal0,xglobalu0,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(DMGlobalToLocal(dau0,xglobalu0,INSERT_VALUES,xlocalu0));

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
  PetscCall(PetscStrncpy(petscinfo0->ksptype0,ksptype0,64));
  PetscCall(SNESGetIterationNumber(snes0,&snesits0));
  petscinfo0->snesits0 = snesits0;
  STOP(section0,timers)
  PetscCall(ClearPetscOptions0());

  PetscCall(MatDestroy(&jctx0.submats[0]));
  PetscCall(MatDestroy(&jctx0.submats[1]));
  PetscCall(MatDestroy(&jctx0.submats[2]));
  PetscCall(MatDestroy(&jctx0.submats[3]));
  PetscCall(PetscFree(jctx0.submats));
  PetscCall(ISDestroy(&fields0[0]));
  PetscCall(ISDestroy(&fields0[1]));
  PetscCall(PetscFree(fields0));
  PetscCall(DMDestroy(&subdms0[0]));
  PetscCall(DMDestroy(&subdms0[1]));
  PetscCall(PetscFree(subdms0));
  PetscCall(VecScatterDestroy(&scatteru0));
  PetscCall(VecScatterDestroy(&scatterv0));
  PetscCall(VecDestroy(&xglobal0));
  PetscCall(VecDestroy(&xglobalu0));
  PetscCall(VecDestroy(&xglobalv0));
  PetscCall(VecDestroy(&xlocal0));
  PetscCall(VecDestroy(&xlocalu0));
  PetscCall(VecDestroy(&xlocalv0));
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
  PetscCall(DMGetApplicationContext(dm0, &ctx0));
  PetscInt count = *numBCPtr0;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* SubTop (middle x, top y) and SubBottom (middle x, bottom y) */
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
      count += 1;
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
      count += 1;
  }
  /* SubLeft (left x, full y) */
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      count += 1;
  }
  /* SubRight (right x, full y) */
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      count += 1;
  }

  *numBCPtr0 = count;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CountBCs1(DM dm0, PetscInt * numBCPtr1)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0, &ctx0));
  PetscInt count = *numBCPtr1;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* SubTop (middle x, top y) and SubBottom (middle x, bottom y) */
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
      count += 1;
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
      count += 1;
  }
  /* SubLeft (left x, full y) */
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      count += 1;
  }
  /* SubRight (right x, full y) */
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      count += 1;
  }

  *numBCPtr1 = count;

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetPointBCs(DM dm0, PetscInt numBC0, PetscInt numBC1)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  PetscCall(DMGetApplicationContext(dm0, &ctx0));
  DMDALocalInfo info;
  PetscCall(DMDAGetLocalInfo(dm0, &info));

  PetscInt k0 = 0;
  PetscInt k1 = 0;
  PetscInt * bcPointsArr0;
  PetscInt * bcPointsArr1;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  /* --- field 0 (v, DOF index 0) --- */
  PetscCall(PetscMalloc1(numBC0, &bcPointsArr0));
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
      bcPointsArr0[k0++] = info.gxm * (ix + 2) + (iy + 2);
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
      bcPointsArr0[k0++] = info.gxm * (ix + 2) + (iy + 2);
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      bcPointsArr0[k0++] = info.gxm * (ix + 2) + (y + 2);
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      bcPointsArr0[k0++] = info.gxm * (ix + 2) + (y + 2);
  }

  /* --- field 1 (u, DOF index 1) --- */
  PetscCall(PetscMalloc1(numBC1, &bcPointsArr1));
  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
      bcPointsArr1[k1++] = info.gxm * (ix + 2) + (iy + 2);
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
      bcPointsArr1[k1++] = info.gxm * (ix + 2) + (iy + 2);
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      bcPointsArr1[k1++] = info.gxm * (ix + 2) + (y + 2);
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
      bcPointsArr1[k1++] = info.gxm * (ix + 2) + (y + 2);
  }

  /* Build IS arrays and register with DM */
  IS bcPointsIS[2];
  IS bcCompsIS[2];
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm0),
            numBC0, bcPointsArr0, PETSC_OWN_POINTER, &bcPointsIS[0]));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm0),
            numBC1, bcPointsArr1, PETSC_OWN_POINTER, &bcPointsIS[1]));

  PetscInt comp0[] = {0};
  PetscInt comp1[] = {1};
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm0),
            1, comp0, PETSC_COPY_VALUES, &bcCompsIS[0]));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm0),
            1, comp1, PETSC_COPY_VALUES, &bcCompsIS[1]));

  /* DMDASetPointBC duplicates all ISes internally */
  PetscCall(DMDASetPointBC(dm0, 2, bcPointsIS, bcCompsIS));

  PetscCall(ISDestroy(&bcPointsIS[0]));
  PetscCall(ISDestroy(&bcPointsIS[1]));
  PetscCall(ISDestroy(&bcCompsIS[0]));
  PetscCall(ISDestroy(&bcCompsIS[1]));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsSetValue(NULL,"-biharm_snes_type","ksponly"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_ksp_type","gmres"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_pc_type","none"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_ksp_rtol","1e-10"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_ksp_atol","1e-50"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_ksp_divtol","100000.0"));
  PetscCall(PetscOptionsSetValue(NULL,"-biharm_ksp_max_it","10000"));

  PetscFunctionReturn(0);
}

PetscErrorCode WholeMatMult0(Mat J, Vec X, Vec Y)
{
  Vec J00X;
  Vec J00Y;
  Vec J10X;
  Vec J10Y;
  Vec J11X;
  Vec J11Y;

  struct SubMatrixCtx * J00ctx;
  struct SubMatrixCtx * J10ctx;
  struct SubMatrixCtx * J11ctx;
  struct JacobianCtx * jctx;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(J,&jctx));
  PetscCall(VecSet(Y,0.0));

  Mat J00 = jctx->submats[0];
  PetscCall(MatShellGetContext(J00,&J00ctx));
  PetscCall(VecGetSubVector(X,*J00ctx->cols,&J00X));
  PetscCall(VecGetSubVector(Y,*J00ctx->rows,&J00Y));
  PetscCall(MatMult(J00,J00X,J00Y));
  PetscCall(VecRestoreSubVector(X,*J00ctx->cols,&J00X));
  PetscCall(VecRestoreSubVector(Y,*J00ctx->rows,&J00Y));
  Mat J10 = jctx->submats[2];
  PetscCall(MatShellGetContext(J10,&J10ctx));
  PetscCall(VecGetSubVector(X,*J10ctx->cols,&J10X));
  PetscCall(VecGetSubVector(Y,*J10ctx->rows,&J10Y));
  PetscCall(MatMult(J10,J10X,J10Y));
  PetscCall(VecRestoreSubVector(X,*J10ctx->cols,&J10X));
  PetscCall(VecRestoreSubVector(Y,*J10ctx->rows,&J10Y));
  Mat J11 = jctx->submats[3];
  PetscCall(MatShellGetContext(J11,&J11ctx));
  PetscCall(VecGetSubVector(X,*J11ctx->cols,&J11X));
  PetscCall(VecGetSubVector(Y,*J11ctx->rows,&J11Y));
  PetscCall(MatMult(J11,J11X,J11Y));
  PetscCall(VecRestoreSubVector(X,*J11ctx->cols,&J11X));
  PetscCall(VecRestoreSubVector(Y,*J11ctx->rows,&J11Y));

  PetscFunctionReturn(0);
}

PetscErrorCode WholeFormFunc0(SNES snes, Vec X, Vec F, void* dummy)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0 = (DM)(dummy);
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  Vec floc;
  DMDALocalInfo info;
  Vec xloc;

  PetscScalar * f_bundle_vec;
  PetscScalar * x_bundle_vec;

  PetscCall(VecSet(F,0.0));
  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&floc));
  PetscCall(VecGetArray(floc,&f_bundle_vec));
  PetscCall(VecGetArray(xloc,&x_bundle_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));
  struct dataobj * f_vec = ctx0->f_vec;

  PetscScalar (* f)[f_vec->size[1]] __attribute__ ((aligned (64))) = (PetscScalar (*)[f_vec->size[1]]) f_vec->data;
  struct Field0 (* f_bundle)[info.gxm] = (struct Field0 (*)[info.gxm]) f_bundle_vec;
  struct Field0 (* x_bundle)[info.gxm] = (struct Field0 (*)[info.gxm]) x_bundle_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      f_bundle[ix + 2][iy + 2].v = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][iy + 2].v;
      x_bundle[ix + 2][iy + 2].v = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      f_bundle[ix + 2][iy + 2].v = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][iy + 2].v;
      x_bundle[ix + 2][iy + 2].v = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_M - ctx0->y_rtkn0 + 1; iy <= ctx0->y_M; iy += 1)
    {
      f_bundle[ix + 2][iy + 2].u = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][iy + 2].u;
      x_bundle[ix + 2][iy + 2].u = 0.0;
    }
    #pragma omp simd
    for (int iy = ctx0->y_m; iy <= ctx0->y_m + ctx0->y_ltkn1 - 1; iy += 1)
    {
      f_bundle[ix + 2][iy + 2].u = (2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x)))*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][iy + 2].u;
      x_bundle[ix + 2][iy + 2].u = 0.0;
    }
  }
  for (int ix = ctx0->x_m; ix <= ctx0->x_m + ctx0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      PetscScalar r6 = 2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x));
      f_bundle[ix + 2][y + 2].v = r6*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][y + 2].v;
      x_bundle[ix + 2][y + 2].v = 0.0;
      f_bundle[ix + 2][y + 2].u = r6*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][y + 2].u;
      x_bundle[ix + 2][y + 2].u = 0.0;
    }
  }
  for (int ix = ctx0->x_M - ctx0->x_rtkn2 + 1; ix <= ctx0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = ctx0->y_m; y <= ctx0->y_M; y += 1)
    {
      PetscScalar r7 = 2.0/((ctx0->h_y*ctx0->h_y)) + 2.0/((ctx0->h_x*ctx0->h_x));
      f_bundle[ix + 2][y + 2].v = r7*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][y + 2].v;
      x_bundle[ix + 2][y + 2].v = 0.0;
      f_bundle[ix + 2][y + 2].u = r7*ctx0->h_x*ctx0->h_y*x_bundle[ix + 2][y + 2].u;
      x_bundle[ix + 2][y + 2].u = 0.0;
    }
  }

  PetscScalar r4 = 1.0/(ctx0->h_x*ctx0->h_x);
  PetscScalar r5 = 1.0/(ctx0->h_y*ctx0->h_y);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd aligned(f:32)
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      f_bundle[ix + 2][iy + 2].v = (2.0*(r4*x_bundle[ix + 2][iy + 2].v + r5*x_bundle[ix + 2][iy + 2].v) - (r4*x_bundle[ix + 1][iy + 2].v + r4*x_bundle[ix + 3][iy + 2].v + r5*x_bundle[ix + 2][iy + 1].v + r5*x_bundle[ix + 2][iy + 3].v + f[ix + 2][iy + 2]))*ctx0->h_x*ctx0->h_y;
      f_bundle[ix + 2][iy + 2].u = (2.0*(r4*x_bundle[ix + 2][iy + 2].u + r5*x_bundle[ix + 2][iy + 2].u) - (r4*x_bundle[ix + 1][iy + 2].u + r4*x_bundle[ix + 3][iy + 2].u + r5*x_bundle[ix + 2][iy + 1].u + r5*x_bundle[ix + 2][iy + 3].u + x_bundle[ix + 2][iy + 2].v))*ctx0->h_x*ctx0->h_y;
    }
  }
  PetscCall(VecRestoreArray(floc,&f_bundle_vec));
  PetscCall(VecRestoreArray(xloc,&x_bundle_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,floc,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(dm0,floc,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&floc));

  PetscFunctionReturn(0);
}

PetscErrorCode ClearPetscOptions0()
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(PetscOptionsClearValue(NULL,"-biharm_snes_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_ksp_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_pc_type"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_ksp_rtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_ksp_atol"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_ksp_divtol"));
  PetscCall(PetscOptionsClearValue(NULL,"-biharm_ksp_max_it"));

  PetscFunctionReturn(0);
}

PetscErrorCode DestroySubMatrixCtx0(Mat J)
{
  PetscFunctionBeginUser;

  struct SubMatrixCtx * subctx;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscCall(MatShellGetContext(J,&subctx));
  PetscCall(PetscFree(subctx));

  PetscFunctionReturn(0);
}

PetscErrorCode J00_MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * o0;
  DM dm0;
  PetscCall(MatGetDM(J,&dm0));
  PetscCall(DMGetApplicationContext(dm0,&o0));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * a0_vec;
  PetscScalar * a1_vec;

  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&a1_vec));
  PetscCall(VecGetArray(xloc,&a0_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* a0)[info.gxm] = (PetscScalar (*)[info.gxm]) a0_vec;
  PetscScalar (* a1)[info.gxm] = (PetscScalar (*)[info.gxm]) a1_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = o0->x_m + o0->x_ltkn0; ix <= o0->x_M - o0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = o0->y_M - o0->y_rtkn0 + 1; iy <= o0->y_M; iy += 1)
    {
      a1[ix + 2][iy + 2] = (2.0/((o0->h_y*o0->h_y)) + 2.0/((o0->h_x*o0->h_x)))*o0->h_x*o0->h_y*a0[ix + 2][iy + 2];
      a0[ix + 2][iy + 2] = 0.0;
    }
    #pragma omp simd
    for (int iy = o0->y_m; iy <= o0->y_m + o0->y_ltkn1 - 1; iy += 1)
    {
      a1[ix + 2][iy + 2] = (2.0/((o0->h_y*o0->h_y)) + 2.0/((o0->h_x*o0->h_x)))*o0->h_x*o0->h_y*a0[ix + 2][iy + 2];
      a0[ix + 2][iy + 2] = 0.0;
    }
  }
  for (int ix = o0->x_m; ix <= o0->x_m + o0->x_ltkn1 - 1; ix += 1)
  {
    #pragma omp simd
    for (int y = o0->y_m; y <= o0->y_M; y += 1)
    {
      a1[ix + 2][y + 2] = (2.0/((o0->h_y*o0->h_y)) + 2.0/((o0->h_x*o0->h_x)))*o0->h_x*o0->h_y*a0[ix + 2][y + 2];
      a0[ix + 2][y + 2] = 0.0;
    }
  }
  for (int ix = o0->x_M - o0->x_rtkn2 + 1; ix <= o0->x_M; ix += 1)
  {
    #pragma omp simd
    for (int y = o0->y_m; y <= o0->y_M; y += 1)
    {
      a1[ix + 2][y + 2] = (2.0/((o0->h_y*o0->h_y)) + 2.0/((o0->h_x*o0->h_x)))*o0->h_x*o0->h_y*a0[ix + 2][y + 2];
      a0[ix + 2][y + 2] = 0.0;
    }
  }

  PetscScalar r0 = 1.0/(o0->h_x*o0->h_x);
  PetscScalar r1 = 1.0/(o0->h_y*o0->h_y);

  for (int ix = o0->x_m + o0->x_ltkn0; ix <= o0->x_M - o0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = o0->y_m + o0->y_ltkn2; iy <= o0->y_M - o0->y_rtkn2; iy += 1)
    {
      a1[ix + 2][iy + 2] = (2.0*(r0*a0[ix + 2][iy + 2] + r1*a0[ix + 2][iy + 2]) - (r0*a0[ix + 1][iy + 2] + r0*a0[ix + 3][iy + 2] + r1*a0[ix + 2][iy + 1] + r1*a0[ix + 2][iy + 3]))*o0->h_x*o0->h_y;
    }
  }
  PetscCall(VecRestoreArray(yloc,&a1_vec));
  PetscCall(VecRestoreArray(xloc,&a0_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode J10_MatMult0(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  struct UserCtx0 * ctx0;
  DM dm0;
  PetscCall(MatGetDM(J,&dm0));
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  PetscScalar * x_v_vec;
  PetscScalar * y_u_vec;

  PetscCall(DMGetLocalVector(dm0,&xloc));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&yloc));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_u_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&info));

  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;
  PetscScalar (* y_u)[info.gxm] = (PetscScalar (*)[info.gxm]) y_u_vec;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  for (int ix = ctx0->x_m + ctx0->x_ltkn0; ix <= ctx0->x_M - ctx0->x_rtkn0; ix += 1)
  {
    #pragma omp simd
    for (int iy = ctx0->y_m + ctx0->y_ltkn2; iy <= ctx0->y_M - ctx0->y_rtkn2; iy += 1)
    {
      y_u[ix + 2][iy + 2] = -ctx0->h_x*ctx0->h_y*x_v[ix + 2][iy + 2];
    }
  }
  PetscCall(VecRestoreArray(yloc,&y_u_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&xloc));
  PetscCall(DMRestoreLocalVector(dm0,&yloc));

  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrices0(Mat J, PetscInt nfields, IS * irow, IS * icol, MatReuse scall, Mat * * submats)
{
  PetscFunctionBeginUser;

  Mat block;
  DM dm0;
  PetscInt dof;

  struct UserCtx0 * ctx0;
  struct JacobianCtx * jctx;
  struct SubMatrixCtx * subctx;

  PetscCall(MatShellGetContext(J,&jctx));
  DM * subdms = jctx->subdms;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  PetscInt nsubmats = nfields*nfields;
  PetscCall(PetscCalloc1(nsubmats,submats));
  PetscCall(MatGetDM(J,&dm0));
  PetscCall(DMGetApplicationContext(dm0,&ctx0));
  PetscCall(DMDAGetInfo(dm0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
  Mat * submat_arr = *submats;

  for (int i = 0; i <= nsubmats - 1; i += 1)
  {
    PetscInt rowidx = i / dof;
    PetscInt colidx = (i)%(dof);
    /* Query constrained global sizes from sub-DMs (accounts for BC-excluded DOFs) */
    Vec rowvec, colvec;
    PetscInt subblockrows, subblockcols;
    PetscCall(DMGetGlobalVector(subdms[rowidx], &rowvec));
    PetscCall(VecGetSize(rowvec, &subblockrows));
    PetscCall(DMRestoreGlobalVector(subdms[rowidx], &rowvec));
    PetscCall(DMGetGlobalVector(subdms[colidx], &colvec));
    PetscCall(VecGetSize(colvec, &subblockcols));
    PetscCall(DMRestoreGlobalVector(subdms[colidx], &colvec));
    PetscCall(MatCreate(PETSC_COMM_WORLD,&block));
    PetscCall(MatSetSizes(block,PETSC_DECIDE,PETSC_DECIDE,subblockrows,subblockcols));
    PetscCall(MatSetType(block,MATSHELL));
    PetscCall(PetscMalloc1(1,&subctx));
    subctx->rows = &irow[rowidx];
    subctx->cols = &icol[colidx];
    PetscCall(DMSetApplicationContext(subdms[rowidx],ctx0));
    PetscCall(MatSetDM(block,subdms[rowidx]));
    PetscCall(MatShellSetContext(block,subctx));
    PetscCall(MatShellSetOperation(block,MATOP_DESTROY,(void (*)(void))DestroySubMatrixCtx0));
    PetscCall(MatSetUp(block));
    submat_arr[i] = block;
  }
  PetscCall(MatShellSetOperation(submat_arr[0],MATOP_MULT,(void (*)(void))J00_MatMult0));
  PetscCall(MatShellSetOperation(submat_arr[2],MATOP_MULT,(void (*)(void))J10_MatMult0));
  PetscCall(MatShellSetOperation(submat_arr[3],MATOP_MULT,(void (*)(void))J00_MatMult0));

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateUserContext0(struct UserCtx0 * ctx0, struct dataobj * f_vec, const PetscScalar h_x, const PetscScalar h_y, const PetscInt x_M, const PetscInt x_ltkn0, const PetscInt x_ltkn1, const PetscInt x_m, const PetscInt x_rtkn0, const PetscInt x_rtkn2, const PetscInt y_M, const PetscInt y_ltkn1, const PetscInt y_ltkn2, const PetscInt y_m, const PetscInt y_rtkn0, const PetscInt y_rtkn2)
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
  ctx0->f_vec = f_vec;

  PetscFunctionReturn(0);
}

PetscErrorCode PopulateMatContext(struct JacobianCtx * jctx, DM * subdms, IS * fields)
{
  PetscFunctionBeginUser;

  /* Flush denormal numbers to zero in hardware */
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

  jctx->subdms = subdms;
  jctx->fields = fields;

  PetscFunctionReturn(0);
}
/* Backdoor edit at Mon Mar  2 19:26:39 2026*/ 
/* Backdoor edit at Mon Mar  2 19:28:58 2026*/ 
/* Backdoor edit at Tue Mar  3 10:15:06 2026*/ 
/* Backdoor edit at Tue Mar  3 10:30:52 2026*/ 
/* Backdoor edit at Tue Mar  3 10:43:27 2026*/ 
/* Backdoor edit at Tue Mar  3 10:44:26 2026*/ 
/* Backdoor edit at Tue Mar  3 10:44:57 2026*/ 
/* Backdoor edit at Tue Mar  3 11:16:06 2026*/ 
/* Backdoor edit at Tue Mar  3 11:16:35 2026*/ 
/* Backdoor edit at Tue Mar  3 11:31:42 2026*/ 
/* Backdoor edit at Tue Mar  3 11:32:14 2026*/ 
/* Backdoor edit at Tue Mar  3 11:32:29 2026*/ 
/* Backdoor edit at Tue Mar  3 11:38:02 2026*/ 
/* Backdoor edit at Tue Mar  3 11:38:28 2026*/ 
/* Backdoor edit at Tue Mar  3 11:38:51 2026*/ 
/* Backdoor edit at Tue Mar  3 11:39:47 2026*/ 
