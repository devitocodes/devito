
// # ref - https://github.com/bueler/p4pdes/blob/master/c/ch7/biharm.c

static char help[] =
"Solve the linear biharmonic equation in 2D.  Equation is\n"
"  Lap^2 u = f\n"
"where Lap = - grad^2 is the positive Laplacian, equivalently\n"
"  u_xxxx + 2 u_xxyy + u_yyyy = f(x,y)\n"
"Domain is unit square  S = (0,1)^2.   Boundary conditions are homogeneous\n"
"simply-supported:  u = 0,  Lap u = 0.  The equation is rewritten as a\n"
"2x2 block system with SPD Laplacian blocks on the diagonal:\n"
"   | Lap |  0  |  | v |   | f | \n"
"   |-----|-----|  |---| = |---| \n"
"   | -I  | Lap |  | u |   | 0 | \n"
"Includes manufactured, polynomial exact solution.  The discretization is\n"
"structured-grid (DMDA) finite differences.  Includes analytical Jacobian.\n"
"Recommended preconditioning combines fieldsplit:\n"
"   -pc_type fieldsplit -pc_fieldsplit_type multiplicative|additive \n"
"with multigrid as the preconditioner for the diagonal blocks:\n"
"   -fieldsplit_v_pc_type mg|gamg -fieldsplit_u_pc_type mg|gamg\n"
"(GMG requires setting levels and Galerkin coarsening.)  One can also do\n"
"monolithic multigrid (-pc_type mg|gamg).\n\n";

#include <petsc.h>

typedef struct {
    PetscReal  v, u;
} Field;

typedef struct {
    PetscReal  (*f)(PetscReal x, PetscReal y);  // right-hand side
} BiharmCtx;

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

static PetscReal c(PetscReal x) {
    return x*x*x * (1.0-x)*(1.0-x)*(1.0-x);
}

static PetscReal ddc(PetscReal x) {
    return 6.0 * x * (1.0-x) * (1.0 - 5.0 * x + 5.0 * x*x);
}

static PetscReal d4c(PetscReal x) {
    return - 72.0 * (1.0 - 5.0 * x + 5.0 * x*x);
}

static PetscReal u_exact_fcn(PetscReal x, PetscReal y) {
    return c(x) * c(y);
}

static PetscReal lap_u_exact_fcn(PetscReal x, PetscReal y) {
    return - ddc(x) * c(y) - c(x) * ddc(y);  // Lap u = - grad^2 u
}

static PetscReal f_fcn(PetscReal x, PetscReal y) {
    return d4c(x) * c(y) + 2.0 * ddc(x) * ddc(y) + c(x) * d4c(y);  // Lap^2 u = grad^4 u
}

extern PetscErrorCode FormExactWLocal(DMDALocalInfo*, Field**, BiharmCtx*);
extern PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void* dummy);
extern PetscErrorCode J00_MatMult(Mat J, Vec X, Vec Y);
extern PetscErrorCode J10_MatMult(Mat J, Vec X, Vec Y);
extern PetscErrorCode J11_MatMult(Mat J, Vec X, Vec Y);
extern PetscErrorCode WholeMatMult(Mat J, Vec X, Vec Y);
PetscErrorCode MatCreateSubMatrices0(Mat J, PetscInt nfields, IS * irow, IS * icol, MatReuse scall, Mat * * submats);
extern PetscErrorCode PopulateMatContext(struct JacobianCtx * jctx, DM * subdms, IS * fields);

int main(int argc,char **argv) {
    DM             da;
    SNES           snes;
    Vec            w, w_initial, w_exact;
    BiharmCtx      user;
    Field          **aW;
    PetscReal      normv, normu, errv, erru;
    DMDALocalInfo  info;
    IS             *fields;
    DM             *subdms;
    PetscInt       nfields;

    struct JacobianCtx jctx0;
    Mat J;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    user.f = &f_fcn;
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                        6,6,PETSC_DECIDE,PETSC_DECIDE,
                        2,1,              // degrees of freedom, stencil width
                        NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMSetMatType(da, MATSHELL));
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0));
    PetscCall(DMDASetFieldName(da,0,"v"));
    PetscCall(DMDASetFieldName(da,1,"u"));
    PetscCall(DMCreateMatrix(da,&J));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(SNESSetFunction(snes,NULL,FormFunction,NULL));
    PetscCall(SNESSetType(snes,SNESKSPONLY));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(SNESSetJacobian(snes,J,J,MatMFFDComputeJacobian,NULL));
    PetscCall(MatShellSetOperation(J,MATOP_MULT,(void (*)(void))WholeMatMult));

    PetscCall(MatSetDM(J,da));
    PetscCall(DMCreateFieldDecomposition(da,&(nfields),NULL,&fields,&subdms));
    PetscCall(PopulateMatContext(&(jctx0),subdms,fields));
    PetscCall(MatShellSetContext(J,&(jctx0)));
    PetscCall(MatCreateSubMatrices0(J,nfields,fields,fields,MAT_INITIAL_MATRIX,&(jctx0.submats)));

    PetscCall(DMGetGlobalVector(da,&w_initial));
    PetscCall(VecSet(w_initial,0.0));
    PetscCall(SNESSolve(snes,NULL,w_initial));
    // PetscCall(VecView(w_initial,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(DMRestoreGlobalVector(da,&w_initial));
    PetscCall(DMDestroy(&da));

    PetscCall(SNESGetSolution(snes,&w));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));

    PetscCall(DMCreateGlobalVector(da,&w_exact));
    PetscCall(DMDAVecGetArray(da,w_exact,&aW));
    PetscCall(FormExactWLocal(&info,aW,&user));
    PetscCall(DMDAVecRestoreArray(da,w_exact,&aW));
    PetscCall(VecStrideNorm(w_exact,0,NORM_INFINITY,&normv));
    PetscCall(VecStrideNorm(w_exact,1,NORM_INFINITY,&normu));
    PetscCall(VecAXPY(w,-1.0,w_exact));
    PetscCall(VecStrideNorm(w,0,NORM_INFINITY,&errv));
    PetscCall(VecStrideNorm(w,1,NORM_INFINITY,&erru));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid ...\n"
        "  errors |v-vex|_inf/|vex|_inf = %.5e, |u-uex|_inf/|uex|_inf = %.5e\n",
        info.mx,info.my,errv/normv,erru/normu));


    PetscCall(ISDestroy(&(fields[0])));
    PetscCall(ISDestroy(&(fields[1])));
    PetscCall(PetscFree(fields));
    PetscCall(DMDestroy(&(subdms[0])));
    PetscCall(DMDestroy(&(subdms[1])));
    PetscCall(PetscFree(subdms));
    PetscCall(VecDestroy(&w_exact));
    PetscCall(MatDestroy(&J));
    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormExactWLocal(DMDALocalInfo *info, Field **aW, BiharmCtx *user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            aW[j][i].u = u_exact_fcn(x,y);
            aW[j][i].v = lap_u_exact_fcn(x,y);
        }
    }
    return 0;
}


PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void * dummy)
{
    Vec xlocal, flocal;
    DMDALocalInfo info;
    DM da;
    PetscScalar *x_vec, *f_vec;

    BiharmCtx *user;

    PetscCall(SNESGetDM(snes,&da));

    PetscCall(DMGetApplicationContext(da,&user));

    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(DMGetLocalVector(da,&xlocal));
    PetscCall(DMGetLocalVector(da,&flocal));

    PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,xlocal));
    PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,xlocal));

    PetscCall(VecGetArray(xlocal,&x_vec));
    PetscCall(VecGetArray(flocal,&f_vec));

    Field (*xx)[info.gxm] = (Field (*)[info.gxm]) x_vec;
    Field (*ff)[info.gxm] = (Field (*)[info.gxm]) f_vec;

    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
               ve, vw, vn, vs, ue, uw, un, us;

    hx = 1. / (info.mx - 1);
    hy = 1. / (info.my - 1);

    darea = hx * hy;               // multiply FD equations by this

    scx = hy / hx;
    scy = hx / hy;
    scdiag = 2.0 * (scx + scy);    // diagonal scaling
    for (j = info.ys; j < info.ys + info.ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                ff[j][i].v = scdiag * xx[j][i].v;
                ff[j][i].u = scdiag * xx[j][i].u;
            } else {
                ve = xx[j][i+1].v;
                vw = xx[j][i-1].v;
                vn = xx[j+1][i].v;
                vs = xx[j-1][i].v;
                ff[j][i].v = scdiag * xx[j][i].v - scx * (vw + ve) - scy * (vs + vn)
                             - darea * (*(user->f))(x,y);
                ue = xx[j][i+1].u;
                uw = xx[j][i-1].u;
                un = xx[j+1][i].u;
                us = xx[j-1][i].u;
                ff[j][i].u = - darea * xx[j][i].v
                             + scdiag * xx[j][i].u - scx * (uw + ue) - scy * (us + un);
            }
        }
    }

    PetscCall(VecRestoreArray(xlocal,&x_vec));
    PetscCall(VecRestoreArray(flocal,&f_vec));

    PetscCall(DMLocalToGlobalBegin(da,flocal,INSERT_VALUES,F));
    PetscCall(DMLocalToGlobalEnd(da,flocal,INSERT_VALUES,F));
    PetscCall(DMRestoreLocalVector(da,&xlocal));
    PetscCall(DMRestoreLocalVector(da,&flocal));

    return 0;
}


PetscErrorCode J00_MatMult(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  DM dm0;
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  BiharmCtx * ctx0;
  PetscScalar * x_v_vec;
  PetscScalar * y_v_vec;

  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(yloc)));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_v_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;
  PetscScalar (* y_v)[info.gxm] = (PetscScalar (*)[info.gxm]) y_v_vec;

  PetscInt   i, j;
  PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
             ve, vw, vn, vs, ue, uw, un, us;

  hx = 1./ (info.mx - 1);
  hy = 1./ (info.my - 1);
  darea = hx * hy;               // multiply FD equations by this
  scx = hy / hx;
  scy = hx / hy;
  scdiag = 2.0 * (scx + scy);    // diagonal scaling
  for (j = info.ys; j < info.ys + info.ym; j++) {
      y = xymin[1] + j * hy;
      for (i = info.xs; i < info.xs + info.xm; i++) {
          x = xymin[0] + i * hx;
          if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
              y_v[j][i] = scdiag * x_v[j][i];
          } else {
            ve = x_v[j][i+1];
            vw = x_v[j][i-1];
            vn = x_v[j+1][i];
            vs = x_v[j-1][i];
            y_v[j][i] = scdiag * x_v[j][i] - scx * (vw + ve) - scy * (vs + vn);

          }
      }
  }

  PetscCall(VecRestoreArray(yloc,&y_v_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

  PetscFunctionReturn(0);
}

PetscErrorCode J10_MatMult(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  DM dm0;
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  BiharmCtx * ctx0;
  PetscScalar * x_v_vec;
  PetscScalar * y_v_vec;

  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(yloc)));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_v_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;
  PetscScalar (* y_v)[info.gxm] = (PetscScalar (*)[info.gxm]) y_v_vec;

  PetscInt   i, j;
  PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
             ve, vw, vn, vs, ue, uw, un, us;

  hx = 1. / (info.mx - 1);
  hy = 1. / (info.my - 1);

  darea = hx * hy;               // multiply FD equations by this
  scx = hy / hx;
  scy = hx / hy;
  scdiag = 2.0 * (scx + scy);    // diagonal scaling
  for (j = info.ys; j < info.ys + info.ym; j++) {
      y = xymin[1] + j * hy;
      for (i = info.xs; i < info.xs + info.xm; i++) {
          x = xymin[0] + i * hx;
          if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
              y_v[j][i] = 0.0;
          } else {
              y_v[j][i] = -darea * x_v[j][i];

          }
      }
  }

  PetscCall(VecRestoreArray(yloc,&y_v_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

  PetscFunctionReturn(0);
}


PetscErrorCode J11_MatMult(Mat J, Vec X, Vec Y)
{
  PetscFunctionBeginUser;

  DM dm0;
  DMDALocalInfo info;
  Vec xloc;
  Vec yloc;

  BiharmCtx * ctx0;
  PetscScalar * x_v_vec;
  PetscScalar * y_v_vec;

  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMGetLocalVector(dm0,&(xloc)));
  PetscCall(DMGlobalToLocalBegin(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGlobalToLocalEnd(dm0,X,INSERT_VALUES,xloc));
  PetscCall(DMGetLocalVector(dm0,&(yloc)));
  PetscCall(VecSet(yloc,0.0));
  PetscCall(VecGetArray(yloc,&y_v_vec));
  PetscCall(VecGetArray(xloc,&x_v_vec));
  PetscCall(DMDAGetLocalInfo(dm0,&(info)));

  PetscScalar (* x_v)[info.gxm] = (PetscScalar (*)[info.gxm]) x_v_vec;
  PetscScalar (* y_v)[info.gxm] = (PetscScalar (*)[info.gxm]) y_v_vec;

  PetscInt   i, j;
  PetscReal  xymin[2], xymax[2], hx, hy, darea, scx, scy, scdiag, x, y,
             ve, vw, vn, vs, ue, uw, un, us;
//   PetscCall(DMGetBoundingBox(info.da,xymin,xymax));
  hx = 1. / (info.mx - 1);
  hy = 1. / (info.my - 1);

  darea = hx * hy;
  scx = hy / hx;
  scy = hx / hy;
  scdiag = 2.0 * (scx + scy);
  for (j = info.ys; j < info.ys + info.ym; j++) {
      y = xymin[1] + j * hy;
      for (i = info.xs; i < info.xs + info.xm; i++) {
          x = xymin[0] + i * hx;
          if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
              y_v[j][i] = scdiag * x_v[j][i];
          } else {
            ve = x_v[j][i+1];
            vw = x_v[j][i-1];
            vn = x_v[j+1][i];
            vs = x_v[j-1][i];
            y_v[j][i] = scdiag * x_v[j][i] - scx * (vw + ve) - scy * (vs + vn);

          }
      }
  }

  PetscCall(VecRestoreArray(yloc,&y_v_vec));
  PetscCall(VecRestoreArray(xloc,&x_v_vec));
  PetscCall(DMLocalToGlobalBegin(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMLocalToGlobalEnd(dm0,yloc,ADD_VALUES,Y));
  PetscCall(DMRestoreLocalVector(dm0,&(xloc)));
  PetscCall(DMRestoreLocalVector(dm0,&(yloc)));

  PetscFunctionReturn(0);
}

PetscErrorCode WholeMatMult(Mat J, Vec X, Vec Y)
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

  PetscFunctionBeginUser;

  PetscCall(MatShellGetContext(J,&(jctx)));

  PetscCall(VecSet(Y,0.0));

  Mat J00 = jctx->submats[0];
  PetscCall(MatShellGetContext(J00,&(J00ctx)));
  PetscCall(VecGetSubVector(X,*(J00ctx->cols),&(J00X)));
  PetscCall(VecGetSubVector(Y,*(J00ctx->rows),&(J00Y)));
  PetscCall(MatMult(J00,J00X,J00Y));
  PetscCall(VecRestoreSubVector(X,*(J00ctx->cols),&(J00X)));
  PetscCall(VecRestoreSubVector(Y,*(J00ctx->rows),&(J00Y)));

  Mat J10 = jctx->submats[2];
  PetscCall(MatShellGetContext(J10,&(J10ctx)));
  PetscCall(VecGetSubVector(X,*(J10ctx->cols),&(J10X)));
  PetscCall(VecGetSubVector(Y,*(J10ctx->rows),&(J10Y)));
  PetscCall(MatMult(J10,J10X,J10Y));
  PetscCall(VecRestoreSubVector(X,*(J10ctx->cols),&(J10X)));
  PetscCall(VecRestoreSubVector(Y,*(J10ctx->rows),&(J10Y)));

  Mat J11 = jctx->submats[3];
  PetscCall(MatShellGetContext(J11,&(J11ctx)));
  PetscCall(VecGetSubVector(X,*(J11ctx->cols),&(J11X)));
  PetscCall(VecGetSubVector(Y,*(J11ctx->rows),&(J11Y)));
  PetscCall(MatMult(J11,J11X,J11Y));
  PetscCall(VecRestoreSubVector(X,*(J11ctx->cols),&(J11X)));
  PetscCall(VecRestoreSubVector(Y,*(J11ctx->rows),&(J11Y)));


  PetscFunctionReturn(0);
}


PetscErrorCode MatCreateSubMatrices0(Mat J, PetscInt nfields, IS * irow, IS * icol, MatReuse scall, Mat * * submats)
{
  PetscFunctionBeginUser;

  PetscInt M;
  PetscInt N;
  Mat block;
  DM dm0;
  PetscInt dof;

  struct UserCtx0 * ctx0;
  struct JacobianCtx * jctx;
  struct SubMatrixCtx * subctx;

  PetscCall(MatShellGetContext(J,&(jctx)));
  DM * subdms = jctx->subdms;

  PetscInt nsubmats = nfields*nfields;
  PetscCall(PetscCalloc1(nsubmats,submats));
  PetscCall(MatGetDM(J,&(dm0)));
  PetscCall(DMGetApplicationContext(dm0,&(ctx0)));
  PetscCall(DMDAGetInfo(dm0,NULL,&(M),&(N),NULL,NULL,NULL,NULL,&(dof),NULL,NULL,NULL,NULL,NULL));
  PetscInt subblockrows = M*N;
  PetscInt subblockcols = M*N;
  Mat * submat_arr = *submats;

  for (int i = 0; i <= nsubmats - 1; i += 1)
  {
    PetscCall(MatCreate(PETSC_COMM_WORLD,&(block)));
    PetscCall(MatSetSizes(block,PETSC_DECIDE,PETSC_DECIDE,subblockrows,subblockcols));
    PetscCall(MatSetType(block,MATSHELL));
    PetscCall(PetscMalloc1(1,&(subctx)));
    PetscInt rowidx = i / dof;
    PetscInt colidx = (i)%(dof);
    subctx->rows = &(irow[rowidx]);
    subctx->cols = &(icol[colidx]);
    PetscCall(DMSetApplicationContext(subdms[rowidx],ctx0));
    PetscCall(MatSetDM(block,subdms[rowidx]));
    PetscCall(MatShellSetContext(block,subctx));
    PetscCall(MatSetUp(block));
    submat_arr[i] = block;
  }
  PetscCall(MatShellSetOperation(submat_arr[0],MATOP_MULT,(void (*)(void))J00_MatMult));
  PetscCall(MatShellSetOperation(submat_arr[2],MATOP_MULT,(void (*)(void))J10_MatMult));
  PetscCall(MatShellSetOperation(submat_arr[3],MATOP_MULT,(void (*)(void))J11_MatMult));

  PetscFunctionReturn(0);
}


PetscErrorCode PopulateMatContext(struct JacobianCtx * jctx, DM * subdms, IS * fields)
{
  PetscFunctionBeginUser;

  jctx->subdms = subdms;
  jctx->fields = fields;

  PetscFunctionReturn(0);
}
