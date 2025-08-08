
static char help[] = "Modified snes tutorial 1\n\n";

#include <petscsnes.h>

extern PetscErrorCode FormJacobian1(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormFunction1(SNES, Vec, Vec, void *);
extern PetscErrorCode SetPetscOption(const char *, const char *);
extern PetscErrorCode SetPetscOptions0();

int main(int argc, char **argv)
{
  SNES        snes;
  KSP         ksp;
  PC          pc;
  Vec         x, r;
  Mat         J;
  PetscMPIInt size;
  PetscBool   flg;
  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  // PetscCall(PetscOptionsSetValue(NULL, "-poisson_ksp_type", "cg"));
  // PetscCall(PetscOptionsInsert(NULL, &argc, &argv, NULL));
  
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "poisson_"));
  PetscCall(SetPetscOptions0());
  // PetscCall(SNESSetType(snes, SNESKSPONLY));
  // PetscCall(PetscOptionsSetValue(NULL, "-poisson_snes_type", "snesksponly"));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 2));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESSetFunction(snes, r, FormFunction1, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian1, NULL));

  // PetscCall(SNESGetKSP(snes, &ksp));
  // PetscCall(KSPGetPC(ksp, &pc));
  // PetscCall(PCSetType(pc, PCNONE));
  // PetscCall(KSPSetTolerances(ksp, 1.e-4, PETSC_CURRENT, PETSC_CURRENT, 20));
  // PetscCall(KSPSetFromOptions(ksp));

//   PetscCall(VecSet(x, pfive));
  PetscCall(SNESSolve(snes, NULL, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormFunction1(SNES snes, Vec x, Vec f, void *ctx)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;

  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));

  /* Compute function */
  ff[0] = xx[0] * xx[0] + xx[0] * xx[1] - 3.0;
  ff[1] = xx[0] * xx[1] + xx[1] * xx[1] - 6.0;

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian1(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar        A[4];
  PetscInt           idx[2] = {0, 1};

  PetscFunctionBeginUser;

  PetscCall(VecGetArrayRead(x, &xx));

  A[0] = 2.0 * xx[0] + xx[1];
  A[1] = xx[0];
  A[2] = xx[1];
  A[3] = xx[0] + 2.0 * xx[1];
  PetscCall(MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES));

  PetscCall(VecRestoreArrayRead(x, &xx));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetPetscOption(const char * option, const char * value)
{
  PetscFunctionBeginUser;

  PetscBool set;

  PetscCall(PetscOptionsHasName(NULL,NULL,option,&(set)));
  if (!set)
  {
    PetscCall(PetscOptionsSetValue(NULL,option,value));
  }

  PetscFunctionReturn(0);
}


PetscErrorCode SetPetscOptions0()
{
  PetscFunctionBeginUser;

  // PetscCall(SetPetscOption("-snes_type","ksponly"));
  PetscCall(SetPetscOption("-poisson_ksp_type","cg"));
  // PetscCall(SetPetscOption("-pc_type","none"));
  // PetscCall(SetPetscOption("-ksp_rtol","1e-05"));
  // PetscCall(SetPetscOption("-ksp_atol","1e-50"));
  // PetscCall(SetPetscOption("-ksp_divtol","100000.0"));
  // PetscCall(SetPetscOption("-ksp_max_it","10000"));
  // PetscCall(SetPetscOption("-ksp_view",NULL));

  PetscFunctionReturn(0);
}