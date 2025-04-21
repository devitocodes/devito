#include <petscsys.h>

extern PetscErrorCode PetscInit();
extern PetscErrorCode PetscFinal();

int main(int argc, char **argv)
{
  PetscInit(argc, argv);
  PetscPrintf(PETSC_COMM_WORLD, "Hello World!\n");
  return PetscFinalize();
}

PetscErrorCode PetscInit(int argc, char **argv)
{
    static char help[] = "Magic help string\n";
    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscFunctionReturn(0);
}

PetscErrorCode PetscFinal()
{
    PetscFunctionBeginUser;
    PetscCall(PetscFinalize());
    PetscFunctionReturn(0);
}
