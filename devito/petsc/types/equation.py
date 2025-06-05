from devito.types.equation import Eq


__all__ = ['EssentialBC']


class EssentialBC(Eq):
    """
    Represents an essential boundary condition for use with PETScSolve.

    Due to ongoing work on PetscSection and DMDA integration (WIP),
    these conditions are imposed as trivial equations. The compiler
    will automatically zero the corresponding rows/columns in the Jacobian
    and lift the boundary terms into the residual RHS.

    Note:
        - To define an essential boundary condition, use:
            Eq(target, boundary_value, subdomain=...),
          where `target` is the Function-like object passed to PETScSolve.
        - SubDomains used for multiple EssentialBCs must not overlap.
    """
    pass


class ZeroRow(EssentialBC):
    """
    Equation used to zero all entries, except the diagonal,
    of a row in the Jacobian.

    Note:
        This is only used interally by the compiler, not by users.
    """
    pass


class ZeroColumn(EssentialBC):
    """
    Equation used to zero the column of the Jacobian.

    Note:
        This is only used interally by the compiler, not by users.
    """
    pass
