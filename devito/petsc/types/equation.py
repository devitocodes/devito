from devito.types.equation import Eq


__all__ = ['EssentialBC']


class EssentialBC(Eq):
    """
    A special equation used to handle essential boundary conditions
    in the PETSc solver. Until PetscSection + DMDA is supported,
    we treat essential boundary conditions as trivial equations
    in the solver, where we place 1.0 (scaled) on the diagonal of
    the jacobian, zero symmetrically and move the boundary
    data to the right-hand side.

    NOTE: When users define essential boundary conditions, they need to ensure that
    the SubDomains do not overlap. Solver will still run but may see unexpected behaviour
    at boundaries. This will be documented in the PETSc examples.
    """
    pass


class ZeroRow(EssentialBC):
    """
    Equation used to zero the row of the Jacobian corresponding
    to an essential BC.
    This is only used interally by the compiler, not by users.
    """
    pass


class ZeroColumn(EssentialBC):
    """
    Equation used to zero the column of the Jacobian corresponding
    to an essential BC.
    This is only used interally by the compiler, not by users.
    """
    pass