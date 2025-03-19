from devito.tools import timed_pass
from devito.petsc.types import LinearSolveExpr


@timed_pass()
def petsc_preprocess(clusters):
    """
    Preprocess the clusters to make them suitable for PETSc
    code generation.
    """
    clusters = petsc_lift(clusters)
    return clusters


def petsc_lift(clusters):
    """
    Lift the iteration space surrounding each PETSc solve to create
    distinct iteration loops.
    """
    processed = []
    for c in clusters:
        if isinstance(c.exprs[0].rhs, LinearSolveExpr):
            ispace = c.ispace.lift(c.exprs[0].rhs.fielddata.space_dimensions)
            processed.append(c.rebuild(ispace=ispace))
        else:
            processed.append(c)
    return processed
