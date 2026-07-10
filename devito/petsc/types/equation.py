from devito.types.equation import Eq, Inc

__all__ = ['CallbackEq', 'EssentialBC']


class CallbackEq(Eq):
    """
    An equation built for a Jacobian/Residual/Diagonal/InitialGuess callback
    body (FormFunction/FormRHS/MatMult/etc).

    When `is_multilevel` is True, the equation stands in for a whole family
    of per-level equations, collapsed into one by PETSc's callback-sharing
    model (one C function is registered per callback kind and reused, via
    runtime ctx-struct arguments, at every multigrid level) - see
    `lower_exprs_petsc` for where fine-grid reads get remapped accordingly.
    """
    __rkwargs__ = Eq.__rkwargs__ + ("is_multilevel",)

    def __new__(cls, *args, is_multilevel=False, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)
        obj._is_multilevel = is_multilevel
        return obj

    @property
    def is_multilevel(self):
        return self._is_multilevel


class EssentialBC(CallbackEq):
    """
    Represents an essential boundary condition for use with `petscsolve`.

    The compiler will automatically zero the corresponding rows/columns in the Jacobian
    and lift the boundary terms into the residual RHS, unless the BC is constrained
    via a `PetscSection`. In which case, they are set once and removed from the
    global solver. Constraining can be enabled in two ways:

    - Globally: pass `constrain_bcs=True` to `petscsolve` to constrain all
      `EssentialBC`s in the solve.
    - Individually: pass `constrain=True` to a specific `EssentialBC` constructor,
      e.g. ``EssentialBC(lhs, rhs, subdomain=..., constrain=True)``.

    Note:
        - To define an essential boundary condition, use:
            Eq(target, boundary_value, subdomain=...),
          where `target` is the Function-like object passed to `petscsolve`.
        - SubDomains used for multiple `EssentialBC`s must not overlap.
    """
    __rkwargs__ = CallbackEq.__rkwargs__ + ("target", "constrain")

    def __new__(cls, *args, target=None, constrain=False, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)

        if target is None:
            target = obj.lhs.function

        obj._target = target
        obj._constrain = constrain
        return obj

    @property
    def target(self):
        return self._target

    @property
    def constrain(self):
        return self._constrain


class ZeroRow(EssentialBC):
    """
    Equation used to zero all entries, except the diagonal,
    of a row in the Jacobian.

    Warnings
    --------
    Created and managed directly by Devito, not by users.
    """
    pass


class ZeroColumn(EssentialBC):
    """
    Equation used to zero the column of the Jacobian.

    Warnings
    --------
    Created and managed directly by Devito, not by users.
    """
    pass


class ConstrainBC(EssentialBC):
    pass


class NoOfEssentialBC(Inc, ConstrainBC):
    """Equation used count essential boundary condition nodes.
    This type of equation is generated inside
    petscsolve if the user sets `constrain_bcs=True`."""
    pass


class PointEssentialBC(ConstrainBC):
    pass
