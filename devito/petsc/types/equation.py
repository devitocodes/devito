from devito.types.equation import Eq, Inc


__all__ = ['EssentialBC']


class EssentialBC(Eq):
    """
    Represents an essential boundary condition for use with `petscsolve`.

    The compiler will automatically zero the corresponding rows/columns in the Jacobian
    and lift the boundary terms into the residual RHS, unless the user
    specifies `constrain_bcs=True` to `petscsolve`.

    Note:
        - To define an essential boundary condition, use:
            Eq(target, boundary_value, subdomain=...),
          where `target` is the Function-like object passed to `petscsolve`.
        - SubDomains used for multiple `EssentialBC`s must not overlap.
    """
    __rkwargs__ = Eq.__rkwargs__ + ("target",)

    def __new__(cls, *args, target=None, **kwargs):
        obj = super().__new__(cls, *args, **kwargs)

        if target is None:
            target = obj.lhs.function

        obj._target = target
        return obj

    @property
    def target(self):
        return self._target


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
