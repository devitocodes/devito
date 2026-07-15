from devito.petsc.types.multigrid import _field_shifts
from devito.types.equation import Eq
from devito.types.grid import SubDomain

__all__ = ['mirror_halo']


class _MirrorSubDomain(SubDomain):
    """
    """

    def __init__(self, grid, dim, side, thickness):
        self._mirror_dim = dim
        self._mirror_side = side
        self._mirror_thickness = thickness
        self.name = f'mirror_{dim.name}_{side}_{id(self)}'
        super().__init__(grid=grid)

    def define(self, dimensions):
        return {d: (self._mirror_side, self._mirror_thickness) if d is self._mirror_dim
                else d for d in dimensions}


def mirror_halo(field, dims=None, r_coeff=1):
    """
    Build Eqs that populate `field`'s halo region via a mirror
    reflection across each global domain boundary, using
    `SubDomain`s.

    Parameters
    ----------
    field : Function
        The Function whose halo should be populated.
    dims : dict, optional
        Maps a space Dimension to 'left', 'right', or the Dimension itself
        (meaning both sides)
    r_coeff : int, optional
        1 mirrors the field value unchanged (Neumann-like, even reflection);
        -1 flips the sign. Default is 1.

    Returns
    -------
    A list of Eq objects.
    """
    if dims is None:
        dims = {d: d for d in field.space_dimensions}

    shifts = dict(zip(field.space_dimensions, _field_shifts(field), strict=True))

    eqs = []
    for dim, side_spec in dims.items():
        sides = ('left', 'right') if side_spec is dim else (side_spec,)
        for side in sides:
            left, right = field.halo[dim]
            thickness = left if side == 'left' else right
            if not thickness:
                continue

            sub = _MirrorSubDomain(field.grid, dim, side, thickness + 1)
            raw = next(sd for orig, sd in zip(field.space_dimensions, sub.dimensions)
                       if orig is dim)
            m, M = dim.symbolic_min, dim.symbolic_max

            staggered = shifts[dim]

            if side == 'left':
                read_off = (raw - 1) if staggered else raw
                eqs.append(Eq(field._subs(dim, m - raw),
                              r_coeff * field._subs(dim, m + read_off)))
            else:
                k = M - raw
                read_off = (k - 1) if staggered else k
                eqs.append(Eq(field._subs(dim, M + k),
                              r_coeff * field._subs(dim, M - read_off)))

    return eqs