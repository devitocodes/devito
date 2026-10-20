import sympy

from devito.finite_differences.differentiable import LocalSum, Mul
from devito.symbolics import INT
from devito.types import ConditionalDimension, CustomDimension, Inc

__all__ = ['SubGridTransfer', 'prolong', 'restrict']


class SubGridTransfer:

    """
    Move data between a parent Grid and a SubGrid whose points fall between the
    parent's, by separable interpolation.

    Subgrid point `i` lies at parent coordinate `i*factor` along each Dimension.
    `factor` need not be an integer, so each subgrid point is a weighted sum of
    the `radius**ndim` parent points surrounding it. Both the base cell index
    and the fractional offset are affine in the subgrid Dimension, hence
    computed in the loop -- unlike the sparse interpolators, which tabulate
    them per point on the host.

    Parameters
    ----------
    subgrid : SubGrid
        The SubGrid one side of the transfer is defined on.
    scheme : str, optional
        Interpolation scheme, one of `SubGridTransfer.schemes`. Defaults to
        'linear'.

    Notes
    -----
    `restrict` and `prolong` are adjoints of one another: for any `u` and
    `g`, `<restrict(u), g> == <u, prolong(g)>` to machine precision.
    """

    # Scheme -> stencil radius, i.e. number of parent points used per Dimension
    schemes = {'nearest': 1, 'linear': 2}

    def __init__(self, subgrid, scheme='linear'):
        if not subgrid.is_interpolated:
            raise ValueError(
                f"SubGrid of factor `{tuple(subgrid.factor)}` has its points on "
                "the parent Grid's, so no interpolation is needed -- use a "
                "plain `Eq` between the two Functions instead"
            )
        if scheme not in self.schemes:
            raise ValueError(f"Unknown interpolation scheme `{scheme}`; "
                             f"available: {sorted(self.schemes)}")

        self.subgrid = subgrid
        self.scheme = scheme

    @property
    def radius(self):
        """Number of parent points used along each Dimension."""
        return self.schemes[self.scheme]

    @property
    def parent(self):
        return self.subgrid.parent

    def _positions(self):
        """
        Per-Dimension `(base index, fractional offset)` of each subgrid point
        within the parent Grid, as expressions in the subgrid Dimensions.
        """
        retval = []
        for ds, f in zip(self.subgrid.dimensions, self.subgrid.factor,
                         strict=True):
            if f == 1:
                # Dimension passed through untouched: the subgrid index *is*
                # the parent index, with no fractional part
                retval.append((ds, sympy.S.Zero))
                continue

            coord = ds*f
            if self.scheme == 'nearest':
                # Snap to the closest parent point; no fractional offset left
                retval.append((INT(sympy.floor(coord + 0.5)), sympy.S.Zero))
            else:
                pos = INT(sympy.floor(coord))
                retval.append((pos, coord - pos))

        return retval

    def _rdims(self, positions):
        """
        One radius Dimension per parent Dimension, guarded so that taps falling
        outside the parent's domain contribute nothing.

        Mirrors `SparseFunction._cond_rdim`: the CustomDimension carries the
        iteration Dimension as parent, and the ConditionalDimension reuses its
        name. Both are needed for the radius symbols to become loop bounds
        rather than Operator parameters.
        """
        retval = []
        for ds, d, (pos, _) in zip(self.subgrid.dimensions,
                                   self.parent.dimensions, positions,
                                   strict=True):
            rd = CustomDimension(f'r{ds.name}', 0, self.radius - 1,
                                 self.radius, ds)
            cond = sympy.And(pos + rd >= d.symbolic_min,
                             pos + rd <= d.symbolic_max)
            retval.append(ConditionalDimension(rd.name, parent=rd,
                                               condition=cond, indirect=True))

        return retval

    def _weights(self, positions, rdims):
        """
        The separable interpolation weight, a product of one factor per
        Dimension.

        Written arithmetically rather than with `sympy.Piecewise`, which does
        not survive Devito's expression manipulation.
        """
        terms = []
        for (_, alpha), rd in zip(positions, rdims, strict=True):
            if self.radius == 1:
                continue                      # nearest: unit weight
            # w[0] = 1 - alpha, w[1] = alpha, i.e. linear in the radius index
            terms.append((1 - alpha) + rd.parent*(2*alpha - 1))

        if not terms:
            return sympy.S.One

        return Mul(*terms)

    def _taps(self, field, positions, rdims):
        """`field` indexed at the stencil taps around each subgrid point."""
        indices = [pos + rd for (pos, _), rd in zip(positions, rdims,
                                                    strict=True)]

        return field[tuple(indices)]

    def restrict(self, field):
        """
        The value of `field`, defined on the parent Grid, interpolated onto the
        SubGrid's points. Returns an expression to be assigned to a SubGrid
        Function.
        """
        positions = self._positions()
        rdims = self._rdims(positions)
        weights = self._weights(positions, rdims)

        return LocalSum(weights*self._taps(field, positions, rdims),
                        cdims=tuple(rdims), dtype=field.dtype)

    def prolong(self, field, expr):
        """
        Scatter `expr`, defined on the SubGrid, onto `field` on the parent
        Grid. Returns the `Inc` performing the accumulation -- the adjoint of
        `restrict`.
        """
        positions = self._positions()
        rdims = self._rdims(positions)
        weights = self._weights(positions, rdims)

        lhs = self._taps(field, positions, rdims)

        return Inc(lhs, weights*expr)


def restrict(field, subgrid, scheme='linear'):
    """
    Interpolate `field`, defined on a parent Grid, onto `subgrid`'s points.

    Parameters
    ----------
    field : Function
        The Function to read, defined on `subgrid.parent`.
    subgrid : SubGrid
        The SubGrid to interpolate onto.
    scheme : str, optional
        Interpolation scheme; see `SubGridTransfer.schemes`.

    Examples
    --------
    >>> from devito import Grid, SubGrid, Function, Eq, Operator, restrict
    >>> grid = Grid(shape=(9, 9), extent=(8., 8.))
    >>> sub = SubGrid(grid, factor=1.5)
    >>> u = Function(name='u', grid=grid)
    >>> g = Function(name='g', grid=sub, space_order=0)
    >>> eq = Eq(g, restrict(u, sub))
    """
    return SubGridTransfer(subgrid, scheme=scheme).restrict(field)


def prolong(field, expr, subgrid, scheme='linear'):
    """
    Scatter `expr`, defined on `subgrid`, onto `field` on the parent Grid.

    This is the adjoint of `restrict`, so it returns an `Inc`: each parent
    point accumulates the contributions of every subgrid point whose stencil
    reaches it.

    Parameters
    ----------
    field : Function
        The Function to accumulate into, defined on `subgrid.parent`.
    expr : expr-like
        The SubGrid-defined value to scatter.
    subgrid : SubGrid
        The SubGrid `expr` is defined on.
    scheme : str, optional
        Interpolation scheme; see `SubGridTransfer.schemes`.
    """
    return SubGridTransfer(subgrid, scheme=scheme).prolong(field, expr)
