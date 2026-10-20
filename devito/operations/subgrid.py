from fractions import Fraction
from itertools import product

import sympy

from devito.symbolics import IntDiv
from devito.types import Inc

__all__ = ['SubGridTransfer', 'prolong', 'restrict']


class SubGridTransfer:

    """
    Move data between a parent Grid and a SubGrid whose points fall between the
    parent's, by separable interpolation.

    Subgrid point `i` lies at parent coordinate `i*factor` along each Dimension.
    `factor` need not be an integer, so each subgrid point is a weighted sum of
    the `radius**ndim` parent points surrounding it.

    The tap count is a compile-time constant and the tap positions are affine
    in the subgrid Dimensions, so this lowers to a **plain stencil**: the taps
    are written out as one expression, exactly as finite-difference weights
    are. There is no accumulator and no per-tap bounds test in the inner loop,
    both of which would block vectorization. (The sparse interpolators use a
    guarded accumulation because their point set is irregular; here it is not.)

    `factor` is converted to an exact rational, so the base index is
    `(num*i)//den` in integer arithmetic -- no `floor` on floats, and the
    fractional offset follows from the same division.

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

            # `factor` as an exact rational, so the base index is integer
            # arithmetic rather than a `floor` on floats
            frac = Fraction(f).limit_denominator(10**6)
            num, den = frac.numerator, frac.denominator

            if self.scheme == 'nearest':
                # Snap to the closest parent point: floor(i*f + 1/2)
                retval.append((IntDiv(2*num*ds + den, 2*den), sympy.S.Zero))
            else:
                pos = IntDiv(num*ds, den)
                # The fractional part of `i*num/den`, i.e. `(i*num mod den)/den`
                alpha = sympy.Rational(1, den)*(num*ds - den*pos)
                retval.append((pos, alpha))

        return retval

    def _stencil(self, field, positions):
        """
        `field` interpolated at each subgrid point, as a flat weighted sum over
        the `radius**ndim` surrounding taps.
        """
        terms = []
        for offsets in product(range(self.radius), repeat=len(positions)):
            weight = sympy.S.One
            for (_, alpha), o in zip(positions, offsets, strict=True):
                if self.radius == 1:
                    continue                       # nearest: unit weight
                # Linear: `1 - alpha` at the lower tap, `alpha` at the upper
                weight = weight*(alpha if o else (1 - alpha))

            indices = tuple(pos + o for (pos, _), o in
                            zip(positions, offsets, strict=True))
            terms.append(weight*field[indices])

        return sympy.Add(*terms)

    def restrict(self, field):
        """
        The value of `field`, defined on the parent Grid, interpolated onto the
        SubGrid's points. Returns an expression to be assigned to a SubGrid
        Function.
        """
        return self._stencil(field, self._positions())

    def prolong(self, field, expr):
        """
        Scatter `expr`, defined on the SubGrid, onto `field` on the parent
        Grid. Returns the `Inc`s performing the accumulation -- the adjoint of
        `restrict`, so one per tap.
        """
        positions = self._positions()

        retval = []
        for offsets in product(range(self.radius), repeat=len(positions)):
            weight = sympy.S.One
            for (_, alpha), o in zip(positions, offsets, strict=True):
                if self.radius == 1:
                    continue
                weight = weight*(alpha if o else (1 - alpha))

            indices = tuple(pos + o for (pos, _), o in
                            zip(positions, offsets, strict=True))
            retval.append(Inc(field[indices], weight*expr))

        return retval


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
