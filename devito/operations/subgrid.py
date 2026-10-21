from fractions import Fraction
from itertools import product

import sympy

from devito.finite_differences.differentiable import Differentiable
from devito.symbolics import IntDiv
from devito.types import Inc

__all__ = ['SubGridTransfer', 'interpolated', 'prolong', 'restrict']


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
    schemes = {'nearest': 1, 'linear': 2, 'cubic': 4}

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
    def offsets(self):
        """
        Tap offsets from the base index, along one Dimension. `nearest` and
        `linear` read forward from it; `cubic` is centred, so it reaches one
        point back.
        """
        lo = -((self.radius - 1)//2) if self.radius > 2 else 0
        return tuple(range(lo, lo + self.radius))

    def _weight(self, alpha, offset):
        """The interpolation weight for one tap along one Dimension."""
        if self.scheme == 'nearest':
            return sympy.S.One
        if self.scheme == 'linear':
            # `1 - alpha` at the lower tap, `alpha` at the upper
            return alpha if offset else (1 - alpha)

        # Catmull-Rom cubic, taps at offsets -1, 0, 1, 2. Reproduces a linear
        # field exactly and the weights sum to one
        a = alpha
        return {
            -1: -a**3/2 + a**2 - a/2,
            0: 3*a**3/2 - 5*a**2/2 + 1,
            1: -3*a**3/2 + 2*a**2 + a/2,
            2: a**3/2 - a**2/2,
        }[offset]

    @property
    def parent(self):
        return self.subgrid.parent

    def _positions(self, invert=False):
        """
        Per-Dimension `(base index, fractional offset)` locating one grid's
        points within the other.

        By default this maps a subgrid index to the parent, i.e. subgrid point
        `i` sits at parent coordinate `i*factor`. With `invert=True` it maps a
        parent index to the subgrid, at `i/factor` -- what is needed to read a
        SubGrid Function from within a parent-grid equation.
        """
        if invert:
            dims = self.parent.dimensions
            factors = [1/f if f != 1 else 1 for f in self.subgrid.factor]
            # Reading the SubGrid at parent points: shift *into* subgrid-local
            src, dst = self.parent, self.subgrid
        else:
            dims = self.subgrid.dimensions
            factors = list(self.subgrid.factor)
            src, dst = self.subgrid, self.parent

        # `pos = floor(i*factor)` holds for a *global* index, but each rank
        # iterates local ones. Lift the index to global, map it, then drop it
        # back into the target's local frame. Both offsets are zero without
        # MPI, so the expressions fold away
        parallel = src.distributor.is_parallel
        src_off = src.ioffset_symbols if parallel else None
        dst_off = dst.ioffset_symbols if parallel else None

        retval = []
        for i, (ds, f) in enumerate(zip(dims, factors, strict=True)):
            if f == 1:
                # Dimension passed through untouched: one index *is* the
                # other, with no fractional part
                shift = (src_off[i] - dst_off[i]) if parallel else sympy.S.Zero
                retval.append((ds + shift, sympy.S.Zero))
                continue

            idx = ds + src_off[i] if parallel else ds

            # `factor` as an exact rational, so the base index is integer
            # arithmetic rather than a `floor` on floats
            frac = Fraction(f).limit_denominator(10**6)
            num, den = frac.numerator, frac.denominator

            if self.scheme == 'nearest':
                # Snap to the closest point: floor(i*f + 1/2)
                pos = IntDiv(2*num*idx + den, 2*den)
                alpha = sympy.S.Zero
            else:
                pos = IntDiv(num*idx, den)
                # The fractional part of `i*num/den`, i.e. `(i*num mod den)/den`
                alpha = sympy.Rational(1, den)*(num*idx - den*pos)

            if parallel:
                pos = pos - dst_off[i]

            retval.append((pos, alpha))

        return retval

    def _taps(self, positions):
        """
        The `radius**ndim` stencil taps, as `(weight, indices)` pairs. The tap
        count is a compile-time constant, hence the plain product.
        """
        retval = []
        for offsets in product(self.offsets, repeat=len(positions)):
            weight = sympy.S.One
            for (_, alpha), o in zip(positions, offsets, strict=True):
                weight = weight*self._weight(alpha, o)

            indices = tuple(pos + o for (pos, _), o in
                            zip(positions, offsets, strict=True))
            retval.append((weight, indices))

        return retval

    def _stencil(self, field, positions):
        """
        `field` interpolated at each target point, as a flat weighted sum over
        the surrounding taps.
        """
        return sympy.Add(*[w*field[i] for w, i in self._taps(positions)])

    def evaluate(self, field):
        """
        `field`, defined on the SubGrid, interpolated onto the *parent's*
        points -- so that it can be read from within a parent-grid equation.
        """
        self._check_halo(field)

        return self._stencil(field, self._positions(invert=True))

    def _check_halo(self, field):
        """
        A stencil wider than one point reaches beyond the base index, so
        `field` needs enough halo to hold the taps -- otherwise the kernel
        reads, or for `prolong` *writes*, out of bounds.

        Under MPI the same halo also stands in for points owned by a
        neighbouring rank. Devito does not derive an exchange for these
        accesses (they are indexed by the *subgrid* Dimensions, so the
        exchange is never placed against the surrounding Iteration), so the
        caller must ensure the halo is valid as well as large enough.
        """
        need = max(-min(self.offsets), max(self.offsets))
        if not need:
            return

        if any(i.left < need or i.right < need for i in field._size_halo):
            raise ValueError(
                f"`{field.name}` needs a halo of at least {need} along every "
                f"Dimension for a `{self.scheme}` SubGrid transfer (has "
                f"{tuple(field._size_halo)}): the stencil reaches "
                f"{min(self.offsets)}..{max(self.offsets)} around each base "
                f"index. Pass `space_order={need}` or more"
            )

    def restrict(self, field):
        """
        The value of `field`, defined on the parent Grid, interpolated onto the
        SubGrid's points. Returns an expression to be assigned to a SubGrid
        Function.

        Notes
        -----
        Under MPI, `field`'s halo must be valid: Devito does not derive a halo
        exchange for the taps, so an `Operator` that writes `field` and then
        restricts it in one go will read stale halo data. Restrict in a
        separate `Operator`, or exchange explicitly.
        """
        self._check_halo(field)

        return self._stencil(field, self._positions())

    def prolong(self, field, expr):
        """
        Scatter `expr`, defined on the SubGrid, onto `field` on the parent
        Grid. Returns the `Inc`s performing the accumulation -- the adjoint of
        `restrict`, so one per tap.
        """
        if self.subgrid.distributor.is_parallel:
            # A tap may land on a parent point owned by a neighbour, so the
            # contribution is written into the halo. Summing those back to
            # their owner is a halo *reduction*, which Devito does not derive
            # for these accesses -- the result would silently not be the
            # adjoint of `restrict` (measured 6.5% off on two ranks)
            raise NotImplementedError(
                "`prolong` is not supported under MPI. Contributions landing "
                "in a neighbour's halo would have to be reduced back to their "
                "owner, which is not yet derived. `restrict` and "
                "`interpolated` do work under MPI, given a sufficient halo"
            )

        return [Inc(field[i], w*expr)
                for w, i in self._taps(self._positions())]


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


class InterpolatedFunction(Differentiable):

    """
    A SubGrid Function viewed from its parent Grid, interpolated on read.

    Lazy: it carries the Function and the scheme, and expands into the
    interpolation stencil when the enclosing equation is evaluated. That lets a
    SubGrid Function be used directly in a parent-grid expression

        u = Function(name='u', grid=grid)
        vp = Function(name='vp', grid=sub)
        Eq(u, interpolated(vp)*u)

    where `vp` is read at `x/factor` and interpolated onto the parent's points,
    so the Operator iterates `x, y` alone rather than gaining `xs, ys`.

    Notes
    -----
    This is deliberately **opt-in**. Using a SubGrid Function in a parent-grid
    equation without wrapping it is already meaningful -- the subgrid
    Dimensions become extra iteration Dimensions -- so interpolating
    implicitly would silently change the meaning of existing code.
    """

    __rargs__ = ('function',)
    __rkwargs__ = ('scheme',)

    def __new__(cls, function, scheme='linear', **kwargs):
        grid = getattr(function, 'grid', None)
        if grid is None or not getattr(grid, 'is_interpolated', False):
            raise ValueError(
                f"`{function}` is not defined on an interpolated SubGrid. Only "
                "a Function whose Grid has a non-integer factor needs "
                "interpolating to be read on the parent Grid"
            )

        obj = super().__new__(cls, function)
        obj._function = function
        obj._scheme = scheme

        return obj

    @property
    def function(self):
        return self._function

    @property
    def scheme(self):
        return self._scheme

    @property
    def subgrid(self):
        return self._function.grid

    def __repr__(self):
        return f"interpolated({self._function}, scheme={self._scheme!r})"

    __str__ = __repr__

    def _evaluate(self, **kwargs):
        transfer = SubGridTransfer(self.subgrid, scheme=self._scheme)
        return transfer.evaluate(self._function)


def interpolated(function, scheme='linear'):
    """
    View a SubGrid Function from its parent Grid, interpolating on read.

    This is what lets a coarse (or fine) Function take part in a parent-grid
    equation directly:

        sub = SubGrid(grid, factor=1.5)
        vp = Function(name='vp', grid=sub, space_order=0)
        u = Function(name='u', grid=grid)

        Eq(u, interpolated(vp)*u)     # `vp` interpolated onto x, y

    The resulting Operator iterates the parent's Dimensions only; `vp` is read
    at `x/factor` through the same separable stencil `restrict` uses.

    Parameters
    ----------
    function : Function
        A Function defined on an interpolated SubGrid, i.e. one whose factor is
        not an integer.
    scheme : str, optional
        Interpolation scheme; see `SubGridTransfer.schemes`.

    Notes
    -----
    Opt-in by design: an unwrapped SubGrid Function in a parent-grid equation
    keeps its existing meaning, where the subgrid Dimensions are iterated.
    """
    return InterpolatedFunction(function, scheme=scheme)
