import numpy as np
import pytest

from devito import Eq, Function, Grid, Operator, SubGrid, prolong, restrict, switchconfig
from devito.operations.subgrid import SubGridTransfer


def linear_field(grid, a=3., b=2.):
    """`a*x + b*y` sampled on `grid`, in grid-index units."""
    idx = np.meshgrid(*[np.arange(n) for n in grid.shape], indexing='ij')
    return a*idx[0] + b*idx[1]


def restricted(fac, scheme='linear', nx=9, a=3., b=2.):
    """Restrict a linear field onto a SubGrid of the given factor."""
    grid = Grid(shape=(nx, nx), extent=(float(nx - 1),)*2)
    sub = SubGrid(grid, factor=fac)

    u = Function(name='u', grid=grid)
    g = Function(name='g', grid=sub, space_order=0)
    u.data[:] = linear_field(grid, a, b).astype(u.dtype)

    Operator(Eq(g, restrict(u, sub, scheme=scheme))).apply()

    return sub, g


class TestSubGridConstruction:

    """Non-integer factors are accepted and classified."""

    @pytest.mark.parametrize('factor,shape,interpolated', [
        (2, (5, 5), False),
        (1, (9, 9), False),
        (1.5, (6, 6), True),
        (0.5, (17, 17), True),
        (2.5, (4, 4), True),
        ((2, 1.5), (5, 6), True),
        ((1.5, 1), (6, 9), True),
    ])
    def test_shape_and_kind(self, factor, shape, interpolated):
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=factor)

        assert sub.shape == shape
        assert sub.is_interpolated is interpolated

    def test_integer_keeps_conditional_dimensions(self):
        """An integer factor still lowers through ConditionalDimensions."""
        sub = SubGrid(Grid(shape=(9, 9)), factor=2)

        assert not sub.is_interpolated
        assert all(d.is_Conditional for d in sub.dimensions)

    def test_non_integer_gets_own_dimensions(self):
        """
        A non-integer factor has no `parent % factor == 0` relation to express,
        so the axes become SpaceDimensions in their own right.
        """
        sub = SubGrid(Grid(shape=(9, 9)), factor=1.5)

        assert sub.is_interpolated
        assert all(d.is_Space and not d.is_Derived for d in sub.dimensions)

    def test_spacing_and_extent(self):
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        assert np.allclose(sub.spacing, [1.5*s for s in grid.spacing])
        # 6 points at spacing 1.5 span 7.5, the last point before the parent's
        # right edge at 8.0
        assert np.allclose(sub.extent, (7.5, 7.5))

    def test_function_needs_no_shape(self):
        sub = SubGrid(Grid(shape=(9, 9)), factor=1.5)

        f = Function(name='f', grid=sub, space_order=0)

        assert f.shape == sub.shape
        assert f.dimensions == sub.dimensions

    @pytest.mark.parametrize('factor', [0, -1, -2.5])
    def test_invalid_factor(self, factor):
        with pytest.raises(ValueError, match='must be > 0'):
            SubGrid(Grid(shape=(9, 9)), factor=factor)


class TestLinear:

    """
    Linear interpolation is exact on a linear field -- the sharpest available
    correctness check, since any error in the position or the weights shows up
    immediately.
    """

    @pytest.mark.parametrize('factor', [
        1.5, 0.5, 2.5, 0.25, 3.5, (2, 1.5), (1.5, 1), (0.5, 2.5),
    ])
    def test_exact_on_linear_field(self, factor):
        sub, g = restricted(factor)

        f0, f1 = sub.factor
        i, j = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]),
                           indexing='ij')
        expected = 3.*(i*f0) + 2.*(j*f1)

        assert np.allclose(g.data, expected, atol=2e-4)

    def test_exact_on_constant_field(self):
        """Weights must sum to one at every subgrid point."""
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        u.data[:] = 2.5

        Operator(Eq(g, restrict(u, sub))).apply()

        assert np.allclose(g.data, 2.5, atol=1e-5)

    def test_refinement(self):
        """`factor < 1` refines rather than coarsens."""
        sub, g = restricted(0.5)

        assert sub.shape == (17, 17)
        assert g.shape == (17, 17)

    @pytest.mark.parametrize('ndim', [1, 2, 3])
    def test_dimensionality(self, ndim):
        n = 9
        grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        u.data[:] = 1.

        Operator(Eq(g, restrict(u, sub))).apply()

        assert g.shape == (6,)*ndim
        assert np.allclose(g.data, 1., atol=1e-5)


class TestNearest:

    def test_snaps_to_nearest(self):
        """
        `nearest` reads a single parent point, so a linear field is reproduced
        only up to the distance to that point.
        """
        sub, g = restricted(1.5, scheme='nearest')

        f0, f1 = sub.factor
        i, j = np.meshgrid(np.arange(g.shape[0]), np.arange(g.shape[1]),
                           indexing='ij')
        expected = 3.*np.floor(i*f0 + 0.5) + 2.*np.floor(j*f1 + 0.5)

        assert np.allclose(g.data, expected, atol=1e-4)

    def test_exact_on_constant_field(self):
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        u.data[:] = -4.

        Operator(Eq(g, restrict(u, sub, scheme='nearest'))).apply()

        assert np.allclose(g.data, -4., atol=1e-6)


class TestAdjoint:

    """
    `restrict` and `prolong` must be adjoints: `<R u, g> == <u, P g>`. This
    pins the weights and the stencil footprint together -- an error in either
    breaks the identity.
    """

    @pytest.mark.parametrize('factor', [1.5, 0.5, 2.5, (2, 1.5)])
    @pytest.mark.parametrize('scheme', ['linear', 'nearest'])
    def test_dot_test(self, factor, scheme):
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid)
        ut = Function(name='ut', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        gt = Function(name='gt', grid=sub, space_order=0)

        rng = np.random.default_rng(7)
        u.data[:] = rng.random(u.shape).astype(u.dtype)
        gt.data[:] = rng.random(gt.shape).astype(gt.dtype)
        ut.data[:] = 0.
        g.data[:] = 0.

        Operator(Eq(g, restrict(u, sub, scheme=scheme))).apply()
        Operator(prolong(ut, gt, sub, scheme=scheme)).apply()

        lhs = np.sum(g.data.astype(np.float64)*gt.data.astype(np.float64))
        rhs = np.sum(u.data.astype(np.float64)*ut.data.astype(np.float64))

        assert np.isclose(lhs, rhs, rtol=1e-5)

    def test_prolong_is_an_increment(self):
        """
        `prolong` accumulates: several subgrid points can contribute to one
        parent point, so it must not overwrite.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=0.5)         # refinement: many taps per point

        ut = Function(name='ut', grid=grid)
        gt = Function(name='gt', grid=sub, space_order=0)
        gt.data[:] = 1.
        ut.data[:] = 0.

        Operator(prolong(ut, gt, sub)).apply()

        # With 17x17 unit-weighted-sum contributions onto 9x9, interior parent
        # points receive more than one
        assert ut.data.max() > 1.


class TestErrors:

    def test_integer_factor_rejected(self):
        """
        An integer-factor SubGrid needs no interpolation, and the message says
        what to use instead.
        """
        sub = SubGrid(Grid(shape=(9, 9)), factor=2)

        with pytest.raises(ValueError, match='no interpolation is needed'):
            SubGridTransfer(sub)

    def test_unknown_scheme(self):
        sub = SubGrid(Grid(shape=(9, 9)), factor=1.5)

        with pytest.raises(ValueError, match='Unknown interpolation scheme'):
            SubGridTransfer(sub, scheme='quintic')

    def test_scheme_list_is_reported(self):
        sub = SubGrid(Grid(shape=(9, 9)), factor=1.5)

        with pytest.raises(ValueError) as e:
            SubGridTransfer(sub, scheme='quintic')

        for known in SubGridTransfer.schemes:
            assert known in str(e.value)


class TestCodegen:

    def test_is_a_plain_stencil(self):
        """
        The taps are written out as one expression: no accumulator loop and no
        per-tap bounds test, both of which would block vectorization. The tap
        count is a compile-time constant, so there is nothing to accumulate
        dynamically.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)

        ccode = str(Operator(Eq(g, restrict(u, sub))).ccode)

        # One loop per subgrid Dimension, and nothing else
        assert ccode.count('for (int xs') == 1
        assert ccode.count('for (int ys') == 1
        assert 'for (int r' not in ccode
        assert 'if (' not in ccode

    @switchconfig(language='openmp')
    def test_vectorizes(self):
        """The innermost subgrid loop must still be SIMD."""
        grid = Grid(shape=(64, 64), extent=(63., 63.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        g = Function(name='g', grid=sub, space_order=0)

        ccode = str(Operator(Eq(g, restrict(u, sub))).ccode)

        assert '#pragma omp simd' in ccode
        assert '#pragma omp for' in ccode

    def test_positions_are_integer_arithmetic(self):
        """
        `factor` is an exact rational, so the base index is `(num*i)//den` in
        integers -- no `floor` on floats, which would cost a call per tap.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)

        ccode = str(Operator(Eq(g, restrict(u, sub))).ccode)

        assert 'floor' not in ccode

    @pytest.mark.parametrize('ndim,taps', [(1, 2), (2, 4), (3, 8)])
    def test_tap_count(self, ndim, taps):
        """A linear stencil reads `2**ndim` parent points, once each."""
        n = 9
        grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)

        ccode = str(Operator(Eq(g, restrict(u, sub))).ccode)

        assert ccode.count('u[') == taps

    @pytest.mark.parametrize('scheme,taps', [('nearest', 1), ('linear', 2)])
    def test_radius_matches_scheme(self, scheme, taps):
        sub = SubGrid(Grid(shape=(9, 9)), factor=1.5)

        assert SubGridTransfer(sub, scheme=scheme).radius == taps
