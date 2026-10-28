import pickle

import numpy as np
import pytest
import sympy

from devito import Eq, Function, Grid, Operator, SubGrid, TimeFunction, switchconfig
from devito.finite_differences.differentiable import (
    EvalDerivative, Weights, WeightsIndexed
)
from devito.finite_differences.subgrid import _phase_weights, weights_at
from devito.finite_differences.tools import kaiser_b, sinc_weights
from devito.ir.iet import Conditional, Expression, FindNodes, FindSymbols, Iteration
from devito.symbolics import retrieve_indexed

COEFFS = (3., 2., 0.5)


def linear(shape):
    """`3*i + 2*j + 0.5*k` over `shape`, in index units."""
    idx = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    return sum(c*i for c, i in zip(COEFFS[:len(shape)], idx, strict=True))


def restricted(factor, n=25, ndim=2):
    """Read a linear SubGrid field at its parent's points."""
    grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
    sub = SubGrid(grid, factor=factor)

    u = Function(name='u', grid=grid, space_order=2)
    vp = Function(name='vp', grid=sub, space_order=2)
    vp.data[:] = linear(sub.shape).astype(vp.dtype)
    u.data[:] = 0.

    Operator(Eq(u, vp)).apply()

    return sub, u, vp


def expected_at_parent(sub, idx):
    """
    `linear` evaluated at the parent points `idx`, in SubGrid index units.

    A parent point `x` sits at SubGrid coordinate `x/factor`, so this is what
    an exact read of a `linear`-valued SubGrid Function must produce.
    """
    return sum(c*(i/float(f)) for c, i, f in
               zip(COEFFS[:len(idx)], idx, sub.factor, strict=True))


def interior(sub, n, ndim):
    """Mask of parent points whose stencil stays inside the SubGrid."""
    idx = np.meshgrid(*[np.arange(n)]*ndim, indexing='ij')
    mask = np.ones_like(idx[0], dtype=bool)
    for i, (ns, f) in enumerate(zip(sub.shape, sub.factor, strict=True)):
        mask &= idx[i] <= int((ns - 1)*float(f)) - 1
    return mask, idx


class TestApi:

    """
    A SubGrid Function is read at the LHS's location, with no wrapper -- the
    same convention a staggered Function follows.
    """

    def test_plain_equation(self):
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = 1.
        vp.data[:] = 2.5

        Operator(Eq(u, vp*u)).apply()

        mask, _ = interior(sub, 13, 2)
        assert np.allclose(np.asarray(u.data)[mask], 2.5, atol=1e-5)

    def test_iterates_parent_dimensions_only(self):
        """
        The SubGrid Dimensions are not iterated: `vp` is read through a
        stencil indexed by the parent's Dimensions.
        """
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

    def test_every_axis_relocated_under_blocking(self):
        """
        In 3D, blocking has already run when the subgrid pass sees the Cluster,
        so a parent Dimension appears in the IterationSpace only through its
        block Dimensions. Every axis must still be relocated -- the symptom of
        getting this wrong is a read left indexed by a SubGrid Dimension.
        """
        grid = Grid(shape=(17, 17, 17), extent=(16.,)*3)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        reads = [i for i in retrieve_indexed(e) if i.function is vp]
        assert reads

        gone = set(sub.dimensions)
        for i in reads:
            assert not (set(i.free_symbols) & gone)

        # And the loop nest is the parent's, not a deeper one
        dims = [i.dim for i in FindNodes(Iteration).visit(op)]
        assert not (set(dims) & gone)

    def test_shared_factor_one_dimension_still_iterates(self):
        """
        An axis of factor 1 shares the parent's own Dimension, so it must stay
        in the IterationSpace -- dropping it leaves a loop bound with no
        runtime value.
        """
        grid = Grid(shape=(13, 13, 13), extent=(12.,)*3)
        sub = SubGrid(grid, factor=(2, 1.5, 1.0))
        assert sub.dimensions[2] is grid.dimensions[2]

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = 2.5

        op = Operator(Eq(u, vp))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

        op.apply()      # must not raise: `z` needs its bounds
        assert np.allclose(np.asarray(u.data)[:-1, :-1, :], 2.5, atol=1e-5)

    def test_operator_pickles(self):
        """
        An Operator carrying a relocated read must survive a round-trip: the
        weight table, the phase arithmetic and the SubGrid's rank offsets all
        have to reconstruct, and the result must be unchanged.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=1.5)

        vp = Function(name='vp', grid=sub, space_order=4, interp_order=4)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u = Function(name='u', grid=grid, space_order=4)

        op = Operator(Eq(u, vp))
        op.apply()
        expected = np.asarray(u.data).copy()

        u.data[:] = 0.
        pickle.loads(pickle.dumps(op)).apply(u=u, vp=vp)

        assert np.array_equal(np.asarray(u.data), expected)

    def test_one_relocation_per_phase(self):
        """
        A staggered read has one `x0` for the whole loop; here `x0` cycles
        with the parent index. The FD scheme needs a numeric evaluation point
        to pick its taps, so there is one relocation per phase.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=2)
        xs, _ = sub.dimensions
        x, _ = grid.dimensions

        vp = Function(name='vp', grid=sub, space_order=4, interp_order=4)

        # One relocation at a *numeric* `x0` is fine -- that is a phase
        phase = vp.diff(xs, deriv_order=0,
                        x0={xs: xs + sympy.Rational(1, 2)*xs.spacing})
        assert phase.evaluate

        # ...but at the parent's Dimension the evaluation point is symbolic
        with pytest.raises(TypeError, match='Cannot convert expression'):
            _ = vp.diff(xs, deriv_order=0, x0={xs: x}).evaluate

        # Which is why the read carries one set of weights per phase
        w = _phase_weights(vp, xs, sympy.Rational(2))
        assert w.nsets == 2

    def test_evaluates_like_a_staggered_read(self):
        """
        A SubGrid read is a relocation, so it evaluates to what a staggered
        one does: an `EvalDerivative` carrying the expression it reads.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=2)
        x, _ = grid.dimensions

        u = Function(name='u', grid=grid, space_order=4)
        vp = Function(name='vp', grid=sub, space_order=4, interp_order=4)
        w = Function(name='w', grid=grid, space_order=4, staggered=x)

        read = Eq(u, vp).evaluate.rhs
        staggered = Eq(u, w).evaluate.rhs
        assert isinstance(staggered, EvalDerivative)
        assert isinstance(read, EvalDerivative)

        # `base` is the Function read, so CIRE treats both as leaves
        assert read.base.function is vp
        assert read.base.is_Function
        # Already evaluated at its x0: no second relocation
        assert read._eval_at(u) is read

        # And it stays one term inside a larger sum, rather than flattening
        c = Function(name='c', grid=grid, space_order=4)
        combined = Eq(u, vp + c).evaluate.rhs
        assert len([i for i in combined.args
                    if isinstance(i, EvalDerivative)]) == 1

    def test_is_a_plain_stencil(self):
        """
        The read is a flat weighted sum: one Expression writing `u`, with no
        guard around it and no reduction temporary, so the loop vectorizes.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        with switchconfig(language='openmp'):
            op = Operator(Eq(u, vp))

        # A single unconditional assignment, not an accumulation
        exprs = [i for i in FindNodes(Expression).visit(op) if i.write is u]
        assert len(exprs) == 1
        assert not exprs[0].is_reduction
        assert not FindNodes(Conditional).visit(op)

        # ...and the innermost loop over `u` is the vectorized one
        assert '#pragma omp simd' in str(op.ccode)

    def test_weights_are_compile_time(self):
        """
        The per-phase weights are a table of literals, not arithmetic per
        point: every entry must be a number once the Operator is built.
        """
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        w = {i.function for i in FindSymbols().visit(op)
             if isinstance(i.function, Weights)}
        assert len(w) == 1
        w = w.pop()

        # `factor = 3/2` read at the parent's points gives 3 phases of 2 taps
        assert w.nsets == 3
        assert w.shape == (6,)
        assert all(i.is_Number for i in w.weights)

    @pytest.mark.parametrize('ndim,taps', [(1, 2), (2, 4), (3, 8)])
    def test_tap_count(self, ndim, taps):
        """Linear interpolation reads `2**ndim` SubGrid points."""
        n = 13
        grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))
        expr, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]

        assert len([i for i in retrieve_indexed(expr.rhs)
                    if i.function is vp]) == taps


class TestExactness:

    """
    Linear interpolation is exact on a linear field -- the sharpest check, as
    any error in a base index or a weight shows up immediately.
    """

    @pytest.mark.parametrize('expand', [True, False])
    def test_inside_a_time_loop(self, expand):
        """
        The read must sit correctly inside a time loop: the relocation touches
        only space, so the SubGrid term is re-read identically at every step
        and the result accumulates linearly. A stepping or loop-placement error
        shows up as drift that a stationary Function cannot reveal.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=1.5)

        vp = Function(name='vp', grid=sub, space_order=4, interp_order=4)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)

        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=1)
        u.data[:] = 0.

        nt = 5
        Operator(Eq(u.forward, u + vp),
                 opt=('advanced', {'expand': expand})).apply(time_M=nt - 1)

        mask, idx = interior(sub, n, 2)
        pad = 4
        for c in idx:
            mask &= (c >= pad) & (c <= n - 1 - pad)

        # `nt` steps, each adding the same interpolated value
        expected = nt*expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data[nt % 2])[mask], expected[mask],
                           atol=1e-3)

    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.5, 0.75, 1.25])
    def test_1d(self, factor):
        n = 25
        sub, u, _ = restricted(factor, n=n, ndim=1)

        mask, idx = interior(sub, n, 1)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.5, (2, 1.5), (0.5, 1.5)])
    def test_2d(self, factor):
        n = 25
        sub, u, _ = restricted(factor, n=n, ndim=2)

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    @pytest.mark.parametrize('factor', [1.5, 2.5, (2, 1.5, 1.)])
    def test_3d(self, factor):
        """
        3D is the case where blocking has already run by the time the subgrid
        pass sees the Cluster, so a parent Dimension shows up only through its
        block Dimensions. Every axis must still be relocated.
        """
        n = 17
        sub, u, _ = restricted(factor, n=n, ndim=3)

        mask, idx = interior(sub, n, 3)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    def test_constant_field(self):
        """The weights must sum to one at every phase."""
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = -3.25
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, _ = interior(sub, n, 2)
        assert np.allclose(np.asarray(u.data)[mask], -3.25, atol=1e-5)

    def test_refinement(self):
        """A factor below 1 refines: every parent point is an exact hit."""
        n = 25
        sub, u, vp = restricted(0.5, n=n, ndim=1)

        assert sub.shape[0] > n
        mask, idx = interior(sub, n, 1)
        assert np.allclose(np.asarray(u.data)[mask], (3.*idx[0]/0.5)[mask],
                           atol=1e-4)


class TestIntegerFactor:

    """
    An integer factor needs no interpolation when *writing* the SubGrid -- its
    points coincide with the parent's, so `Eq(g, u)` is a strided copy and
    keeps Part 1's lowering.

    *Reading* one on the parent grid is a different matter: `Eq(u, g)` is
    evaluated at every parent point, and the points in between two SubGrid
    points do need interpolating.
    """

    def test_not_interpolated(self):
        grid = Grid(shape=(12, 12))

        assert not SubGrid(grid, factor=3).is_interpolated
        assert SubGrid(grid, factor=1.5).is_interpolated

    def test_read_writes_every_parent_point(self):
        """
        `Eq(u, vp*u)` writes `u`, so every parent point must be computed. The
        Cluster still carries the SubGrid's GuardFactors at this stage; leaving
        them behind lets `subsampling` strengthen them into strided iteration
        and only every `factor`-th point of `u` gets written -- silently wrong
        results, not a crash.
        """
        n = 9
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = 1.
        vp.data[:] = 3.

        op = Operator(Eq(u, vp*u))

        # The loop nest is the parent's, not the SubGrid's
        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

        op.apply()

        assert not np.any(np.asarray(u.data) == 1.)
        assert np.allclose(np.asarray(u.data), 3., atol=1e-5)

    def test_read_is_exact_on_a_linear_field(self):
        """
        Reading an integer-factor SubGrid at *every* parent point interpolates
        at the points falling between two SubGrid points, and linear
        interpolation is exact there.
        """
        n = 17
        grid = Grid(shape=(n,), extent=(float(n - 1),))
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = (3.*np.arange(sub.shape[0])).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        idx = np.arange(n)
        inside = idx <= (sub.shape[0] - 1)*2 - 1
        expected = 3.*(idx/2.)

        assert np.allclose(np.asarray(u.data)[inside], expected[inside],
                           atol=1e-5)

    def test_write_stays_a_strided_copy(self):
        """Writing the SubGrid must not acquire interpolation weights."""
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(vp, u))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in sub.dimensions]
        assert 'float w' not in str(op)

    @pytest.mark.parametrize('factor', [3, (3, 1), (2, 3)])
    def test_scatter_per_axis(self, factor):
        """Each axis scatters on its own factor, including a factor-1 axis."""
        n = 13
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = -1.
        vp.data[:] = 7.

        Operator(Eq(u, vp, subdomain=sub)).apply()

        a = np.asarray(u.data)
        f0, f1 = [int(v) for v in sub.factor]
        assert np.all(a[::f0, ::f1] == 7.)
        # Nothing outside the aligned points was touched
        assert int((a != -1.).sum()) == a[::f0, ::f1].size

    @pytest.mark.parametrize('factor', [3, (3, 1), (2, 3)])
    def test_gather_per_axis_is_exact(self, factor):
        """
        Reading at every parent point is exact on a linear field for per-axis
        integer factors too, where the aligned and interpolated axes mix.
        """
        n = 17
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    def test_explicit_scatter(self):
        """
        `subdomain=sub` iterates the SubGrid's points alone, so writing the
        parent from it scatters: every `factor`-th point is written and the
        rest keep their previous values.
        """
        n = 9
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = -1.
        vp.data[:] = 7.

        op = Operator(Eq(u, vp, subdomain=sub))

        # The strided loop over the SubGrid, not the parent's
        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in sub.dimensions]
        assert 'float w' not in str(op)

        op.apply()

        a = np.asarray(u.data)
        assert np.all(a[::2, ::2] == 7.)
        assert np.all(a[1::2, :] == -1.)
        assert np.all(a[:, 1::2] == -1.)

    def test_scatter_rejected_when_interpolated(self):
        """
        With a rational factor only every q-th SubGrid point falls on the
        parent, so iterating the SubGrid would also write the ones in
        between, which no parent point scatters to.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        # Rejected where the mistake is made, at Eq construction
        with pytest.raises(ValueError, match='only some of its points'):
            Eq(u, vp, subdomain=sub)

        # The overlap is real -- factor 3/2 retains every other point -- so
        # it is the gaps that make the scatter ill-posed
        p, q = sympy.Rational(3, 2).p, sympy.Rational(3, 2).q
        on_parent = [j for j in range(sub.shape[0]) if (j*p) % q == 0]
        assert on_parent == [0, 2, 4] and len(on_parent) < sub.shape[0]


class TestSchemes:

    """
    The stencil width follows `interp_order`, exactly as a staggered
    relocation does -- the weights come from the same FD path. `sinc` swaps
    the weights for Kaiser windowed ones at the same width.
    """

    # `sinc` has no radius-1 window tabulated, so it starts at 4 taps
    @pytest.mark.parametrize('interpolation,interp_order', [
        ('fd', 2), ('fd', 4), ('fd', 8),
        ('sinc', 4), ('sinc', 8),
    ])
    def test_interp_order_sets_the_width(self, interp_order, interpolation):
        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation=interpolation)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8,
                      interp_order=interp_order)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        reads = {i for i in retrieve_indexed(e) if i.function is vp}

        assert len(reads) == interp_order

    @pytest.mark.parametrize('interp_order', [4, 8])
    @pytest.mark.parametrize('x0', [0., 1/3, 0.5, 2/3])
    def test_sinc_matches_the_shared_window(self, interp_order, x0):
        """
        The SubGrid `sinc` weights must come from the same windowed sinc, with
        the same Kaiser parameter, as `SincInterpolator`. That table is keyed
        by *radius*: indexing it by `interp_order` instead silently picks the
        wrong window, and asks for radii Hicks (2002) never tabulated.
        """
        r = interp_order//2
        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation='sinc')
        f = Function(name='f', grid=sub, space_order=8,
                     interp_order=interp_order)

        # Through the registry, so this also pins that `sinc` dispatches here
        row = weights_at('sinc', f, sub.dimensions[0], x0)
        offsets = sorted(row)

        expected = sinc_weights([o - x0 for o in offsets], r, kaiser_b[r],
                                normalize=True)

        assert np.allclose([float(row[o]) for o in offsets],
                           [float(i) for i in expected], rtol=0, atol=0)
        # Partition of unity, or a constant field picks up a bias
        assert np.isclose(sum(float(row[o]) for o in offsets), 1., atol=1e-12)

    def test_sinc_rejects_untabulated_radius(self):
        """
        `interp_order=2` is radius 1, which the Kaiser table does not cover.
        Reject it at construction rather than with a `KeyError` once the
        weights are built.
        """
        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation='sinc')

        with pytest.raises(ValueError, match='not supported by the `sinc`'):
            Function(name='f', grid=sub, space_order=8, interp_order=2)

    def test_defaults(self):
        grid = Grid(shape=(12,))
        sub = SubGrid(grid, factor=1.5)

        assert sub.interpolation == 'fd'
        assert Function(name='f', grid=sub).interp_order == 2

    def test_unknown_scheme_rejected(self):
        grid = Grid(shape=(12,))

        with pytest.raises(ValueError, match='Unknown interpolation scheme'):
            SubGrid(grid, factor=1.5, interpolation='quintic')

    @pytest.mark.parametrize('factor', [np.pi, 100.5, 65, 500])
    def test_too_many_phases_rejected(self, factor):
        """
        Reading a `p/q` SubGrid at its parent's points repeats with period `p`,
        and every phase carries its own set of weights, so `p` sizes the weight
        table. Capping the factor's *denominator* does not bound it: `100.5`
        is `201/2`, a denominator of 2 but 201 phases -- and a large *integer*
        factor is `p` outright.
        """
        grid = Grid(shape=(1001,))
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        v = Function(name='v', grid=sub, space_order=2)

        with pytest.raises(ValueError, match='phases, more than'):
            Operator(Eq(u, v))

    @pytest.mark.parametrize('factor', [65, 500])
    def test_big_integer_factor_still_writes(self, factor):
        """
        The flip side: the cap is on the *read*. A large integer factor is a
        perfectly good strided write, which builds no weight table at all.
        """
        grid = Grid(shape=(1001,))
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        v = Function(name='v', grid=sub, space_order=2)
        u.data[:] = 1.

        Operator(Eq(v, u)).apply()

        assert np.allclose(v.data, 1.)

    @pytest.mark.parametrize('factor', [1.5, 2.5, 2])
    def test_staggered_target(self, factor):
        """
        A staggered target is read at its own location: `indices_ref` sets the
        evaluation point, exactly as it does for an ordinary FD relocation. It
        is not a special case -- note `factor=2` staggered lands on quarter
        cells, so the phases are not simply `k/p`.
        """
        n = 25
        grid = Grid(shape=(n,), extent=(float(n - 1),))
        x, = grid.dimensions
        sub = SubGrid(grid, factor=factor)

        vp = Function(name='vp', grid=sub, space_order=8, interp_order=4)
        vp.data[:] = np.arange(sub.shape[0], dtype=vp.dtype)

        got = {}
        for stagger in (None, x):
            u = Function(name=f'u{int(stagger is not None)}', grid=grid,
                         space_order=8, staggered=stagger)
            Operator(Eq(u, vp)).apply()
            got[stagger] = np.asarray(u.data).copy()

        # A linear field, so the read is exact at either location
        for stagger, shift in ((None, 0.), (x, 0.5)):
            expected = (np.arange(n) + shift)/factor
            assert np.allclose(got[stagger][6:-6], expected[6:-6], atol=2e-4)

        # ...and the two differ, or the stagger was ignored
        assert not np.allclose(got[None][6:-6], got[x][6:-6], atol=1e-3)

    @pytest.mark.parametrize('interp_order', [0, 1])
    def test_interp_order_below_two_rejected(self, interp_order):
        """
        `0` would mean "no relocation", which is meaningful for a staggered
        shift but not here: a SubGrid's Dimensions are not the parent's, so a
        read without relocation leaves both in the same Eq.
        """
        grid = Grid(shape=(17,))
        sub = SubGrid(grid, factor=2)

        with pytest.raises(ValueError, match='at least 2'):
            Function(name='f', grid=sub, space_order=4,
                     interp_order=interp_order)

    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.75, 1.25, 2, 3])
    def test_usable_factors_accepted(self, factor):
        """A ratio of small integers must still be allowed."""
        grid = Grid(shape=(25,))
        sub = SubGrid(grid, factor=factor)

        assert sub.rational_factor[0].numerator <= 64

    def test_space_order_must_hold_the_taps(self):
        """
        An interpolated SubGrid read reaches `interp_order//2` past the base
        index, which under MPI must fit in the halo. Devito's own
        `interp_order <= space_order` check is skipped for `space_order <= 1`,
        so `space_order=0` would otherwise keep `interp_order=2` with no halo
        at all -- silently wrong results on every rank but the first.
        """
        grid = Grid(shape=(25, 25))
        sub = SubGrid(grid, factor=1.5)

        with pytest.raises(ValueError, match='too small for'):
            Function(name='f', grid=sub, space_order=0)

        # One layer is enough for the default width
        assert Function(name='g', grid=sub, space_order=1).interp_order == 2

        # An integer-factor SubGrid is written, not interpolated, so it is
        # exempt
        plain = SubGrid(grid, factor=2, name='k')
        assert Function(name='h', grid=plain, space_order=0) is not None

    def test_interp_order_bounded_by_space_order(self):
        """`interp_order <= space_order` is the existing Function invariant;
        it is also what guarantees the taps stay inside the halo."""
        grid = Grid(shape=(12,))
        sub = SubGrid(grid, factor=1.5)

        with pytest.raises(ValueError, match='interp_order'):
            Function(name='f', grid=sub, space_order=2, interp_order=4)

    @pytest.mark.parametrize('interpolation', ['fd', 'sinc'])
    def test_weights_sum_to_one(self, interpolation):
        """
        Partition of unity at every phase, otherwise a constant field picks up
        a bias. The windowed sinc needs explicit normalisation for this.
        """
        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation=interpolation)
        vp = Function(name='vp', grid=sub, space_order=8, interp_order=4)

        weights = _phase_weights(vp, sub.dimensions[0],
                                 sub.rational_factor[0])
        values = [float(i) for i in weights.initvalue]
        for phase in range(weights.nsets):
            at_phase = values[phase*weights.nweights:(phase + 1)*weights.nweights]
            assert np.isclose(sum(at_phase), 1., atol=1e-12)

    @pytest.mark.parametrize('interp_order', [2, 4, 8])
    def test_fd_is_exact_on_a_linear_field(self, interp_order):
        """
        The FD weights are proper Lagrange coefficients, so they reproduce a
        linear field exactly at every width.
        """
        n = 33
        grid = Grid(shape=(n,), extent=(float(n - 1),))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8,
                      interp_order=interp_order)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        pad = interp_order//2 + 2
        idx = np.arange(n)
        mask = (idx >= pad) & (idx <= int((sub.shape[0] - 1)*1.5) - pad)

        assert np.allclose(np.asarray(u.data)[mask], 3.*(idx[mask]/1.5),
                           atol=2e-4)

    def test_accuracy_improves_with_interp_order(self):
        """Wider stencils are more accurate on a field the SubGrid only
        marginally resolves -- which is what `interp_order` is for."""
        n = 129
        fac = 1.5
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        k = 2*np.pi*8/(n - 1)        # ~11 SubGrid points per cycle

        errs = {}
        for interp_order in (2, 4, 8):
            sub = SubGrid(grid, factor=fac, name=f's{interp_order}')
            u = Function(name=f'u{interp_order}', grid=grid, space_order=8)
            vp = Function(name=f'v{interp_order}', grid=sub, space_order=8,
                          interp_order=interp_order)

            i, j = np.meshgrid(*[np.arange(s) for s in sub.shape],
                               indexing='ij')
            vp.data[:] = np.sin(k*i*fac)*np.cos(k*j*fac)
            u.data[:] = 0.

            Operator(Eq(u, vp)).apply()

            pad = interp_order//2 + 2
            px, py = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            exact = np.sin(k*px)*np.cos(k*py)
            mask = np.ones_like(px, dtype=bool)
            for q, c in enumerate((px, py)):
                mask &= (c >= pad) & (c <= int((sub.shape[q] - 1)*fac) - pad)

            errs[interp_order] = np.abs(np.asarray(u.data)[mask]
                                        - exact[mask]).max()

        assert errs[4] < errs[2]/5.
        assert errs[8] < errs[4]/5.


class TestUnexpanded:

    """
    `expand=False` leaves the read as a nest of derivatives, one loop per
    relocated Dimension, instead of unrolling `interp_order**ndim` terms.
    """

    @pytest.mark.parametrize('interp_order', [2, 4, 8])
    def test_emits_a_stencil_loop(self, interp_order):
        grid = Grid(shape=(33, 33), extent=(32., 32.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8,
                      interp_order=interp_order)

        op = Operator(Eq(u, vp), opt=('advanced', {'expand': False}))

        dims = [i.dim for i in FindNodes(Iteration).visit(op)]
        stencils = [d for d in dims if d.is_Stencil]

        # One stencil loop per relocated Dimension, and the parent's own
        assert len(stencils) == 2
        assert all(d.symbolic_size == interp_order for d in stencils)

    @pytest.mark.parametrize('interp_order', [2, 4, 8])
    @pytest.mark.parametrize('ndim', [1, 2, 3])
    def test_matches_expanded(self, interp_order, ndim):
        """
        The two lowerings must agree: same weights, same taps, only the
        schedule differs. Both are also checked against the analytic result,
        so a phase or weight error shared by the two cannot pass.
        """
        n = 25 if ndim < 3 else 21
        grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
        sub = SubGrid(grid, factor=1.5)

        vp = Function(name='vp', grid=sub, space_order=8,
                      interp_order=interp_order)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)

        got = []
        for expand in (True, False):
            u = Function(name=f'u{int(expand)}', grid=grid, space_order=8)
            u.data[:] = 0.
            Operator(Eq(u, vp),
                     opt=('advanced', {'expand': expand})).apply()
            got.append(np.asarray(u.data).copy())

        mask, idx = interior(sub, n, ndim)
        pad = interp_order//2 + 2
        for k, c in enumerate(idx):
            mask &= (c >= pad) & (c <= int((sub.shape[k] - 1)*1.5) - pad)

        assert np.allclose(got[0][mask], got[1][mask], atol=1e-5)

        # ...and an independent oracle: the FD weights are Lagrange
        # coefficients, so a linear field is reproduced exactly
        expected = expected_at_parent(sub, idx)
        for v in got:
            assert np.allclose(v[mask], expected[mask], atol=2e-4)

    def test_reexpansion_keeps_the_table(self):
        """
        Expansion substitutes a weight per tap, so it needs the table index to
        resolve to a number. A phase index does not -- the set is chosen at run
        time -- so those reads stay as they are rather than being substituted
        with one arbitrary phase's values.
        """
        grid = Grid(shape=(33, 33), extent=(32., 32.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8, interp_order=4)

        unexpanded = Eq(u, vp)._evaluate(expand=False)
        expanded = unexpanded._evaluate(expand=True)

        # The taps are spelled out, but each still reads the table
        reads = expanded.find(WeightsIndexed)
        assert reads
        assert all(not i.indices[-1].is_Integer for i in reads)

    def test_weights_table_is_flat(self):
        """
        The phase rows live end to end in one *flat* table, indexed by
        `phase*ntaps + i`. A `(phase, tap)`-shaped one would reach a device
        kernel as a bare `float*` and lose its row stride, so the shape must
        stay one-dimensional -- and must span every row, not just the first.
        """
        grid = Grid(shape=(33, 33), extent=(32., 32.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8, interp_order=4)

        op = Operator(Eq(u, vp), opt=('advanced', {'expand': False}))

        w = {i.function for i in FindSymbols().visit(op)
             if isinstance(i.function, Weights)}
        assert len(w) == 1
        w = w.pop()

        # 3 phases of 4 taps, declared as a single flat row
        assert w.nsets == 3
        assert len(w.dimensions) == 1
        assert w.shape == (12,)
        assert w.symbolic_shape == (12,)

        # ...and the C declaration agrees, with no 2-D subscript anywhere
        src = str(op)
        assert 'float w0[12]' in src
        assert '[3][4]' not in src

    @pytest.mark.parametrize('expand', [True, False])
    @pytest.mark.parametrize('factors, ntables', [((1.5, 2.5), 2),
                                                  ((1.5, 1.5), 1)])
    def test_several_subgrids_in_one_expression(self, factors, ntables,
                                                expand):
        """
        Two SubGrid reads in one expression each get their own table, unless
        the weights coincide -- `lower_index_derivatives` keys the table on its
        content, so identical ones collapse to a single C declaration.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        subs = [SubGrid(grid, factor=i, name=f'sub{n}')
                for n, i in enumerate(factors)]

        u = Function(name='u', grid=grid, space_order=8)
        vs = [Function(name=f'v{n}', grid=i, space_order=8, interp_order=4)
              for n, i in enumerate(subs)]
        # A *linear* field per SubGrid, so each read's own factor shows up in
        # the result: a constant one would agree whichever table were used
        for v, sub in zip(vs, subs, strict=True):
            v.data[:] = linear(sub.shape).astype(v.dtype)

        op = Operator(Eq(u, vs[0] + vs[1]),
                      opt=('advanced', {'expand': expand}))

        w = {i.function for i in FindSymbols().visit(op)
             if isinstance(i.function, Weights)}
        assert len(w) == ntables

        op.apply()

        # Each term read at its own factor, so the sum pins both
        mask, idx = interior(subs[0], 25, 2)
        m1, _ = interior(subs[1], 25, 2)
        mask &= m1
        for c in idx:
            mask &= (c >= 6) & (c <= 25 - 7)
        expected = (expected_at_parent(subs[0], idx) +
                    expected_at_parent(subs[1], idx))

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)


class TestMPI:

    """
    Interpolated reads under domain decomposition.

    The parent Grid and the SubGrid are decomposed independently, so a rank's
    parent points generally need SubGrid points its neighbour owns. Two things
    have to be right for this to work: the halo exchange must happen at all,
    and the base index must be computed on the *global* index.
    """

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.75, 2])
    def test_read_matches_serial(self, mode, factor):
        """
        Linear interpolation is exact on a linear field, on every rank. Getting
        the index space wrong leaves rank 0 correct and the others wrong, so
        this must assert on all of them.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        # Compare on this rank's share of the global domain
        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('factor', [1.5, 2])
    def test_staggered_read_matches_serial(self, mode, factor):
        """
        A staggered target is read half a cell off its own index, so the phase
        differs from the unstaggered case. That has to hold on every rank: the
        shift enters the same non-affine index arithmetic as the base.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        x, _ = grid.dimensions
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=4, staggered=x)
        vp = Function(name='vp', grid=sub, space_order=4, interp_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 2)
        for c in idx:
            mask &= (c >= 2) & (c <= n - 3)

        # `linear` at the staggered location: half a parent cell along `x`
        shifted = (idx[0] + 0.5, idx[1])
        expected = expected_at_parent(sub, shifted)

        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('factor', [2, 1.5])
    @pytest.mark.parametrize('expand', [True, False])
    def test_wide_read_matches_serial(self, mode, factor, expand):
        """
        A wide stencil reaches several points past the base index, so it needs
        the neighbour's points. An *integer* factor is not `is_interpolated`,
        yet reading it at every parent point still interpolates and so still
        spans `interp_order` taps -- its halo must be exchanged too.

        Both lowerings are covered: the unexpanded one indexes the phase table
        at run time, which is the combination most sensitive to the global
        index arithmetic.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=4)
        vp = Function(name='vp', grid=sub, space_order=4, interp_order=4)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp), opt=('advanced', {'expand': expand})).apply()

        # Stay clear of the global boundary, where the stencil is truncated
        mask, idx = interior(sub, n, 2)
        for c in idx:
            mask &= (c >= 4) & (c <= n - 5)
        expected = expected_at_parent(sub, idx)

        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    def test_base_index_is_global(self, mode):
        """
        `IntDiv`/`Mod` are non-linear, so they must be evaluated on the global
        index. Computing them on the local one is correct only on the rank
        whose offsets are zero, which is why `test_read_matches_serial` has to
        check every rank.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        syms = {s.name for s in e.free_symbols}
        for d, ds in zip(grid.dimensions, sub.dimensions, strict=True):
            assert f'oi_{d.name}' in syms
            assert f'oi_{ds.name}' in syms

    @pytest.mark.parallel(mode=[2, 4])
    def test_read_matches_serial_3d(self, mode):
        """
        3D is where blocking has already run when the subgrid pass sees the
        Cluster, so it exercises the block-promotion and the global index
        together.
        """
        n = 17
        grid = Grid(shape=(n,)*3, extent=(float(n - 1),)*3)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 3)
        expected = expected_at_parent(sub, idx)
        loc = u.local_indices

        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('interp_order', [2, 4, 8])
    def test_every_width_matches_serial(self, mode, interp_order):
        """
        A wider stencil reaches further into the halo, but `interp_order <=
        space_order` keeps it inside -- no separate halo bookkeeping needed.

        Uses the FD weights, which are exact on a linear field at every width.
        `sinc` is covered by `test_sinc_matches_serial`, which cannot assert
        exactness: a narrow windowed sinc does not match the field's first
        moment.
        """
        interpolation = 'fd'
        n, fac = 65, 1.5
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=fac, interpolation=interpolation)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8,
                      interp_order=interp_order)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        pad = interp_order//2 + 2
        px, py = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        expected = 3.*(px/fac) + 2.*(py/fac)
        mask = np.ones_like(px, dtype=bool)
        for q, c in enumerate((px, py)):
            mask &= (c >= pad) & (c <= int((sub.shape[q] - 1)*fac) - pad)

        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=5e-3)

    @pytest.mark.parallel(mode=[2, 4])
    def test_sinc_matches_serial(self, mode):
        """
        `sinc` is not exact on a linear field at a narrow width, so this
        asserts agreement with the *serial* result rather than with the exact
        field -- which is what the MPI path has to preserve.
        """
        n, fac, io = 65, 1.5, 8
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=fac, interpolation='sinc')

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8, interp_order=io)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        # `sinc` at this width tracks the linear field to ~1e-3
        pad = io//2 + 2
        px, py = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        expected = 3.*(px/fac) + 2.*(py/fac)
        mask = np.ones_like(px, dtype=bool)
        for q, c in enumerate((px, py)):
            mask &= (c >= pad) & (c <= int((sub.shape[q] - 1)*fac) - pad)

        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], rtol=2e-3, atol=2e-2)
