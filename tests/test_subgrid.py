import numpy as np
import pytest

from devito import Eq, Function, Grid, Operator, SubGrid, switchconfig
from devito.ir.iet import Expression, FindNodes, HaloSpot, Iteration
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

    def test_is_a_plain_stencil(self):
        """
        No accumulator and no per-tap branch: the taps are a flat weighted
        sum, so the inner loop stays vectorizable.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        with switchconfig(language='openmp'):
            ccode = str(Operator(Eq(u, vp)).ccode)

        assert 'for (int r' not in ccode
        assert 'if (' not in ccode
        assert 'floor' not in ccode
        assert '#pragma omp simd' in ccode

    def test_weights_are_compile_time(self):
        """The per-phase weights are a stack table, not computed per point."""
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        ccode = str(Operator(Eq(u, vp)).ccode)

        # `factor = 3/2` reading at the parent's points gives 3 phases of 2
        # taps, i.e. a 6-entry table
        assert '[6]' in ccode

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

    def test_strided_copy(self):
        n = 24
        grid = Grid(shape=(n, n))
        sub = SubGrid(grid, factor=3)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        data = np.arange(n*n, dtype=u.dtype).reshape(n, n)
        u.data[:] = data

        Operator(Eq(g, u)).apply()

        assert np.array_equal(np.asarray(g.data), data[::3, ::3])

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
        An interpolated SubGrid has no points on the parent grid, so there is
        nothing to scatter onto -- it must raise, not silently gather.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        # Rejected where the mistake is made, at Eq construction
        with pytest.raises(ValueError, match='fall \\*between\\*'):
            Eq(u, vp, subdomain=sub)

    def test_self_reference(self):
        """
        A SubGrid Function read at a shift of its own Dimensions is a
        recurrence, not a relocation -- it must not be interpolated.
        """
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=(3, 1))
        xs = sub.dimensions[0]

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        f.data[:] = 1.
        g.data[:] = 0.

        Operator(Eq(g, g.subs(xs, xs - 1) + f)).apply()

        assert np.allclose(np.asarray(g.data)[:, 0], [0., 1., 2., 3.])


class TestSchemes:

    """
    The stencil width follows `interp_order`, exactly as a staggered
    relocation does -- the weights come from the same FD path. `sinc` swaps
    the weights for Kaiser windowed ones at the same width.
    """

    @pytest.mark.parametrize('interp_order', [2, 4, 8])
    @pytest.mark.parametrize('interpolation', ['fd', 'sinc'])
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

    def test_defaults(self):
        grid = Grid(shape=(12,))
        sub = SubGrid(grid, factor=1.5)

        assert sub.interpolation == 'fd'
        assert Function(name='f', grid=sub).interp_order == 2

    def test_unknown_scheme_rejected(self):
        grid = Grid(shape=(12,))

        with pytest.raises(ValueError, match='Unknown interpolation scheme'):
            SubGrid(grid, factor=1.5, interpolation='quintic')

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
        from devito.finite_differences.subgrid import _phase_table

        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation=interpolation)
        vp = Function(name='vp', grid=sub, space_order=8, interp_order=4)

        w, offsets, nphases = _phase_table(vp, sub.rational_factor[0])
        values = [float(i) for i in w.initvalue]
        for ph in range(nphases):
            row = values[ph*len(offsets):(ph + 1)*len(offsets)]
            assert np.isclose(sum(row), 1., atol=1e-12)

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


class TestMPI:

    """
    Interpolated reads under domain decomposition.

    The parent Grid and the SubGrid are decomposed independently, so a rank's
    parent points generally need SubGrid points its neighbour owns. Two things
    have to be right for this to work: the halo exchange must happen at all,
    and the base index must be computed on the *global* index.
    """

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.75])
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

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_is_emitted(self, mode):
        """
        `vp` is never written, but its halo is stale on entry -- `.data` only
        fills the local domain -- so the exchange must not be dropped.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        assert len(FindNodes(HaloSpot).visit(op)) == 0      # already lowered
        assert 'haloupdate' in str(op)
