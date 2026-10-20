import numpy as np
import pytest
from sympy import Or

from devito import (
    CondEq, ConditionalDimension, Constant, Eq, Function, Grid, Operator, TimeFunction,
    switchconfig
)
from devito.ir.iet import Conditional, FindNodes, Iteration


def iteration(op, name):
    """The Iteration over the Dimension named `name` in `op`."""
    for i in FindNodes(Iteration).visit(op):
        if i.dim.name == name:
            return i
    raise KeyError(name)


def subsampled(shape, factor, ndim=2):
    """
    A grid, a Function `f` over it, a Function `g` subsampled by `factor` along
    the leading Dimension, and the ConditionalDimension tying them together.
    """
    grid = Grid(shape=shape[:ndim])
    dims = grid.dimensions
    cd = ConditionalDimension(name='xs', parent=dims[0], factor=factor)

    f = Function(name='f', grid=grid)
    gshape = ((shape[0] + factor - 1)//factor,) + tuple(shape[1:ndim])
    g = Function(name='g', shape=gshape, dimensions=(cd,) + dims[1:],
                 grid=grid, space_order=0)

    return grid, f, g, cd


class TestStridedCodegen:

    """Subsampling lowers to a loop stride, not a guard."""

    def test_codegen(self):
        _, f, g, cd = subsampled((11, 11), 2)
        op = Operator(Eq(g, f))

        # No guard...
        assert not FindNodes(Conditional).visit(op)
        # ...the parent Dimension's loop is strided instead
        assert iteration(op, 'x').limits[2] == cd.symbolic_factor

    @pytest.mark.parametrize('factor', [2, 3, 4, 5])
    def test_factor_in_loop_header(self, factor):
        _, f, g, cd = subsampled((16, 8), factor)
        op = Operator(Eq(g, f))

        assert f'+= {cd.symbolic_factor.name}' in str(op.ccode)
        assert not FindNodes(Conditional).visit(op)

    def test_innermost_strided(self):
        """
        The innermost Dimension is strided too. The alternative is a guard
        *inside* the SIMD loop -- a predicated store on every iteration --
        which is worse than striding on both instruction count and branching.
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        ys = ConditionalDimension(name='ys', parent=y, factor=3)
        u = Function(name='u', grid=grid)
        usub = Function(name='usub', grid=grid, dimensions=(x, ys),
                        shape=(12, 4), space_order=0)

        op = Operator(Eq(usub, u))

        assert iteration(op, 'y').limits[2] == ys.symbolic_factor
        assert not FindNodes(Conditional).visit(op)

    def test_innermost_numerics(self):
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        ys = ConditionalDimension(name='ys', parent=y, factor=3)
        u = Function(name='u', grid=grid)
        usub = Function(name='usub', grid=grid, dimensions=(x, ys),
                        shape=(12, 4), space_order=0)
        data = np.arange(144, dtype=u.dtype).reshape(12, 12)
        u.data[:] = data

        Operator(Eq(usub, u)).apply()

        assert np.array_equal(usub.data, data[:, ::3])

    def test_multiple_dimensions_strided(self):
        """Each subsampled Dimension gets its own independent stride."""
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        xs = ConditionalDimension(name='xs', parent=x, factor=3)
        ys = ConditionalDimension(name='ys', parent=y, factor=2)
        u = Function(name='u', grid=grid)
        usub = Function(name='usub', grid=grid, dimensions=(xs, ys),
                        shape=(4, 6), space_order=0)

        op = Operator(Eq(usub, u))

        assert not FindNodes(Conditional).visit(op)
        assert iteration(op, 'x').limits[2] == xs.symbolic_factor
        assert iteration(op, 'y').limits[2] == ys.symbolic_factor

    def test_multiple_dimensions_numerics(self):
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        xs = ConditionalDimension(name='xs', parent=x, factor=3)
        ys = ConditionalDimension(name='ys', parent=y, factor=2)
        u = Function(name='u', grid=grid)
        usub = Function(name='usub', grid=grid, dimensions=(xs, ys),
                        shape=(4, 6), space_order=0)
        data = np.arange(144, dtype=u.dtype).reshape(12, 12)
        u.data[:] = data

        Operator(Eq(usub, u)).apply()

        assert np.array_equal(usub.data, data[::3, ::2])

    def test_anchored_min(self):
        """The loop min is rounded up onto the subsampling grid."""
        _, f, g, cd = subsampled((12, 12), 3)
        op = Operator(Eq(g, f))

        _min = iteration(op, 'x').limits[0]
        # Not simply `x_m`: it carries the round-up to the next multiple
        assert _min != f.grid.dimensions[0].symbolic_min
        assert cd.symbolic_factor in _min.free_symbols


class TestNumerics:

    """Results must match an independent reference."""

    @pytest.mark.parametrize('shape,factor,ndim', [
        ((12,), 2, 1), ((12,), 3, 1), ((13,), 4, 1),
        ((12, 12), 2, 2), ((12, 12), 3, 2), ((11, 9), 4, 2),
        ((9, 9, 9), 3, 3),
    ])
    def test_against_numpy(self, shape, factor, ndim):
        _, f, g, _ = subsampled(shape, factor, ndim=ndim)
        data = np.arange(f.data.size, dtype=f.dtype).reshape(f.data.shape)
        f.data[:] = data
        Operator(Eq(g, f)).apply()

        assert np.array_equal(g.data, data[::factor])

    @pytest.mark.parametrize('x_m,x_M', [
        (0, 11), (1, 11), (2, 11), (3, 11), (0, 9), (4, 10),
    ])
    def test_anchoring(self, x_m, x_M):
        """
        Exactly the points of `[x_m, x_M]` that lie on the subsampling grid
        must be written; `x_m` need not be a multiple of `factor`.
        """
        factor = 3
        _, f, g, _ = subsampled((12, 12), factor)
        f.data[:] = 1.
        g.data[:] = -1.
        Operator(Eq(g, f)).apply(x_m=x_m, x_M=x_M)

        written = {i//factor for i in range(x_m, x_M + 1) if i % factor == 0}
        for i in range(g.data.shape[0]):
            expected = 1. if i in written else -1.
            assert np.all(g.data[i] == expected), (i, x_m, x_M)

    def test_constant_factor(self):
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        factor = Constant(name='factor', dtype=np.int32, value=3)
        cd = ConditionalDimension(name='xs', parent=x, factor=factor)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(4, 12), dimensions=(cd, y), grid=grid,
                     space_order=0)
        data = np.arange(144, dtype=f.dtype).reshape(12, 12)
        f.data[:] = data
        Operator(Eq(g, f)).apply(factor=3)

        assert np.array_equal(g.data, data[::3])

    def test_time_subsampling(self):
        """
        Time is SEQUENTIAL, so this keeps its guard -- but must still save the
        right slices.
        """
        nt, factor = 9, 4
        grid = Grid(shape=(5, 5))
        ts = ConditionalDimension(name='ts', parent=grid.time_dim,
                                  factor=factor)
        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_dim=ts,
                             save=(nt + factor - 1)//factor)

        Operator([Eq(u.forward, u + 1.), Eq(usave, u)]).apply(time_M=nt-2)

        # time runs 0..nt-2, so only the slots whose time index is reached
        # are written; `u` holds the step count at that point
        nwritten = (nt - 2)//factor + 1
        for i in range(nwritten):
            assert np.allclose(usave.data[i], i*factor)


class TestEquivalence:

    """The strided loop must visit exactly the guard's point set."""

    @pytest.mark.parametrize('factor', [1, 2, 3, 4, 5, 6])
    def test_point_set_identical(self, factor):
        """
        Exhaustive check of the rewrite's arithmetic: iterating
        `range(anchor, x_M + 1, factor)` with
        `anchor = x_m + (factor - x_m%factor)%factor` enumerates exactly the
        points of `[x_m, x_M]` that satisfy `x%factor == 0`.
        """
        for x_m in range(0, 14):
            for x_M in range(x_m, 20):
                guarded = [x for x in range(x_m, x_M + 1) if x % factor == 0]
                anchor = x_m + (factor - x_m % factor) % factor
                strided = list(range(anchor, x_M + 1, factor))

                assert guarded == strided, (factor, x_m, x_M)


class TestLegality:

    """Cases that must keep the pointwise guard."""

    def test_mixed_guarded_and_unguarded(self):
        """
        An unguarded expression sharing the loop nest must keep running at
        every point, so the guard cannot become a stride.
        """
        grid = Grid(shape=(12,))
        x, = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=4)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(3,), dimensions=(cd,), grid=grid,
                     space_order=0)

        op = Operator([Eq(f, 1), Eq(g, f)])

        assert len(FindNodes(Conditional).visit(op)) == 1
        assert iteration(op, 'x').limits[2] == 1

    def test_mixed_guards_reach_one_callback(self):
        """
        The guarded and unguarded Clusters must be handed to the pass
        *together*, as one group: that is what lets it see the nest is shared
        and refuse the rewrite. Grouping by guard instead would hide the
        unguarded Cluster and silently drop the guard.
        """
        import importlib

        module = importlib.import_module('devito.passes.clusters.subsampling')
        subsampling_queue = vars(module)['Subsampling']

        seen = []
        original = subsampling_queue.callback

        def callback(self, clusters, prefix):
            if prefix:
                d = prefix[-1].dim
                seen.append({type(c.guards.get(d)).__name__ for c in clusters})
            return original(self, clusters, prefix)

        grid = Grid(shape=(12,))
        x, = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=4)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(3,), dimensions=(cd,), grid=grid,
                     space_order=0)

        subsampling_queue.callback = callback
        try:
            Operator([Eq(f, 1), Eq(g, f)])
        finally:
            subsampling_queue.callback = original

        assert {'BooleanTrue', 'GuardFactorEq'} in seen

    def test_mixed_numerics(self):
        grid = Grid(shape=(12,))
        x, = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=4)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(3,), dimensions=(cd,), grid=grid,
                     space_order=0)
        g.data[:] = -1

        Operator([Eq(f, x), Eq(g, f)]).apply()

        assert np.array_equal(f.data, np.arange(12))
        assert np.array_equal(g.data, np.arange(12)[::4])

    def test_non_affine_access(self):
        """
        An indirectly indexed read is not affine in the subsampled Dimension,
        so the points the stride would skip cannot be shown to be unneeded.
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=3)
        idx = Function(name='idx', grid=grid, dtype=np.int32)
        u = Function(name='u', grid=grid)
        g = Function(name='g', shape=(4, 12), dimensions=(cd, y), grid=grid,
                     space_order=0)

        op = Operator(Eq(g, u[idx[x, y], y]))

        assert iteration(op, 'x').limits[2] == 1
        assert FindNodes(Conditional).visit(op)

    def test_user_condition(self):
        """A `condition=` must still be evaluated pointwise."""
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        f = Function(name='f', grid=grid)
        cd = ConditionalDimension(name='xs', parent=x, factor=3,
                                  condition=CondEq(x % 2, 0))
        g = Function(name='g', shape=(4, 12), dimensions=(cd, y), grid=grid,
                     space_order=0)

        f.data[:] = 1.
        op = Operator(Eq(g, f))
        op.apply()

        assert FindNodes(Conditional).visit(op)
        assert iteration(op, 'x').limits[2] == 1

    def test_or_relation(self):
        """
        A factor guard `Or`-ed with another condition fires on more points
        than the subsampling grid, so it cannot become a stride.
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        c_even = ConditionalDimension(name='ceven', parent=x, factor=2,
                                      relation=Or)
        c_edge = ConditionalDimension(name='cedge', parent=x,
                                      condition=CondEq(x, x.symbolic_max),
                                      relation=Or)
        f = Function(name='f', grid=grid, dimensions=(c_even, y),
                     shape=(5, 10), space_order=0)

        op = Operator(Eq(f, 1, implicit_dims=c_edge))

        assert FindNodes(Conditional).visit(op)
        assert iteration(op, 'x').limits[2] == 1

    def test_uniformly_guarded_sequential(self):
        """
        The case that isolates the parallelism gate: the subsampled Dimension
        `x` is *uniformly* guarded (so the shared-nest check does not apply)
        and is *not* innermost (so the SIMD check does not apply), but it
        carries a flow dependence and is therefore SEQUENTIAL. Only the
        parallelism gate can reject it; striding here would skip the
        iterations that feed the retained ones.
        """
        grid = Grid(shape=(12, 12))
        x, y = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=3)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(4, 12), dimensions=(cd, y), grid=grid,
                     space_order=0)

        op = Operator(Eq(g, g.subs(cd, cd-1) + f))

        assert iteration(op, 'x').limits[2] == 1
        assert FindNodes(Conditional).visit(op)

    def test_sequential_dimension(self):
        """
        A dependence carried along the subsampled Dimension forbids skipping
        points: the discarded iterations feed the retained ones.
        """
        grid = Grid(shape=(12,))
        x, = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=3)
        f = Function(name='f', grid=grid)
        g = Function(name='g', shape=(4,), dimensions=(cd,), grid=grid,
                     space_order=0)

        # `f[x] = f[x-1] + 1` carries a flow dependence along `x`
        op = Operator([Eq(f, f.subs(x, x-1) + 1), Eq(g, f)])
        op.apply()

        assert iteration(op, 'x').limits[2] == 1
        assert np.array_equal(g.data, f.data[::3])


class TestInteractions:

    def test_with_blocking(self):
        """
        Blocking rewrites `x` into BlockDimensions, so the subsampled
        Dimension's parent leaves the iteration space and the guard is kept.
        Correct, if not strided.
        """
        grid = Grid(shape=(32, 32))
        x, y = grid.dimensions
        cd = ConditionalDimension(name='xs', parent=x, factor=4)
        u = Function(name='u', grid=grid, space_order=4)
        g = Function(name='g', shape=(8, 32), dimensions=(cd, y), grid=grid,
                     space_order=0)
        u.data[:] = np.arange(32*32).reshape(32, 32).astype(u.dtype)

        op = Operator(Eq(g, u.dx2 + u.dy2), opt=('advanced',
                                                 {'blockrelax': True}))
        op.apply()

        assert len(FindNodes(Conditional).visit(op)) == 1
        assert np.all(np.isfinite(g.data))

    @switchconfig(language='openmp')
    def test_with_openmp(self):
        _, f, g, _ = subsampled((16, 16), 2)
        data = np.arange(256, dtype=f.dtype).reshape(16, 16)
        f.data[:] = data
        Operator(Eq(g, f)).apply()

        assert np.array_equal(g.data, data[::2])
