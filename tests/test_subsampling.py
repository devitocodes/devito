import re

import numpy as np
import pytest
from sympy import Or

from devito import (
    CondEq, ConditionalDimension, Constant, Eq, Function, Grid, Operator, SubGrid,
    TimeFunction, switchconfig
)
from devito.ir.iet import Conditional, FindNodes, Iteration
from devito.tools import as_tuple


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

    """
    Subsampling iterates the subsampled Dimension directly, with unit stride,
    rescaling parent accesses -- rather than striding the parent's loop. A
    non-unitary increment would escape the Lamport-style dependence analysis
    the compiler relies on.
    """

    def test_codegen(self):
        _, f, g, cd = subsampled((11, 11), 2)
        op = Operator(Eq(g, f))

        # No guard...
        assert not FindNodes(Conditional).visit(op)
        # ...instead the loop is over `cd` itself, with unit stride
        assert iteration(op, cd.name).limits[2] == 1
        with pytest.raises(KeyError):
            iteration(op, 'x')

    @pytest.mark.parametrize('factor', [2, 3, 4, 5])
    def test_parent_access_rescaled(self, factor):
        """Parent accesses become `cd*factor`; subgrid accesses become `cd`."""
        _, f, g, cd = subsampled((16, 8), factor)
        op = Operator(Eq(g, f))
        ccode = str(op.ccode)

        fname = cd.symbolic_factor.name
        assert f'{cd.name}*{fname}' in ccode          # parent, rescaled
        assert f'g[{cd.name}]' in ccode               # subgrid, direct
        assert f'+= {fname}' not in ccode             # no strided loop
        assert not FindNodes(Conditional).visit(op)

    def test_unit_stride_everywhere(self):
        """Every Iteration keeps a unit increment, so DDA stays applicable."""
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=(3, 2))
        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)

        op = Operator(Eq(f, u))

        for i in FindNodes(Iteration).visit(op):
            assert i.limits[2] == 1

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

        assert iteration(op, ys.name).limits[2] == 1
        assert f'{ys.name}*{ys.symbolic_factor.name}' in str(op.ccode)
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
        for d in (xs, ys):
            assert iteration(op, d.name).limits[2] == 1
            assert f'{d.name}*{d.symbolic_factor.name}' in str(op.ccode)

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

    def test_properties_inherited(self):
        """
        The subsampled Dimension must inherit the parent's computational
        properties: it covers the same points, just in subsampled coordinates.
        Losing them costs the loop its PARALLEL/VECTORIZED status, and with it
        all of its OpenMP.
        """
        grid = Grid(shape=(32, 32))
        x, y = grid.dimensions
        sub = SubGrid(grid, factor=2)
        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)

        # What the same copy gets on the parent Grid...
        v = Function(name='v', grid=grid)
        ref = Operator(Eq(v, u))
        expected = {i.dim.name: i.properties
                    for i in FindNodes(Iteration).visit(ref)}

        # ...the subsampled version must get too
        op = Operator(Eq(f, u))
        for i in FindNodes(Iteration).visit(op):
            parent = i.dim.parent if i.dim.is_Conditional else i.dim
            assert i.properties == expected[parent.name], i.dim.name

    @switchconfig(language='openmp')
    def test_openmp_emitted(self):
        """The subsampled loop nest is still parallelized and vectorized."""
        grid = Grid(shape=(32, 32))
        sub = SubGrid(grid, factor=2)
        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)

        ccode = str(Operator(Eq(f, u)).ccode)

        assert '#pragma omp for' in ccode
        assert '#pragma omp simd' in ccode

    def test_bounds_are_subsampled(self):
        """
        The loop runs over the subsampled Dimension's own runtime bounds,
        which `ConditionalDimension._arg_values` already derives as
        `ceil(x_m/factor) .. ceil(x_M/factor)`. No anchoring is needed.
        """
        _, f, g, cd = subsampled((12, 12), 3)
        op = Operator(Eq(g, f))

        assert iteration(op, cd.name).limits[0] == cd.symbolic_min
        args = op.arguments()
        assert args[cd.min_name] == 0
        assert args[cd.max_name] == 3


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

        # The loop runs over the subsampled Dimension's own bounds,
        # `ceil(x_m/factor) .. floor(x_M/factor)`
        written = set(range(-(-x_m//factor), x_M//factor + 1))
        for i in range(g.data.shape[0]):
            expected = 1. if i in written else -1.
            assert np.all(g.data[i] == expected), (i, x_m, x_M)

    @pytest.mark.parametrize('x_M,expected', [
        (8, 2), (9, 3), (10, 3), (11, 3),
    ])
    def test_max_bound_floors(self, x_M, expected):
        """
        The subsampled max is `floor(x_M/factor)`, the last subsampled point at
        or before `x_M` -- not `ceil`, which would run one point past the end
        of the Function and corrupt the heap.
        """
        _, f, g, cd = subsampled((12, 12), 3)
        op = Operator(Eq(g, f))

        args = op.arguments(x_M=x_M)

        assert args[cd.max_name] == expected
        assert args[cd.max_name] < g.shape[0]

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


class TestSubGrid:

    """The `SubGrid` API."""

    def test_derived_properties(self):
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=2)

        assert sub.parent is grid
        assert sub.factor == (2, 2)
        assert sub.shape == (5, 5)
        assert all(d.is_Conditional for d in sub.dimensions)
        assert all(d.parent is p
                   for d, p in zip(sub.dimensions, grid.dimensions, strict=True))

    def test_spacing_scaled(self):
        """Physical spacing is `factor` times the parent's; extent is kept."""
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=2)

        assert np.allclose(sub.spacing, [2*s for s in grid.spacing])
        assert np.allclose(sub.extent, grid.extent)
        assert np.allclose(sub.origin, grid.origin)

    def test_function_needs_no_shape(self):
        """`Function(grid=sub)` derives both dimensions and shape."""
        grid = Grid(shape=(9, 9))
        sub = SubGrid(grid, factor=2)

        f = Function(name='f', grid=sub, space_order=0)

        assert f.dimensions == sub.dimensions
        assert f.shape == (5, 5)

    @pytest.mark.parametrize('shape,factor,expected', [
        ((9, 9), 2, (5, 5)),
        ((9, 9), (2, 1), (5, 9)),
        ((10,), 3, (4,)),
        ((10,), 5, (2,)),
        ((8, 8, 8), (2, 4, 1), (4, 2, 8)),
    ])
    def test_shape_derivation(self, shape, factor, expected):
        sub = SubGrid(Grid(shape=shape), factor=factor)

        assert sub.shape == expected

    @pytest.mark.parametrize('factor', [2, 3, (2, 1), (1, 3), (3, 2)])
    def test_copy_from_parent(self, factor):
        """Writing a SubGrid Function from a parent Function is a strided copy."""
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)
        data = np.arange(144, dtype=u.dtype).reshape(12, 12)
        u.data[:] = data

        Operator(Eq(f, u)).apply()

        fx, fy = sub.factor
        assert np.array_equal(f.data, data[::fx, ::fy])

    def test_copy_is_strided(self):
        """
        No guard is emitted, every loop keeps a unit stride, and the Operator
        runs to the right answer.
        """
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=3)

        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)
        data = np.arange(144, dtype=u.dtype).reshape(12, 12)
        u.data[:] = data

        op = Operator(Eq(f, u))
        op.apply()

        assert not FindNodes(Conditional).visit(op)
        for d in sub.dimensions:
            assert iteration(op, d.name).limits[2] == 1
        assert np.array_equal(f.data, data[::3, ::3])

    @switchconfig(language='openmp')
    def test_openmp_and_numerics(self):
        """A SubGrid Operator is parallelized, vectorized, and correct."""
        grid = Grid(shape=(32, 32))
        sub = SubGrid(grid, factor=(4, 2))

        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=sub, space_order=0)
        data = np.arange(32*32, dtype=u.dtype).reshape(32, 32)
        u.data[:] = data

        op = Operator(Eq(f, u))
        op.apply()

        ccode = str(op.ccode)
        assert '#pragma omp for' in ccode
        assert '#pragma omp simd' in ccode
        assert np.array_equal(f.data, data[::4, ::2])

    def test_shared_nest_numerics(self):
        """
        A SubGrid Function sharing a nest with an unguarded Eq: the guard is
        kept and both results are right.
        """
        grid = Grid(shape=(12,))
        sub = SubGrid(grid, factor=4)

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        g.data[:] = -1.

        op = Operator([Eq(f, 1), Eq(g, f)])
        op.apply()

        assert FindNodes(Conditional).visit(op)
        assert np.all(f.data == 1.)
        assert np.all(g.data == 1.)

    def test_sequential_numerics(self):
        """A SubGrid Function accumulating along the subsampled Dimension."""
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=(3, 1))
        xs = sub.dimensions[0]

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        f.data[:] = 1.
        g.data[:] = 0.

        Operator(Eq(g, g.subs(xs, xs-1) + f)).apply()

        # Row 0 reads the halo (zero), then each row adds one
        assert np.allclose(g.data[:, 0], [0., 1., 2., 3.])

    def test_factor_one_is_passthrough(self):
        """A factor of 1 leaves the Dimension untouched."""
        grid = Grid(shape=(9, 9))
        sub = SubGrid(grid, factor=(2, 1))

        assert sub.dimensions[0].is_Conditional
        assert sub.dimensions[1] is grid.dimensions[1]

    def test_shares_parent_time_dimensions(self):
        """
        A SubGrid coarsens space only, so a TimeFunction on it must step over
        the same buffer indices as one on the parent.
        """
        grid = Grid(shape=(9, 9))
        sub = SubGrid(grid, factor=2)

        assert sub.time_dim is grid.time_dim
        assert sub.stepping_dim is grid.stepping_dim

        u = TimeFunction(name='u', grid=grid)
        us = TimeFunction(name='us', grid=sub, space_order=0)

        assert u.dimensions[0] is us.dimensions[0]

    def test_time_stepping_operator(self):
        grid = Grid(shape=(9, 9))
        sub = SubGrid(grid, factor=2)

        u = TimeFunction(name='u', grid=grid)
        us = TimeFunction(name='us', grid=sub, space_order=0)

        Operator([Eq(u.forward, u + 1.), Eq(us.forward, u.forward)]).apply(time_M=4)

        assert np.allclose(us.data[-1], u.data[-1][::2, ::2])

    def test_derivative_uses_coarse_spacing(self):
        """
        `f.dx` on a SubGrid Function differentiates over the coarse spacing:
        `ConditionalDimension.spacing` is `factor*h_x`, so the stencil steps
        `factor*h_x` and the weights scale by `1/(factor*h_x)`.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        sub = SubGrid(grid, factor=2)
        xs = sub.dimensions[0]

        assert xs.spacing == 2*grid.dimensions[0].spacing

        f = Function(name='f', grid=sub, space_order=2)
        expr = f.dx.evaluate

        # Steps by the coarse spacing, with correspondingly halved weights
        assert str(expr) == '-0.5*f(xs, ys)/h_x + 0.5*f(xs + 2*h_x, ys)/h_x'

    def test_derivative_numerics(self):
        """A linear field differentiates exactly on the SubGrid."""
        grid = Grid(shape=(21, 21), extent=(20., 20.))
        sub = SubGrid(grid, factor=2)
        xs, ys = sub.dimensions

        f = Function(name='f', grid=sub, space_order=2)
        d = Function(name='d', grid=sub, space_order=2)

        # f = 3*x on the coarse grid, so df/dx == 3 everywhere inside
        for i in range(f.shape[0]):
            f.data[i, :] = 3.*i*float(sub.spacing[0])

        Operator(Eq(d, f.dx)).apply()

        assert np.allclose(d.data[1:-1, 1:-1], 3., atol=1e-4)

    @pytest.mark.parametrize('factor,err', [
        (0, 'must be >= 1'),
        (-1, 'must be >= 1'),
        (2.5, 'Non-integer'),
        ((2, 2, 2), 'Expected 1 or 2 factors'),
    ])
    def test_invalid_factor(self, factor, err):
        grid = Grid(shape=(9, 9))

        with pytest.raises(ValueError, match=err):
            SubGrid(grid, factor=factor)

    @pytest.mark.parametrize('f0,f1,combined', [
        (2, 3, (6, 6)),
        (2, (3, 1), (6, 2)),
        ((3, 2), (2, 4), (6, 8)),
    ])
    def test_cannot_nest(self, f0, f1, combined):
        """
        Nesting is rejected, and the message names the equivalent single-level
        call rather than leaving the user to work out the arithmetic.
        """
        grid = Grid(shape=(24, 24))
        sub = SubGrid(grid, factor=f0)

        with pytest.raises(ValueError) as e:
            SubGrid(sub, factor=f1)

        assert f'factor={combined}' in str(e.value)

    @pytest.mark.parametrize('f0,f1,combined', [
        (2, 3, (6, 6)),
        (2, (3, 1), (6, 2)),
        ((3, 2), (2, 4), (6, 8)),
    ])
    def test_nesting_suggestion_is_equivalent(self, f0, f1, combined):
        """
        The combined factor the error suggests must really select the points a
        two-step subsampling would have.
        """
        grid = Grid(shape=(24, 24))
        sub = SubGrid(grid, factor=f0)
        suggested = SubGrid(grid, factor=combined, name='c')

        u = Function(name='u', grid=grid)
        f = Function(name='f', grid=suggested, space_order=0)
        data = np.arange(24*24, dtype=u.dtype).reshape(24, 24)
        u.data[:] = data

        Operator(Eq(f, u)).apply()

        # Two-step reference: coarsen by `f0`, then by `f1`
        g0, g1 = sub.factor
        h0, h1 = as_tuple(f1)*2 if len(as_tuple(f1)) == 1 else as_tuple(f1)
        two_step = data[::g0, ::g1][::h0, ::h1]

        assert np.array_equal(f.data, two_step)

    def test_non_integer_factor_suggests_valid_factors(self):
        """The suggested fallbacks must themselves be legal factors."""
        grid = Grid(shape=(9, 9))

        for bad in (0.5, 1.2, 2.5):
            with pytest.raises(ValueError) as e:
                SubGrid(grid, factor=bad)
            suggested = re.findall(r'use `(\d+)` or `(\d+)`', str(e.value))
            assert suggested, str(e.value)
            for v in suggested[0]:
                assert int(v) >= 1
                SubGrid(grid, factor=int(v))        # must not raise

    def test_parent_must_be_grid(self):
        with pytest.raises(ValueError, match='must be a Grid'):
            SubGrid(Grid(shape=(9, 9)).interior, factor=2)

    def test_custom_dimension_names(self):
        grid = Grid(shape=(9, 9))
        sub = SubGrid(grid, factor=2, name='c')

        assert [d.name for d in sub.dimensions] == ['xc', 'yc']
