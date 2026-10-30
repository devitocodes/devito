"""
CurvilinearGrid: its maps, its physical derivatives and their transposes, its
SparseFunctions, and the resampling of Functions from a Cartesian Grid.
"""

import pickle

import numpy as np
import pytest

from devito import (
    CurvilinearGrid, CurvilinearMap, Eq, Function, Grid, Inc, MeshMap, Operator,
    SparseFunction, SubDomain, TerrainFollowing, resample
)
from devito.mpi import MPI

EXTENT = (1000., 800.)

# The wavenumbers of the analytic field `wave`
KX, KZ = 2*np.pi/400., 2*np.pi/500.


def sine_surface(x):
    """A surface 60 m high, in m, positive down."""
    return 20. + 60.*np.sin(2*np.pi*x/1000.)


def sine_grid(n, order=8, dtype=np.float64, **kwargs):
    """A 2D CurvilinearGrid of `n` points a side under `sine_surface`."""
    return CurvilinearGrid(shape=(n, n), extent=EXTENT, dtype=dtype,
                           mapping=TerrainFollowing(sine_surface, order=order), **kwargs)


def hill_surface(x, y):
    """A smooth 3D surface 40 m high, in m, positive down."""
    return 40.*np.sin(2*np.pi*x/1000.)*np.cos(2*np.pi*y/1000.)


def hill_grid(n, order=4):
    """A 3D CurvilinearGrid of `n` points a side under `hill_surface`."""
    return CurvilinearGrid(shape=(n, n, n), extent=(1000., 1000., 800.),
                           dtype=np.float64,
                           mapping=TerrainFollowing(hill_surface, order=order))


def wave(x, z):
    """An analytic field of the physical coordinates."""
    return np.sin(KX*x)*np.cos(KZ*z)


def wave_dx(x, z):
    return KX*np.cos(KX*x)*np.cos(KZ*z)


def wave_dy(x, z):
    return -KZ*np.sin(KX*x)*np.sin(KZ*z)


def wave_dx2(x, z):
    return -KX**2*wave(x, z)


def wave_dy2(x, z):
    return -KZ**2*wave(x, z)


def wave_dxdy(x, z):
    return -KX*KZ*np.cos(KX*x)*np.sin(KZ*z)


def wave_laplace(x, z):
    return -(KX**2 + KZ**2)*wave(x, z)


# The derivatives of `wave`, by name of the Derivative
DERIVATIVES = {'dx': wave_dx, 'dy': wave_dy, 'dx2': wave_dx2, 'dy2': wave_dy2,
               'dxdy': wave_dxdy, 'laplace': wave_laplace}


def physical(grid, staggered=()):
    """
    The physical coordinates of the domain points of `grid`, at the nodes or
    staggered by half a cell along `staggered`, one array per Dimension.
    """
    vectors = [float(o) + float(h)*(np.arange(n) + (.5 if d in staggered else 0.))
               for d, o, h, n in zip(grid.dimensions, grid.origin, grid.spacing,
                                     grid.shape, strict=True)]
    computational = np.meshgrid(*vectors, indexing='ij')
    points = np.stack([c.ravel() for c in computational], axis=1)
    return [p.reshape(grid.shape) for p in grid.to_physical(points).T]


def field(name, grid, staggered=(), values=wave):
    """A Function holding `values` of the physical coordinates at its points."""
    f = Function(name=name, grid=grid, space_order=8, staggered=staggered or None)
    f.data[:] = values(*physical(grid, staggered))
    return f


def random_field(name, grid, seed, staggered=None, space_order=8):
    """A Function of random values on the whole domain."""
    f = Function(name=name, grid=grid, space_order=space_order, staggered=staggered)
    f.data[:] = np.random.default_rng(seed).uniform(-1, 1, f.shape)
    return f


def weighted(f, g):
    """`sum(J f g)`, the inner product weighted by the Jacobian where `f` is."""
    grid = f.grid
    location = {d: f.indices_ref[d] for d in grid.dimensions}
    j = Function(name=f'j_{f.name}', grid=grid, space_order=8, staggered=f.staggered)
    Operator(Eq(j, grid.jacobian_at(location))).apply()
    return np.sum(np.asarray(j.data)*np.asarray(f.data)*np.asarray(g.data))


def dot_test(forward, back, w, u):
    """
    The relative difference of `<forward, w>` and `<u, back>` in the inner
    product weighted by the Jacobian, `forward` reading `u` and `back` `w`.
    """
    fu = Function(name='fu', grid=w.grid, space_order=8, staggered=w.staggered)
    tw = Function(name='tw', grid=w.grid, space_order=8, staggered=u.staggered)
    Operator([Eq(fu, forward), Eq(tw, back)]).apply()
    lhs, rhs = weighted(fu, w), weighted(tw, u)
    return abs(lhs - rhs)/abs(lhs)


def term_values(term):
    """The values of a metric term, a Function or a number."""
    return np.asarray(term.data) if term.is_Function else float(term)


def order_of(errors):
    """The order of convergence of `errors` at n = 41 and 81."""
    return np.log2(errors[0]/errors[1])


class Top(SubDomain):
    """The first row of the grid."""

    name = 'top'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('left', 1)}


class Rows(SubDomain):
    """The rows of the grid but 15 at each end of the vertical Dimension."""

    name = 'rows'

    def define(self, dimensions):
        x, y = dimensions
        return {x: x, y: ('middle', 15, 15)}


class Middle(SubDomain):
    """The grid without its 10 outer nodes along each Dimension."""

    name = 'middle'

    def define(self, dimensions):
        return {d: ('middle', 10, 10) for d in dimensions}


class Folded(CurvilinearMap):
    """A map turning cells inside out, `z = -eta`."""

    def forward(self, grid, computational):
        return computational[0], -computational[1]


class Bounded(CurvilinearMap):
    """A map whose depths stay within 100 m, `z = 100 tanh(eta/100)`."""

    def forward(self, grid, computational):
        return computational[0], 100.*np.tanh(computational[1]/100.)


class TestMaps:

    def test_terrain_following(self):
        """The surface is the row at depth 0, the bottom row stays flat."""
        x = np.linspace(0., 1000., 11)
        elevation = 50.*np.exp(-((x - 500.)/200.)**2)
        grid = CurvilinearGrid(shape=(11, 21), extent=(1000., 1000.),
                               mapping=TerrainFollowing(-elevation))
        assert np.allclose(grid.coordinates[-1].data[:, 0], -elevation)
        assert np.allclose(grid.coordinates[-1].data[:, -1], 1000.)
        assert np.allclose(grid.coordinates[0].data[:, 0], x)

    def test_constant_metrics(self):
        """
        The metric terms constant by construction of a terrain-following map are
        exact numbers, all of them on a flat surface, the identity of a Grid,
        and so are the second derivatives of the map there.
        """
        grid = sine_grid(21)
        x, y = grid.dimensions
        assert grid.metrics[(x, x)] == 1 and grid.metrics[(y, x)] == 0
        assert grid.metrics[(x, y)].is_Function and grid.metrics[(y, y)].is_Function

        flat = CurvilinearGrid(shape=(21, 21), extent=EXTENT,
                               mapping=TerrainFollowing(np.zeros(21)))
        assert flat.metrics == {(a, b): int(a is b)
                                for a in flat.dimensions for b in flat.dimensions}
        assert flat.jacobian == 1
        assert set(flat.curvatures_at({}).values()) == {0}
        assert set(flat.normal_ratios_at({}).values()) == {0}

    def test_metrics(self):
        """The metric terms converge to the closed form of the map."""
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            x, y = grid.dimensions
            xs, _ = physical(grid)
            eta = np.meshgrid(*[float(h)*np.arange(s)
                                for h, s in zip(grid.spacing, grid.shape, strict=True)],
                              indexing='ij')[1]
            bottom = EXTENT[1]
            slope = 2*np.pi/1000.*60.*np.cos(2*np.pi*xs/1000.)
            height = bottom - sine_surface(xs)
            exact = {(y, y): bottom/height, (x, y): slope*(eta - bottom)/height}
            errors.append(max(np.abs(grid.metrics[k].data - v).max()
                              for k, v in exact.items()))
        assert order_of(errors) > 6

    def test_names(self):
        """
        The names of the geometric Functions of a grid are its own, so that a
        Function of the same name, e.g. `jacobian`, lives next to them.
        """
        grid, other = sine_grid(21), sine_grid(21)
        jacobian = Function(name='jacobian', grid=grid, space_order=8)
        jacobian.data[:] = 1
        u = random_field('u', grid, 1)
        out = Function(name='out', grid=grid, space_order=8)
        ref = Function(name='ref', grid=grid, space_order=8)
        Operator([Eq(out, jacobian*u.dx), Eq(ref, u.dx)]).apply()
        assert np.allclose(out.data, ref.data, rtol=0, atol=0)
        assert grid.jacobian.name != other.jacobian.name != 'jacobian'

    @pytest.mark.parametrize('staggered', [False, True])
    def test_normal_ratios(self, staggered):
        """
        The ratios of the normal of the rows are those of the metric terms
        `d xi_z/d x_i` on the domain, at the nodes and the half points, and 0
        in the halo, where the metric terms are 0.
        """
        grid = sine_grid(21)
        x, y = grid.dimensions
        location = {x: x + x.spacing/2, y: y + y.spacing/2} if staggered else {}
        stagger = (x, y) if staggered else None
        ratio = Function(name='ratio', grid=grid, space_order=4, staggered=stagger)
        n_y, n_x = grid.metrics_at(location)[(y, y)], grid.metrics_at(location)[(x, y)]
        Operator(Eq(ratio, grid.normal_ratios_at(location)[x])).apply()
        stored = grid.normal_ratios_at(location)[x].function
        assert np.allclose(ratio.data, np.asarray(n_x.function.data) /
                           np.asarray(n_y.function.data), rtol=1e-12, atol=1e-14)
        assert np.abs(stored.data).max() > 0
        assert not np.any(stored.data_with_halo[:stored.halo[0][0]])

    def test_metrics_3d(self):
        """In 3D, the lateral metric term converges to the closed form."""
        errors = []
        for n in (11, 21):
            grid = hill_grid(n, order=4)
            x, y, z = grid.dimensions
            xs, ys, _ = physical(grid)
            zeta = np.meshgrid(*[float(h)*np.arange(s)
                                 for h, s in zip(grid.spacing, grid.shape, strict=True)],
                               indexing='ij')[2]
            k = 2*np.pi/1000.
            slope = 40.*k*np.cos(k*xs)*np.cos(k*ys)
            height = 800. - hill_surface(xs, ys)
            exact = slope*(zeta - 800.)/height
            errors.append(np.abs(grid.metrics[(x, z)].data - exact).max())
        assert np.log2(errors[0]/errors[1]) > 3.5

    def test_mesh_map(self):
        """A MeshMap of the nodes of a map is that map."""
        grid = sine_grid(41)
        coordinates = tuple(np.asarray(c.data) for c in grid.coordinates)
        mesh = CurvilinearGrid(shape=(41, 41), extent=EXTENT, dtype=np.float64,
                               mapping=MeshMap(coordinates))
        for k, v in grid.metrics.items():
            assert np.allclose(term_values(mesh.metrics[k]), term_values(v), atol=1e-9)

    def test_mesh_map_padding(self):
        """A MeshMap with padding places the mesh past the padding nodes."""
        grid = sine_grid(41)
        coordinates = tuple(np.asarray(c.data)[5:-5, 3:] for c in grid.coordinates)
        mesh = CurvilinearGrid(shape=(41, 41), extent=EXTENT, dtype=np.float64,
                               mapping=MeshMap(coordinates, padding=(5, 3)))
        for a, b in zip(mesh.coordinates, grid.coordinates, strict=True):
            assert np.allclose(a.data[5:-5, 3:], b.data[5:-5, 3:])

    @pytest.mark.parametrize('kind', ['map', 'mesh'])
    def test_inverse(self, kind):
        """`to_computational` inverts `to_physical`."""
        grid = sine_grid(41)
        if kind == 'mesh':
            coordinates = tuple(np.asarray(c.data) for c in grid.coordinates)
            grid = CurvilinearGrid(shape=(41, 41), extent=EXTENT, dtype=np.float64,
                                   mapping=MeshMap(coordinates))
        points = np.random.default_rng(1).uniform(50., 750., (20, 2))
        assert np.allclose(grid.to_computational(grid.to_physical(points)), points,
                           rtol=0, atol=1e-8)

    def test_inverse_unreachable(self):
        """A point the map cannot reach is refused, not placed anywhere."""
        grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.), mapping=Bounded())
        with pytest.raises(ValueError, match='cannot place'):
            grid.to_computational(np.array([[50., 500.]]))

    def test_singular(self):
        """A map that turns cells inside out has no metric terms."""
        grid = CurvilinearGrid(shape=(11, 11), extent=(100., 100.), mapping=Folded())
        with pytest.raises(ValueError, match='inside out'):
            grid.metrics  # noqa: B018

    def test_invalid(self):
        """A map that cannot place the nodes of a grid is refused."""
        with pytest.raises(TypeError):
            CurvilinearGrid(shape=(11, 11), mapping=None)
        with pytest.raises(ValueError, match='lateral'):
            CurvilinearGrid(shape=(11,), mapping=TerrainFollowing(np.zeros(1)))
        with pytest.raises(ValueError, match='shape'):
            CurvilinearGrid(shape=(11, 11), mapping=TerrainFollowing(np.zeros(5)))
        with pytest.raises(ValueError, match='even'):
            TerrainFollowing(np.zeros(11), order=3)

    def test_min_spacing(self):
        """The smallest cell of a flat map is that of a Grid, a hill squeezes it."""
        flat = CurvilinearGrid(shape=(21, 21), extent=EXTENT,
                               mapping=TerrainFollowing(np.zeros(21)))
        cartesian = Grid(shape=(21, 21), extent=EXTENT)
        assert np.isclose(flat.min_spacing, cartesian.min_spacing)
        assert sine_grid(21).min_spacing < flat.min_spacing

    def test_pickle(self):
        """A pickled grid rebuilds its geometry from its map."""
        grid = sine_grid(21)
        copy = pickle.loads(pickle.dumps(grid))
        x, y = grid.dimensions
        assert np.allclose(copy.metrics[(x, y)].data, grid.metrics[(x, y)].data)


class TestDerivatives:

    @pytest.mark.parametrize('name', sorted(DERIVATIVES))
    def test_convergence(self, name):
        """Physical derivatives at the nodes converge at the order of the stencils."""
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            u = field('u', grid)
            out = Function(name='out', grid=grid, space_order=8)
            Operator(Eq(out, getattr(u, name))).apply()
            exact = DERIVATIVES[name](*physical(grid))
            errors.append(np.abs(out.data - exact)[10:-10, 10:-10].max())
        assert order_of(errors) > 5.5

    @pytest.mark.parametrize('name', ['dx', 'dy'])
    def test_staggered_convergence(self, name):
        """A derivative from the nodes to the half points converges at its order."""
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            dim = grid.dimensions[name == 'dy']
            u = field('u', grid)
            out = Function(name='out', grid=grid, space_order=8, staggered=dim)
            Operator(Eq(out, getattr(u, name)(x0=dim + dim.spacing/2))).apply()
            exact = DERIVATIVES[name](*physical(grid, (dim,)))
            errors.append(np.abs(out.data - exact)[10:-10, 10:-10].max())
        assert order_of(errors) > 5.5

    def test_compact_second_derivative(self):
        """
        A mixed second derivative is a second derivative of the chain rule, not
        nested first derivatives: its stencil reaches half its order along
        each Dimension, where nested ones would reach its order.
        """
        grid = sine_grid(21)
        x, y = grid.dimensions
        u = Function(name='u', grid=grid, space_order=8)
        reads = [a for a in u.dxdy.evaluate.atoms(Function) if a.function is u]
        for d in (x, y):
            assert max(abs((a.indices[d] - d)/d.spacing) for a in reads) == 4

    def test_staggered_pair(self):
        """The pair `u.dx(x0=x + h_x/2).dx(x0=x - h_x/2)` is a second derivative."""
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            x = grid.dimensions[0]
            u = field('u', grid)
            out = Function(name='out', grid=grid, space_order=8)
            pair = u.dx(x0=x + x.spacing/2).dx(x0=x - x.spacing/2)
            Operator(Eq(out, pair)).apply()
            exact = DERIVATIVES['dx2'](*physical(grid))
            errors.append(np.abs(out.data - exact)[10:-10, 10:-10].max())
        assert order_of(errors) > 5.5

    def test_computational(self):
        """`computational=True` is the derivative w.r.t. the computational coordinate."""
        grid = sine_grid(21)
        cartesian = Grid(shape=(21, 21), extent=EXTENT, dtype=np.float64)
        u, v = (Function(name=n, grid=g, space_order=8)
                for n, g in (('u', grid), ('v', cartesian)))
        u.data[:] = v.data[:] = np.random.default_rng(1).uniform(size=(21, 21))
        out, ref = (Function(name=n, grid=g, space_order=8)
                    for n, g in (('out', grid), ('ref', cartesian)))
        Operator([Eq(out, u.dy(computational=True)), Eq(ref, v.dy)]).apply()
        assert np.allclose(out.data, ref.data, rtol=0, atol=1e-12)

    def test_flat_is_cartesian(self):
        """
        Under a flat surface, the derivatives are those of a Grid, staggered,
        shifted and transposed ones included.
        """
        flat = CurvilinearGrid(shape=(11, 11), extent=(100., 100.),
                               mapping=TerrainFollowing(np.zeros(11)))
        cartesian = Grid(shape=(11, 11), extent=(100., 100.), dimensions=flat.dimensions)
        x, y = flat.dimensions
        u, v = (Function(name='u', grid=g, space_order=4, staggered=x)
                for g in (flat, cartesian))
        half = {x: x + x.spacing/2, y: y + y.spacing/2}
        for a, b in [(u.dx, v.dx), (u.dx2, v.dx2), (u.dxdy, v.dxdy),
                     (u.laplace, v.laplace), (u.dx(x0={x: x}), v.dx(x0={x: x})),
                     (u.dy(x0={x: x, y: y + y.spacing/2}),
                      v.dy(x0={x: x, y: y + y.spacing/2})),
                     (u.dx(x0={x: x}).T, v.dx(x0={x: x}).T), (u.dy2.T, v.dy2.T),
                     ((u.dx(x0=half)*u.dy(x0=half)).dx,
                      (v.dx(x0=half)*v.dy(x0=half)).dx)]:
            assert str(a.evaluate) == str(b.evaluate)

    @pytest.mark.parametrize('case', ['dx', 'dy', 'dx2', 'dy2', 'dxdy', 'nested',
                                      'half', 'interpolation', 'across', 'across y'])
    def test_transpose(self, case):
        """
        `.T` is the transpose in the inner product weighted by the Jacobian:
        at the nodes, from the half points back to the nodes, and for
        Functions staggered along another Dimension, as the elastic stress
        and velocity.
        """
        grid = sine_grid(41)
        x, y = grid.dimensions
        hx, hy = x.spacing, y.spacing
        u, w = random_field('u', grid, 1), random_field('w', grid, 2)
        vx = random_field('vx', grid, 3, staggered=x)
        uy = random_field('uy', grid, 4, staggered=y)
        vxy = random_field('vxy', grid, 5, staggered=(x, y))
        forward, back, ww, uu = {
            'dx': (u.dx, w.dx.T, w, u),
            'dy': (u.dy, w.dy.T, w, u),
            'dx2': (u.dx2, w.dx2.T, w, u),
            'dy2': (u.dy2, w.dy2.T, w, u),
            'dxdy': (u.dxdy, w.dxdy.T, w, u),
            'nested': (u.dx.dy, w.dy.T.dx.T, w, u),
            'half': (u.dx(x0={x: x + hx/2}), vx.dx(x0={x: x}).T, vx, u),
            'interpolation': (u.diff((x, 0), x0={x: x + hx/2}),
                              vx.diff((x, 0), x0={x: x}).T, vx, u),
            'across': (uy.dx(x0={x: x + hx/2}), vxy.dx(x0={x: x}).T, vxy, uy),
            'across y': (vx.dy(x0={y: y + hy/2}), vxy.dy(x0={y: y}).T, vxy, vx),
        }[case]
        assert dot_test(forward, back, ww, uu) < 1e-12

    def test_transpose_held(self):
        """
        The transpose of a derivative of an expression holding a derivative at
        the half points, as in a flux, is that of the outer derivative.
        """
        grid = sine_grid(41)
        x = grid.dimensions[0]
        b, u = random_field('b', grid, 1), random_field('u', grid, 2)
        forward = (b*u.dx(x0={x: x + x.spacing/2})).dx(x0={x: x - x.spacing/2})
        q = random_field('q', grid, 4)
        back = (b*q.dx(x0={x: x - x.spacing/2}).T).dx(x0={x: x + x.spacing/2}).T
        assert dot_test(forward, back, q, u) < 1e-12

    def test_offsets(self):
        """
        On the first row, a horizontal physical derivative, its vertical
        chain-rule term one-sided on the points it is given, converges at the
        order of these points.
        """
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            x, y = grid.dimensions
            u = field('u', grid)
            out = Function(name='out', grid=grid, space_order=8)
            top = Top(grid=grid)
            Operator(Eq(out, u.dx(offsets={y: (0, 1, 2)}), subdomain=top)).apply()
            exact = DERIVATIVES['dx'](*physical(grid))[:, 0]
            errors.append(np.abs(out.data[10:-10, 0] - exact[10:-10]).max())
        assert order_of(errors) > 1.8

    def test_offsets_invalid(self):
        """Offsets along a Dimension a derivative does not take, or past the halo."""
        cartesian = Grid(shape=(11, 11))
        x, y = cartesian.dimensions
        u = Function(name='u', grid=cartesian, space_order=2)
        with pytest.raises(ValueError, match='only apply'):
            u.dx(offsets={y: (0, 1)})
        with pytest.raises(ValueError, match='halo'):
            u.dx(offsets=(0, 1, 2, 3))
        grid = sine_grid(11)
        v = Function(name='v', grid=grid, space_order=2)
        y = grid.dimensions[1]
        assert v.dx(offsets={y: (0, 1)}).offsets == {y: (0, 1)}

    def test_subdomain_function(self):
        """A Function on a SubDomain has the physical derivative of the grid."""
        grid = sine_grid(41)
        middle = Middle(grid=grid)
        u = field('u', grid)
        a = Function(name='a', grid=middle, space_order=8)
        a.data[:] = np.asarray(u.data)[10:-10, 10:-10]
        out = Function(name='out', grid=middle, space_order=8)
        ref = Function(name='ref', grid=grid, space_order=8)
        Operator([Eq(out, a.dx), Eq(ref, u.dx)]).apply()
        assert np.allclose(out.data[4:-4, 4:-4], ref.data[14:-14, 14:-14],
                           rtol=0, atol=1e-12)

    def test_halo_zero(self):
        """
        With `halo=0`, a physical derivative on rows of the grid is that of its
        argument extended by zero outside them, over the rows grown by the
        reach of its vertical chain-rule term, interpolated along `x`.
        """
        grid = sine_grid(41)
        x, y = grid.dimensions
        rows = Rows(grid=grid)
        g = random_field('g', grid, 1, staggered=x)
        g_zero = Function(name='g_zero', grid=grid, space_order=8, staggered=x)
        g_zero.data[:, 15:-15] = g.data[:, 15:-15]
        out, ref = (Function(name=n, grid=grid, space_order=8) for n in ('out', 'ref'))
        Operator([Eq(out, g.dx(x0={x: x}, halo=0), subdomain=rows),
                  Eq(ref, g_zero.dx(x0={x: x}))]).apply()
        assert np.linalg.norm(ref.data) > 0
        assert np.allclose(out.data, ref.data, rtol=0, atol=1e-12)

    @pytest.mark.parametrize('staggered', [(), ('x',), ('y',), ('x', 'y')])
    @pytest.mark.parametrize('along', ['x', 'y'])
    def test_halo_zero_transpose(self, staggered, along):
        """
        The transpose of a physical derivative on rows of the grid to the
        points half a cell away, with `halo=0`, reads its argument as zero
        outside the rows and writes past them: it is the transpose of the
        derivative on the rows, in the inner product weighted by the Jacobian,
        for the staggered pairs of the elastic scheme. Its interpolations
        along the rows and its derivatives across them read the argument as
        zero there first.
        """
        grid = sine_grid(41)
        dims = {d.name: d for d in grid.dimensions}
        dim = dims[along]
        source = tuple(dims[d] for d in staggered)
        target = tuple(sorted(set(source) ^ {dim}, key=grid.dimensions.index))
        rows = Rows(grid=grid)
        u = random_field('u', grid, 1, staggered=source or None)
        a = random_field('a', grid, 2, staggered=target or None)
        fu = Function(name='fu', grid=grid, space_order=8, staggered=target or None)
        ta = Function(name='ta', grid=grid, space_order=8, staggered=source or None)
        to = dim + dim.spacing/2 if dim in target else dim
        Operator([Eq(fu, u.diff(dim, x0={dim: to}), subdomain=rows),
                  Inc(ta, a.diff(dim, x0={dim: 2*dim - u.indices_ref[dim]}, halo=0).T,
                      subdomain=rows)]).apply()
        lhs, rhs = weighted(fu, a), weighted(ta, u)
        assert abs(lhs - rhs) < 1e-12*abs(lhs)

    def test_rsfd(self):
        """
        A rotated staggered derivative, from the nodes to the centers of the
        cells, converges at its order; its transpose is refused.
        """
        errors = []
        for n in (41, 81):
            grid = sine_grid(n)
            x, y = grid.dimensions
            u = field('u', grid)
            centers = {x: x + x.spacing/2, y: y + y.spacing/2}
            out = Function(name='out', grid=grid, space_order=8, staggered=(x, y))
            Operator(Eq(out, u.dx(method='RSFD', x0=centers))).apply()
            exact = wave_dx(*physical(grid, (x, y)))
            errors.append(np.abs(out.data - exact)[10:-10, 10:-10].max())
        assert order_of(errors) > 5.5
        with pytest.raises(NotImplementedError, match='rotated'):
            u.dx(method='RSFD', x0=centers).T.evaluate  # noqa: B018

    def test_sum(self):
        """
        A sum of Functions at different points is differentiated term by term,
        each with the metric terms of its own points.
        """
        grid = sine_grid(41)
        x, y = grid.dimensions
        u, w = field('u', grid), field('w', grid, staggered=(x,))
        whole, parts = (Function(name=n, grid=grid, space_order=8) for n in 'ab')
        Operator([Eq(whole, (u + w).dy(x0={x: x})),
                  Eq(parts, u.dy(x0={x: x}) + w.dy(x0={x: x}))]).apply()
        assert np.allclose(whole.data, parts.data, rtol=0, atol=1e-12)

    def test_3d(self):
        """
        In 3D, the Laplacian converges, and the transposes of first, second and
        mixed derivatives are exact.
        """
        errors = []
        for n in (11, 21):
            grid = hill_grid(n, order=4)
            u = Function(name='u', grid=grid, space_order=4)
            xs, ys, zs = physical(grid)
            u.data[:] = np.sin(KX*xs)*np.cos(KX*ys)*np.cos(KZ*zs)
            out = Function(name='out', grid=grid, space_order=4)
            Operator(Eq(out, u.laplace)).apply()
            exact = -(2*KX**2 + KZ**2)*np.asarray(u.data)
            errors.append(np.abs(out.data - exact)[4:-4, 4:-4, 4:-4].max())
        assert np.log2(errors[0]/errors[1]) > 3.5

        grid = hill_grid(13, order=4)
        u = random_field('u', grid, 1, space_order=4)
        w = random_field('w', grid, 2, space_order=4)
        for forward, back in [(u.dx + u.dz, w.dx.T + w.dz.T),
                              (u.dx2 + u.dz2, w.dx2.T + w.dz2.T),
                              (u.dxdz + u.dydz, w.dxdz.T + w.dydz.T)]:
            assert dot_test(forward, back, w, u) < 1e-12


class TestSparse:

    def test_interpolation(self):
        """
        A SparseFunction at physical coordinates interpolates exactly a field
        linear in the computational coordinates, the frame it interpolates in.
        """
        grid = sine_grid(41)
        xi, eta = np.meshgrid(*[float(h)*np.arange(n) for h, n in
                                zip(grid.spacing, grid.shape, strict=True)],
                              indexing='ij')
        f = Function(name='f', grid=grid, space_order=8)
        f.data[:] = 2*xi - 3*eta + 1
        points = np.random.default_rng(3).uniform(100., 700., (10, 2))
        s = SparseFunction(name='s', grid=grid, npoint=10, coordinates=points)
        Operator(s.interpolate(expr=f)).apply()
        computational = grid.to_computational(points)
        exact = 2*computational[:, 0] - 3*computational[:, 1] + 1
        assert np.allclose(s.data, exact, rtol=0, atol=1e-10)

    def test_injection(self):
        """Injection is the transpose of interpolation, weighted by the Jacobian."""
        grid = sine_grid(41)
        u = random_field('u', grid, 1)
        v = Function(name='v', grid=grid, space_order=8)
        points = np.random.default_rng(3).uniform(200., 600., (10, 2))
        s, r = (SparseFunction(name=n, grid=grid, npoint=10, coordinates=points)
                for n in 'sr')
        r.data[:] = np.random.default_rng(4).standard_normal(10)
        Operator(s.interpolate(expr=u) + r.inject(field=v, expr=r)).apply()
        assert np.isclose(np.dot(s.data, r.data), weighted(v, u), rtol=1e-12)


def cartesian_linear(grid, staggered=False, comm=None):
    """
    A Cartesian Function covering `grid`, linear in the physical coordinates,
    staggered half a cell along `x` if `staggered`, distributed over `comm`,
    that of `grid` by default.
    """
    cartesian = Grid(shape=(41, 61), extent=(1000., 1200.), origin=(0., -200.),
                     dtype=np.float64, comm=comm or grid.comm)
    x, _ = cartesian.dimensions
    c = Function(name='c', grid=cartesian, space_order=0,
                 staggered=x if staggered else None)
    xc, zc = np.meshgrid(*[np.linspace(o, o + e, n) for o, e, n in
                           zip(cartesian.origin, cartesian.extent, cartesian.shape,
                               strict=True)], indexing='ij')
    c.data[:] = 2*(xc + .5*cartesian.spacing[0]*staggered) - zc
    return c


class TestResample:

    @pytest.mark.parametrize('staggered', [False, True])
    def test_linear(self, staggered):
        """
        `resample` reads a field linear in the physical coordinates exactly, at
        the points of a Cartesian field staggered or not, where the field covers
        the grid: before its first point, half a cell in when staggered, it
        takes the value of that point.
        """
        grid = sine_grid(21)
        f = Function(name='f', grid=grid, space_order=0)
        c = cartesian_linear(grid, staggered)
        resample(f, c)
        xs, zs = physical(grid)
        covered = xs >= .5*c.grid.spacing[0]*staggered
        assert np.allclose(f.data[covered], (2*xs - zs)[covered], rtol=0, atol=1e-10)

    def test_staggered(self):
        """
        A staggered Function is resampled at its own points, the last ones,
        past the Cartesian grid, at its edge.
        """
        grid = sine_grid(21)
        x = grid.dimensions[0]
        f = Function(name='f', grid=grid, space_order=0, staggered=x)
        resample(f, cartesian_linear(grid))
        xs, zs = physical(grid, (x,))
        assert np.allclose(f.data[:-1], (2*xs - zs)[:-1], rtol=0, atol=1e-10)
        assert np.allclose(f.data[-1], 2*1000. - zs[-1], rtol=0, atol=1e-10)

    def test_subdomain(self):
        """A Function on a SubDomain is refused."""
        grid = sine_grid(41)
        f = Function(name='f', grid=Middle(grid=grid), space_order=0)
        with pytest.raises(ValueError, match='SubDomain'):
            resample(f, cartesian_linear(grid))

    def test_bounds(self):
        """Resampling a sharp interface stays within the values of the field."""
        grid = sine_grid(21)
        c = cartesian_linear(grid)
        zc = np.meshgrid(*[np.linspace(o, o + e, n) for o, e, n in
                           zip(c.grid.origin, c.grid.extent, c.grid.shape, strict=True)],
                         indexing='ij')[1]
        c.data[:] = np.where(zc > 300., 3., 1.5)
        f = Function(name='f', grid=grid, space_order=0)
        resample(f, c)
        assert f.data.min() == 1.5 and f.data.max() == 3.


class TestMPI:

    @pytest.mark.parallel(mode=[2, 4])
    def test_geometry(self, mode):
        """The geometry on each rank is that of the serial grid."""
        grid = sine_grid(41)
        serial = sine_grid(41, comm=MPI.COMM_SELF)
        x, y = grid.dimensions
        gathered = grid.metrics[(x, y)].data_gather()
        if gathered is not None:
            assert np.allclose(gathered, serial.metrics[(x, y)].data)
        assert np.isclose(grid.min_spacing, serial.min_spacing)

    @pytest.mark.parallel(mode=[2, 4])
    def test_laplace(self, mode):
        """The Laplacian on each rank is that of the serial grid."""
        results = []
        for kwargs in ({}, {'comm': MPI.COMM_SELF}):
            grid = sine_grid(41, **kwargs)
            u = field('u', grid)
            out = Function(name='out', grid=grid, space_order=8)
            Operator(Eq(out, u.laplace)).apply()
            results.append(out.data_gather())
        if results[0] is not None:
            assert np.allclose(results[0], results[1], rtol=0, atol=1e-12)

    @pytest.mark.parallel(mode=[2, 4])
    def test_resample(self, mode):
        """
        Under MPI, `resample` gives the serial result, from a Cartesian field
        distributed over the ranks of the grid or held whole by every rank.
        """
        serial = sine_grid(21, comm=MPI.COMM_SELF)
        reference = Function(name='f', grid=serial, space_order=0)
        resample(reference, cartesian_linear(serial))

        grid = sine_grid(21)
        for comm in (grid.comm, MPI.COMM_SELF):
            f = Function(name='f', grid=grid, space_order=0)
            resample(f, cartesian_linear(grid, comm=comm))
            gathered = f.data_gather()
            if gathered is not None:
                assert np.allclose(gathered, reference.data, rtol=0, atol=1e-12)

    @pytest.mark.parallel(mode=4)
    def test_resample_other_ranks(self, mode):
        """`resample` refuses a Cartesian field distributed over other ranks."""
        grid = sine_grid(21)
        half = MPI.COMM_WORLD.Split(MPI.COMM_WORLD.rank // 2)
        f = Function(name='f', grid=grid, space_order=0)
        with pytest.raises(ValueError, match='other ranks'):
            resample(f, cartesian_linear(grid, comm=half))

    @pytest.mark.parallel(mode=[2, 4])
    def test_sparse(self, mode):
        """Interpolation and injection at physical points are those of one rank."""
        results = []
        points = np.random.default_rng(3).uniform(100., 700., (10, 2))
        for comm in (MPI.COMM_WORLD, MPI.COMM_SELF):
            grid = sine_grid(41, comm=comm)
            u = field('u', grid)
            v = Function(name='v', grid=grid, space_order=8)
            s, r = (SparseFunction(name=n, grid=grid, npoint=10, coordinates=points)
                    for n in 'sr')
            r.data[:] = np.arange(10.)
            Operator(s.interpolate(expr=u) + r.inject(field=v, expr=r)).apply()
            results.append((s.data_gather(), v.data_gather()))
        if results[0][1] is not None:
            for parallel, serial in zip(*results, strict=True):
                assert np.allclose(parallel, serial, rtol=1e-12, atol=1e-12)

    @pytest.mark.parallel(mode=[2, 4])
    def test_mesh_map(self, mode):
        """A MeshMap gives each rank the metric terms of the serial grid."""
        coordinates = tuple(np.asarray(c.data)
                            for c in sine_grid(41, comm=MPI.COMM_SELF).coordinates)
        results = []
        for comm in (MPI.COMM_WORLD, MPI.COMM_SELF):
            grid = CurvilinearGrid(shape=(41, 41), extent=EXTENT, dtype=np.float64,
                                   mapping=MeshMap(coordinates), comm=comm)
            x, y = grid.dimensions
            results.append(grid.metrics[(x, y)].data_gather())
        if results[0] is not None:
            assert np.allclose(results[0], results[1], rtol=0, atol=1e-12)
