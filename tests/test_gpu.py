import numpy as np
from sympy import cos

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, solve, Operator, switchconfig,
                    norm)
from devito.ir.iet import retrieve_iteration_tree
from examples.seismic import TimeAxis, RickerSource, Receiver

pytestmark = skipif(['yask', 'ops'])


class TestOffloading(object):

    """
    Test GPU offloading via OpenMP.
    """

    @switchconfig(platform='nvidiaX')
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert op.body[1].header[0].value ==\
            ('omp target enter data map(to: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].footer[0].value ==\
            ('omp target exit data map(from: u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')

    @switchconfig(platform='nvidiaX')
    def test_multiple_eqns(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        op = Operator([Eq(u.forward, u + v + 1), Eq(v.forward, u + v + 4)],
                      dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        for i, f in enumerate([u, v]):
            assert op.body[2].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[2].footer[i].value ==\
                ('omp target exit data map(from: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})

    @switchconfig(platform='nvidiaX')
    def test_multiple_loops(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)
        v = TimeFunction(name='v', grid=grid, space_order=2)

        eqns = [Eq(f, g*2),
                Eq(u.forward, u + v*f),
                Eq(v.forward, u.forward.dx + v*f + 4)]

        op = Operator(eqns, dle=('noop', {'openmp': True}))

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3

        # All loop nests must have been parallelized
        assert trees[0][0].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[1][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'
        assert trees[2][1].pragmas[0].value ==\
            'omp target teams distribute parallel for collapse(3)'

        # Check `u` and `v`
        for i, f in enumerate([u, v], 1):
            assert op.body[4].header[i].value ==\
                ('omp target enter data map(to: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})
            assert op.body[4].footer[i].value ==\
                ('omp target exit data map(from: %(n)s[0:%(n)s_vec->size[0]]'
                 '[0:%(n)s_vec->size[1]][0:%(n)s_vec->size[2]][0:%(n)s_vec->size[3]])' %
                 {'n': f.name})

        # Check `f`
        assert op.body[4].header[0].value ==\
            ('omp target enter data map(to: f[0:f_vec->size[0]]'
             '[0:f_vec->size[1]][0:f_vec->size[2]])')
        assert op.body[4].footer[0].value ==\
            ('omp target exit data map(from: f[0:f_vec->size[0]]'
            '[0:f_vec->size[1]][0:f_vec->size[2]])')

        # Check `g` -- note that unlike `f`, this one should be `delete` upon
        # exit, not `from`
        assert op.body[4].header[3].value ==\
            ('omp target enter data map(to: g[0:g_vec->size[0]]'
             '[0:g_vec->size[1]][0:g_vec->size[2]])')
        assert op.body[4].footer[3].value ==\
            ('omp target exit data map(delete: g[0:g_vec->size[0]]'
            '[0:g_vec->size[1]][0:g_vec->size[2]])')

    @switchconfig(platform='nvidiaX')
    def test_arrays(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)

        eqn = Eq(u.forward, u*cos(f*2))

        op = Operator(eqn, dse='aggressive', dle=('noop', {'openmp': True}))

        assert str(op.body[2].header[0]) == 'float (*r1)[y_size][z_size];'
        assert op.body[2].header[1].contents[0].text ==\
            'posix_memalign((void**)&r1, 64, sizeof(float[x_size][y_size][z_size]))'
        assert op.body[2].header[1].contents[1].value ==\
            'omp target enter data map(alloc: r1[0:x_size][0:y_size][0:z_size])'
        assert op.body[2].footer[0].contents[0].value ==\
            'omp target exit data map(delete: r1[0:x_size][0:y_size][0:z_size])'
        assert op.body[2].footer[0].contents[1].text == 'free(r1)'

    def test_op_apply(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1), dle=('advanced', {'openmp': True}))

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    def test_iso_ac(self):

        shape = (101, 101)
        extent = (1000, 1000)
        origin = (0., 0.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.
        tn = 1000.
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name='src', grid=grid, f0=f0,
                           npoint=1, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size*.5
        src.coordinates.data[0, -1] = 20.

        rec = Receiver(name='rec', grid=grid, npoint=101, time_range=time_range)
        rec.coordinates.data[:, 0] = np.linspace(0, domain_size[0], num=101)
        rec.coordinates.data[:, 1] = 20.

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        pde = m * u.dt2 - u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term, dle=('advanced', {'openmp': True}))
        op(time=time_range.num-1, dt=dt)

        assert np.isclose(norm(rec), 490.54, atol=1e-2, rtol=0)
