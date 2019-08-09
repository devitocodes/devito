import numpy as np
from math import floor

from conftest import skipif
from devito import (Grid, Function, TimeFunction, Eq, solve, Operator, SubDomainSet,
                    Dimension)

pytestmark = skipif(['yask', 'ops'])


class TestSubdomains(object):
    """
    Class for testing SubDomains
    """

    def test_iterate_NDomains(self):
        """
        Test that a set of subdomains are iterated upon correctly.
        """

        n_domains = 10

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = n_domains-1-j
            bounds_ym[j] = floor(j/2)
            bounds_yM[j] = floor(j/2)

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, bounds_ym[j]:n_domains-bounds_yM[j]] = 10

        assert((np.array(result) == expected).all())

    def test_multi_eq(self):
        """
        Test SubDomainSet functionality when multiple equations are
        present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains(SubDomainSet):
            name = 'mydomains'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = 1
        bounds_yM = 1
        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)
        my_sd = MySubdomains(N=n_domains, bounds=bounds)
        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)
        h = Function(name='h', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
        eq2 = Eq(g, g+1)
        eq3 = Eq(h, h+2, subdomain=grid.subdomains['mydomains'])

        op = Operator([eq1, eq2, eq3])
        op.apply()

        expected1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        expected2 = np.full((10, 10), 1, dtype=np.int32)
        expected3 = 2*expected1

        assert((np.array(f.data) == expected1).all())
        assert((np.array(g.data) == expected2).all())
        assert((np.array(h.data) == expected3).all())

    def test_multi_sets(self):
        """
        Check functionality for when multiple subdomain sets are present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        n = Dimension(name='n')
        m = Dimension(name='m')

        class MySubdomains1(SubDomainSet):
            name = 'mydomains1'
            implicit_dimension = n

        class MySubdomains2(SubDomainSet):
            name = 'mydomains2'
            implicit_dimension = m

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(1)
        bounds_yM = int(Ny/2+1)
        bounds1 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(Ny/2+1)
        bounds_yM = int(1)
        bounds2 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        my_sd1 = MySubdomains1(N=n_domains, bounds=bounds1)
        my_sd2 = MySubdomains2(N=n_domains, bounds=bounds2)

        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd1, my_sd2))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains1'])
        eq2 = Eq(g, g+2, subdomain=grid.subdomains['mydomains2'])

        op = Operator([eq1, eq2])
        op.apply()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        assert((np.array(f.data[:]+g.data[:]) == expected).all())
