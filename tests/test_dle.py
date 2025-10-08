from functools import reduce
from operator import mul

import sympy
import numpy as np
import pytest

from conftest import assert_structure, assert_blocking, _R, skipif
from devito import (Grid, Function, TimeFunction, SparseTimeFunction, SpaceDimension,
                    CustomDimension, Dimension, DefaultDimension, SubDimension,
                    PrecomputedSparseTimeFunction, Eq, Inc, ReduceMin, ReduceMax,
                    Operator, configuration, dimensions, info, cos)
from devito.exceptions import InvalidArgument
from devito.ir.iet import (Iteration, FindNodes, IsPerfectIteration,
                           retrieve_iteration_tree, Expression)
from devito.passes.iet.languages.openmp import Ompizer, OmpRegion
from devito.tools import as_tuple
from devito.types import Barrier, Scalar, Symbol


def get_blocksizes(op, opt, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if opt[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}


def _new_operator2(shape, time_order, blockshape=None, opt=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, opt=opt)

    blocksizes = get_blocksizes(op, opt, grid, blockshape)
    op(infield=infield, outfield=outfield, t=10, **blocksizes)

    return outfield, op


def _new_operator3(shape, blockshape0=None, blockshape1=None, opt=None):
    blockshape0 = as_tuple(blockshape0)
    blockshape1 = as_tuple(blockshape1)

    grid = Grid(shape=shape, extent=shape, dtype=np.float64)

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=(2, 2, 2))
    u.data[0, :] = np.linspace(-1, 1, reduce(mul, shape)).reshape(shape)

    # Derive the stencil according to devito conventions
    op = Operator(Eq(u.forward, 0.5 * u.laplace + u), opt=opt)

    blocksizes0 = get_blocksizes(op, opt, grid, blockshape0, 0)
    blocksizes1 = get_blocksizes(op, opt, grid, blockshape1, 1)
    op.apply(u=u, t=10, **blocksizes0, **blocksizes1)

    return u.data[1, :], op


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator2(shape, time_order=2, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order=2, opt='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("blockinner, openmp, expected", [
    (False, True, 't,x0_blk0,y0_blk0,x,y,z'), (False, False, 't,x0_blk0,y0_blk0,x,y,z'),
    (True, True, 't,x0_blk0,y0_blk0,z0_blk0,x,y,z'),
    (True, False, 't,x0_blk0,y0_blk0,z0_blk0,x,y,z')
])
def test_cache_blocking_structure(blockinner, openmp, expected):
    # Check code structure
    _, op = _new_operator2((10, 31, 45), time_order=2,
                           opt=('blocking', {'openmp': openmp, 'blockinner': blockinner,
                                'par-collapse-ncores': 1}))

    assert_structure(op, [expected])

    # Check presence of openmp pragmas at the right place
    if openmp:
        trees = retrieve_iteration_tree(op)
        assert len(trees[0][1].pragmas) == 1
        assert 'omp for' in trees[0][1].pragmas[0].ccode.value


def test_cache_blocking_structure_subdims():
    """
    Test that:

        * With local SubDimensions no-blocking is expected.
        * With non-local SubDimensions, blocking is expected.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions
    xi, yi, zi = grid.interior.dimensions
    t = grid.stepping_dim
    xl = SubDimension.left(name='xl', parent=x, thickness=4)

    f = TimeFunction(name='f', grid=grid)

    assert xl.local

    # Local SubDimension -> no blocking expected
    op = Operator(Eq(f[t+1, xl, y, z], f[t, xl, y, z] + f[t, xl, y + 1, z] + 1))

    assert_blocking(op, {})

    # Non-local SubDimension -> blocking expected
    op = Operator(Eq(f.forward, f.dx + 1, subdomain=grid.interior))

    bns, _ = assert_blocking(op, {'x0_blk0'})

    trees = retrieve_iteration_tree(bns['x0_blk0'])
    tree = trees[0]
    assert len(tree) == 5
    assert tree[0].dim.is_Block and tree[0].dim.parent.name == 'ix' and\
        tree[0].dim.root is x
    assert tree[1].dim.is_Block and tree[1].dim.parent.name == 'iy' and\
        tree[1].dim.root is y
    assert tree[2].dim.is_Block and tree[2].dim.parent is tree[0].dim and\
        tree[2].dim.root is x
    assert tree[3].dim.is_Block and tree[3].dim.parent is tree[1].dim and\
        tree[3].dim.root is y
    # zi is rebuilt with name z, so check symbolic max and min are preserved
    # Also check the zi was rebuilt
    assert not tree[4].dim.is_Block and tree[4].dim is not zi and\
        str(tree[4].dim.symbolic_min) == 'z_m + z_ltkn0' and\
        str(tree[4].dim.symbolic_max) == 'z_M - z_rtkn0' and\
        tree[4].dim.parent is z


@pytest.mark.parallel(mode=[(1, 'full')])  # Shortcut to put loops in nested efuncs
def test_cache_blocking_structure_distributed(mode):
    """
    Test cache blocking in multiple nested elemental functions.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions

    u = TimeFunction(name="u", grid=grid, space_order=2)
    U = TimeFunction(name="U", grid=grid, space_order=2)
    src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1,
                             coordinates=np.array([(0.5, 0.5, 0.5)]))

    eqns = [Eq(u.forward, u.dx)]
    eqns += src.inject(field=u.forward, expr=src)
    eqns += [Eq(U.forward, U.dx + u.forward)]

    op = Operator(eqns)
    op.cfunction

    bns0, _ = assert_blocking(op._func_table['compute0'].root, {'x0_blk0'})
    bns1, _ = assert_blocking(op._func_table['compute2'].root, {'x1_blk0'})

    for i in [bns0['x0_blk0'], bns1['x1_blk0']]:
        iters = FindNodes(Iteration).visit(i)
        assert len(iters) == 5
        assert iters[0].dim.parent is x
        assert iters[1].dim.parent is y
        assert iters[2].dim.parent is iters[0].dim
        assert iters[3].dim.parent is iters[1].dim
        assert iters[4].dim is z


class TestBlockingOptRelax:

    def test_basic(self):
        grid = Grid(shape=(8, 8, 8))

        u = TimeFunction(name="u", grid=grid, space_order=2)
        src = SparseTimeFunction(name="src", grid=grid, nt=3, npoint=1,
                                 coordinates=np.array([(0.5, 0.5, 0.5)]))

        eqns = [Eq(u.forward, u.dx)]
        eqns += src.inject(field=u.forward, expr=src)

        op = Operator(eqns, opt=('advanced', {'blockrelax': True}))

        bns, _ = assert_blocking(op, {'x0_blk0', 'p_src0_blk0'})

        iters = FindNodes(Iteration).visit(bns['p_src0_blk0'])
        assert len(iters) == 5
        assert iters[0].dim.is_Block
        assert iters[1].dim.is_Block

    def test_customdim(self):
        grid = Grid(shape=(8, 8, 8))
        d = CustomDimension(name='d', symbolic_size=2)
        x, y, z = grid.dimensions

        u = TimeFunction(name="u", grid=grid)
        f = Function(name="f", grid=grid, dimensions=(d, x, y, z),
                     shape=(2,) + grid.shape)

        eqn = Eq(f, u[d, x, y, z] + u[d, x + 1, y, z])

        op = Operator(eqn, opt=('advanced', {'blockrelax': True}))

        assert_blocking(op, {'x0_blk0'})
        assert_structure(op, ['d,x0_blk0,y0_blk0,z0_blk0,x,y,z'],
                         'd,x0_blk0,y0_blk0,z0_blk0,x,y,z')

    def test_defaultdim_alone(self):
        grid = Grid(shape=(8, 8, 8))
        d = DefaultDimension(name='d', default_value=2)
        time = grid.time_dim
        x, y, z = grid.dimensions

        u = TimeFunction(name="u", grid=grid)
        f = Function(name="f", grid=grid, dimensions=(d, x, y, z),
                     shape=(2,) + grid.shape)

        eqn = Inc(f, u*cos(time*d))

        op = Operator(eqn, opt=('advanced', {'blockrelax': 'device-aware'}))

        assert_blocking(op, {'d0_blk0', 'x0_blk0'})
        assert_structure(op,
                         ['t,d0_blk0,d', 't,d,x0_blk0,y0_blk0,z0_blk0,x,y,z'],
                         't,d0_blk0,d,d,x0_blk0,y0_blk0,z0_blk0,x,y,z')

    def test_leftright_subdims(self):
        grid = Grid(shape=(12, 12))
        nbl = 3

        damp = Function(name='damp', grid=grid)

        eqns = [Eq(damp, 0.)]
        for d in damp.dimensions:
            # Left
            dl = SubDimension.left(name='%sl' % d.name, parent=d, thickness=nbl)
            eqns.extend([Inc(damp.subs({d: dl}), 1.)])
            # right
            dr = SubDimension.right(name='%sr' % d.name, parent=d, thickness=nbl)
            eqns.extend([Inc(damp.subs({d: dr}), 1.)])

        op = Operator(eqns, opt=('fission', 'blocking', {'blockrelax': 'device-aware'}))

        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0', 'x2_blk0'})
        assert all(IsPerfectIteration().visit(i) for i in bns.values())
        assert all(len(FindNodes(Iteration).visit(i)) == 4 for i in bns.values())

    @pytest.mark.parametrize('opt, expected', [('noop', ('ijk', 'ikl')),
                             (('advanced', {'blockinner': True, 'blockrelax': True}),
                             ('i0_blk0ijk', 'i0_blk0ikl'))])
    def test_linalg(self, opt, expected):
        mat_shape = (4, 4)

        i, j, k, l = dimensions('i j k l')
        A = Function(name='A', shape=mat_shape, dimensions=(i, j))
        B = Function(name='B', shape=mat_shape, dimensions=(j, k))
        C = Function(name='C', shape=mat_shape, dimensions=(j, k))
        D = Function(name='D', shape=mat_shape, dimensions=(i, k))
        E = Function(name='E', shape=mat_shape, dimensions=(k, l))
        F = Function(name='F', shape=mat_shape, dimensions=(i, l))

        eqs = [Inc(D, A*B + A*C), Inc(F, D*E)]

        A.data[:] = 1
        B.data[:] = 1
        C.data[:] = 1
        E.data[:] = 1

        op0 = Operator(eqs, opt=opt)
        op0.apply()
        assert_structure(op0, expected)
        assert np.linalg.norm(D.data) == 32.0
        assert np.linalg.norm(F.data) == 128.0

    def test_prec_inject(self):
        grid = Grid(shape=(10, 10))
        dt = grid.stepping_dim.spacing

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)

        # The values we put it don't matter, we won't run an Operator
        points = [(0.05, 0.9), (0.01, 0.8), (0.07, 0.84)]
        gridpoints = [(5, 90), (1, 80), (7, 84)]
        interpolation_coeffs = np.ndarray(shape=(3, 2, 2))
        sf = PrecomputedSparseTimeFunction(
            name='s', grid=grid, r=2, npoint=len(points), nt=5,
            gridpoints=gridpoints, interpolation_coeffs=interpolation_coeffs
        )

        eqns = sf.inject(field=u.forward, expr=sf * dt**2)

        op = Operator(eqns, opt=('advanced', {'blockrelax': 'device-aware',
                                              'openmp': True,
                                              'par-collapse-ncores': 1}))

        assert_structure(op, ['t', 't,p_s0_blk0,p_s,rsx,rsy'],
                         't,p_s0_blk0,p_s,rsx,rsy')


class TestBlockingParTile:

    @pytest.mark.parametrize('par_tile,expected', [
        ((16, 16, 16), ((16, 16, 16), (16, 16, 16))),
        ((32, 4, 4), ((4, 4, 32), (4, 4, 32))),
        (((16, 4, 4), (16, 16, 16)), ((4, 4, 16), (16, 16, 16))),
        (((32, 4, 4), None), ((4, 4, 32), (4, 4, 32))),
        (((32, 4, 4), None, 'tag0'), ((4, 4, 32), (4, 4, 32))),
        ((((32, 4, 8), None, 'tag0'), ((32, 8, 4), None)), ((8, 4, 32), (4, 8, 32))),
    ])
    def test_structure(self, par_tile, expected):
        grid = Grid(shape=(8, 8, 8))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dx),
                Eq(v.forward, u.forward.dx)]

        op = Operator(eqns, opt=('advanced', {'par-tile': par_tile,
                                              'blockinner': True}))

        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0'})
        assert len(bns) == len(expected)
        for root, v in zip(bns.values(), expected):
            iters = FindNodes(Iteration).visit(root)
            iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
            assert len(iters) == len(v)
            assert all(i.step == j for i, j in zip(iters, v))

    def test_structure_2p5D(self):
        grid = Grid(shape=(80, 80, 80))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dx),
                Eq(v.forward, u.forward.dx)]

        par_tile = (16, 4)

        op = Operator(eqns, opt=('advanced', {'par-tile': par_tile,
                                              'blockinner': True}))

        # 3D grid, but par-tile has only 2 entries => generates so called
        # 2.5D blocking

        bns, _ = assert_blocking(op, {'y0_blk0', 'y1_blk0'})
        for root in bns.values():
            iters = FindNodes(Iteration).visit(root)
            iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
            assert len(iters) == 2
            # NOTE: par-tile are applied in reverse order
            assert iters[0].step == par_tile[1]
            assert iters[1].step == par_tile[0]

    def test_custom_rule0(self):
        grid = Grid(shape=(8, 8, 8))

        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dz.dy + u.dx.dz + u.dy.dx),
                Eq(v.forward, u.forward.dx)]

        # "Apply par-tile=(4, 4, 4) to the loop nest (kernel) with id (rule)=1,
        # and use default for the rest!"
        par_tile = (4, 4, 4)
        rule = 1

        op = Operator(eqns, opt=('advanced-fsg', {'par-tile': (par_tile, rule),
                                                  'blockinner': True}))

        # Check generated code. By having specified "1" as rule, we expect the
        # given par-tile to be applied to the kernel with id 1
        bns, _ = assert_blocking(op, {'z0_blk0', 'x0_blk0', 'z2_blk0'})
        root = bns['x0_blk0']
        iters = FindNodes(Iteration).visit(root)
        iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
        assert len(iters) == 3
        assert all(i.step == j for i, j in zip(iters, par_tile))

    def test_custom_rule1(self):
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)
        u = TimeFunction(name="u", grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eqns = [Eq(u.forward, u.dz.dy + u.dx.dz + u.dy.dx + cos(f)*cos(f[x+1, y, z])),
                Eq(v.forward, u.forward.dx)]

        # "Apply par-tile=(4, 4, 4) to the loop nests (kernels) embedded within
        # the time loop, and use default for the rest!"
        par_tile = (4, 4, 4)
        rule = grid.time_dim.name  # We must be able to infer it from str

        op = Operator(eqns, opt=('advanced-fsg', {'par-tile': (par_tile, rule),
                                                  'blockinner': True,
                                                  'blockrelax': True}))

        # Check generated code. By having specified "time" as rule, we expect the
        # given par-tile to be applied to the kernel within the time loop
        bns, _ = assert_blocking(op, {'x0_blk0', 'x1_blk0', 'x2_blk0'})
        for i in ['x0_blk0', 'x1_blk0', 'x2_blk0']:
            root = bns[i]
            iters = FindNodes(Iteration).visit(root)
            iters = [i for i in iters if i.dim.is_Block and i.dim._depth == 1]
            assert len(iters) == 3
            assert all(i.step == j for i, j in zip(iters, par_tile))


@pytest.mark.parametrize("shape", [(10,), (10, 45), (20, 33), (10, 31, 45), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (3, 3), (9, 20), (2, 9, 11), (7, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking, _ = _new_operator2(shape, time_order, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   opt=('blocking', {'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (25, 25, 46)),
    ((25, 25, 46), (7, 25, 46)),
    ((25, 25, 46), (25, 25, 7)),
    ((25, 25, 46), (25, 7, 46)),
    ((25, 25, 46), (5, 25, 7)),
    ((25, 25, 46), (10, 3, 46)),
    ((25, 25, 46), (25, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (25, 7)),
    ((25, 46), (7, 46))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    time_order = 2
    wo_blocking, _ = _new_operator2(shape, time_order, opt='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   opt=('blocking', {'blockinner': True}))
    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((3, 3), (3, 3)),
    ((4, 4), (3, 4)),
    ((5, 5), (3, 4)),
    ((6, 6), (3, 4)),
    ((7, 7), (3, 4)),
    ((8, 8), (3, 4)),
    ((9, 9), (3, 4)),
    ((10, 10), (3, 4)),
    ((11, 11), (3, 4)),
    ((12, 12), (3, 4)),
    ((13, 13), (3, 4)),
    ((14, 14), (3, 4)),
    ((15, 15), (3, 4))
])
def test_cache_blocking_edge_cases_highorder(shape, blockshape):
    wo_blocking, a = _new_operator3(shape, opt='noop')
    w_blocking, b = _new_operator3(shape, blockshape, opt=('blocking',
                                                           {'blockinner': True}))

    assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)


@pytest.mark.parametrize("blockshape0,blockshape1,exception", [
    ((24, 24, 40), (24, 24, 40), False),
    ((24, 24, 40), (4, 4, 4), False),
    ((24, 24, 40), (8, 8, 8), False),
    ((20, 20, 12), (4, 4, 4), False),
    ((28, 32, 16), (14, 16, 8), False),
    ((12, 12, 60), (4, 12, 4), False),
    ((12, 12, 60), (4, 5, 4), True),  # not a perfect divisor
    ((12, 12, 60), (24, 4, 4), True),  # bigger than outer block
])
def test_cache_blocking_hierarchical(blockshape0, blockshape1, exception):
    shape = (51, 102, 71)

    wo_blocking, a = _new_operator3(shape, opt='noop')
    try:
        w_blocking, b = _new_operator3(shape, blockshape0, blockshape1,
                                       opt=('blocking', {'blockinner': True,
                                                         'blocklevels': 2}))
        assert not exception
        assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)
    except InvalidArgument:
        assert exception
    except:
        assert False


@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_imperfect_nest(blockinner):
    """
    Test that a non-perfect Iteration nest is blocked correctly.
    """
    grid = Grid(shape=(4, 4, 4), dtype=np.float64)

    u = TimeFunction(name='u', grid=grid, space_order=2)
    v = TimeFunction(name='v', grid=grid, space_order=2)

    eqns = [Eq(u.forward, v.laplace),
            Eq(v.forward, u.forward.dz)]

    op0 = Operator(eqns, opt='noop')
    op1 = Operator(eqns, opt=('advanced', {'blockinner': blockinner}))

    # First, check the generated code
    bns, _ = assert_blocking(op1, {'x0_blk0'})
    trees = retrieve_iteration_tree(bns['x0_blk0'])
    assert len(trees) == 2
    assert len(trees[0]) == len(trees[1])
    assert all(i is j for i, j in zip(trees[0][:4], trees[1][:4]))
    assert trees[0][4] is not trees[1][4]
    assert trees[0].root.dim.is_Block
    assert trees[1].root.dim.is_Block
    assert op1.parameters[7] is trees[0][0].step
    assert op1.parameters[10] is trees[0][1].step

    u.data[:] = 0.2
    v.data[:] = 1.5
    op0(time_M=0)

    u1 = TimeFunction(name='u1', grid=grid, space_order=2)
    v1 = TimeFunction(name='v1', grid=grid, space_order=2)

    u1.data[:] = 0.2
    v1.data[:] = 1.5
    op1(u=u1, v=v1, time_M=0)

    assert np.all(u.data == u1.data)
    assert np.all(v.data == v1.data)


@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_imperfect_nest_v2(blockinner):
    """
    Test that a non-perfect Iteration nest is blocked correctly. This
    is slightly different than ``test_cache_blocking_imperfect_nest``
    as here only one Iteration gets blocked.
    """
    shape = (16, 16, 16)
    grid = Grid(shape=shape, dtype=np.float64)

    u = TimeFunction(name='u', grid=grid, space_order=4)
    u.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    eq = Eq(u.forward, 0.01*u.dy.dy)

    op0 = Operator(eq, opt='noop')
    op1 = Operator(eq, opt=('cire-sops', {'blockinner': blockinner}))
    op2 = Operator(eq, opt=('advanced-fsg', {'blockinner': blockinner,
                                             'blockrelax': True}))
    op3 = Operator(eq, opt=('advanced-fsg', {'blockinner': blockinner}))

    # First, check the generated code
    bns, _ = assert_blocking(op2, {'x0_blk0'})
    trees = retrieve_iteration_tree(bns['x0_blk0'])
    assert len(trees) == 2
    assert len(trees[0]) == len(trees[1])
    assert all(i is j for i, j in zip(trees[0][:2], trees[1][:2]))
    assert trees[0][2] is not trees[1][2]
    assert trees[0].root.dim.is_Block
    assert trees[1].root.dim.is_Block
    assert op2.parameters[4] is trees[0].root.step
    # No blocking expected in `op3` because the blocking heuristics prevent it
    # when there would be only one TILABLE Dimension
    _, _ = assert_blocking(op3, {})

    op0(time_M=0)

    u1 = TimeFunction(name='u1', grid=grid, space_order=4)
    u1.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    op1(time_M=0, u=u1)

    u2 = TimeFunction(name='u2', grid=grid, space_order=4)
    u2.data[:] = np.linspace(0, 1, reduce(mul, shape), dtype=np.float64).reshape(shape)

    op2(time_M=0, u=u2)

    assert np.allclose(u.data, u1.data, rtol=1e-07)
    assert np.allclose(u.data, u2.data, rtol=1e-07)


def test_cache_blocking_reuse_blk_dims():
    grid = Grid(shape=(16, 16, 16))
    time = grid.time_dim

    u = TimeFunction(name='u', grid=grid, space_order=4)
    v = TimeFunction(name='v', grid=grid, space_order=4)
    w = TimeFunction(name='w', grid=grid, space_order=4)
    r = TimeFunction(name='r', grid=grid, space_order=4)

    # Use barriers to prevent fusion of otherwise fusible expressions; I could
    # have created data dependencies to achieve the same effect, but that would
    # have made the test more complex
    class DummyBarrier(sympy.Function, Barrier):
        pass

    eqns = [Eq(u.forward, u.dx + v.dy),
            Eq(Symbol('dummy0'), DummyBarrier(time)),
            Eq(v.forward, v.dx),
            Eq(Symbol('dummy1'), DummyBarrier(time)),
            Eq(w.forward, w.dx),
            Eq(Symbol('dummy2'), DummyBarrier(time)),
            Eq(r.forward, r.dy + 1)]

    op = Operator(eqns, language='C')

    unique = 't,x0_blk0,y0_blk0,x,y,z'
    reused = 't,x1_blk0,y1_blk0,x,y,z'
    assert_structure(op, [unique, 't', reused, reused, reused],
                     unique+reused[1:]+reused[1:]+reused[1:])


class TestNodeParallelism:

    def test_nthreads_generation(self):
        grid = Grid(shape=(10, 10))

        f = TimeFunction(name='f', grid=grid)

        eq = Eq(f.forward, f + 1)

        op0 = Operator(eq, opt=('advanced', {'openmp': True}))

        # `nthreads` must appear among the Operator parameters
        assert op0.nthreads in op0.parameters

        # `nthreads` is bindable to a runtime value
        assert op0.nthreads._arg_values(nthreads=3)['nthreads'] == 3

    @pytest.mark.parametrize('exprs,expected', [
        # trivial 1D
        (['Eq(fa[x], fa[x] + fb[x])'],
         (True,)),
        # trivial 1D
        (['Eq(t0, fa[x] + fb[x])', 'Eq(fa[x], t0 + 1)'],
         (True,)),
        # trivial 2D
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y], t0 + 1)'],
         (True, False)),
        # outermost parallel, innermost sequential
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y+1], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost parallel
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x+1,y], t0 + 1)'],
         (False, True)),
        # outermost sequential, innermost parallel
        (['Eq(fc[x,y], fc[x+1,y+1] + fc[x-1,y])'],
         (False, True)),
        # outermost parallel w/ repeated dimensions (hence irregular dependencies)
        # both `x` and `y` are parallel-if-atomic loops
        (['Inc(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost sequential (classic skewing example)
        (['Eq(fc[x,y], fc[x,y+1] + fc[x-1,y])'],
         (False, False)),
        # skewing-like over two Eqs
        (['Eq(t0, fc[x,y+2] + fc[x-1,y+2])', 'Eq(fc[x,y+1], t0 + 1)'],
         (False, False)),
        # two nests, each nest: outermost parallel, innermost sequential
        (['Eq(fc[x,y], fc[x,y+1] + fd[x-1,y])', 'Eq(fd[x-1,y+1], fd[x-1,y] + fc[x,y+1])'],
         (True, False, False)),
        # outermost sequential, innermost parallel w/ mixed dimensions
        (['Eq(fc[x+1,y], fc[x,y+1] + fc[x,y])', 'Eq(fc[x+1,y], 2. + fc[x,y+1])'],
         (False, True)),
    ])
    def test_iterations_ompized(self, exprs, expected):
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa

        fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa
        fc = Function(name='fc', grid=grid)  # noqa
        fd = Function(name='fd', grid=grid)  # noqa
        t0 = Scalar(name='t0')  # noqa

        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        op = Operator(eqns, opt='openmp')

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == len(expected)

        # Check for presence of pragma omp
        for i, j in zip(iterations, expected):
            pragmas = i.pragmas
            if j is True:
                assert len(pragmas) == 1
                pragma = pragmas[0]
                assert 'omp for' in pragma.ccode.value
            else:
                for k in pragmas:
                    assert 'omp for' not in k.ccode.value

    def test_dynamic_nthreads(self):
        grid = Grid(shape=(16, 16, 16))
        f = TimeFunction(name='f', grid=grid)
        sf = SparseTimeFunction(name='sf', grid=grid, npoint=1, nt=5)

        eqns = [Eq(f.forward, f + 1)]
        eqns += sf.interpolate(f)

        op = Operator(eqns, opt='openmp')

        parregions = FindNodes(OmpRegion).visit(op)
        assert len(parregions) == 2

        # Check suitable `num_threads` appear in the generated code
        # Not very elegant, but it does the trick
        assert 'num_threads(nthreads)' in str(parregions[0].header[0])
        assert 'num_threads(nthreads_nonaffine)' in str(parregions[1].header[0])

        # Check `op` accepts the `nthreads*` kwargs
        op.apply(time=0)
        op.apply(time_m=1, time_M=1, nthreads=4)
        op.apply(time_m=1, time_M=1, nthreads=4, nthreads_nonaffine=2)
        op.apply(time_m=1, time_M=1, nthreads_nonaffine=2)
        assert np.all(f.data[0] == 2.)

        # Check the actual value assumed by `nthreads` and `nthreads_nonaffine`
        assert op.arguments(time=0, nthreads=123)['nthreads'] == 123
        assert op.arguments(time=0, nthreads_nonaffine=100)['nthreads_nonaffine'] == 100

    @pytest.mark.parametrize('eqns,expected,blocking', [
        ('[Eq(f, 2*f)]', [2, 0, 0], False),
        ('[Eq(u, 2*u)]', [0, 2, 0, 0], False),
        ('[Eq(u, 2*u + f)]', [0, 3, 0, 0, 0, 0, 0], True),
        ('[Eq(u, 2*u), Eq(f, u.dzr)]', [0, 2, 0, 0, 0], False)
    ])
    def test_collapsing(self, eqns, expected, blocking):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)  # noqa
        u = TimeFunction(name='u', grid=grid)  # noqa

        eqns = eval(eqns)

        if blocking:
            op = Operator(eqns, opt=('blocking', 'simd', 'openmp',
                                     {'blockinner': True, 'par-collapse-ncores': 1,
                                      'par-collapse-work': 0}))
            assert_structure(op, ['t,x0_blk0,y0_blk0,z0_blk0,x,y,z'])
        else:
            op = Operator(eqns, opt=('simd', 'openmp', {'par-collapse-ncores': 1,
                                                        'par-collapse-work': 0}))

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == len(expected)

        # Check for presence of pragma omp + collapse clause
        for i, j in zip(iterations, expected):
            if j > 0:
                assert len(i.pragmas) == 1
                pragma = i.pragmas[0]
                assert 'omp for collapse(%d)' % j in pragma.ccode.value
            else:
                for k in i.pragmas:
                    assert 'omp for collapse' not in k.ccode.value

    def test_collapsing_v2(self):
        """
        MFE from issue #1478.
        """
        n = 8
        m = 8
        nx, ny, nchi, ncho = 12, 12, 1, 1
        x, y = SpaceDimension("x"), SpaceDimension("y")
        ci, co = Dimension("ci"), Dimension("co")
        i, j = Dimension("i"), Dimension("j")
        grid = Grid((nx, ny), dtype=np.float32, dimensions=(x, y))

        X = Function(name="xin", dimensions=(ci, x, y),
                     shape=(nchi, nx, ny), grid=grid, space_order=n//2)
        dy = Function(name="dy", dimensions=(co, x, y),
                      shape=(ncho, nx, ny), grid=grid, space_order=n//2)
        dW = Function(name="dW", dimensions=(co, ci, i, j), shape=(ncho, nchi, n, m),
                      grid=grid)

        eq = [Eq(dW[co, ci, i, j],
                 dW[co, ci, i, j] + dy[co, x, y]*X[ci, x+i-n//2, y+j-m//2])
              for i in range(n) for j in range(m)]

        op = Operator(eq, opt=('advanced', {'openmp': True}))

        assert_structure(op, ['co,ci,x,y'])
        iterations = FindNodes(Iteration).visit(op)
        assert iterations[0].ncollapsed == 1
        assert iterations[1].is_Vectorized
        assert iterations[2].is_Sequential
        assert iterations[3].is_Sequential

    def test_scheduling(self):
        """
        Affine iterations -> #pragma omp ... schedule(dynamic,1) ...
        Non-affine iterations -> #pragma omp ... schedule(dynamic,chunk_size) ...
        """
        grid = Grid(shape=(11, 11))

        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=1)
        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)

        eqns = [Eq(u.forward, u + 1)]
        eqns += sf1.interpolate(u)

        op = Operator(eqns, opt=('openmp', {'par-dynamic-work': 0}))

        iterations = FindNodes(Iteration).visit(op)

        assert len(iterations) == 6
        assert iterations[1].is_Affine
        assert 'schedule(dynamic,1)' in iterations[1].pragmas[0].ccode.value
        assert not iterations[3].is_Affine
        assert 'schedule(dynamic,chunk_size)' in iterations[3].pragmas[0].ccode.value

    @skipif('cpu64-icc')
    @pytest.mark.parametrize('so', [0, 1, 2])
    @pytest.mark.parametrize('dim', [0, 1, 2])
    def test_array_sum_reduction(self, so, dim):
        """
        Test generation of OpenMP sum-reduction clauses involving Function's.
        """
        grid = Grid(shape=(3, 3, 3))
        d = grid.dimensions[dim]

        f = Function(name='f', shape=(3,), dimensions=(d,), grid=grid, space_order=so)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Inc(f, u + 1), opt=('openmp', {'par-collapse-ncores': 1}))

        iterations = FindNodes(Iteration).visit(op)
        parallelized = iterations[dim+1]
        assert parallelized.pragmas
        if parallelized.dim is iterations[-1]:
            # With the `f[z] += u[t0][x + 1][y + 1][z + 1] + 1` expr, the innermost
            # `z` Iteration gets parallelized, nothing is collapsed, hence no
            # reduction is required
            assert "reduction" not in parallelized.pragmas[0].ccode.value
        elif Ompizer._support_array_reduction(configuration['compiler']):
            if "collapse" in parallelized.pragmas[0].ccode.value:
                assert ("reduction(+:f[0:f_vec->size[0]])"
                        in parallelized.pragmas[0].ccode.value)
        else:
            # E.g. old GCC's
            assert "atomic update" in str(iterations[-1])

        try:
            op(time_M=1)
        except:
            # Older gcc <6.1 don't support reductions on array
            info("Un-supported older gcc version for array reduction")
            assert True
            return

        assert np.allclose(f.data, 18)

    def test_reduction_local(self):
        grid = Grid((11, 11))
        d = Dimension("i")
        n = Function(name="n", dimensions=(d,), shape=(1,))
        u = Function(name="u", grid=grid)
        u.data.fill(1.)

        op = Operator(Inc(n[0], u))
        op()

        cond = FindNodes(Expression).visit(op)
        iterations = FindNodes(Iteration).visit(op)
        # Should not creat any temporary for the reduction
        nlin = 2 if op._options['linearize'] else 0
        assert len(cond) == 1 + nlin
        if configuration['language'] in ['CXX', 'C']:
            pass
        elif Ompizer._support_array_reduction(configuration['compiler']):
            i = '0:1' if op._options['linearize'] else '0'
            assert f"reduction(+:n[{i}])" in iterations[0].pragmas[0].ccode.value
        else:
            # E.g. old GCC's
            assert "atomic update" in str(iterations[-1])

        assert n.data[0] == 11*11

    def test_mapify_reduction_sparse(self):
        grid = Grid((11, 11))
        s = SparseTimeFunction(name="s", grid=grid, npoint=1, nt=11)
        s.data.fill(1.)
        r = Symbol(name="r", dtype=np.float32)
        n0 = Function(name="n0", dimensions=(Dimension("noi"),), shape=(1,))

        eqns = [Eq(r, 0), Inc(r, s*s), Eq(n0[0], r)]
        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=('advanced', {'mapify-reduce': True}))

        expr0 = FindNodes(Expression).visit(op0)
        nlin = 2 if op0._options['linearize'] else 0
        assert len(expr0) == 3 + nlin
        assert expr0[1+nlin].is_reduction

        expr1 = FindNodes(Expression).visit(op1)
        nlin = 2 if op0._options['linearize'] else 0
        assert len(expr1) == 4 + nlin
        assert expr1[1+nlin].expr.lhs.indices == s.indices
        assert expr1[2+nlin].expr.rhs.is_Indexed
        assert expr1[2+nlin].is_reduction

        op0()
        assert n0.data[0] == 11
        op1()
        assert n0.data[0] == 11

    def test_array_max_reduction(self):
        """
        Test generation of OpenMP max-reduction clauses involving Function's.
        """
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', grid=grid)
        n = Function(name='n', grid=grid, shape=(1,), dimensions=(i,))

        f.data[:] = np.arange(0, 27).reshape((3, 3, 3))

        eqn = ReduceMax(n[0], f)

        if Ompizer._support_array_reduction(configuration['compiler']):
            op = Operator(eqn, opt=('advanced', {'openmp': True}))

            iterations = FindNodes(Iteration).visit(op)
            i = '0:1' if op._options['linearize'] else '0'
            assert f"reduction(max:n[{i}])" in iterations[0].pragmas[0].ccode.value

            op()
            assert n.data[0] == 26
        else:
            # Unsupported min/max reductions with obsolete compilers
            with pytest.raises(NotImplementedError):
                Operator(eqn, opt=('advanced', {'openmp': True}))

    def test_array_minmax_reduction(self):
        """
        Test generation of OpenMP combined min- and max-reduction clauses
        involving Function's.
        """
        grid = Grid(shape=(3, 3, 3))
        i = Dimension(name='i')

        f = Function(name='f', grid=grid)
        n = Function(name='n', grid=grid, shape=(2,), dimensions=(i,))
        r0 = Symbol(name='r0', dtype=grid.dtype)
        r1 = Symbol(name='r1', dtype=grid.dtype)

        f.data[:] = np.arange(0, 27).reshape((3, 3, 3))

        eqns = [ReduceMax(r0, f),
                ReduceMin(r1, f),
                Eq(n[0], r0),
                Eq(n[1], r1)]

        if not Ompizer._support_array_reduction(configuration['compiler']):
            return

        op = Operator(eqns)

        if 'openmp' in configuration['language']:
            iterations = FindNodes(Iteration).visit(op)
            expected = "reduction(max:r0) reduction(min:r1)"
            assert expected in iterations[0].pragmas[0].ccode.value

        op()
        assert n.data[0] == 26
        assert n.data[1] == 0

    def test_incs_no_atomic(self):
        """
        Test that `Inc`'s don't get a `#pragma omp atomic` if performing
        an increment along a fully parallel loop.
        """
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        # Format: u(t, x, nastyness) += 1
        uf = u[t, x, f, z]

        # All loops get collapsed, but the `y` and `z` loops are PARALLEL_IF_ATOMIC,
        # hence an atomic pragma is expected
        op0 = Operator(Inc(uf, 1), opt=('advanced', {'openmp': True,
                                                     'par-collapse-ncores': 1,
                                                     'par-collapse-work': 0}))
        assert 'omp for schedule' in str(op0)
        assert 'collapse' not in str(op0)
        assert 'atomic' not in str(op0)

        # Now only `x` is parallelized
        op1 = Operator([Eq(v[t, x, 0, 0], v[t, x, 0, 0] + 1), Inc(uf, 1)],
                       opt=('advanced', {'openmp': True,
                                         'par-collapse-ncores': 1}))

        assert 'omp for' in str(op1)
        assert 'collapse' not in str(op1)
        assert 'atomic' not in str(op1)

    def test_incr_perfect_outer(self):
        grid = Grid((5, 5))
        d = Dimension(name="d")

        u = Function(name="u", dimensions=(*grid.dimensions, d),
                     grid=grid, shape=(*grid.shape, 5), )
        v = Function(name="v", dimensions=(*grid.dimensions, d),
                     grid=grid, shape=(*grid.shape, 5))
        w = Function(name="w", grid=grid)

        u.data.fill(1)
        v.data.fill(2)

        summation = Inc(w, u*v)

        op = Operator([summation], opt=('advanced', {'openmp': True}))
        assert 'reduction' not in str(op)
        assert 'omp for' in str(op)

        op()
        assert np.all(w.data == 10)

    def test_incr_perfect_sparse_outer(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)
        s = SparseTimeFunction(name='u', grid=grid, npoint=1, nt=11)

        eqns = s.inject(u, expr=s)

        op = Operator(eqns, opt=('advanced', {'par-collapse-ncores': 0,
                                              'openmp': True}))

        iters = FindNodes(Iteration).visit(op)
        assert len(iters) == 5
        assert iters[0].is_Sequential
        assert all(i.is_ParallelAtomic for i in iters[1:])
        assert iters[1].pragmas[0].ccode.value ==\
            'omp for schedule(dynamic,chunk_size)'
        assert all(not i.pragmas for i in iters[2:])

    @pytest.mark.parametrize('exprs,simd_level,expected', [
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Inc(h1[0, 0], 1, implicit_dims=(t, x, y))'],
         None, [6, 0, 0]),
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],  # 1695
         2, [0, 1, 2]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
         'Eq(h1[0, y], 3 - y, implicit_dims=(t, x, y))'],
         2, [3, 2, 1]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         2, [0, 1, 2]),
        (['Eq(y.symbolic_min, g[0, 0], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         None, [0, 1, 2]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Eq(h1[0, x], y.symbolic_min, implicit_dims=(t, x))'],
         1, [2, 2, 2]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Eq(h1[0, x], y.symbolic_max, implicit_dims=(t, x))'],
         1, [2, 2, 2]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, y], y, implicit_dims=(t, x, y))'],
         2, [0, 2, 6]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y, implicit_dims=(t, x, y))'],
         None, [3, 3, 2]),
        (['Eq(y.symbolic_min, g[0, 0], implicit_dims=(t, x))',
          'Inc(h1[0, y], x, implicit_dims=(t, x, y))'],
         2, [3, 3, 3]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y.symbolic_min, implicit_dims=(t, x))'],
         None, [2, 2, 2]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y.symbolic_min, implicit_dims=(t, x, y))'],
         None, [2, 2, 2]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, x]-1, implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         2, [0, 0, 0])
    ])
    def test_edge_cases(self, exprs, simd_level, expected):
        # Tests for issue #1695
        t, x, y = dimensions('t x y')

        g = TimeFunction(name='g', shape=(1, 3), dimensions=(t, x),
                         time_order=0, dtype=np.int32)
        g.data[0, :] = [0, 1, 2]
        h1 = TimeFunction(name='h1', shape=(1, 3), dimensions=(t, y), time_order=0)
        h1.data[0, :] = 0

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs, opt=('advanced', {'openmp': True,
                                               'par-collapse-ncores': 1}))

        iterations = FindNodes(Iteration).visit(op)
        parallel = [i for i in iterations if i.is_Parallel]
        try:
            assert 'omp for' in iterations[0].pragmas[0].ccode.value
            if len(parallel) > 1 and simd_level is not None and simd_level > 1:
                assert 'collapse' in iterations[0].pragmas[0].ccode.value
            if simd_level:
                assert 'omp simd' in iterations[simd_level].pragmas[0].ccode.value
        except:
            # E.g. gcc-5 doesn't support array reductions, so the compiler will
            # generate different legal code
            assert not Ompizer._support_array_reduction(configuration['compiler'])
            assert any('omp for' in i.pragmas[0].ccode.value
                       for i in iterations if i.pragmas)

        op.apply()
        assert (h1.data == expected).all()

    def test_simd_space_invariant(self):
        """
        Similar to test_space_invariant_v3, testing simd vectorization happens
        in the correct place.
        """
        grid = Grid(shape=(10, 10, 10))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)
        eq = Inc(f, cos(x*y) + cos(x*z))

        op = Operator(eq, opt=('advanced', {'openmp': True}))
        iterations = FindNodes(Iteration).visit(op)

        assert 'omp for schedule(static,1)' in iterations[0].pragmas[0].ccode.value
        assert 'omp simd' in iterations[1].pragmas[0].ccode.value
        assert 'omp simd' in iterations[3].pragmas[0].ccode.value

        op.apply()
        assert np.isclose(np.linalg.norm(f.data), 37.1458, rtol=1e-5)

    def test_parallel_prec_inject(self):
        grid = Grid(shape=(10, 10))
        dt = grid.stepping_dim.spacing

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)

        # The values we put it don't matter, we won't run an Operator
        points = [(0.05, 0.9), (0.01, 0.8), (0.07, 0.84)]
        gridpoints = [(5, 90), (1, 80), (7, 84)]
        interpolation_coeffs = np.ndarray(shape=(3, 2, 2))
        sf = PrecomputedSparseTimeFunction(
            name='s', grid=grid, r=2, npoint=len(points), nt=5,
            gridpoints=gridpoints, interpolation_coeffs=interpolation_coeffs
        )

        eqns = sf.inject(field=u.forward, expr=sf * dt**2)

        op0 = Operator(eqns, opt=('advanced', {'openmp': True,
                                               'par-collapse-ncores': 2000}))
        iterations = FindNodes(Iteration).visit(op0)

        assert not iterations[0].pragmas
        assert 'omp for' in iterations[1].pragmas[0].ccode.value
        assert 'collapse' not in iterations[1].pragmas[0].ccode.value

        op0 = Operator(eqns, opt=('advanced', {'openmp': True,
                                               'par-collapse-ncores': 1,
                                               'par-collapse-work': 1}))
        iterations = FindNodes(Iteration).visit(op0)

        assert not iterations[0].pragmas
        assert 'omp for collapse' in iterations[1].pragmas[0].ccode.value


class TestNestedParallelism:

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + f + 1),
                      opt=('blocking', 'openmp', {'par-nested': 0,
                                                  'par-collapse-ncores': 10000,
                                                  'par-dynamic-work': 0}))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        # Try again but this time supplying specific values for the num_threads
        u.data[:] = 0.
        op.apply(t_M=9, nthreads=1, nthreads_nested=2)
        assert np.all(u.data[0] == 10)
        assert op.arguments(t_M=9, nthreads_nested=2)['nthreads_nested'] == 2

        bns, _ = assert_blocking(op, {'x0_blk0'})

        iterations = FindNodes(Iteration).visit(bns['x0_blk0'])
        assert iterations[0].pragmas[0].ccode.value == 'omp for schedule(dynamic,1)'
        assert iterations[2].pragmas[0].ccode.value ==\
            'omp parallel for schedule(dynamic,1) num_threads(nthreads_nested)'

    def test_collapsing(self):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + f + 1),
                      opt=('blocking', 'openmp', {'par-nested': 0,
                                                  'par-collapse-ncores': 1,
                                                  'par-collapse-work': 0,
                                                  'par-dynamic-work': 0}))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)

        assert np.all(u.data[0] == 10)

        bns, _ = assert_blocking(op, {'x0_blk0'})

        iterations = FindNodes(Iteration).visit(bns['x0_blk0'])
        assert iterations[0].pragmas[0].ccode.value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert iterations[2].pragmas[0].ccode.value ==\
            ('omp parallel for collapse(2) schedule(dynamic,1) '
             'num_threads(nthreads_nested)')

    def test_multiple_subnests_v0(self):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=3)

        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3.*f +
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.*f) + 1.)
        op = Operator(eqn, opt=('advanced', {'openmp': True,
                                             'cire-mingain': 0,
                                             'cire-schedule': 1,
                                             'par-nested': 0,
                                             'par-collapse-ncores': 1,
                                             'par-dynamic-work': 0}))

        bns, _ = assert_blocking(op, {'x0_blk0'})

        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2

        assert trees[0][0] is trees[1][0]
        assert trees[0][0].pragmas[0].ccode.value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert trees[0][2].pragmas[0].ccode.value ==\
            ('omp parallel for collapse(2) schedule(dynamic,1) '
             'num_threads(nthreads_nested)')
        assert trees[1][2].pragmas[0].ccode.value ==\
            ('omp parallel for collapse(2) schedule(dynamic,1) '
             'num_threads(nthreads_nested)')

    def test_multiple_subnests_v1(self):
        """
        Unlike ``test_multiple_subnestes_v0``, now we use the ``cire-rotate=True``
        option, which trades some of the inner parallelism for a smaller working set.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=3)

        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3.*f +
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.*f) + 1.)
        op = Operator(eqn, opt=('advanced', {'openmp': True,
                                             'cire-mingain': 0,
                                             'cire-schedule': 1,
                                             'cire-rotate': True,
                                             'par-nested': 0,
                                             'par-collapse-ncores': 1,
                                             'par-dynamic-work': 0}))

        bns, _ = assert_blocking(op, {'x0_blk0'})

        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 4

        assert len(set(i.root for i in trees)) == 1
        assert trees[-2].root.pragmas[0].ccode.value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert not trees[-2][2].pragmas
        assert not trees[-2][3].pragmas
        assert trees[-2][4].pragmas[0].ccode.value ==\
            'omp parallel for schedule(dynamic,1) num_threads(nthreads_nested)'
        assert not trees[-1][2].pragmas
        assert trees[-1][3].pragmas[0].ccode.value ==\
            'omp parallel for schedule(dynamic,1) num_threads(nthreads_nested)'

    @pytest.mark.parametrize('blocklevels', [1, 2])
    def test_nested_cache_blocking_structure_subdims(self, blocklevels):
        """
        Test that:

            * With non-local SubDimensions, nested blocking works fine when expected.
            * With non-local SubDimensions, hierarchical nested blocking works fine
            when expected.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        xi, yi, zi = grid.interior.dimensions
        xl = SubDimension.left(name='xl', parent=x, thickness=4)

        f = TimeFunction(name='f', grid=grid)

        assert xl.local

        # Non-local SubDimension -> nested blocking can works as expected
        op = Operator(Eq(f.forward, f.dx + 1, subdomain=grid.interior),
                      opt=('blocking', 'openmp',
                           {'par-nested': 0, 'blocklevels': blocklevels,
                            'par-collapse-ncores': 2,
                            'par-dynamic-work': 0}))

        bns, _ = assert_blocking(op, {'x0_blk0'})

        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 1
        tree = trees[0]
        assert len(tree) == 5 + (blocklevels - 1) * 2
        assert tree[0].dim.is_Block and tree[0].dim.parent.name == 'ix' and\
            tree[0].dim.root is x
        assert tree[1].dim.is_Block and tree[1].dim.parent.name == 'iy' and\
            tree[1].dim.root is y
        assert tree[2].dim.is_Block and tree[2].dim.parent is tree[0].dim and\
            tree[2].dim.root is x
        assert tree[3].dim.is_Block and tree[3].dim.parent is tree[1].dim and\
            tree[3].dim.root is y

        if blocklevels == 1:
            assert not tree[4].dim.is_Block and tree[4].dim is not zi and\
                str(tree[4].dim.symbolic_min) == 'z_m + z_ltkn0' and\
                str(tree[4].dim.symbolic_max) == 'z_M - z_rtkn0' and\
                tree[4].dim.parent is z
        elif blocklevels == 2:
            assert tree[3].dim.is_Block and tree[3].dim.parent is tree[1].dim and\
                tree[3].dim.root is y
            assert tree[4].dim.is_Block and tree[4].dim.parent is tree[2].dim and\
                tree[4].dim.root is x
            assert tree[5].dim.is_Block and tree[5].dim.parent is tree[3].dim and\
                tree[5].dim.root is y
            assert not tree[6].dim.is_Block and tree[6].dim is not zi and\
                str(tree[6].dim.symbolic_min) == 'z_m + z_ltkn0' and\
                str(tree[6].dim.symbolic_max) == 'z_M - z_rtkn0' and\
                tree[6].dim.parent is z

        assert trees[0][0].pragmas[0].ccode.value ==\
            'omp for collapse(2) schedule(dynamic,1)'
        assert trees[0][2].pragmas[0].ccode.value ==\
            ('omp parallel for collapse(2) schedule(dynamic,1) '
             'num_threads(nthreads_nested)')

    @pytest.mark.parametrize('exprs,collapsed,scheduling', [
        (['Eq(u.forward, u.dx)'], '2', 'static'),
        (['Eq(u.forward, u.dy)'], '2', 'static'),
        (['Eq(u.forward, u.dx.dx)'], '2', 'dynamic'),
        (['Eq(u.forward, u.dy.dy)'], '2', 'dynamic'),
    ])
    def test_collapsing_w_wo_halo(self, exprs, collapsed, scheduling):
        """
        This test ensures correct number of collapsed loops and scheduling for several
        expressions based on the amount of work (par-dynamic-work). For issue #1723
        """

        grid = Grid(shape=(10, 10, 10))
        u = TimeFunction(name='u', grid=grid, space_order=4) # noqa
        eqns = []
        for e in exprs:
            eqns.append(eval(e))

        op = Operator(eqns, opt=('blocking', 'openmp',
                                 {'par-collapse-ncores': 1,
                                  'par-dynamic-work': 20}))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction
        iterations = FindNodes(Iteration).visit(op)

        ompfor_string = "".join(['omp for collapse(', collapsed, ')'])
        scheduling_string = "".join([' schedule(', scheduling, ',1)'])

        assert iterations[1].pragmas[0].ccode.value ==\
            "".join([ompfor_string, scheduling_string])
