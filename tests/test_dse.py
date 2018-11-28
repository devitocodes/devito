from conftest import EVAL
from sympy import sin  # noqa
import numpy as np
import pytest
from conftest import x, y, z, skipif_backend  # noqa

from devito import (Eq, Inc, Constant, Function, TimeFunction, SparseFunction,  # noqa
                    Grid, Operator, switchconfig, configuration)
from devito.ir import Stencil, FlowGraph, FindSymbols, retrieve_iteration_tree
from devito.dle import BlockDimension
from devito.dse import common_subexprs_elimination, collect
from devito.symbolics import (xreplace_constrained, iq_timeinvariant, iq_timevarying,
                              estimate_cost, pow_to_mul)
from devito.types import Scalar
from devito.tools import generator
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver

pytestmark = pytest.mark.skipif(configuration['backend'] == 'yask' or
                                configuration['backend'] == 'ops',
                                reason="testing is currently restricted")


# Acoustic

def run_acoustic_forward(dse=None):
    shape = (50, 50, 50)
    spacing = (10., 10., 10.)
    nbpml = 10
    nrec = 101
    t0 = 0.0
    tn = 250.0

    # Create two-layer model from preset
    model = demo_model(preset='layers-isotropic', vp_top=3., vp_bottom=4.5,
                       spacing=spacing, shape=shape, nbpml=nbpml)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Ricker', f0=0.010)

    solver = AcousticWaveSolver(model, geometry, dse=dse, dle='basic')
    rec, u, _ = solver.forward(save=False)

    return u, rec


def test_acoustic_rewrite_basic():
    ret1 = run_acoustic_forward(dse=None)
    ret2 = run_acoustic_forward(dse='basic')

    assert np.allclose(ret1[0].data, ret2[0].data, atol=10e-5)
    assert np.allclose(ret1[1].data, ret2[1].data, atol=10e-5)


# TTI

def tti_operator(dse=False, dle='advanced', space_order=4):
    nrec = 101
    t0 = 0.0
    tn = 250.
    nbpml = 10
    shape = (50, 50, 50)
    spacing = (20., 20., 20.)

    # Two layer model for true velocity
    model = demo_model('layers-tti', ratio=3, nbpml=nbpml, space_order=space_order,
                       shape=shape, spacing=spacing)

    # Source and receiver geometries
    src_coordinates = np.empty((1, len(spacing)))
    src_coordinates[0, :] = np.array(model.domain_size) * .5
    src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

    rec_coordinates = np.empty((nrec, len(spacing)))
    rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec_coordinates[:, 1:] = src_coordinates[0, 1:]

    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=t0, tn=tn, src_type='Gabor', f0=0.010)

    return AnisotropicWaveSolver(model, geometry, space_order=space_order, dse=dse)


@pytest.fixture(scope="session")
def tti_nodse():
    operator = tti_operator(dse=None)
    rec, u, v, _ = operator.forward()
    return v, rec


def test_tti_rewrite_basic(tti_nodse):
    operator = tti_operator(dse='basic')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-3)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-3)


def test_tti_rewrite_advanced(tti_nodse):
    operator = tti_operator(dse='advanced')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


def test_tti_rewrite_speculative(tti_nodse):
    operator = tti_operator(dse='speculative')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


def test_tti_rewrite_aggressive(tti_nodse):
    operator = tti_operator(dse='aggressive')
    rec, u, v, _ = operator.forward(kernel='centered', save=False)

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)

    # Also check that DLE's loop blocking with DSE=aggressive does the right thing
    # There should be exactly two BlockDimensions; bugs in the past were generating
    # either code with no blocking (zero BlockDimensions) or code with four
    # BlockDimensions (i.e., Iteration folding was somewhat broken)
    op = operator.op_fwd(kernel='centered', save=False)
    block_dims = [i for i in op.dimensions if isinstance(i, BlockDimension)]
    assert len(block_dims) == 2

    # Also, in this operator, we expect six temporary Arrays, two on the stack and
    # four on the heap
    arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
    assert len([i for i in arrays if i._mem_stack]) == 2
    assert len([i for i in arrays if i._mem_heap]) == 4
    assert len([i for i in arrays if i._mem_external]) == 0


@switchconfig(profiling='advanced')
@skipif_backend(['yask', 'ops'])
@pytest.mark.parametrize('kernel,space_order,expected', [
    ('shifted', 8, 355), ('shifted', 16, 622),
    ('centered', 8, 168), ('centered', 16, 300)
])
def test_tti_rewrite_aggressive_opcounts(kernel, space_order, expected):
    operator = tti_operator(dse='aggressive', space_order=space_order)
    _, _, _, summary = operator.forward(kernel=kernel, save=False)
    assert summary['section1'].ops == expected


# DSE manipulation


def test_scheduling_after_rewrite():
    """Tests loop scheduling after DSE-induced expression hoisting."""
    grid = Grid((10, 10))
    u1 = TimeFunction(name="u1", grid=grid, save=10, time_order=2)
    u2 = TimeFunction(name="u2", grid=grid, time_order=2)
    sf1 = SparseFunction(name='sf1', grid=grid, npoint=1, ntime=10)
    const = Function(name="const", grid=grid, space_order=2)

    # Deliberately inject into u1, rather than u1.forward, to create a WAR
    eqn1 = Eq(u1.forward, u1 + sin(const))
    eqn2 = sf1.inject(u1.forward, expr=sf1)
    eqn3 = Eq(u2.forward, u2 - u1.dt2 + sin(const))

    op = Operator([eqn1] + eqn2 + [eqn3])
    trees = retrieve_iteration_tree(op)

    # Check loop nest structure
    assert len(trees) == 4
    assert all(i.dim == j for i, j in zip(trees[0], grid.dimensions))  # time invariant
    assert trees[1][0].dim == trees[2][0].dim == trees[3][0].dim == grid.time_dim


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(tu, ti0 + ti1 + 5.)'],
     ['ti0[x, y, z] + ti1[x, y, z]']),
    # more ops
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)',
      'Eq(tw, 2. + ti0*t1)', 'Eq(tu, (ti0*ti1*t0) + (ti1*tv) + (t1 + ti1)*tw)'],
     ['t1*ti0[x, y, z]', 't1 + ti1[x, y, z]', 't0*ti0[x, y, z]*ti1[x, y, z]']),
    # wrapped
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)', 'Eq(tv, 2.4)',
      'Eq(tu, ((ti0*ti1*t0)*tv + (ti0*ti1*tv)*t1))'],
     ['t0*ti0[x, y, z]*ti1[x, y, z]', 't1*ti0[x, y, z]*ti1[x, y, z]']),
])
def test_xreplace_constrained_time_invariants(tu, tv, tw, ti0, ti1, t0, t1,
                                              exprs, expected):
    exprs = EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1)
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timeinvariant(FlowGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(ti0, 3.)', 'Eq(tv, 2.4)', 'Eq(tu, tv + 5. + ti0)'],
     ['tv[t, x, y, z] + 5.0']),
    # more ops
    (['Eq(tv, 2.4)', 'Eq(tw, tv*2.3)', 'Eq(ti1, 4.)', 'Eq(ti0, 3. + ti1)',
      'Eq(tu, tv*tw*4.*ti0 + ti1*tv)'],
     ['4.0*tv[t, x, y, z]*tw[t, x, y, z]']),
    # wrapped
    (['Eq(tv, 2.4)', 'Eq(tw, tv*tw*2.3)', 'Eq(ti1, 4.)', 'Eq(ti0, 3. + ti1)',
      'Eq(tu, ((tv + 4.)*ti0*ti1 + (tv + tw)/3.)*ti1*t0)'],
     ['tv[t, x, y, z] + 4.0',
      '0.333333333333333*tv[t, x, y, z] + 0.333333333333333*tw[t, x, y, z]']),
])
def test_xreplace_constrained_time_varying(tu, tv, tw, ti0, ti1, t0, t1,
                                           exprs, expected):
    exprs = EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1)
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timevarying(FlowGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(tv[t, x, y, z] + tw[t, x, y, z] + 5.0)']),
    # across expressions
    (['Eq(tu, tv*4 + tw*5 + tw*5*t0)', 'Eq(tv, tw*5)'],
     ['5*tw[t, x, y, z]', 'r0 + 5*t0*tw[t, x, y, z] + 4*tv[t, x, y, z]', 'r0']),
    # intersecting
    pytest.param(['Eq(tu, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                 ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'],
                 marks=pytest.mark.xfail),
])
def test_common_subexprs_elimination(tu, tv, tw, ti0, ti1, t0, t1, exprs, expected):
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed = common_subexprs_elimination(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1),
                                            make)
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@pytest.mark.parametrize('exprs,expected', [
    (['Eq(t0, 3.)', 'Eq(t1, 7.)', 'Eq(ti0, t0*3. + 2.)', 'Eq(ti1, t1 + t0 + 1.5)',
      'Eq(tv, (ti0 + ti1)*t0)', 'Eq(tw, (ti0 + ti1)*t1)',
      'Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     '{tu: {tu, tv, tw, ti0, ti1, t0, t1}, tv: {ti0, ti1, t0, tv},\
tw: {ti0, ti1, t1, tw}, ti0: {ti0, t0}, ti1: {ti1, t1, t0}, t0: {t0}, t1: {t1}}'),
])
def test_graph_trace(tu, tv, tw, ti0, ti1, t0, t1, exprs, expected):
    g = FlowGraph(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1))
    mapper = eval(expected)
    for i in [tu, tv, tw, ti0, ti1, t0, t1]:
        assert set([j.lhs for j in g.trace(i)]) == mapper[i]


@pytest.mark.parametrize('exprs,expected', [
    # trivial
    (['Eq(t0, 1.)', 'Eq(t1, fa[x] + fb[x])'],
     '{t0: False, t1: False}'),
    # trivial
    (['Eq(t0, 1)', 'Eq(t1, fa[t0] + fb[x])'],
     '{t0: True, t1: False}'),
    # simple
    (['Eq(t0, 1)', 'Eq(t1, fa[t0*4 + 1] + fb[x])'],
     '{t0: True, t1: False}'),
    # two-steps
    (['Eq(t0, 1.)', 'Eq(t1, t0 + 4)', 'Eq(t2, fa[t1*4 + 1] + fb[x])'],
     '{t0: False, t1: True, t2: False}'),
    # indirect
    pytest.param(['Eq(t0, 1)', 'Eq(t1, fa[fb[t0]] + fb[x])'],
                 '{t0: True, t1: False}',
                 marks=pytest.mark.xfail),
])
def test_graph_isindex(fa, fb, fc, t0, t1, t2, exprs, expected):
    g = FlowGraph(EVAL(exprs, fa, fb, fc, t0, t1, t2))
    mapper = eval(expected)
    for k, v in mapper.items():
        assert g.is_index(k) == v


@pytest.mark.parametrize('expr,expected', [
    ('2*fa[x] + fb[x]', '2*fa[x] + fb[x]'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('fa[x]**2 + fb[x]**3', 'fa[x]*fa[x] + fb[x]*fb[x]*fb[x]'),
    ('3*fa[x]**4', '3*(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('1/(fa[x]**2)', '1/(fa[x]*fa[x])'),
    ('1/(fa[x] + fb[x])', '1/(fa[x] + fb[x])'),
    ('3*sin(fa[x])**2', '3*(sin(fa[x])*sin(fa[x]))'),
])
def test_pow_to_mul(fa, fb, expr, expected):
    assert str(pow_to_mul(eval(expr))) == expected


@pytest.mark.parametrize('exprs,expected', [
    # none (different distance)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x])'],
     {'fa[x] + fb[x]': None, 'fa[x+1] + fb[x]': None}),
    # none (different dimension)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] + fb[y])'],
     {'fa[x] + fb[x]': None, 'fa[x] + fb[y]': None}),
    # none (different operation)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] - fb[x])'],
     {'fa[x] + fb[x]': None, 'fa[x] - fb[x]': None}),
    # simple
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x+1])', 'Eq(t2, fa[x-1] + fb[x-1])'],
     {'fa[x] + fb[x]': Stencil([(x, {-1, 0, 1})])}),
    # 2D simple
    (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(t1, fc[x+1,y+1] + fd[x+1,y+1])'],
     {'fc[x,y] + fd[x,y]': Stencil([(x, {0, 1}), (y, {0, 1})])}),
    # 2D with stride
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])'],
     {'fc[x,y] + fd[x+1,y+2]': Stencil([(x, {0, 1}), (y, {0, 1})])}),
    # complex (two 2D aliases with stride inducing relaxation)
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])',
      'Eq(t2, fc[x-2,y-2]*3. + fd[x+2,y+2])', 'Eq(t3, fc[x-4,y-4]*3. + fd[x,y])'],
     {'fc[x,y] + fd[x+1,y+2]': Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})]),
      '3.*fc[x-3,y-3] + fd[x+1,y+1]': Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})])}),
])
def test_collect_aliases(fa, fb, fc, fd, t0, t1, t2, t3, exprs, expected):
    scope = [fa, fb, fc, fd, t0, t1, t2, t3]
    mapper = dict([(EVAL(k, *scope), v) for k, v in expected.items()])
    _, aliases = collect(EVAL(exprs, *scope))
    for k, v in aliases.items():
        assert k in mapper
        assert (len(v.aliased) == 1 and mapper[k] is None) or v.anti_stencil == mapper[k]


@pytest.mark.parametrize('expr,expected', [
    ('Eq(t0, t1)', 0),
    ('Eq(t0, fa[x] + fb[x])', 1),
    ('Eq(t0, fa[x + 1] + fb[x - 1])', 1),
    ('Eq(t0, fa[fb[x+1]] + fa[x])', 1),
    ('Eq(t0, fa[fb[x+1]] + fc[x+2, y+1])', 1),
    ('Eq(t0, t1*t2)', 1),
    ('Eq(t0, 2.*t0*t1*t2)', 3),
    ('Eq(t0, cos(t1*t2))', 2),
    ('Eq(t0, 2.*t0*t1*t2 + t0*fa[x+1])', 5),
    ('Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0)', 7),
    ('[Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0), Eq(t0, cos(t1*t2))]', 9),
])
def test_estimate_cost(fa, fb, fc, t0, t1, t2, expr, expected):
    # Note: integer arithmetic isn't counted
    assert estimate_cost(EVAL(expr, fa, fb, fc, t0, t1, t2)) == expected


@pytest.mark.parametrize('exprs,exp_u,exp_v', [
    (['Eq(s, 0)', 'Eq(s, s + 4)', 'Eq(u, s)'], 4, 0),
    (['Eq(s, 0)', 'Eq(s, s + s + 4)', 'Eq(s, s + 4)', 'Eq(u, s)'], 8, 0),
    (['Eq(s, 0)', 'Inc(s, 4)', 'Eq(u, s)'], 4, 0),
    (['Eq(s, 0)', 'Inc(s, 4)', 'Eq(v, s)', 'Eq(u, s)'], 4, 4),
    (['Eq(s, 0)', 'Inc(s, 4)', 'Eq(v, s)', 'Eq(s, s + 4)', 'Eq(u, s)'], 8, 4),
    (['Eq(s, 0)', 'Inc(s, 4)', 'Eq(v, s)', 'Inc(s, 4)', 'Eq(u, s)'], 8, 4),
    (['Eq(u, 0)', 'Inc(u, 4)', 'Eq(v, u)', 'Inc(u, 4)'], 8, 4),
    (['Eq(u, 1)', 'Eq(v, 4)', 'Inc(u, v)', 'Inc(v, u)'], 5, 9),
])
def test_makeit_ssa(exprs, exp_u, exp_v):
    """
    A test building Operators with non-trivial sequences of input expressions
    that push hard on the `makeit_ssa` utility function.
    """
    grid = Grid(shape=(4, 4))
    u = Function(name='u', grid=grid)  # noqa
    v = Function(name='v', grid=grid)  # noqa
    s = Scalar(name='s')  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)

    op = Operator(exprs)
    op.apply()

    assert np.all(u.data == exp_u)
    assert np.all(v.data == exp_v)
