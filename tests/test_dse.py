from conftest import EVAL

from sympy import sin  # noqa
import numpy as np
import pytest
from conftest import x, y, z, time, skipif_yask  # noqa

from devito import Eq  # noqa
from devito.ir import Stencil, clusterize, TemporariesGraph
from devito.dse import rewrite, common_subexprs_elimination, collect
from devito.symbolics import (xreplace_constrained, iq_timeinvariant, iq_timevarying,
                              estimate_cost, pow_to_mul, indexify)
from devito.types import Scalar
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, RickerSource, GaborSource, Receiver
from examples.seismic.tti import AnisotropicWaveSolver


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

    # Derive timestepping from model spacing
    dt = model.critical_dt
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time_values = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time=time_values)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = 20.

    # Define receiver geometry (same as source, but spread across x)
    rec = Receiver(name='nrec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    solver = AcousticWaveSolver(model, source=src, receiver=rec, dse=dse, dle='basic')
    rec, u, _ = solver.forward(save=False)

    return u, rec


@skipif_yask
def test_acoustic_rewrite_basic():
    ret1 = run_acoustic_forward(dse=None)
    ret2 = run_acoustic_forward(dse='basic')

    assert np.allclose(ret1[0].data, ret2[0].data, atol=10e-5)
    assert np.allclose(ret1[1].data, ret2[1].data, atol=10e-5)


# TTI

def tti_operator(dse=False, space_order=4):
    nrec = 101
    t0 = 0.0
    tn = 250.
    nbpml = 10
    shape = (50, 50, 50)
    spacing = (20., 20., 20.)

    # Two layer model for true velocity
    model = demo_model('layers-tti', ratio=3, nbpml=nbpml,
                       shape=shape, spacing=spacing)

    # Derive timestepping from model spacing
    # Derive timestepping from model spacing
    dt = model.critical_dt
    nt = int(1 + (tn-t0) / dt)  # Number of timesteps
    time_values = np.linspace(t0, tn, nt)  # Discretized time axis

    # Define source geometry (center of domain, just below surface)
    src = GaborSource(name='src', grid=model.grid, f0=0.01, time=time_values)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]

    # Define receiver geometry (spread across x, lust below surface)
    rec = Receiver(name='nrec', grid=model.grid, ntime=nt, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    return AnisotropicWaveSolver(model, source=src, receiver=rec,
                                 time_order=2, space_order=space_order, dse=dse)


@pytest.fixture(scope="session")
def tti_nodse():
    operator = tti_operator(dse=None)
    rec, u, v, _ = operator.forward()
    return v, rec


@skipif_yask
def test_tti_clusters_to_graph():
    solver = tti_operator()

    expressions = solver.op_fwd('centered').args['expressions']
    subs = solver.op_fwd('centered').args['subs']
    expressions = [indexify(s) for s in expressions]
    expressions = [s.xreplace(subs) for s in expressions]
    stencils = solver.op_fwd('centered')._retrieve_stencils(expressions)
    clusters = clusterize(expressions, stencils)
    assert len(clusters) == 3

    main_cluster = clusters[0]
    n_output_tensors = len(main_cluster.trace)

    clusters = rewrite([main_cluster], mode='basic')
    assert len(clusters) == 1
    main_cluster = clusters[0]

    graph = main_cluster.trace
    assert len([v for v in graph.values() if v.is_tensor]) == n_output_tensors  # u and v
    assert all(v.reads or v.readby for v in graph.values())


@skipif_yask
def test_tti_rewrite_basic(tti_nodse):
    operator = tti_operator(dse='basic')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-3)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-3)


@skipif_yask
def test_tti_rewrite_advanced(tti_nodse):
    operator = tti_operator(dse='advanced')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


@skipif_yask
def test_tti_rewrite_speculative(tti_nodse):
    operator = tti_operator(dse='speculative')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


@skipif_yask
def test_tti_rewrite_aggressive(tti_nodse):
    operator = tti_operator(dse='aggressive')
    rec, u, v, _ = operator.forward()

    assert np.allclose(tti_nodse[0].data, v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1].data, rec.data, atol=10e-1)


@skipif_yask
@pytest.mark.parametrize('kernel,space_order,expected', [
    ('shifted', 8, 355), ('shifted', 16, 811),
    ('centered', 8, 168), ('centered', 16, 300)
])
def test_tti_rewrite_aggressive_opcounts(kernel, space_order, expected):
    operator = tti_operator(dse='aggressive', space_order=space_order)
    _, _, _, summary = operator.forward(kernel=kernel, save=False)
    assert summary['main'].ops == expected


# DSE manipulation

@skipif_yask
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
    make = lambda i: Scalar(name='r%d' % i).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timeinvariant(TemporariesGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@skipif_yask
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
    make = lambda i: Scalar(name='r%d' % i).indexify()
    processed, found = xreplace_constrained(exprs, make,
                                            iq_timevarying(TemporariesGraph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@skipif_yask
@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(tv[t, x, y, z] + tw[t, x, y, z] + 5.0)']),
    # across expressions
    (['Eq(tu, tv*4 + tw*5 + tw*5*t0)', 'Eq(tv, tw*5)'],
     ['5*tw[t, x, y, z]', 'r0 + 5*t0*tw[t, x, y, z] + 4*tv[t, x, y, z]', 'r0']),
    # intersecting
    pytest.mark.xfail((['Eq(tu, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                       ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'])),
])
def test_common_subexprs_elimination(tu, tv, tw, ti0, ti1, t0, t1, exprs, expected):
    make = lambda i: Scalar(name='r%d' % i).indexify()
    processed = common_subexprs_elimination(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1),
                                            make)
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@skipif_yask
@pytest.mark.parametrize('exprs,expected', [
    (['Eq(t0, 3.)', 'Eq(t1, 7.)', 'Eq(ti0, t0*3. + 2.)', 'Eq(ti1, t1 + t0 + 1.5)',
      'Eq(tv, (ti0 + ti1)*t0)', 'Eq(tw, (ti0 + ti1)*t1)',
      'Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     '{tu: {tu, tv, tw, ti0, ti1, t0, t1}, tv: {ti0, ti1, t0, tv},\
tw: {ti0, ti1, t1, tw}, ti0: {ti0, t0}, ti1: {ti1, t1, t0}, t0: {t0}, t1: {t1}}'),
])
def test_graph_trace(tu, tv, tw, ti0, ti1, t0, t1, exprs, expected):
    g = TemporariesGraph(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1))
    mapper = eval(expected)
    for i in [tu, tv, tw, ti0, ti1, t0, t1]:
        assert set([j.lhs for j in g.trace(i)]) == mapper[i]


@skipif_yask
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
    pytest.mark.xfail((['Eq(t0, 1)', 'Eq(t1, fa[fb[t0]] + fb[x])'],
                      '{t0: True, t1: False}')),
])
def test_graph_isindex(fa, fb, fc, t0, t1, t2, exprs, expected):
    g = TemporariesGraph(EVAL(exprs, fa, fb, fc, t0, t1, t2))
    mapper = eval(expected)
    for k, v in mapper.items():
        assert g.is_index(k) == v


@skipif_yask
@pytest.mark.parametrize('expr,expected', [
    ('2*fa[x] + fb[x]', '2*fa[x] + fb[x]'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('fa[x]**2 + fb[x]**3', 'fa[x]*fa[x] + fb[x]*fb[x]*fb[x]'),
    ('3*fa[x]**4', '3*(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('1/(fa[x]**2)', 'fa[x]**(-2)'),
    ('1/(fa[x] + fb[x])', '1/(fa[x] + fb[x])'),
    ('3*sin(fa[x])**2', '3*(sin(fa[x])*sin(fa[x]))'),
])
def test_pow_to_mul(fa, fb, expr, expected):
    assert str(pow_to_mul(eval(expr))) == expected


@skipif_yask
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


@skipif_yask
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
