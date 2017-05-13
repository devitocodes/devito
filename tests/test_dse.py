import numpy as np
import pytest
from sympy import Eq, Symbol  # noqa

from devito.dse import (clusterize, rewrite, xreplace_constrained, iq_timeinvariant,
                        iq_timevarying, estimate_cost, temporaries_graph,
                        common_subexprs_elimination, collect_aliases)
from devito import Dimension, x, y, z, time, TimeData, clear_cache  # noqa
from devito.nodes import Expression
from devito.stencil import Stencil
from devito.visitors import FindNodes
from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IShot
from examples.seismic import Model
from examples.source_type import SourceLike
from examples.tti.tti_example import setup
from examples.tti.tti_operators import ForwardOperator


# Acoustic


def run_acoustic_forward(dse=None):
    # TODO: temporary work around to issue #225 on GitHub
    clear_cache()

    dimensions = (50, 50, 50)
    origin = (0., 0., 0.)
    spacing = (10., 10., 10.)
    nbpml = 10

    # True velocity
    true_vp = np.ones(dimensions) + 2.0
    true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

    model = Model(origin, spacing, dimensions, true_vp, nbpml=nbpml)

    # Define seismic data.
    data = IShot()
    src = IShot()
    f0 = .010
    dt = model.critical_dt
    t0 = 0.0
    tn = 250.0
    nt = int(1+(tn-t0)/dt)

    t = np.linspace(t0, tn, nt)
    r = (np.pi * f0 * (t - 1./f0))
    # Source geometry
    time_series = np.zeros((nt, 1))
    time_series[:, 0] = (1-2.*r**2)*np.exp(-r**2)

    location = np.zeros((1, 3))
    location[0, 0] = origin[0] + dimensions[0] * spacing[0] * 0.5
    location[0, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    location[0, 2] = origin[1] + 2 * spacing[2]
    src.set_receiver_pos(location)
    src.set_shape(nt, 1)
    src.set_traces(time_series)

    # Receiver geometry
    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(0, origin[0] +
                                        dimensions[0] * spacing[0], num=101)
    receiver_coords[:, 1] = origin[1] + dimensions[1] * spacing[1] * 0.5
    receiver_coords[:, 2] = location[0, 1]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)
    acoustic = Acoustic_cg(model, data, src, dse=dse)
    rec, u, _, _, _ = acoustic.Forward(save=False, dse=dse)

    return rec, u


def test_acoustic_rewrite_basic():
    output1 = run_acoustic_forward(dse=None)
    output2 = run_acoustic_forward(dse='basic')

    assert np.allclose(output1[0], output2[0], atol=10e-5)
    assert np.allclose(output1[1].data, output2[1].data, atol=10e-5)


# TTI

def tti_operator(dse=False):
    problem = setup(dimensions=(50, 50, 50), time_order=2,
                    space_order=4, tn=250.0)
    nt, nrec = problem.data.shape
    nsrc = problem.source.shape[1]
    ndim = len(problem.model.shape)
    dt = problem.dt
    h = problem.model.get_spacing()
    dtype = problem.model.dtype
    nbpml = problem.model.nbpml

    u = TimeData(name="u", shape=problem.model.shape_domain,
                 time_dim=nt, time_order=2, space_order=2, dtype=dtype)
    v = TimeData(name="v", shape=problem.model.shape_domain,
                 time_dim=nt, time_order=2, space_order=2, dtype=dtype)

    # Create source symbol
    p_src = Dimension('p_src', size=nsrc)
    src = SourceLike(name="src", dimensions=[time, p_src], npoint=nsrc, nt=nt,
                     dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                     coordinates=problem.source.receiver_coords)
    src.data[:] = .5 * problem.source.traces[:]

    # Create receiver symbol
    p_rec = Dimension('p_rec', size=nrec)
    rec = SourceLike(name="rec", dimensions=[time, p_rec], npoint=nrec, nt=nt,
                     dt=dt, h=h, ndim=ndim, nbpml=nbpml, dtype=dtype,
                     coordinates=problem.data.receiver_coords)

    handle = ForwardOperator(problem.model, u, v, src, rec,
                             problem.data, time_order=problem.t_order,
                             spc_order=problem.s_order, save=False,
                             cache_blocking=None, dse=dse, dle='basic')
    return handle, v, rec


@pytest.fixture(scope="session")
def tti_nodse():
    # FIXME: note that np.copy is necessary because of the broken caching system
    operator, v, rec = tti_operator(dse=None)
    operator.apply()
    return (np.copy(v.data), np.copy(rec.data))


def test_tti_clusters_to_graph():
    operator, _, _ = tti_operator()

    nodes = FindNodes(Expression).visit(operator.elemental_functions)
    expressions = [n.expr for n in nodes]
    stencils = operator._retrieve_stencils(expressions)
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


def test_tti_rewrite_basic(tti_nodse):
    operator, v, rec = tti_operator(dse='basic')
    operator.apply()

    assert np.allclose(tti_nodse[0], v.data, atol=10e-3)
    assert np.allclose(tti_nodse[1], rec.data, atol=10e-3)


def test_tti_rewrite_factorizer(tti_nodse):
    operator, v, rec = tti_operator(dse=('basic', 'factorize'))
    operator.apply()

    assert np.allclose(tti_nodse[0], v.data, atol=10e-3)
    assert np.allclose(tti_nodse[1], rec.data, atol=10e-3)


def test_tti_rewrite_trigonometry(tti_nodse):
    operator, v, rec = tti_operator(dse=('basic', 'approx-trigonometry'))
    operator.apply()

    assert np.allclose(tti_nodse[0], v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1], rec.data, atol=10e-1)


def test_tti_rewrite_advanced(tti_nodse):
    operator, v, rec = tti_operator(dse='advanced')
    operator.apply()

    assert np.allclose(tti_nodse[0], v.data, atol=10e-1)
    assert np.allclose(tti_nodse[1], rec.data, atol=10e-1)


# DSE manipulation

@pytest.mark.parametrize('expr,expected', [
    # simple
    ('Eq(u, ti0 + ti1 + 5.)',
     ['ti0[x, y, z] + ti1[x, y, z]']),
    # more ops
    ('Eq(u, (ti0*ti1*t0) + (ti1*v) + (t1 + ti1)*w)',
     ['t1 + ti1[x, y, z]', 't0*ti0[x, y, z]*ti1[x, y, z]']),
    # wrapped
    ('Eq(u, ((ti0*ti1*t0)*v + (ti0*ti1*v)*t1))',
     ['t0*ti0[x, y, z]*ti1[x, y, z]', 't1*ti0[x, y, z]*ti1[x, y, z]']),
])
def test_xreplace_constrained_time_invariants(u, v, w, ti0, ti1, t0, t1, expr, expected):
    exprs = [eval(expr)]
    processed, found = xreplace_constrained(exprs,
                                            lambda i: Symbol('r%d' % i),
                                            iq_timeinvariant(temporaries_graph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('expr,expected', [
    # simple
    ('Eq(u, v + w + 5. + ti0)',
     ['v[t, x, y, z] + w[t, x, y, z] + 5.0']),
    # more ops
    ('Eq(u, v*w*4.*ti0 + ti1*v)',
     ['4.0*v[t, x, y, z]*w[t, x, y, z]']),
    # wrapped
    ('Eq(u, ((v + 4.)*ti0*ti1 + (v + w)/3.)*ti1*t0)',
     ['v[t, x, y, z] + 4.0',
      '0.333333333333333*v[t, x, y, z] + 0.333333333333333*w[t, x, y, z]']),
])
def test_xreplace_constrained_time_varying(u, v, w, ti0, ti1, t0, t1, expr, expected):
    exprs = [eval(expr)]
    processed, found = xreplace_constrained(exprs,
                                            lambda i: Symbol('r%d' % i),
                                            iq_timevarying(temporaries_graph(exprs)),
                                            lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(u, (v + w + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(v[t, x, y, z] + w[t, x, y, z] + 5.0)']),
    # across expressions
    (['Eq(u, v*4 + w*5 + w*5*t0)', 'Eq(v, w*5)'],
     ['5*w[t, x, y, z]', '5*t0*w[t, x, y, z] + r0 + 4*v[t, x, y, z]', 'r0']),
    # intersecting
    pytest.mark.xfail((['Eq(u, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                       ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'])),
])
def test_common_subexprs_elimination(u, v, w, ti0, ti1, t0, t1, exprs, expected):
    processed = common_subexprs_elimination([eval(i) for i in exprs],
                                            lambda i: Symbol('r%d' % i))
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@pytest.mark.parametrize('exprs,expected', [
    (['Eq(t0, 3.)', 'Eq(t1, 7.)', 'Eq(ti0, t0*3. + 2.)', 'Eq(ti1, t1 + t0 + 1.5)',
      'Eq(v, (ti0 + ti1)*t0)', 'Eq(w, (ti0 + ti1)*t1)',
      'Eq(u, (v + w + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     '{u: {u, v, w, ti0, ti1, t0, t1}, v: {ti0, ti1, t0, v}, w: {ti0, ti1, t1, w},\
ti0: {ti0, t0}, ti1: {ti1, t1, t0}, t0: {t0}, t1: {t1}}'),
])
def test_graph_trace(u, v, w, ti0, ti1, t0, t1, exprs, expected):
    g = temporaries_graph([eval(i) for i in exprs])
    mapper = eval(expected)
    for i in [u, v, w, ti0, ti1, t0, t1]:
        assert set([j.lhs for j in g.trace(i)]) == mapper[i]


@pytest.mark.parametrize('exprs,expected', [
    # none (different distance)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x])'],
     {}),
    # none (different dimension)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] + fb[y])'],
     {}),
    # none (different operation)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] - fb[x])'],
     {}),
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
     {'fc[x,y] + fd[x+1,y+2]': Stencil([(x, {0, 1, 2}), (y, {0, 1, 2})]),
      '3.*fc[x-4,y-4] + fd[x,y]': Stencil([(x, {0, 1, 2}), (y, {0, 1, 2})])}),
])
def test_collect_aliases(fa, fb, fc, fd, t0, t1, t2, t3, exprs, expected):
    mapper = dict([(eval(k), v) for k, v in expected.items()])
    _, aliases = collect_aliases([eval(i) for i in exprs])
    for k, v in aliases.items():
        assert k in mapper
        assert v.anti_stencil == mapper[k]
