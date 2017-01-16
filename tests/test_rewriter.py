import numpy as np

import pytest

from devito.dse.graph import temporaries_graph
from devito.dse.symbolics import rewrite

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
from examples.tti.tti_example import setup
from examples.tti.tti_operators import ForwardOperator


# Acoustic


def run_acoustic_forward(dse=None):
    dimensions = (50, 50, 50)
    origin = (0., 0., 0.)
    spacing = (10., 10., 10.)

    # True velocity
    true_vp = np.ones(dimensions) + 2.0
    true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

    model = IGrid(origin, spacing, true_vp)

    # Define seismic data.
    data = IShot()
    src = IShot()
    f0 = .010
    dt = model.get_critical_dt()
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

    assert np.allclose(output1[0], output2[0], atol=10e-6)
    assert np.allclose(output1[1].data, output2[1].data, atol=10e-6)


# TTI

def tti_operator(dse=False):
    problem = setup(dimensions=(50, 50, 50), time_order=2,
                    space_order=4, tn=250.0, nbpml=10)
    handle = ForwardOperator(problem.model, problem.src, problem.damp,
                             problem.data, time_order=problem.t_order,
                             spc_order=problem.s_order, save=False,
                             cache_blocking=None, dse=dse)
    return handle


@pytest.fixture(scope="session")
def tti_nodse():
    # FIXME: note that np.copy is necessary because of the broken caching system
    output = tti_operator(dse=None).apply()
    return (np.copy(output[0].data), np.copy(output[1].data))


def test_tti_rewrite_temporaries_graph():
    operator = tti_operator()
    handle = rewrite(operator.stencils, mode='basic')

    graph = temporaries_graph(handle.exprs)

    assert len([v for v in graph.values() if v.is_terminal]) == 2  # u and v
    assert len(graph) == len(handle.exprs)
    assert all(v.reads or v.readby for v in graph.values())


def test_tti_rewrite_basic(tti_nodse):
    output = tti_operator(dse='basic').apply()

    assert np.allclose(tti_nodse[0], output[0].data, atol=10e-3)
    assert np.allclose(tti_nodse[1], output[1].data, atol=10e-3)


def test_tti_rewrite_factorizer(tti_nodse):
    output = tti_operator(dse=('basic', 'factorize')).apply()

    assert np.allclose(tti_nodse[0], output[0].data, atol=10e-3)
    assert np.allclose(tti_nodse[1], output[1].data, atol=10e-3)


def test_tti_rewrite_trigonometry(tti_nodse):
    output = tti_operator(dse=('basic', 'approx-trigonometry')).apply()

    assert np.allclose(tti_nodse[0], output[0].data, atol=10e-1)
    assert np.allclose(tti_nodse[1], output[1].data, atol=10e-1)


def test_tti_rewrite_advanced(tti_nodse):
    output = tti_operator(dse='advanced').apply()

    assert np.allclose(tti_nodse[0], output[0].data, atol=10e-1)
    assert np.allclose(tti_nodse[1], output[1].data, atol=10e-1)
