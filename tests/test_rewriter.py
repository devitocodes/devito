import numpy as np

from examples.acoustic.Acoustic_codegen import Acoustic_cg
from examples.containers import IGrid, IShot
from examples.tti.tti_example import setup
from examples.tti.tti_operators import ForwardOperator


# Acoustic


def run_acoustic_forward(cse=False):
    dimensions = (50, 50, 50)
    model = IGrid()
    model0 = IGrid()
    model1 = IGrid()
    model.shape = dimensions
    model0.shape = dimensions
    model1.shape = dimensions
    origin = (0., 0.)
    spacing = (20., 20.)

    # True velocity
    true_vp = np.ones(dimensions) + 2.0
    true_vp[:, :, int(dimensions[0] / 2):int(dimensions[0])] = 4.5

    model.create_model(origin, spacing, true_vp)

    # Define seismic data.
    data = IShot()

    f0 = .010
    dt = model.get_critical_dt()
    t0 = 0.0
    tn = 250.0
    nt = int(1+(tn-t0)/dt)

    # Source
    t = np.linspace(t0, tn, nt)
    r = (np.pi * f0 * (t - 1./f0))
    time_series = (1-2.*r**2)*np.exp(-r**2)
    location = (origin[0] + dimensions[0] * spacing[0] * 0.5, 500,
                origin[1] + 2 * spacing[1])
    data.set_source(time_series, dt, location)

    receiver_coords = np.zeros((101, 3))
    receiver_coords[:, 0] = np.linspace(50, 950, num=101)
    receiver_coords[:, 1] = 500
    receiver_coords[:, 2] = location[2]
    data.set_receiver_pos(receiver_coords)
    data.set_shape(nt, 101)
    acoustic = Acoustic_cg(model, data, cse=cse)
    rec, u, _, _, _ = acoustic.Forward(save=False, cse=cse)

    return rec, u


def test_acoustic_rewrite_basic():
    output1 = run_acoustic_forward(cse=True)
    output2 = run_acoustic_forward(cse=False)

    assert np.allclose(output1[0], output2[0], 10e-7)
    assert np.allclose(output1[1].data, output2[1].data, 10e-7)


# TTI

def tti_operator(cse=False):
    problem = setup(dimensions=(16, 16, 16), time_order=2, space_order=2, tn=10.0)
    handle = ForwardOperator(problem.model, problem.src, problem.damp,
                             problem.data, time_order=problem.t_order,
                             spc_order=problem.s_order, save=False,
                             cache_blocking=None, cse=cse)
    return handle


def test_tti_rewrite_basic():
    output1 = tti_operator(cse=False).apply()
    output2 = tti_operator(cse='basic').apply()

    assert np.allclose(output1[0].data, output2[0].data, rtol=10e-6)
    assert np.allclose(output1[1].data, output2[1].data, rtol=10e-6)


def test_tti_rewrite_advanced():
    output1 = tti_operator(cse=False).apply()
    output2 = tti_operator(cse='advanced').apply()

    assert np.allclose(output1[0].data, output2[0].data, rtol=10e-6)
    assert np.allclose(output1[1].data, output2[1].data, rtol=10e-6)
