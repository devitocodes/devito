import numpy as np

from devito.symbolics import Rewriter

from examples.tti.tti_example import setup
from examples.tti.tti_operators import ForwardOperator


def operator(rewrite):
    problem = setup(dimensions=(16, 16, 16), time_order=2, space_order=2, tn=10.0,
                    cse=rewrite, auto_tuning=False, cache_blocking=None)
    operator = ForwardOperator(problem.model, problem.src, problem.damp,
                               problem.data, time_order=problem.t_order,
                               spc_order=problem.s_order, save=False,
                               cache_blocking=None, cse=False)
    return operator


def test_tti_rewrite_output():
    output1 = operator(False).apply()
    output2 = operator(False).apply()

    for o1, o2 in zip(output1, output2):
        assert np.isclose(np.linalg.norm(o1.data - o2.data), 0.0)
