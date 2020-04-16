import numpy as np
import pytest

from devito.logger import info
from examples.seismic.skew_self_adjoint import *

class TestUtils(object):

    @pytest.mark.parametrize('shape', [(200, 200), (200, 200, 200)])
    @pytest.mark.parametrize('npad', [50, ])
    @pytest.mark.parametrize('qmin', [0.1, 1.0])
    @pytest.mark.parametrize('qmax', [10.0, 100.0])
    @pytest.mark.parametrize('sigma', [None, 11])
    def test_setupWOverQ(self, shape, npad, qmin, qmax, sigma):
        """
        Test for the function that sets up the w/Q attenuation model.
        This is not a correctness test, we just ensure that the output w/Q model 
        had value w/Qmax in the interior, w/Qmin at the edge of the model, and 
        monotonically intermediate value in between.
        """
        print("test_setupWOverQ(shape, npad, qmin, qmax, sigma)", 
             shape, npad, qmin, qmax, sigma)

#         assert np.isclose((term1 - term2)/term1, 0., atol=1.e-12)