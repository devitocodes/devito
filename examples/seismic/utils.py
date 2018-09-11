import numpy as np
from scipy import ndimage

from devito import Operator, Eq
__all__ = ['smooth', 'scipy_smooth']


# Velocity models
def smooth(dest, f):
    """
    Run an n-point moving average kernel over ``f`` and store the result
    into ``dest``. The average is computed along the innermost ``f`` dimension.
    """
    if f.is_Constant:
        # Return a scaled version of the input if it's a Constant
        dest.data[:] = .9 * f.data
    else:
        Operator(Eq(dest, f.avg(dims=f.dimensions[-1])), name='smoother').apply()


def scipy_smooth(img, sigma=5):
    """
    Smooth the input with scipy ndimage utility
    """
    return ndimage.gaussian_filter(img, sigma=sigma)
