from devito.dimension import Dimension, time
from devito.pointdata import PointData
from devito.logger import error

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['PointSource', 'Receiver', 'Shot', 'RickerSource']


class PointSource(PointData):
    """Symbolic data object for a set of sparse points sources

    :param name: Name of the symbol representing this source
    :param coordinates: Point coordinates for this source
    :param data: (Optional) Data values to initialise point data
    :param ntime: (Optional) Number of timesteps for which to allocate data
    :param npoint: (Optional) Number of sparse points represented by this source
    :param dimension: :(Optional) class:`Dimension` object for
                       representing the number of points in this source

    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """

    def __new__(cls, name, ntime=None, npoint=None, ndim=None,
                data=None, coordinates=None, **kwargs):
        p_dim = kwargs.get('dimension', Dimension('p_%s' % name))
        ndim = ndim or coordinates.shape[1]
        npoint = npoint or coordinates.shape[0]
        if data is None:
            if ntime is None:
                error('Either data or ntime are required to'
                      'initialise source/receiver objects')
        else:
            ntime = ntime or data.shape[0]

        # Create the underlying PointData object
        obj = PointData.__new__(cls, name=name, dimensions=[time, p_dim],
                                npoint=npoint, nt=ntime, ndim=ndim,
                                coordinates=coordinates, **kwargs)

        # If provided, copy initial data into the allocated buffer
        if data is not None:
            obj.data[:] = data
        return obj

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PointSource, self).__init__(*args, **kwargs)


Receiver = PointSource
Shot = PointSource


class RickerSource(PointSource):
    """
    Symbolic object to encapsulate a set of sources with a
    pre-defined Ricker wavelet.

    :param name: Name for the reuslting symbol
    :param f0: Peak frequency for Ricker wavelet
    :param time: Discretized values of time
    :param ndim: Number of spatial dimensions
    """

    def __new__(cls, *args, **kwargs):
        time = kwargs.get('time')
        f0 = kwargs.get('f0')
        npoint = kwargs.get('npoint', 1)
        kwargs['ntime'] = len(time)
        kwargs['npoint'] = npoint
        obj = PointSource.__new__(cls, *args, **kwargs)
        for p in range(npoint):
            obj.data[:, p] = obj.wavelet(f0, time)
        return obj

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(RickerSource, self).__init__(*args, **kwargs)

    def wavelet(self, f0, t):
        """
        Create Ricker wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency
        :param t: Discretized values of time
        """
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)

    def show(self, f, t):
        """
        Plot the signal of the specified source data.
        """
        plt.figure()
        plt.plot(t, f)
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (km/s)')
        plt.tick_params()
        plt.show()
