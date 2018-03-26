from devito import Dimension
from devito.function import SparseTimeFunction
from devito.logger import error

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None

__all__ = ['PointSource', 'Receiver', 'Shot', 'RickerSource', 'GaborSource']


class PointSource(SparseTimeFunction):
    """Symbolic data object for a set of sparse point sources

    :param name: Name of the symbol representing this source
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: Point coordinates for this source
    :param data: (Optional) Data values to initialise point data
    :param ntime: (Optional) Number of timesteps for which to allocate data
    :param npoint: (Optional) Number of sparse points represented by this source
    :param time_order: (Optional) Time discretization order (defaults to 2)
    :param dimension: :(Optional) class:`Dimension` object for
                       representing the number of points in this source

    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """

    def __new__(cls, name, grid, ntime=None, npoint=None, data=None,
                coordinates=None, **kwargs):
        p_dim = kwargs.get('dimension', Dimension(name='p_%s' % name))
        time_order = kwargs.get('time_order', 2)
        npoint = npoint or coordinates.shape[0]
        if data is None:
            if ntime is None:
                error('Either data or ntime are required to'
                      'initialise source/receiver objects')
        else:
            ntime = ntime or data.shape[0]

        # Create the underlying SparseTimeFunction object
        obj = SparseTimeFunction.__new__(cls, name=name, grid=grid,
                                         dimensions=[grid.time_dim, p_dim],
                                         npoint=npoint, nt=ntime, time_order=time_order,
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


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def __new__(cls, *args, **kwargs):
        time = kwargs.get('time')
        npoint = kwargs.get('npoint', 1)
        kwargs['ntime'] = len(time)
        kwargs['npoint'] = npoint
        obj = PointSource.__new__(cls, *args, **kwargs)

        obj.time = time
        obj.f0 = kwargs.get('f0')
        for p in range(npoint):
            obj.data[:, p] = obj.wavelet(obj.f0, obj.time)
        return obj

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(WaveletSource, self).__init__(*args, **kwargs)

    def wavelet(self, f0, t):
        """
        Defines a wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        raise NotImplementedError('Wavelet not defined')

    def show(self, idx=0, time=None, wavelet=None):
        """
        Plot the wavelet of the specified source.

        :param idx: Index of the source point for which to plot wavelet
        :param wavelet: Prescribed wavelet instead of one from this symbol
        :param time: Prescribed time instead of time from this symbol
        """
        wavelet = wavelet or self.data[:, idx]
        time = time or self.time
        plt.figure()
        plt.plot(time, wavelet)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.tick_params()
        plt.show()


class RickerSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:

    http://subsurfwiki.org/wiki/Ricker_wavelet

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Ricker wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)


class GaborSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Gabor wavelet:

    https://en.wikipedia.org/wiki/Gabor_wavelet

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Gabor wavelet with a peak frequency f0 at time t.

        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        agauss = 0.5 * f0
        tcut = 1.5 / agauss
        s = (t-tcut) * agauss
        return np.exp(-2*s**2) * np.cos(2 * np.pi * s)
