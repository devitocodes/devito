from scipy import signal
from scipy.interpolate import CubicSpline
from devito import Dimension
from devito.function import SparseTimeFunction

from cached_property import cached_property

import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None

__all__ = ['PointSource', 'Receiver', 'Shot', 'WaveletSource',
           'RickerSource', 'GaborSource', 'TimeAxis']


class TimeAxis(object):
    """ Data object to store the time axis. Exactly three of the four key arguments
        must be prescribed. Because of remainder values it is not possible to create
        a time axis that exactly adhears to the inputs therefore start, stop, step
        and num values should be taken from the TimeAxis object rather than relying
        upon the input values.

        The four possible cases are:
        start is None: start = step*(1 - num) + stop
        step is None: step = (stop - start)/(num - 1)
        num is None: num = ceil((stop - start + step)/step);
                     because of remainder stop = step*(num - 1) + start
        stop is None: stop = step*(num - 1) + start

    :param start:(Optional) Start of time axis.
    :param step: (Optional) Time interval.
    :param: num: (Optional) Number of values (Note: this is the number of intervals + 1).
                 stop value is reset to correct for remainder.
    :param stop: (Optional) End time.
    """
    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step*(1 - num) + stop
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step)/step))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = step*(num - 1) + start
            else:
                raise ValueError("Only three of start, step, num and stop may be set")
        except:
            raise ValueError("Three of args start, step, num and stop may be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of type int")

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)


class PointSource(SparseTimeFunction):
    """Symbolic data object for a set of sparse point sources

    :param name: Name of the symbol representing this source.
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: Point coordinates for this source.
    :param time_range: :class:`TimeAxis` TimeAxis(start, step, num) object.
    :param data: (Optional) Data values to initialise point data.
    :param npoint: (Optional) Number of sparse points represented by this source.
    :param time_order: (Optional) Time discretization order (defaults to 2).
    :param dimension: :(Optional) class:`Dimension` object for
                       representing the number of points in this source.
    """

    def __new__(cls, name, grid, time_range, npoint=None,
                data=None, coordinates=None, **kwargs):
        p_dim = kwargs.get('dimension', Dimension(name='p_%s' % name))
        time_order = kwargs.get('time_order', 2)
        npoint = npoint or coordinates.shape[0]

        # Create the underlying SparseTimeFunction object
        obj = SparseTimeFunction.__new__(cls, name=name, grid=grid,
                                         dimensions=[grid.time_dim, p_dim],
                                         npoint=npoint, nt=time_range.num,
                                         time_order=time_order,
                                         coordinates=coordinates, **kwargs)

        obj._time_range = time_range._rebuild()

        # If provided, copy initial data into the allocated buffer
        if data is not None:
            obj.data[:] = data

        return obj

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PointSource, self).__init__(*args, **kwargs)

    @cached_property
    def time_values(self):
        return self._time_range.time_values

    @property
    def time_range(self):
        return self._time_range

    def resample(self, dt=None, num=None, rtol=1e-5):
        # Only one of dt or num may be set.
        if dt is None:
            assert num is not None
        else:
            assert num is None

        start, stop = self._time_range.start, self._time_range.stop
        dt0 = self._time_range.step

        if dt is None:
            new_time_range = TimeAxis(start=start, stop=stop, num=num)
            dt = new_time_range.step
        else:
            new_time_range = TimeAxis(start=start, stop=stop, step=dt)

        npad = int(np.ceil(np.log2(self._time_range.num)))
        for n in range(npad, 28):
            if abs(2**n*dt0/np.ceil(2**n*dt0/dt) - dt)/dt < rtol:
                npad = 2**n
                break

        # Create resampled data.
        npoint = self.coordinates.shape[0]
        new_data = np.zeros((new_time_range.num, npoint))
        scratch = np.zeros(npad)
        scratch_time_range = TimeAxis(start=start, step=self._time_range.step, num=npad)
        for i in range(npoint):
            scratch[0:self.data.shape[0]] = self.data[:, i]
            resample_num = int(round((scratch_time_range.stop -
                                      scratch_time_range.start)/dt))
            approx_data, t = signal.resample(scratch, resample_num,
                                             t=scratch_time_range.time_values)

            spline = CubicSpline(t, approx_data, extrapolate=True)

            new_data[:, i] = spline(new_time_range.time_values)

        # Return new object
        return PointSource(self.name, self.grid, data=new_data, time_range=new_time_range,
                           coordinates=self.coordinates.data)


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.

    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time_values: Discretized values of time in ms
    """

    def __new__(cls, *args, **kwargs):
        npoint = kwargs.get('npoint', 1)
        kwargs['npoint'] = npoint
        obj = PointSource.__new__(cls, *args, **kwargs)

        obj.f0 = kwargs.get('f0')
        for p in range(npoint):
            obj.data[:, p] = obj.wavelet(obj.f0, obj.time_values)
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

    def show(self, idx=0, wavelet=None):
        """
        Plot the wavelet of the specified source.

        :param idx: Index of the source point for which to plot wavelet
        :param wavelet: Prescribed wavelet instead of one from this symbol
        :param time: Prescribed time instead of time from this symbol
        """
        wavelet = wavelet or self.data[:, idx]
        plt.figure()
        plt.plot(self.time_values, wavelet)
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
