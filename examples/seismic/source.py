from functools import cached_property
from scipy import interpolate
import sympy
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    plt = None

from devito.types import SparseTimeFunction

__all__ = ['PointSource', 'Receiver', 'Shot', 'WaveletSource',
           'RickerSource', 'GaborSource', 'DGaussSource', 'TimeAxis']


class TimeAxis:
    """
    Data object to store the TimeAxis. Exactly three of the four key arguments
    must be prescribed. Because of remainder values, it is not possible to create
    a TimeAxis that exactly adheres to the inputs; therefore, start, stop, step
    and num values should be taken from the TimeAxis object rather than relying
    upon the input values.

    The four possible cases are:
    start is None: start = step*(1 - num) + stop
    step is None: step = (stop - start)/(num - 1)
    num is None: num = ceil((stop - start + step)/step);
                 because of remainder stop = step*(num - 1) + start
    stop is None: stop = step*(num - 1) + start

    Parameters
    ----------
    start : float, optional
        Start of time axis.
    step : float, optional
        Time interval.
    num : int, optional
        Number of values (Note: this is the number of intervals + 1).
        Stop value is reset to correct for remainder.
    stop : float, optional
        End time.
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

        self.start = float(start)
        self.stop = float(stop)
        self.step = float(step)
        self.num = int(num)

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

    Parameters
    ----------
    name : str
        Name of the symbol representing this source.
    grid : Grid
        The computational domain.
    time_range : TimeAxis
        TimeAxis(start, step, num) object.
    npoint : int, optional
        Number of sparse points represented by this source.
    data : ndarray, optional
        Data values to initialise point data.
    coordinates : ndarray, optional
        Point coordinates for this source.
    space_order : int, optional
        Space discretization order.
    time_order : int, optional
        Time discretization order (defaults to 2).
    dtype : data-type, optional
        Data type of the buffered data.
    dimension : Dimension, optional
        Represents the number of points in this source.
    """

    __rkwargs__ = list(SparseTimeFunction.__rkwargs__) + ['time_range']
    __rkwargs__.remove('nt')  # `nt` is inferred from `time_range`

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs['nt'] = kwargs['time_range'].num

        # Either `npoint` or `coordinates` must be provided
        npoint = kwargs.get('npoint', kwargs.get('npoint_global'))
        if npoint is None:
            coordinates = kwargs.get('coordinates', kwargs.get('coordinates_data'))
            if coordinates is None:
                raise TypeError("Need either `npoint` or `coordinates`")
            kwargs['npoint'] = coordinates.shape[0]

        return args, kwargs

    def __init_finalize__(self, *args, **kwargs):
        time_range = kwargs.pop('time_range')
        data = kwargs.pop('data', None)

        kwargs.setdefault('time_order', 2)
        super().__init_finalize__(*args, **kwargs)

        self._time_range = time_range._rebuild()

        # If provided, copy initial data into the allocated buffer
        if data is not None:
            self.data[:] = data

    @cached_property
    def time_values(self):
        return self._time_range.time_values

    @property
    def time_range(self):
        return self._time_range

    def resample(self, dt=None, num=None, rtol=1e-5, order=3):
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

        if np.isclose(dt, dt0):
            return self

        nsamples, ntraces = self.data.shape

        new_traces = np.zeros((new_time_range.num, ntraces))

        for i in range(ntraces):
            tck = interpolate.splrep(self._time_range.time_values,
                                     self.data[:, i], k=order)
            new_traces[:, i] = interpolate.splev(new_time_range.time_values, tck)

        # Return new object
        return PointSource(name=self.name, grid=self.grid, data=new_traces,
                           time_range=new_time_range, coordinates=self.coordinates.data)


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):

    """
    Base class for symbolic objects that encapsulates a set of
    sources with a pre-defined source signal wavelet.

    Parameters
    ----------
    name : str
        Name for the resulting symbol.
    grid : Grid
        The computational domain.
    f0 : float
        Peak frequency for Ricker wavelet in kHz.
    time_values : TimeAxis
        Discretized values of time in ms.
    a : float, optional
        Amplitude of the wavelet (defaults to 1).
    t0 : float, optional
        Firing time (defaults to 1 / f0)
    wavelet: str, optional
        The type of wavelet to generate one of
        {'gauss_soliton', 'dgauss', 'ricker', 'gabor'}
    """

    __rkwargs__ = PointSource.__rkwargs__ + ['f0', 'a', 't0']

    @classmethod
    def __args_setup__(cls, *args, **kwargs):
        kwargs.setdefault('npoint', 1)

        return super().__args_setup__(*args, **kwargs)

    def __init_finalize__(self, *args, **kwargs):
        super().__init_finalize__(*args, **kwargs)

        self.f0 = kwargs.get('f0')
        self.a = kwargs.get('a')
        self.t0 = kwargs.get('t0')
        self.wavelet_type = kwargs.get('wavelet')
        self.wavelet_kwargs = {}

        if isinstance(self.wavelet_type, str):
            if self.wavelet_type == 'dgauss':
                self.wavelet_kwargs['n'] = kwargs.get('n', 1)

            if self.wavelet_type == 'gabor':
                self.wavelet_kwargs['gamma'] = kwargs.get('gamma', 1)
                self.wavelet_kwargs['phi'] = kwargs.get('phi', 0)

        if not self.alias:
            for p in range(kwargs['npoint']):
                self.data[:, p] = self.wavelet

    @property
    def wavelet(self):
        """
        Return a wavelet with a peak frequency ``f0`` at time ``t0``.
        """
        if isinstance(self.wavelet_type, str):
            return wavelet[self.wavelet_type](
                self.time_values,
                self.f0,
                1 if self.a is None else self.a,
                self.t0,
                **self.wavelet_kwargs
            )
        elif any(self.wavelet_type):
            return self.wavelet_type
        else:
            raise NotImplementedError('Wavelet not defined')

    def show(self, idx=0, wavelet=None, ax=plt):
        """
        Plot the wavelet of the specified source.

        Parameters
        ----------
        idx : int
            Index of the source point for which to plot wavelet.
        wavelet : ndarray or callable
            Prescribed wavelet instead of one from this symbol.
        """
        wavelet = wavelet or self.data[:, idx]
        if ax == plt:
            ax.figure()
        lines = ax.plot(self.time_values, wavelet)
        if ax == plt:
            ax.xlabel('Time (ms)')
            ax.ylabel('Amplitude')
        else:
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude')
        ax.tick_params()
        if ax == plt:
            ax.show()
        return lines


# Define parameters
_t, _t0, b, _n, _C = sympy.symbols('t t_0 b n C')
_A, _f, _gamma, _phi = sympy.symbols('A f 𝛾 𝜙')
_gauss = sympy.exp(-(b*_t)**2)


def dgauss(n=1):
    """
    Return a symbolic expression for the n-th derivative of a Gaussian
    """
    dngauss = sympy.diff(_gauss, (_t, n))
    two = sympy.Integer(2)
    half = sympy.Integer(1)/sympy.Integer(2)

    if n == 0:
        scale = 1
    elif n == 1:
        scale = sympy.sqrt(two)*b*sympy.exp(-half)
    elif n == 2:
        scale = 2*b**2
    else:
        scale = _C

    dngauss /= scale
    return _A*dngauss.subs(
        {_t: _t - _t0, b: sympy.pi*_f*sympy.sqrt(two/sympy.Integer(n))}
    )


def _scale_dgauss(f, n):
    """
    Return the scaling of n-th derivative of Gaussian at frequency f, for n =>3.
    For n<3 the expression returned by `dgauss` is already scaled symbolically.
    In all but the smallest cases the scaling factors are too cumbersome to
    derive symbolically so they are calculated numerically.
    """
    dngauss = sympy.diff(_gauss, (_t, n))
    y = sympy.lambdify((_t, b), dngauss)
    if n < 3:
        # These cases are handled analytically
        scale = 1
    if n % 2 == 0:
        # Even derivatives attain their min/max at 0
        scale = np.abs(y(0, np.pi*f*np.sqrt(2/n)))
    else:
        # Odd derivatives we have to sample to approximate the maximum
        # o/w we need to do very hard maths to work out where it is
        xs = np.linspace(0, np.pi/(2*f), 101)
        ys = y(xs, np.pi*f*np.sqrt(2/n))
        scale = np.max(np.abs(ys))
    return scale


# Define the Gaussian soliton
gauss = _A*_gauss.subs({_t: _t - _t0})
# Define the various wavelets
# Ricker
#  As defined in: https://wiki.seg.org/wiki/Dictionary:Ricker_wavelet
ricker = -dgauss(n=2)
# Gabor
#  As defined in: https://wiki.seg.org/wiki/Dictionary:Gabor_wavelet
gabor = _A*_gauss.subs({_t: _t - _t0, b: sympy.pi*_f/_gamma})
gabor *= sympy.cos(2*b*_t + _phi).subs({_t: _t - _t0, b: sympy.pi*_f})

# Dictionary of wavelet functions
wavelet = {
    'gauss_soliton': lambda t, f, A=1, t0=None:
        sympy.lambdify((_t, b, _A, _t0), gauss)(
            t, f, A, 3/(f*np.sqrt(2)) if t0 is None else t0
        ),
    # Note: The offset grows with O(n^1/2) just like the upperbound [Szego (1939) pg. 132]
    # for the largest root of the n-th Hermite polynomial
    'dgauss': lambda t, f, A=1, t0=None, n=1:
        sympy.lambdify((_t, _f, _A, _t0, _C), dgauss(n))(
            t, f, A, np.sqrt(n/2)/f if t0 is None else t0, _scale_dgauss(f, n)
        ),
    'ricker': lambda t, f, A=1, t0=None:
        sympy.lambdify((_t, _f, _A, _t0), ricker)(
            t, f, A, 1/f if t0 is None else t0
        ),
    'gabor': lambda t, f, A=1, t0=None, gamma=1, phi=0:
        sympy.lambdify((_t, _f, _A, _t0, _gamma, _phi), gabor)(
            t, f, A, (3*gamma)/(f*np.pi*np.sqrt(2)) if t0 is None else t0, gamma, phi
        )
}


# Legacy classes for backwards compatibility
def RickerSource(**kwargs):
    """
    Legacy constructor do not use
    """
    return WaveletSource(**kwargs, wavelet='ricker')


def DGaussSource(**kwargs):
    """
    Legacy constructor do not use
    """
    f0 = np.sqrt(kwargs.pop('a', 1)/(2*np.pi**2))
    t0_fallback = 1/kwargs.pop('f0')
    t0 = kwargs.pop('t0', t0_fallback)
    a = 2*np.pi*f0*np.exp(-1/2)
    return WaveletSource(f0=f0, a=a, t0=t0, wavelet='dgauss', **kwargs)


def GaborSource(**kwargs):
    """
    Legacy constructor do not use
    """
    f0 = kwargs.pop('f0')/2
    gamma = np.pi/np.sqrt(2)
    t0 = kwargs.pop('t0', 1.5/f0)
    return WaveletSource(f0=f0, gamma=gamma, t0=t0, wavelet='gabor', **kwargs)
