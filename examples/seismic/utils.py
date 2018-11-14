from cached_property import cached_property
from scipy import ndimage

from devito import error

from .source import *
__all__ = ['scipy_smooth', 'AcquisitionGeometry']


def scipy_smooth(img, sigma=5):
    """
    Smooth the input with scipy ndimage utility
    """
    return ndimage.gaussian_filter(img, sigma=sigma)


class AcquisitionGeometry(object):
    """
    Encapsulate the geometry of an acquisition:
    - receiver positions and number
    - source positions and number

    In practice this would only point to a segy file with the
    necessary information
    """

    def __init__(self, model, rec_pos, src_pos, t0, tn, **kwargs):
        """
        In practice woyuld be __init__(segyfile) and all below parameters
        would come from a segy_read (at property call rather than at init)
        """
        self.rec_positions = rec_pos
        self._nrec = rec_pos.shape[0]
        self.src_positions = src_pos
        self._nsrc = src_pos.shape[0]
        self._src_type = kwargs.get('src_type')
        assert self.src_type in sources
        self._f0 = kwargs.get('f0')
        if self._src_type is not None and self._f0 is None:
            error("Peak frequency must be provided in KH" +
                  " for source of type %s" % self._src_type)

        self._grid = model.grid
        self._dt = model.critical_dt
        self._t0 = t0
        self._tn = tn

    def resample(self, dt):
        self._dt = dt
        return self

    @property
    def time_axis(self):
        return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

    @cached_property
    def src_type(self):
        return self._src_type

    @cached_property
    def grid(self):
        return self._grid

    @cached_property
    def f0(self):
        return self._f0

    @cached_property
    def tn(self):
        return self._tn

    @cached_property
    def t0(self):
        return self._t0

    @cached_property
    def dt(self):
        return self._dt

    @cached_property
    def nt(self):
        return self.time_axis.num

    @property
    def nrec(self):
        return self._nrec

    @property
    def nsrc(self):
        return self._nsrc

    @cached_property
    def dtype(self):
        return self.grid.dtype

    @property
    def rec(self):
        return Receiver(name='rec', grid=self.grid,
                        time_range=self.time_axis, npoint=self.nrec,
                        coordinates=self.rec_positions)

    @property
    def src(self):
        if self.src_type is None:
            return PointSource(name='src', grid=self.grid,
                               time_range=self.time_axis, npoint=self.nsrc,
                               coordinates=self.src_positions)
        else:
            return sources[self.src_type](name='src', grid=self.grid, f0=self.f0,
                                          time_range=self.time_axis, npoitn=self.nsrc,
                                          coordinates=self.src_positions)

    _pickle_args = ['model', 'rec_positions', 'src_positions', 't0', 'tn']
    _pickle_kwargs = ['src_type', 'f0']


sources = {'Wavelet': WaveletSource, 'Ricker': RickerSource, 'Gabor': GaborSource}
