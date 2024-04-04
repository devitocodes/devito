import numpy as np
from argparse import Action, ArgumentError, ArgumentParser

from devito import error, configuration, warning
from devito.tools import Pickable
from devito.types.sparse import _default_radius

from .source import *

__all__ = ['AcquisitionGeometry', 'setup_geometry', 'seismic_args']


def setup_geometry(model, tn, f0=0.010, interpolation='linear', **kwargs):
    # Source and receiver geometries
    src_coordinates = np.empty((1, model.dim))
    if model.dim > 1:
        src_coordinates[0, :] = np.array(model.domain_size) * .5
        src_coordinates[0, -1] = model.origin[-1] + model.spacing[-1]
    else:
        src_coordinates[0, 0] = 2 * model.spacing[0]

    rec_coordinates = setup_rec_coords(model)

    r = kwargs.get('r', _default_radius[interpolation])
    geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                   t0=0.0, tn=tn, src_type='Ricker', f0=f0,
                                   interpolation=interpolation, r=r)

    return geometry


def setup_rec_coords(model):
    nrecx = model.shape[0]
    recx = np.linspace(model.origin[0], model.domain_size[0], nrecx)

    if model.dim == 1:
        return recx.reshape((nrecx, 1))
    elif model.dim == 2:
        rec_coordinates = np.empty((nrecx, model.dim))
        rec_coordinates[:, 0] = recx
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates
    else:
        nrecy = model.shape[1]
        recy = np.linspace(model.origin[1], model.domain_size[1], nrecy)
        rec_coordinates = np.empty((nrecx*nrecy, model.dim))
        rec_coordinates[:, 0] = np.array([recx[i] for i in range(nrecx)
                                          for j in range(nrecy)])
        rec_coordinates[:, 1] = np.array([recy[j] for i in range(nrecx)
                                          for j in range(nrecy)])
        rec_coordinates[:, -1] = model.origin[-1] + 2 * model.spacing[-1]
        return rec_coordinates


class AcquisitionGeometry(Pickable):
    """
    Encapsulate the geometry of an acquisition:
    - source positions and number
    - receiver positions and number

    In practice this would only point to a segy file with the
    necessary information
    """

    __rargs__ = ('grid', 'rec_positions', 'src_positions', 't0', 'tn')
    __rkwargs__ = ('f0', 'src_type', 'interpolation', 'r')

    def __init__(self, model, rec_positions, src_positions, t0, tn, **kwargs):
        """
        In practice would be __init__(segyfile) and all below parameters
        would come from a segy_read (at property call rather than at init)
        """
        src_positions = np.reshape(src_positions, (-1, model.dim))
        rec_positions = np.reshape(rec_positions, (-1, model.dim))
        self.rec_positions = rec_positions
        self._nrec = rec_positions.shape[0]
        self.src_positions = src_positions
        self._nsrc = src_positions.shape[0]
        self._src_type = kwargs.get('src_type')
        assert (self.src_type in sources or self.src_type is None)
        self._f0 = kwargs.get('f0')
        self._a = kwargs.get('a', None)
        self._t0w = kwargs.get('t0w', None)
        if self._src_type is not None and self._f0 is None:
            error("Peak frequency must be provided in KHz" +
                  " for source of type %s" % self._src_type)

        self._grid = model.grid
        self._model = model
        self._dt = model.critical_dt
        self._t0 = t0
        self._tn = tn
        self._interpolation = kwargs.get('interpolation', 'linear')
        self._r = kwargs.get('r', _default_radius[self.interpolation])

        # Initialize to empty, created at new src/rec
        self._src_coordinates = None
        self._rec_coordinates = None

    def resample(self, dt):
        self._dt = dt
        return self

    @property
    def time_axis(self):
        return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

    @property
    def src_type(self):
        return self._src_type

    @property
    def grid(self):
        return self._grid

    @property
    def model(self):
        warning("Model is kept for backward compatibility but should not be"
                "obtained from the geometry")
        return self._model

    @property
    def f0(self):
        return self._f0

    @property
    def tn(self):
        return self._tn

    @property
    def t0(self):
        return self._t0

    @property
    def dt(self):
        return self._dt

    @property
    def nt(self):
        return self.time_axis.num

    @property
    def nrec(self):
        return self._nrec

    @property
    def nsrc(self):
        return self._nsrc

    @property
    def dtype(self):
        return self.grid.dtype

    @property
    def r(self):
        return self._r

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def src_coords(self):
        return self._src_coordinates

    @property
    def rec_coords(self):
        return self._rec_coordinates

    @property
    def rec(self):
        return self.new_rec()
        self._rec_coordinates = rec.coordinates
        return rec

    def new_rec(self, name='rec'):
        coords = self.rec_coords or self.rec_positions
        rec = Receiver(name=name, grid=self.grid,
                       time_range=self.time_axis, npoint=self.nrec,
                       interpolation=self.interpolation, r=self._r,
                       coordinates=coords)

        if self.rec_coords is None:
            self._rec_coordinates = rec.coordinates

        return rec

    @property
    def adj_src(self):
        if self.src_type is None:
            return self.new_rec()
        coords = self.rec_coords or self.rec_positions
        adj_src = sources[self.src_type](name='rec', grid=self.grid, f0=self.f0,
                                         time_range=self.time_axis, npoint=self.nrec,
                                         interpolation=self.interpolation, r=self._r,
                                         coordinates=coords, t0=self._t0w, a=self._a)
        # Revert time axis to have a proper shot record and not compute on zeros
        for i in range(self.nrec):
            adj_src.data[:, i] = adj_src.wavelet[::-1]
        return adj_src

    @property
    def src(self):
        return self.new_src()

    def new_src(self, name='src', src_type='self'):
        coords = self.src_coords or self.src_positions
        if self.src_type is None or src_type is None:
            warning("No source type defined, returning uninitiallized (zero) source")
            src = PointSource(name=name, grid=self.grid,
                              time_range=self.time_axis, npoint=self.nsrc,
                              coordinates=coords,
                              interpolation=self.interpolation, r=self._r)
        else:
            src = sources[self.src_type](name=name, grid=self.grid, f0=self.f0,
                                         time_range=self.time_axis, npoint=self.nsrc,
                                         coordinates=coords,
                                         t0=self._t0w, a=self._a,
                                         interpolation=self.interpolation, r=self._r)

        if self.src_coords is None:
            self._src_coordinates = src.coordinates

        return src


sources = {'Wavelet': WaveletSource, 'Ricker': RickerSource, 'Gabor': GaborSource}


def seismic_args(description):
    """
    Command line options for the seismic examples
    """

    class _dtype_store(Action):
        def __call__(self, parser, args, values, option_string=None):
            values = {'float32': np.float32, 'float64': np.float64}[values]
            setattr(args, self.dest, values)

    class _opt_action(Action):
        def __call__(self, parser, args, values, option_string=None):
            try:
                # E.g., `('advanced', {'par-tile': True})`
                values = eval(values)
                if not isinstance(values, tuple) and len(values) >= 1:
                    raise ArgumentError(self, ("Invalid choice `%s` (`opt` must be "
                                               "either str or tuple)" % str(values)))
                opt = values[0]
            except NameError:
                # E.g. `'advanced'`
                opt = values
            if opt not in configuration._accepted['opt']:
                raise ArgumentError(self, ("Invalid choice `%s` (choose from %s)"
                                           % (opt, str(configuration._accepted['opt']))))
            setattr(args, self.dest, values)

    parser = ArgumentParser(description=description)
    parser.add_argument("-nd", dest="ndim", default=3, type=int,
                        help="Number of dimensions")
    parser.add_argument("-d", "--shape", default=(51, 51, 51), type=int, nargs="+",
                        help="Number of grid points along each axis")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument("-so", "--space_order", default=4,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Use checkpointing, default is False")
    parser.add_argument("-opt", default="advanced", action=_opt_action,
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    parser.add_argument("-tn", "--tn", default=0,
                        type=float, help="Simulation time in millisecond")
    parser.add_argument("-dtype", action=_dtype_store, dest="dtype", default=np.float32,
                        choices=['float32', 'float64'])
    parser.add_argument("-interp", dest="interp", default="linear",
                        choices=['linear', 'sinc'])
    return parser
