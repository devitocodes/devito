from devito.dimension import Dimension, time
from devito.pointdata import PointData
from devito.logger import error


__all__ = ['PointSource', 'Receiver', 'Shot']


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
        obj = PointData(name=name, dimensions=[time, p_dim],
                        npoint=npoint, nt=ntime, ndim=ndim,
                        coordinates=coordinates, **kwargs)

        # If provided, copy initial data into the allocated buffer
        if data is not None:
            obj.data[:] = data
        return obj


Receiver = PointSource
Shot = PointSource
