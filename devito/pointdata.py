from devito.dimension import d, p, t
from devito.interfaces import DenseData

__all__ = ['PointData']


class PointData(DenseData):
    """
    Data object for sparse point data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param npoint: Number of points to sample
    :param nt: Size of the time dimension for point data
    :param ndim: Dimension of the coordinate data
    :param coordinates: Optional coordinate data for the sparse points

    :param dtype: Data type of the buffered data
    """

    is_PointData = True

    def __init__(self, *args, **kwargs):
        if not self._cached():
            self.nt = kwargs.get('nt')
            self.npoint = kwargs.get('npoint')
            self.ndim = kwargs.get('ndim')
            kwargs['shape'] = (self.nt, self.npoint)
            super(PointData, self).__init__(self, *args, **kwargs)
            self.coordinates = DenseData(name='%s_coords' % self.name,
                                         dimensions=[self.indices[1], d],
                                         shape=(self.npoint, self.ndim))
            coordinates = kwargs.get('coordinates', None)
            if coordinates is not None:
                self.coordinates.data[:] = coordinates[:]

    def __new__(cls, *args, **kwargs):
        nt = kwargs.get('nt')
        npoint = kwargs.get('npoint')
        kwargs['shape'] = (nt, npoint)

        return DenseData.__new__(cls, *args, **kwargs)

    @classmethod
    def _indices(cls, **kwargs):
        """Return the default dimension indices for a given data shape

        :param shape: Shape of the spatial data
        :return: indices used for axis.
        """
        dimensions = kwargs.get('dimensions', None)
        return dimensions or [t, p]
