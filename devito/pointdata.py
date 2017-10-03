from collections import OrderedDict

from sympy import Function, Matrix, symbols

from devito.cgen_utils import INT, FLOAT
from devito.dimension import d, p, t, time, x, y, z
from devito.dse.inspection import indexify, retrieve_indexed
from devito.dse.extended_sympy import Eq
from devito.interfaces import DenseData, CompositeData
from devito.logger import error

__all__ = ['PointData']


class PointData(CompositeData):
    """
    Data object for sparse point data that acts as a Function symbol

    :param name: Name of the resulting :class:`sympy.Function` symbol
    :param npoint: Number of points to sample
    :param nt: Size of the time dimension for point data
    :param ndim: Dimension of the coordinate data, eg. 2D or 3D
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

            # Allocate and copy coordinate data
            self.coordinates = DenseData(name='%s_coords' % self.name,
                                         dimensions=[self.indices[1], d],
                                         shape=(self.npoint, self.ndim))
            self._children.append(self.coordinates)
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
        return dimensions or [time, p]

    @property
    def coefficients(self):
        """Symbolic expression for the coefficients for sparse point
        interpolation according to:
        https://en.wikipedia.org/wiki/Bilinear_interpolation.

        :returns: List of coefficients, eg. [b_11, b_12, b_21, b_22]
        """
        # Grid indices corresponding to the corners of the cell
        x1, y1, z1, x2, y2, z2 = symbols('x1, y1, z1, x2, y2, z2')
        # Coordinate values of the sparse point
        px, py, pz = self.point_symbols
        if self.ndim == 2:
            A = Matrix([[1, x1, y1, x1*y1],
                        [1, x1, y2, x1*y2],
                        [1, x2, y1, x2*y1],
                        [1, x2, y2, x2*y2]])

            p = Matrix([[1],
                        [px],
                        [py],
                        [px*py]])

        elif self.ndim == 3:
            A = Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                        [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                        [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                        [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                        [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                        [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                        [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                        [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])

            p = Matrix([[1],
                        [px],
                        [py],
                        [pz],
                        [px*py],
                        [px*pz],
                        [py*pz],
                        [px*py*pz]])
        else:
            error('Point interpolation only supported for 2D and 3D')
            raise NotImplementedError('Interpolation coefficients not '
                                      'implemented for %d dimensions.'
                                      % self.ndim)

        # Map to reference cell
        reference_cell = {x1: 0, y1: 0, z1: 0, x2: x.spacing, y2: y.spacing,
                          z2: z.spacing}
        A = A.subs(reference_cell)
        return A.inv().T.dot(p)

    @property
    def point_symbols(self):
        """Symbol for coordinate value in each dimension of the point"""
        return symbols('px, py, pz')

    @property
    def point_increments(self):
        """Index increments in each dimension for each point symbol"""
        if self.ndim == 2:
            return ((0, 0), (0, 1), (1, 0), (1, 1))
        elif self.ndim == 3:
            return ((0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1),
                    (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1))
        else:
            error('Point interpolation only supported for 2D and 3D')
            raise NotImplementedError('Point increments not defined '
                                      'for %d dimensions.' % self.ndim)

    @property
    def coordinate_symbols(self):
        """Symbol representing the coordinate values in each dimension"""
        p_dim = self.indices[1]
        return tuple([self.coordinates.indexify((p_dim, i))
                      for i in range(self.ndim)])

    @property
    def coordinate_indices(self):
        """Symbol for each grid index according to the coordinates"""
        indices = (x, y, z)
        return tuple([INT(Function('floor')(c / i.spacing))
                      for c, i in zip(self.coordinate_symbols, indices[:self.ndim])])

    @property
    def coordinate_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        indices = (x, y, z)
        return tuple([FLOAT(c - idx * i.spacing)
                      for c, idx, i in zip(self.coordinate_symbols,
                                           self.coordinate_indices,
                                           indices[:self.ndim])])

    def interpolate(self, expr, offset=0, **kwargs):
        """Creates a :class:`sympy.Eq` equation for the interpolation
        of an expression onto this sparse point collection.

        :param expr: The expression to interpolate.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into
                    field data in `expr`.
        :param p_t: (Optional) time index to use for indexing into
                    the sparse point data.
        """
        u_t = kwargs.get('u_t', None)
        p_t = kwargs.get('p_t', None)
        expr = indexify(expr)

        # Apply optional time symbol substitutions to expr
        if u_t is not None:
            expr = expr.subs(t, u_t).subs(time, u_t)

        variables = list(retrieve_indexed(expr))
        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]
        # Generate index substituions for all grid variables
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.ndim] + idx])
                      for v in variables]
            idx_subs += [OrderedDict(v_subs)]
        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        rhs = sum([expr.subs(vsub) * b.subs(subs)
                   for b, vsub in zip(self.coefficients, idx_subs)])

        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self if p_t is None else self.subs(self.indices[0], p_t)
        return [Eq(lhs, rhs)]

    def inject(self, field, expr, offset=0, **kwargs):
        """Symbol for injection of an expression onto a grid

        :param field: The grid field into which we inject.
        :param expr: The expression to inject.
        :param offset: Additional offset from the boundary for
                       absorbing boundary conditions.
        :param u_t: (Optional) time index to use for indexing into `field`.
        :param p_t: (Optional) time index to use for indexing into `expr`.
        """
        u_t = kwargs.get('u_t', None)
        p_t = kwargs.get('p_t', None)

        expr = indexify(expr)
        field = indexify(field)
        variables = list(retrieve_indexed(expr)) + [field]

        # Apply optional time symbol substitutions to field and expr
        if u_t is not None:
            field = field.subs(field.indices[0], u_t)
        if p_t is not None:
            expr = expr.subs(self.indices[0], p_t)

        # List of indirection indices for all adjacent grid points
        index_matrix = [tuple(idx + ii + offset for ii, idx
                              in zip(inc, self.coordinate_indices))
                        for inc in self.point_increments]

        # Generate index substituions for all grid variables except
        # the sparse `PointData` types
        idx_subs = []
        for i, idx in enumerate(index_matrix):
            v_subs = [(v, v.base[v.indices[:-self.ndim] + idx])
                      for v in variables if not v.base.function.is_PointData]
            idx_subs += [OrderedDict(v_subs)]

        # Substitute coordinate base symbols into the coefficients
        subs = OrderedDict(zip(self.point_symbols, self.coordinate_bases))
        return [Eq(field.subs(vsub),
                   field.subs(vsub) + expr.subs(subs).subs(vsub) * b.subs(subs))
                for b, vsub in zip(self.coefficients, idx_subs)]
