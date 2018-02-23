from itertools import product as crossproduct
from abc import ABCMeta

import sympy

from devito.tools import prod


class GenericInterpolator(metaclass=ABCMeta):
    def __init__(self, point_symbols, dimensions, r):
        assert(isinstance(r, int))
        self.r = r
        self.point_symbols = point_symbols
        self.dimensions = dimensions
        self.ndim = len(dimensions)
        # Make sure we have a coefficient for every point
        assert(len(self.point_increments) == len(self.coefficients))

    @property
    def point_increments(self):
        """Index increments in each dimension for each point symbol"""
        r = self.r
        ndim = self.ndim
        return list(crossproduct(*[[i for i in range(-r + 1, r + 1)]
                                   for d in range(ndim)]))


class LinearInterpolator(GenericInterpolator):
    def __init__(self, point_symbols, dimensions):
        super(LinearInterpolator, self).__init__(point_symbols, dimensions, 1)

    @property
    def coefficients(self):
        """Symbolic expression for the coefficients for sparse point
        interpolation according to:
        https://en.wikipedia.org/wiki/Bilinear_interpolation.

        :returns: List of coefficients, eg. [b_11, b_12, b_21, b_22]
        """
        # Grid indices corresponding to the corners of the cell
        x1, y1, z1, x2, y2, z2 = sympy.symbols('x1, y1, z1, x2, y2, z2')
        # Coordinate values of the sparse point
        px, py, pz = self.point_symbols
        if self.ndim == 2:
            A = sympy.Matrix([[1, x1, y1, x1*y1],
                              [1, x1, y2, x1*y2],
                              [1, x2, y1, x2*y1],
                              [1, x2, y2, x2*y2]])

            p = sympy.Matrix([[1],
                              [px],
                              [py],
                              [px*py]])

            # Map to reference cell
            x, y = self.dimensions
            reference_cell = {x1: 0, y1: 0, x2: x.spacing, y2: y.spacing}

        elif self.ndim == 3:
            A = sympy.Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                              [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                              [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                              [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                              [1, x2, y1, z1, x2*y1, x2*z1, y1*z1, x2*y1*z1],
                              [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                              [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                              [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])

            p = sympy.Matrix([[1],
                              [px],
                              [py],
                              [pz],
                              [px*py],
                              [px*pz],
                              [py*pz],
                              [px*py*pz]])

            # Map to reference cell
            x, y, z = self.dimensions
            reference_cell = {x1: 0, y1: 0, z1: 0, x2: x.spacing,
                              y2: y.spacing, z2: z.spacing}
        else:
            raise NotImplementedError('Interpolation coefficients not implemented '
                                      'for %d dimensions.' % self.grid.dim)

        A = A.subs(reference_cell)
        coeffs = A.inv().T.dot(p)
        return coeffs


class LanczosInterpolator(GenericInterpolator):
    def __init__(self, point_symbols, dimensions, a):
        super(LanczosInterpolator, self).__init__(point_symbols, dimensions, a)

    @property
    def coefficients(self):
        x, a = sympy.symbols("x a")
        f = sympy.sinc(x)*sympy.sinc(x/a)
        vectors = []
        for dim in self.point_symbols:
            vector = []
            for i in range(-self.r+1, self.r+1):
                vector.append(f.subs(x, dim-i).subs(a, self.r))
            vectors.append(vector)

        return [prod(list(c)) for c in list(crossproduct(*vectors))]
