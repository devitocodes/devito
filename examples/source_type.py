from sympy import Eq, Function, Matrix, symbols
from sympy.abc import p

from devito.dimension import t
from devito.interfaces import PointData
from devito.iteration import Iteration


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, *args, **kwargs):
        self.dt = kwargs.get('dt')
        self.h = kwargs.get('h')
        self.ndim = kwargs.get('ndim')
        self.nbpml = kwargs.get('nbpml')
        PointData.__init__(self, *args, **kwargs)
        x1, y1, z1, x2, y2, z2 = symbols('x1, y1, z1, x2, y2, z2')

        if self.ndim == 2:
            A = Matrix([[1, x1, z1, x1*z1],
                        [1, x1, z2, x1*z2],
                        [1, x2, z1, x2*z1],
                        [1, x2, z2, x2*z2]])
            self.increments = (0, 0), (0, 1), (1, 0), (1, 1)
            self.rs = symbols('rx, rz')
            rx, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [rz],
                        [rx*rz]])
        else:
            A = Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                        [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                        [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                        [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                        [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                        [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                        [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                        [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])
            self.increments = (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), \
                              (0, 1, 1), (1, 0, 1), (1, 1, 1)
            self.rs = symbols('rx, ry, rz')
            rx, ry, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [ry],
                        [rz],
                        [rx*ry],
                        [rx*rz],
                        [ry*rz],
                        [rx*ry*rz]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (y1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (y2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)
        self.bs = A.inv().T.dot(p)

    @property
    def sym_coordinates(self):
        """Symbol representing the coordinate values in each dimension"""
        return tuple([self.coordinates.indexed[p, i]
                      for i in range(self.ndim)])

    @property
    def sym_coord_indices(self):
        """Symbol for each grid index according to the coordinates"""
        return tuple([Function('INT')(Function('floor')(x / self.h))
                      for x in self.sym_coordinates])

    @property
    def sym_coord_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        return tuple([Function('FLOAT')(x - idx * self.h)
                      for x, idx in zip(self.sym_coordinates,
                                        self.sym_coord_indices)])

    def point2grid(self, u, m, t=t):
        """Generates an expression for generic point-to-grid interpolation"""
        dt = self.dt
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        eqns = [Eq(u.indexed[(t, ) + idx], u.indexed[(t, ) + idx]
                   + self.indexed[t, p] * dt * dt / m.indexed[idx] * b.subs(subs))
                for idx, b in zip(index_matrix, self.bs)]
        return eqns

    def grid2point(self, u, t=t):
        """Generates an expression for generic grid-to-point interpolation"""
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        return sum([b.subs(subs) * u.indexed[(t, ) + idx]
                    for idx, b in zip(index_matrix, self.bs)])

    def read(self, u):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u))
        return [Iteration(interp_expr, index=p, limits=self.shape[1])]

    def read2(self, u, v):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u) + self.grid2point(v))
        return [Iteration(interp_expr, index=p, limits=self.shape[1])]

    def add(self, m, u, t=t):
        """Iteration loop over points performing point-to-grid interpolation."""
        return [Iteration(self.point2grid(u, m, t), index=p, limits=self.shape[1])]