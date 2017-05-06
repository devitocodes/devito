from sympy import Eq, Function, Matrix, symbols

from devito.dimension import t
from devito.interfaces import PointData, DenseData
from devito.nodes import Iteration


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
        p_dim = self.indices[1]
        return tuple([self.coordinates.indexed[p_dim, i]
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

    @property
    def index_matrix(self):
        """Spatial indexing expressions for all grid points
        contributing to spatial interpolation.
        """
        return [tuple([idx + ii + self.nbpml for ii, idx
                       in zip(inc, self.sym_coord_indices)])
                for inc in self.increments]

    def point2grid(self, u, m, u_t=t, p_t=t):
        """Generates an expression for point-to-grid interpolation

        :param u: Wavefield to interpoalte into
        :param m: Model parameter m
        :param u_t: Time index for wavefield u
        :param p_t: Time index for source term
        """
        dt = self.dt
        subs = dict(zip(self.rs, self.sym_coord_bases))
        p_idx = self.indexed[(p_t, ) + tuple(self.indices[1:])]
        eqns = []
        for idx, b in zip(self.index_matrix, self.bs):
            if isinstance(m, DenseData):
                exprm = m.indexed[idx]
            else:
                exprm = m
            u_idx = u.indexed[(u_t, ) + idx]
            expr = p_idx * dt * dt / exprm * b.subs(subs)
            eqns += [Eq(u_idx, u_idx + expr)]
        return eqns

    def grid2point(self, u, t=t):
        """Generates an expression for generic grid-to-point interpolation"""
        subs = dict(zip(self.rs, self.sym_coord_bases))
        return sum([b.subs(subs) * u.indexed[(t, ) + idx]
                    for idx, b in zip(self.index_matrix, self.bs)])

    def read(self, u):
        """Read the value of the wavefield u at point locations with grid2point."""
        sym_p = self.indices[1]
        interp_expr = Eq(self.indexed[t, sym_p], self.grid2point(u))
        return [Iteration(interp_expr, dimension=sym_p, limits=self.shape[1])]

    def read2(self, u, v):
        """Read the value of the wavefield (u+v) at point locations with grid2point."""
        sym_p = self.indices[1]
        interp_expr = Eq(self.indexed[t, sym_p], self.grid2point(u) + self.grid2point(v))
        return [Iteration(interp_expr, dimension=sym_p, limits=self.shape[1])]

    def add(self, m, u, u_t=t, p_t=t):
        """Add a point source term to the wavefield u at time t

        :param m: Model parameter m
        :param u: Wavefield to interpoalte into
        :param u_t: Time index for wavefield u
        :param p_t: Time index for source term
        """
        return [Iteration(self.point2grid(u, m, u_t=u_t, p_t=p_t),
                          dimension=self.indices[1], limits=self.shape[1])]
