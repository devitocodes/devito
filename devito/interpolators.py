import sympy
import numpy as np

from abc import ABC, abstractmethod

from devito.equation import Eq, Inc
from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, indexify, INT
from devito.tools import Evaluable, powerset, flatten, prod
from devito.types.basic import Scalar
from devito.types.dense import SubFunction
from devito.types.dimension import ConditionalDimension, Dimension, DefaultDimension


class UnevaluatedSparseOperation(Evaluable):

    """
    Represents an Injection or an Interpolation operation performed on a
    SparseFunction. Evaluates to a list of Eq objects.
    objects.

    Parameters
    ----------
    interpolator : Interpolator
        Interpolator object that will be used to evaluate the Operation
    *args, **kwargs
        The arguments passed to the corresponding method
    """
    subdomain = None

    def __init__(self, interpolator, *args, **kwargs):
        self.interpolator = interpolator
        self._args = args
        self._kwargs = kwargs

    def __add__(self, other):
        return flatten([self, other])

    def __radd__(self, other):
        return flatten([other, self])


class Injection(UnevaluatedSparseOperation):

    """
    Represents an Injection operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    @property
    def evaluate(self):
        return_value = self.interpolator.inject(*self._args, **self._kwargs)
        assert(all(isinstance(i, Eq) for i in return_value))
        return return_value


class Interpolation(UnevaluatedSparseOperation):

    """
    Represents an Injection operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    @property
    def evaluate(self):
        return_value = self.interpolator.interpolate(*self._args, **self._kwargs)
        assert(all(isinstance(i, Eq) for i in return_value))
        return return_value


class GenericInterpolator(ABC):

    """
    Abstract base class defining the interface for an interpolator
    """

    @abstractmethod
    def inject(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpolate(self, *args, **kwargs):
        pass


class LinearInterpolator(GenericInterpolator):
    def __init__(self, grid, obj):
        self.grid = grid
        self.obj = obj

    @property
    def _interpolation_coeffs(self):
        """
        Symbolic expression for the coefficients for sparse point interpolation
        according to:

            https://en.wikipedia.org/wiki/Bilinear_interpolation.

        Returns
        -------
        Matrix of coefficient expressions.
        """
        # Grid indices corresponding to the corners of the cell ie x1, y1, z1
        indices1 = tuple(sympy.symbols('%s1' % d) for d in self.grid.dimensions)
        indices2 = tuple(sympy.symbols('%s2' % d) for d in self.grid.dimensions)
        # 1, x1, y1, z1, x1*y1, ...
        indices = list(powerset(indices1))
        indices[0] = (1,)
        point_sym = list(powerset(self.obj._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self.obj._point_increments:
            # substitute x1 by x2 if increment in that dimension
            subs = dict((indices1[d], indices2[d] if i[d] == 1 else indices1[d])
                        for d in range(len(i)))
            A += [[1] + [a.subs(subs) for a in ref_A[1:]]]

        A = sympy.Matrix(A)
        # Coordinate values of the sparse point
        p = sympy.Matrix([[np.prod(ind)] for ind in point_sym])

        # reference cell x1:0, x2:h_x
        left = dict((a, 0) for a in indices1)
        right = dict((b, dim.spacing) for b, dim in zip(indices2, self.grid.dimensions))
        reference_cell = {**left, **right}
        # Substitute in interpolation matrix
        A = A.subs(reference_cell)
        return A.inv().T * p

    def _interpolation_indices(self, variables, offset=0, field_offset=0):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        index_matrix, points = self.obj._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                lb = sympy.And(p >= d.symbolic_min - self.obj._radius, evaluate=False)
                ub = sympy.And(p <= d.symbolic_max + self.obj._radius, evaluate=False)
                condition = sympy.And(lb, ub, evaluate=False)
                mapper[d] = ConditionalDimension(p.name, self.obj._sparse_dim,
                                                 condition=condition, indirect=True)

            # Track Indexed substitutions
            idx_subs.append(mapper)

        # Temporaries for the indirection dimensions
        temps = [Eq(v, k, implicit_dims=self.obj.dimensions) for k, v in points.items()]
        # Temporaries for the coefficients
        temps.extend([Eq(p, c, implicit_dims=self.obj.dimensions)
                      for p, c in zip(self.obj._point_symbols,
                                      self.obj._coordinate_bases(field_offset))])

        return idx_subs, temps

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            expr = expr.evaluate
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            pass

        variables = list(retrieve_function_carriers(expr))

        # Need to get origin of the field in case it is staggered
        # TODO: handle each variable staggereing spearately
        field_offset = variables[0].origin
        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interpolation_indices(variables, offset,
                                                      field_offset=field_offset)

        # Substitute coordinate base symbols into the interpolation coefficients
        args = [expr.xreplace(v_sub) * b.xreplace(v_sub)
                for b, v_sub in zip(self._interpolation_coeffs, idx_subs)]

        # Accumulate point-wise contributions into a temporary
        rhs = Scalar(name='sum', dtype=self.obj.dtype)
        summands = [Eq(rhs, 0., implicit_dims=self.obj.dimensions)]
        summands.extend([Inc(rhs, i, implicit_dims=self.obj.dimensions) for i in args])

        # Write/Incr `self`
        lhs = self.obj.subs(self_subs)
        last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

        return temps + summands + last

    def inject(self, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            expr = expr.evaluate
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            pass

        variables = list(retrieve_function_carriers(expr)) + [field]

        # Need to get origin of the field in case it is staggered
        field_offset = field.origin
        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interpolation_indices(variables, offset,
                                                      field_offset=field_offset)

        # Substitute coordinate base symbols into the interpolation coefficients
        eqns = [Inc(field.xreplace(vsub), expr.xreplace(vsub) * b,
                    implicit_dims=self.obj.dimensions)
                for b, vsub in zip(self._interpolation_coeffs, idx_subs)]

        return temps + eqns


class PrecomputedInterpolator(GenericInterpolator):
    def __init__(self, obj, r, gridpoints_data, coefficients_data):
        if not isinstance(r, int):
            raise TypeError('Need `r` int argument')
        if r <= 0:
            raise ValueError('`r` must be > 0')
        self.r = r
        self.obj = obj
        self._npoint = obj._npoint
        gridpoints = SubFunction(name="%s_gridpoints" % self.obj.name, dtype=np.int32,
                                 dimensions=(self.obj.indices[-1], Dimension(name='d')),
                                 shape=(self._npoint, self.obj.grid.dim), space_order=0,
                                 parent=self)

        assert(gridpoints_data is not None)
        gridpoints.data[:] = gridpoints_data[:]
        self.obj._gridpoints = gridpoints

        interpolation_coeffs = SubFunction(name="%s_interpolation_coeffs" % self.obj.name,
                                           dimensions=(self.obj.indices[-1],
                                                       Dimension(name='d'),
                                                       Dimension(name='i')),
                                           shape=(self.obj.npoint, self.obj.grid.dim,
                                                  self.r),
                                           dtype=self.obj.dtype, space_order=0,
                                           parent=self)
        assert(coefficients_data is not None)
        interpolation_coeffs.data[:] = coefficients_data[:]
        self.obj._interpolation_coeffs = interpolation_coeffs
        warning("Ensure that the provided interpolation coefficient and grid point " +
                "values are computed on the final grid that will be used for other " +
                "computations.")

    def interpolate(self, expr, offset=0, increment=False, self_subs={}):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        offset : int, optional
            Additional offset from the boundary.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        """
        expr = indexify(expr)

        p, _, _ = self.obj.interpolation_coeffs.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.obj.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.obj.gridpoints[p, i])))
            coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
        # Apply optional time symbol substitutions to lhs of assignment
        lhs = self.obj.subs(self_subs)
        rhs = prod(coeffs) * expr.subs(dim_subs)

        return [Eq(lhs, lhs + rhs)]

    def inject(self, field, expr, offset=0):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        offset : int, optional
            Additional offset from the boundary.
        """
        expr = indexify(expr)
        field = indexify(field)

        p, _ = self.gridpoints.indices
        dim_subs = []
        coeffs = []
        for i, d in enumerate(self.grid.dimensions):
            rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
            dim_subs.append((d, INT(rd + self.gridpoints[p, i])))
            coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
        rhs = prod(coeffs) * expr
        field = field.subs(dim_subs)
        return [Eq(field, field + rhs.subs(dim_subs))]
