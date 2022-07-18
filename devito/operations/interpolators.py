from abc import ABC, abstractmethod
import sympy
import numpy as np
from cached_property import cached_property

from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, indexify, INT
from devito.tools import powerset, flatten, prod
from devito.types import (ConditionalDimension, Dimension, DefaultDimension, Eq, Inc,
                          Evaluable, Symbol, SubFunction)

__all__ = ['LinearInterpolator', 'PrecomputedInterpolator',
           'CubicInterpolator', 'SincInterpolator']


class UnevaluatedSparseOperation(sympy.Expr, Evaluable):

    """
    Represents an Injection or an Interpolation operation performed on a
    SparseFunction. Evaluates to a list of Eq objects.

    Parameters
    ----------
    interpolator : Interpolator
        Interpolator object that will be used to evaluate the operation.
    callback : callable
        A routine generating the symbolic expressions for the operation.
    """

    subdomain = None

    def __new__(cls, interpolator, callback):
        obj = super().__new__(cls)

        obj.interpolator = interpolator
        obj.callback = callback

        return obj

    @property
    def evaluate(self):
        return_value = self.callback()
        assert(all(isinstance(i, Eq) for i in return_value))
        return return_value

    def __add__(self, other):
        return flatten([self, other])

    def __radd__(self, other):
        return flatten([other, self])


class Interpolation(UnevaluatedSparseOperation):

    """
    Represents an Interpolation operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    def __new__(cls, expr, offset, increment, self_subs, interpolator, callback):
        obj = super().__new__(cls, interpolator, callback)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.expr = expr
        obj.offset = offset
        obj.increment = increment
        obj.self_subs = self_subs

        return obj

    def __repr__(self):
        return "Interpolation(%s into %s)" % (repr(self.expr),
                                              repr(self.interpolator.sfunction))


class Injection(UnevaluatedSparseOperation):

    """
    Represents an Injection operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    def __new__(cls, field, expr, offset, interpolator, callback):
        obj = super().__new__(cls, interpolator, callback)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.field = field
        obj.expr = expr
        obj.offset = offset

        return obj

    def __repr__(self):
        return "Injection(%s into %s)" % (repr(self.expr), repr(self.field))


class GenericInterpolator(ABC):

    """
    Abstract base class defining the interface for an interpolator.
    """

    @abstractmethod
    def inject(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpolate(self, *args, **kwargs):
        pass


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
                                 parent=self.obj)

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
                                           parent=self.obj)
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
        def callback():
            _expr = indexify(expr)

            p, _, _ = self.obj.interpolation_coeffs.indices
            dim_subs = []
            coeffs = []
            for i, d in enumerate(self.obj.grid.dimensions):
                rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
                dim_subs.append((d, INT(rd + self.obj.gridpoints[p, i])))
                coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
            # Apply optional time symbol substitutions to lhs of assignment
            lhs = self.obj.subs(self_subs)
            rhs = prod(coeffs) * _expr.subs(dim_subs)

            return [Eq(lhs, lhs + rhs)]

        return Interpolation(expr, offset, increment, self_subs, self, callback)

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
        def callback():
            _expr = indexify(expr)
            _field = indexify(field)

            p, _ = self.obj.gridpoints.indices
            dim_subs = []
            coeffs = []
            for i, d in enumerate(self.obj.grid.dimensions):
                rd = DefaultDimension(name="r%s" % d.name, default_value=self.r)
                dim_subs.append((d, INT(rd + self.obj.gridpoints[p, i])))
                coeffs.append(self.obj.interpolation_coeffs[p, i, rd])
            rhs = prod(coeffs) * _expr
            _field = _field.subs(dim_subs)
            return [Eq(_field, _field + rhs.subs(dim_subs))]

        return Injection(field, expr, offset, self, callback)


class PolynomialInterpolator(GenericInterpolator):
    """
    Implementation of a parent class that provides inheritance to interpolation classes,
    allowing code reuse
    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """

    def __init__(self, sfunction):
        self.sfunction = sfunction

    @property
    def grid(self):
        return self.sfunction.grid

    @property
    def r(self):
        return self.sfunction._radius

    def _interpolation_indices(self, variables, offset=0, field_offset=0):

        index_matrix, points = self.sfunction._index_matrix(offset)

        idx_subs = []
        for i, idx in enumerate(index_matrix):
            # Introduce ConditionalDimension so that we don't go OOB
            mapper = {}
            for j, d in zip(idx, self.grid.dimensions):
                p = points[j]
                # Only needs Conditional Dimensions if radius > variables.space_order
                if all(list(map((lambda x: self.r <= x.space_order
                       if hasattr(x, 'space_order') else False), variables))):
                    mapper[d] = p
                else:
                    lb = sympy.And(p >= d.symbolic_min - self.r,
                                   evaluate=False)
                    ub = sympy.And(p <= d.symbolic_max + self.r,
                                   evaluate=False)
                    condition = sympy.And(lb, ub, evaluate=False)
                    mapper[d] = ConditionalDimension(p.name, self.sfunction._sparse_dim,
                                                     condition=condition, indirect=True)
            # Track Indexed substitutions
            idx_subs.append(mapper)

        # Temporaries for the position
        temps = [Eq(v, k, implicit_dims=self.sfunction.dimensions)
                 for k, v in self.sfunction._position_map.items()]

        # Temporaries for the indirection dimensions
        temps.extend([Eq(v, k.subs(self.sfunction._position_map),
                         implicit_dims=self.sfunction.dimensions)
                      for k, v in points.items()])

        return idx_subs, temps

    def eq_relative_position(self):
        return [Eq(v, k.subs(self.sfunction._position_map),
                implicit_dims=self.sfunction.dimensions)
                for k, v in self.sfunction._relative_position_map.items()]

    def interpolate(self, expr, idx_subs, increment=False, self_subs={}):
        rhs = Symbol(name='sum', dtype=self.sfunction.dtype)
        summands = [Eq(rhs, 0., implicit_dims=self.sfunction.dimensions)]

        # Verify if is a 2D or 3D interpolation
        dim_pos = self.sfunction.grid.dimensions

        summands.extend([Inc(rhs, v, implicit_dims=self.sfunction.dimensions)
                         for v in self.coeffs(expr, idx_subs, dim_pos=dim_pos)])

        # Write/Incr `self`
        lhs = self.sfunction.subs(self_subs)
        last = [Inc(lhs, rhs)] if increment else [Eq(lhs, rhs)]

        return summands, last

    def inject(self, field, expr, idx_subs):
        dim_pos = field.space_dimensions
        result = self._eqs(expr, idx_subs, dim_pos=dim_pos)

        size = np.prod(np.shape(result))
        eqs = [e[0] for e in np.reshape(result, (size, 1))]

        eqns = [Inc(field.xreplace(vsub), b,
                    implicit_dims=self.sfunction.dimensions)
                for b, vsub in zip(eqs, idx_subs)]

        return eqns


class LinearInterpolator(PolynomialInterpolator):

    """
    Concrete implementation of GenericInterpolator implementing a Linear interpolation
    scheme, i.e. Bilinear for 2D and Trilinear for 3D problems.

    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """

    def coeffs(self, expr, idx_subs, dim_pos):
        return [expr.xreplace(v_sub) * b.xreplace(v_sub)
                for b, v_sub in zip(self._interpolation_coeffs, idx_subs)]

    @cached_property
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
        point_sym = list(powerset(self.sfunction._point_symbols))
        point_sym[0] = (1,)
        # 1, px. py, pz, px*py, ...
        A = []
        ref_A = [np.prod(ind) for ind in indices]
        # Create the matrix with the same increment order as the point increment
        for i in self.sfunction._point_increments:
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
        idx_subs, temps = super(LinearInterpolator, self)._interpolation_indices(
            variables, offset, field_offset)

        # Temporaries for the coefficients
        temps.extend([Eq(p, c.subs(self.sfunction._position_map),
                         implicit_dims=self.sfunction.dimensions)
                      for p, c in zip(self.sfunction._point_symbols,
                                      self.sfunction._coordinate_bases(field_offset))])

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
        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr))

            # Need to get origin of the field in case it is staggered
            # TODO: handle each variable staggereing spearately
            field_offset = variables[0].origin
            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            summands, last = super(LinearInterpolator, self).interpolate(_expr, idx_subs,
                                                                         increment,
                                                                         self_subs)

            return temps + summands + last

        return Interpolation(expr, offset, increment, self_subs, self, callback)

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
        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr)) + [field]

            # Need to get origin of the field in case it is staggered
            field_offset = field.origin
            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            # Substitute coordinate base symbols into the interpolation coefficients
            eqns = [Inc(field.xreplace(vsub), _expr.xreplace(vsub) * b,
                        implicit_dims=self.sfunction.dimensions)
                    for b, vsub in zip(self._interpolation_coeffs, idx_subs)]

            return temps + eqns

        return Injection(field, expr, offset, self, callback)


class CubicInterpolator(PolynomialInterpolator):
    """
    Concrete implementation of GenericInterpolator implementing a cubic interpolation
    scheme.
    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """
    def __init__(self, sfunction):
        super(CubicInterpolator, self).__init__(sfunction)
        self._eqs = self._ncubic_interpolation

    def coeffs(self, expr, idx_subs, dim_pos):
        return [sympy.Add(*v) for v in self._eqs(expr, idx_subs, dim_pos=dim_pos)]

    def _cubic_equation(self, expr, position, dim_pos, idx_subs, idx2d, idx3d=None):
        """
        Generate the basic cubic equation.
        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        position: list
            Elements which represents the position of the interpolation point multiplied
            by their respective coefficients.
        pn: List, optional
            Contains the symbols which will be used as the neighbors points, if is that
            the case.
        idx_subs: dict, optional
            Structure responsible for mapping the order of substitution of dimensions in
            expr.
        idx2d: int
            Defines which iteration of second dimension is being used.
        idx3d: int, optional
            defines which iteration of third dimension is being used,
            if it is a 3D interpolation.
        When idx_subs is not None, expr is used as element which represents the neighbors
        points. Otherwise, pn is used.
        """
        def cubicConv(arg, option):
            """
            option = 1, point immediately adjacent to the interpolation point.
            Where 0 < |arg| < 1
            option = 2, endpoints of the neighborhood.
            Where 1 < |arg| < 2
            """
            a = -1/2
            if option == 1:
                return (a+2)*(abs(arg)**3) - (a+3)*(abs(arg)**2) + 1
            elif option == 2:
                return a*(abs(arg)**3) - 5*a*(abs(arg)**2) + 8*a*abs(arg) - 4*a

        # Values ​​that define which cubic kernel will be used
        option = [2, 1, 1, 2]

        pos = sympy.Matrix(position)

        dim_pos = sympy.Matrix(dim_pos)

        # Neighboring points(interpolation) or source value (injection)
        p = [expr.xreplace(subs) for subs in idx_subs]
        p = sympy.Matrix(p)

        # Defines the kernel witch will be multiplied by the neighboring
        # points(or source value)
        base_index = [idx3d, idx2d]
        index_opt = [b for b in base_index if b is not None]
        arg_kernel = [pp - dim for pp, dim in zip(pos, dim_pos)]

        kernel = [cubicConv(arg, option[opt]) for arg, opt in zip(arg_kernel, index_opt)]
        kernel = np.prod(kernel)
        kernel = [kernel*cubicConv(arg_kernel[-1], option[ii]) for ii in range(4)]

        # Defining the final shape of the cubic equation
        points = [point*k.subs(idx) for point, k, idx in zip(p, kernel, idx_subs)]

        return points

    def _ncubic_interpolation(self, expr, idx_subs, dim_pos=None):
        pos = self.sfunction._point_symbols
        n = self.r*2

        if self.sfunction.grid.dim == 2:
            eqs = [self._cubic_equation(expr, pos, dim_pos,
                   idx_subs=idx_subs[ii*n:(ii+1)*n], idx2d=ii)
                   for ii in range(n)]
        else:
            eqs = []
            eqs.extend([self._cubic_equation(expr, pos, dim_pos,
                        idx_subs=idx_subs[(ind*n)+(n*n*ii):((1+ind)*n)+n*n*ii],
                        idx2d=ind, idx3d=ii)
                        for ii in range(n) for ind in range(n)])
        return eqs

    def _interpolation_indices(self, variables, offset=0, field_offset=0):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        idx_subs, temps = super(CubicInterpolator, self)._interpolation_indices(
            variables, offset, field_offset)
        temps.extend(self.eq_relative_position())

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
        def callback():

            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr))

            # Need to get origin of the field in case it is staggered
            # TODO: handle each variable staggereing spearately
            field_offset = variables[0].origin

            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            summands, last = super(CubicInterpolator, self).interpolate(_expr, idx_subs,
                                                                        increment,
                                                                        self_subs)

            return temps + summands + last

        return Interpolation(expr, offset, increment, self_subs, self, callback)

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
        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr)) + [field]

            # Need to get origin of the field in case it is staggered
            field_offset = field.origin

            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            eqns = super(CubicInterpolator, self).inject(field, _expr, idx_subs)

            return temps + eqns

        return Injection(field, expr, offset, self, callback)


class SincInterpolator(PolynomialInterpolator):

    def __init__(self, sfunction):
        super(SincInterpolator, self).__init__(sfunction)
        self._eqs = self._nsinc_equation

    def coeffs(self, expr, idx_subs, dim_pos):
        return [sympy.Add(*v) for v in self._eqs(expr, idx_subs, dim_pos=dim_pos)]

    def _interpolation_indices(self, variables, offset=0, field_offset=0):

        idx_subs, temps = super(SincInterpolator, self)._interpolation_indices(
            variables, offset, field_offset)
        temps.extend(self.eq_relative_position())
        return idx_subs, temps

    def _sinc_equation(self, expr, position, dim_pos, idx_subs, idx2d, idx3d=None):
        """
        Generate the basic sinc equation.
        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        position: List
            Elements which represents the position of the interpolation point
            multiplied by their respective coefficients.
        dim_pos: Symbol
            Symbol representing the dimension wich will be used to form the sinc
            equations.
        idx_subs: dict
            Structure responsible for mapping the order of substitution of dimensions
            in expr.
        idx2d: int
            Defines which iteration of second dimension is being used.
        idx3d: int, optional
            defines which iteration of third dimension is being used,
            if it is a 3D interpolation.
        """

        # Defining the symbolic function responsible for accessing
        # the pre-computed coefficient value
        def sincKernel(args):
            coeffs = sympy.Function(name="acessCoeffs")(args)
            return coeffs

        npoints = self.r*2

        pos = sympy.Matrix(position)

        dim_pos = sympy.Matrix(dim_pos)

        p = [expr.xreplace(subs) for subs in idx_subs]
        p = sympy.Matrix(p)

        index_opt = [idx3d, idx2d]
        index_opt = [b for b in index_opt if b is not None]
        arg_kernel = [pos[ii] - dim for ii, dim in enumerate(dim_pos)]

        kernel = [sincKernel(arg) for arg, index in zip(arg_kernel, enumerate(index_opt))]

        kernel = np.prod(kernel)
        kernel = [kernel*sincKernel(arg_kernel[-1]) for ii in range(npoints)]

        points = [point*k.subs(idx) for point, k, idx in zip(p, kernel, idx_subs)]

        return points

    def _nsinc_equation(self, expr, idx_subs, dim_pos=None):
        """
        Generate equations responsible for 2D sinc interpolation's computation.
        """
        # Gets the list containing the symbols px and py
        pos = self.sfunction._point_symbols

        # n = number of neighbor points
        n = self.r*2

        # Generating equations
        if self.sfunction.grid.dim == 2:
            eqs = [self._sinc_equation(expr, pos, dim_pos, idx_subs[ii*n:(ii+1)*n],
                                       idx2d=ii)
                   for ii in range(n)]
        else:
            eqs = []
            eqs.extend([self._sinc_equation(expr, pos, dim_pos,
                        idx_subs=idx_subs[(ind*n)+(n*n*ii):((1+ind)*n)+n*n*ii],
                        idx2d=ind, idx3d=ii) for ii in range(n) for ind in range(n)])
        return eqs

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
        def callback():

            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr))

            # Need to get origin of the field in case it is staggered
            # TODO: handle each variable staggereing spearately
            field_offset = variables[0].origin

            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)

            summands, last = super(SincInterpolator, self).interpolate(_expr, idx_subs,
                                                                       increment,
                                                                       self_subs)

            # Creates the symbolic equation that calls the populate function
            err = Symbol(name='err', dtype=np.int32)
            populate = [Eq(err, sympy.Function(name="populate")(),
                        implicit_dims=self.sfunction.dimensions[0])]

            return populate + temps + summands + last

        return Interpolation(expr, offset, increment, self_subs, self, callback)

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
        def callback():

            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr)) + [field]

            # Need to get origin of the field in case it is staggered
            field_offset = field.origin
            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interpolation_indices(variables, offset,
                                                          field_offset=field_offset)


            eqns = super(SincInterpolator, self).inject(field, _expr, idx_subs)

            err = Symbol(name='err', dtype=np.int32)
            populate = [Eq(err, sympy.Function(name="populate")(),
                        implicit_dims=self.sfunction.dimensions[0])]

            return populate + temps + eqns

        return Injection(field, expr, offset, self, callback)
