from abc import ABC, abstractmethod

import sympy
from cached_property import cached_property

from devito.finite_differences.elementary import floor
from devito.symbolics import retrieve_function_carriers, INT
from devito.tools import as_tuple, flatten, prod
from devito.types import (ConditionalDimension, Eq, Inc, Evaluable, Symbol,
                          CustomDimension)
from devito.types.utils import DimensionTuple

__all__ = ['LinearInterpolator', 'PrecomputedInterpolator']


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

    def _evaluate(self, **kwargs):
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

    def __new__(cls, expr, increment, self_subs, interpolator, callback):
        obj = super().__new__(cls, interpolator, callback)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.expr = expr
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

    def __new__(cls, field, expr, interpolator, callback):
        obj = super().__new__(cls, interpolator, callback)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.field = field
        obj.expr = expr

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


class WeightedInterpolator(GenericInterpolator):

    """
    Represent an Interpolation operation on a SparseFunction that is separable
    in space, meaning the coefficients are defined for each Dimension separately
    and multiplied at a given point: `w[x, y] = wx[x] * wy[y]`
    """

    def __init__(self, sfunction):
        self.sfunction = sfunction

    @property
    def grid(self):
        return self.sfunction.grid

    @property
    def _weights(self):
        raise NotImplementedError

    @property
    def _gdim(self):
        return self.grid.dimensions

    @property
    def r(self):
        return self.sfunction.r

    @cached_property
    def _rdim(self):
        dims = []
        # Enforce ordering
        prevdim = self.sfunction._sparse_dim
        for d in self._gdim:
            rd = CustomDimension("r%s%s" % (self.sfunction.name, d.name),
                                 -self.r+1, self.r, len(range(-self.r+1, self.r+1)),
                                 prevdim)
            prevdim = rd
            dims.append(rd)

        return DimensionTuple(*dims, getters=self._gdim)

    def _augment_implicit_dims(self, implicit_dims):
        return as_tuple(implicit_dims) + self.sfunction.dimensions

    def _coeff_temps(self, implicit_dims):
        return []

    def _positions(self, implicit_dims):
        return [Eq(v, INT(floor(k)), implicit_dims=implicit_dims)
                for k, v in self.sfunction._position_map.items()]

    def _interp_idx(self, variables, implicit_dims=None):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        idx_subs = []
        mapper = defaultdict(list)

        # Temporaries for the position
        temps = self._positions(implicit_dims)

        # Coefficient symbol expression
        temps.extend(self._coeff_temps(implicit_dims))

        # Create positions and indices temporaries/indirections
        prev = self.sfunction.dimensions[-1]
        for ((di, d), rd) in zip(enumerate(self._gdim), self._rdim):
            # Add conditional to avoid OOB
            lb = sympy.And(rd >= d.symbolic_min, evaluate=False)
            ub = sympy.And(rd <= d.symbolic_max, evaluate=False)
            cond = sympy.And(lb, ub, evaluate=False)
            mapper[d] = ConditionalDimension(rd.name, prev,
                                             condition=cond, indirect=True)
            prev = rd

        # Substitution mapper for variables
        idx_subs = {v: v.subs({k: c - v.origin.get(k, 0) for (k, c) in mapper.items()})
                    for v in variables}

        return idx_subs, temps

    def subs_coords(self, _expr, *idx_subs):
        return [_expr.xreplace(v_sub) * b.xreplace(v_sub)
                for b, v_sub in zip(self._interpolation_coeffs, idx_subs)]

    def subs_coords_eq(self, field, _expr, *idx_subs, implicit_dims=None):
        return [Inc(field.xreplace(vsub), _expr.xreplace(vsub) * b,
                    implicit_dims=implicit_dims)
                for b, vsub in zip(self._interpolation_coeffs, idx_subs)]

    def _interpolation_indices(self, variables, offset=0, field_offset=0,
                               implicit_dims=None):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        idx_subs = []
        points = {d: [] for d in self._gdim}
        mapper = {d: [] for d in self._gdim}
        pdim = self.sfunction._sparse_dim
    
        # Positon map and temporaries for it
        pmap = self.sfunction._coordinate_indices

        # Temporaries for the position
        temps = self._positions(implicit_dims)

        # Coefficient symbol expression
        temps.extend(self._coeff_temps(implicit_dims))

        # Create positions and indices temporaries/indirections
        pr = []
        for ((di, d), pos, rd) in zip(enumerate(self._gdim), pmap, self._rdim):
            p = Symbol(name='ii_%s_%s' % (self.sfunction.name, d.name))
            temps.extend([Eq(p, pos + rd, implicit_dims=implicit_dims + tuple(pr))])

            # Add conditional to avoid OOB
            lb = sympy.And(p >= d.symbolic_min-self.r, evaluate=False)
            ub = sympy.And(p <= d.symbolic_max+self.r, evaluate=False)
            condition = sympy.And(lb, ub, evaluate=False)
            mapper[d] = ConditionalDimension(p.name, self.sfunction._sparse_dim,
                                             condition=condition, indirect=True)
            pr.append(rd)

        # Substitution mapper for variables
        idx_subs = {v: v.subs({k: c - v.origin.get(k, 0) for (k, c) in mapper.items()})
                    for v in variables}

        return idx_subs, temps

    def interpolate(self, expr, increment=False, self_subs={}, implicit_dims=None):
        """
        Generate equations interpolating an arbitrary expression into ``self``.

        Parameters
        ----------
        expr : expr-like
            Input expression to interpolate.
        increment: bool, optional
            If True, generate increments (Inc) rather than assignments (Eq).
        implicit_dims : Dimension or list of Dimension, optional
            An ordered list of Dimensions that do not explicitly appear in the
            interpolation expression, but that should be honored when constructing
            the operator.
        """
        implicit_dims = self._augment_implicit_dims(implicit_dims)

        def callback():
            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _expr = expr.evaluate
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _expr = expr

            variables = list(retrieve_function_carriers(_expr))

            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interp_idx(variables, implicit_dims=implicit_dims)

            # Accumulate point-wise contributions into a temporary
            rhs = Symbol(name='sum', dtype=self.sfunction.dtype)
            summands = [Eq(rhs, 0., implicit_dims=implicit_dims)]
            # Substitute coordinate base symbols into the interpolation coefficients
            summands.extend([Inc(rhs, _expr.xreplace(idx_subs) * self._weights,
                                 implicit_dims=implicit_dims + self._rdim)])

            # Write/Incr `self`
            lhs = self.sfunction.subs(self_subs)
            ecls = Inc if increment else Eq
            last = [ecls(lhs, rhs, implicit_dims=implicit_dims)]

            return [summands[0]] + temps + summands[1:] + last

        return Interpolation(expr, increment, self_subs, self, callback)

    def inject(self, field, expr, implicit_dims=None):
        """
        Generate equations injecting an arbitrary expression into a field.

        Parameters
        ----------
        field : Function
            Input field into which the injection is performed.
        expr : expr-like
            Injected expression.
        implicit_dims : Dimension or list of Dimension, optional
            An ordered list of Dimensions that do not explicitly appear in the
            injection expression, but that should be honored when constructing
            the operator.
        """
        implicit_dims = self._augment_implicit_dims(implicit_dims)

        def callback():
            # Make iterable to support inject((u, v), expr=expr)
            # or inject((u, v), expr=(expr1, expr2))
            fields, exprs = as_tuple(field), as_tuple(expr)
            # Provide either one expr per field or on expr for all fields
            if len(fields) > 1:
                if len(exprs) == 1:
                    exprs = tuple(exprs[0] for _ in fields)
                else:
                    assert len(exprs) == len(fields)

            # Derivatives must be evaluated before the introduction of indirect accesses
            try:
                _exprs = tuple(e.evaluate for e in exprs)
            except AttributeError:
                # E.g., a generic SymPy expression or a number
                _exprs = exprs

            variables = list(v for e in _exprs for v in retrieve_function_carriers(e))
            variables = variables + list(fields)

            # List of indirection indices for all adjacent grid points
            idx_subs, temps = self._interp_idx(variables, implicit_dims=implicit_dims)

            # Substitute coordinate base symbols into the interpolation coefficients
            eqns = [Inc(_field.xreplace(idx_subs),
                        _expr.xreplace(idx_subs) * self._weights,
                        implicit_dims=implicit_dims + self._rdim)
                    for (_field, _expr) in zip(fields, _exprs)]

            return temps + eqns

        return Injection(field, expr, self, callback)


class LinearInterpolator(WeightedInterpolator):
    """
    Concrete implementation of WeightedInterpolator implementing a Linear interpolation
    scheme, i.e. Bilinear for 2D and Trilinear for 3D problems.

    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """
    @property
    def _weights(self):
        c = [(1 - p) * (1 - (rd - rd._symbolic_min)) + (rd - rd._symbolic_min) * p
             for (p, d, rd) in zip(self._point_symbols, self._gdim, self._rdim)]
        return prod(c)

    @cached_property
    def _point_symbols(self):
        """Symbol for coordinate value in each Dimension of the point."""
        return DimensionTuple(*(Symbol(name='p%s' % d, dtype=self.sfunction.dtype)
                                for d in self.grid.dimensions),
                              getters=self.grid.dimensions)

    def _coeff_temps(self, implicit_dims):
        # Positions
        pmap = self.sfunction._position_map
        poseq = [Eq(self._point_symbols[d], pos - floor(pos),
                    implicit_dims=implicit_dims)
                 for (d, pos) in zip(self._gdim, pmap.keys())]
        return poseq


class PrecomputedInterpolator(WeightedInterpolator):
    """
    Concrete implementation of WeightedInterpolator implementing a Precomputed
    interpolation scheme, i.e. an interpolation with user provided precomputed
    weigths/coefficients.

    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """

    def _positions(self, implicit_dims):
        if self.sfunction.gridpoints is None:
            return super()._positions(implicit_dims)
        # No position temp as we have directly the gridpoints
        return [Eq(p, k, implicit_dims=implicit_dims)
                for (k, p) in self.sfunction._position_map.items()]

    @property
    def interpolation_coeffs(self):
        return self.sfunction.interpolation_coeffs

    @property
    def _weights(self):
        ddim, cdim = self.interpolation_coeffs.dimensions[1:]
        return prod([self.interpolation_coeffs.subs({ddim: ri, cdim: rd-rd._symbolic_min})
                     for (ri, rd) in enumerate(self._rdim)])
