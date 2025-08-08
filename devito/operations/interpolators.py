from abc import ABC, abstractmethod
from functools import wraps, cached_property

import sympy
import numpy as np

try:
    from scipy.special import i0
except ImportError:
    from numpy import i0

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.elementary import floor
from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, retrieve_functions, INT
from devito.tools import as_tuple, flatten, filter_ordered, Pickable, memoized_meth
from devito.types import (ConditionalDimension, Eq, Inc, Evaluable, Symbol,
                          CustomDimension, SubFunction)
from devito.types.utils import DimensionTuple

__all__ = ['LinearInterpolator', 'PrecomputedInterpolator', 'SincInterpolator']


def check_radius(func):
    @wraps(func)
    def wrapper(interp, *args, **kwargs):
        r = interp.sfunction.r
        funcs = set().union(*[retrieve_functions(a) for a in args])
        so = min({f.space_order for f in funcs if not f.is_SparseFunction} or {r})
        if so < r:
            raise ValueError(f"Space order {so} too small for interpolation r {r}")
        return func(interp, *args, **kwargs)
    return wrapper


def _extract_subdomain(variables):
    """
    Check if any of the variables provided are defined on a SubDomain
    and extract it if this is the case.
    """
    sdms = set()
    for v in variables:
        try:
            if v.grid.is_SubDomain:
                sdms.add(v.grid)
        except AttributeError:
            # Variable not on a grid (Indexed for example)
            pass

    if len(sdms) > 1:
        raise NotImplementedError("Sparse operation on multiple Functions defined on"
                                  " different SubDomains currently unsupported")
    elif len(sdms) == 1:
        return sdms.pop()
    return None


class UnevaluatedSparseOperation(sympy.Expr, Evaluable, Pickable):

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
    __rargs__ = ('interpolator',)

    def __new__(cls, interpolator):
        obj = super().__new__(cls)

        obj.interpolator = interpolator

        return obj

    def _evaluate(self, **kwargs):
        return_value = self.operation(**kwargs)
        assert(all(isinstance(i, Eq) for i in return_value))
        return return_value

    @abstractmethod
    def operation(self, **kwargs):
        pass

    def __add__(self, other):
        return flatten([self, other])

    def __radd__(self, other):
        return flatten([other, self])


class Interpolation(UnevaluatedSparseOperation):

    """
    Represents an Interpolation operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    __rargs__ = ('expr', 'increment', 'implicit_dims', 'self_subs') + \
        UnevaluatedSparseOperation.__rargs__

    def __new__(cls, expr, increment, implicit_dims, self_subs, interpolator):
        obj = super().__new__(cls, interpolator)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.expr = expr
        obj.increment = increment
        obj.self_subs = self_subs
        obj.implicit_dims = implicit_dims

        return obj

    def operation(self, **kwargs):
        return self.interpolator._interpolate(expr=self.expr, increment=self.increment,
                                              self_subs=self.self_subs,
                                              implicit_dims=self.implicit_dims)

    def __repr__(self):
        return (f"Interpolation({repr(self.expr)} into "
                f"{repr(self.interpolator.sfunction)})")


class Injection(UnevaluatedSparseOperation):

    """
    Represents an Injection operation performed on a SparseFunction.
    Evaluates to a list of Eq objects.
    """

    __rargs__ = ('field', 'expr', 'implicit_dims') + UnevaluatedSparseOperation.__rargs__

    def __new__(cls, field, expr, implicit_dims, interpolator):
        obj = super().__new__(cls, interpolator)

        # TODO: unused now, but will be necessary to compute the adjoint
        obj.field = field
        obj.expr = expr
        obj.implicit_dims = implicit_dims

        return obj

    def operation(self, **kwargs):
        return self.interpolator._inject(expr=self.expr, field=self.field,
                                         implicit_dims=self.implicit_dims)

    def __repr__(self):
        return f"Injection({repr(self.expr)} into {repr(self.field)})"


class GenericInterpolator(ABC):

    """
    Abstract base class defining the interface for an interpolator.
    """

    _name = "generic"

    @abstractmethod
    def inject(self, *args, **kwargs):
        pass

    @abstractmethod
    def interpolate(self, *args, **kwargs):
        pass

    @property
    def name(self):
        return self._name

    def _arg_defaults(self, **args):
        return {}


class WeightedInterpolator(GenericInterpolator):

    """
    Represent an Interpolation operation on a SparseFunction that is separable
    in space, meaning the coefficients are defined for each Dimension separately
    and multiplied at a given point: `w[x, y] = wx[x] * wy[y]`
    """

    _name = 'weighted'

    def __init__(self, sfunction):
        self.sfunction = sfunction

    @property
    def grid(self):
        return self.sfunction.grid

    @property
    def r(self):
        return self.sfunction.r

    @memoized_meth
    def _weights(self, subdomain=None):
        raise NotImplementedError

    @property
    def _gdims(self):
        return self.grid.dimensions

    @cached_property
    def _cdim(self):
        """Base CustomDimensions used to construct _rdim"""
        parent = self.sfunction._sparse_dim
        dims = [CustomDimension(f"r{self.sfunction.name}{d.name}",
                                -self.r+1, self.r, 2*self.r, parent)
                for d in self._gdims]
        return dims

    @memoized_meth
    def _rdim(self, subdomain=None):
        # If the interpolation operation is limited to a SubDomain,
        # use the SubDimensions of that SubDomain
        if subdomain:
            gdims = tuple(subdomain.dimension_map[d] for d in self._gdims)
        else:
            gdims = self._gdims

        # Make radius dimension conditional to avoid OOB
        rdims = []
        pos = self.sfunction._position_map.values()

        for (d, rd, p) in zip(gdims, self._cdim, pos):
            # Add conditional to avoid OOB
            lb = sympy.And(rd + p >= d.symbolic_min - self.r, evaluate=False)
            ub = sympy.And(rd + p <= d.symbolic_max + self.r, evaluate=False)
            cond = sympy.And(lb, ub, evaluate=False)

            # Insert a check to catch cases where interpolation/injection is
            # into an empty rank. This depends on the injection field or interpolated
            # expression, and so must be inserted here.
            if subdomain and subdomain.distributor.is_parallel:
                rank_populated = subdomain.distributor.rank_populated
                cond = sympy.And(rank_populated, cond)

            rdims.append(ConditionalDimension(rd.name, rd, condition=cond,
                                              indirect=True))

        return DimensionTuple(*rdims, getters=gdims)

    def _augment_implicit_dims(self, implicit_dims, extras=None):
        if extras is not None:
            # If variables are defined on a SubDomain of the Grid, then omit the
            # dimensions of that SubDomain from any extra dimensions found
            edims = []
            for v in extras:
                try:
                    if v.grid.is_SubDomain:
                        edims.extend([d for d in v.grid.dimensions
                                      if d.is_Sub and d.root in self._gdims])
                except AttributeError:
                    pass

            gdims = filter_ordered(edims + list(self._gdims))
            extra = filter_ordered([i for v in extras for i in v.dimensions
                                    if i not in gdims and
                                    i not in self.sfunction.dimensions])
            extra = tuple(extra)
        else:
            extra = tuple()

        if self.sfunction._sparse_position == -1:
            idims = self.sfunction.dimensions + as_tuple(implicit_dims) + extra
        else:
            idims = extra + as_tuple(implicit_dims) + self.sfunction.dimensions
        return tuple(idims)

    def _coeff_temps(self, implicit_dims):
        return []

    def _positions(self, implicit_dims):
        return [Eq(v, INT(floor(k)), implicit_dims=implicit_dims)
                for k, v in self.sfunction._position_map.items()]

    def _interp_idx(self, variables, implicit_dims=None, pos_only=(), subdomain=None):
        """
        Generate interpolation indices for the DiscreteFunctions in ``variables``.
        """
        pos = self.sfunction._position_map.values()

        # Temporaries for the position
        temps = self._positions(implicit_dims)

        # Coefficient symbol expression
        temps.extend(self._coeff_temps(implicit_dims))

        # Substitution mapper for variables
        mapper = self._rdim(subdomain=subdomain).getters

        # Index substitution to make in variables
        subs = {ki: c + p for ((k, c), p)
                in zip(mapper.items(), pos) for ki in {k, k.root}}

        idx_subs = {v: v.subs(subs) for v in variables}

        # Position only replacement, not radius dependent.
        # E.g src.inject(vp(x)*src) needs to use vp[posx] at all points
        # not vp[posx + rx]
        idx_subs.update({v: v.subs({k: p for (k, p) in zip(mapper, pos)})
                         for v in pos_only})

        return idx_subs, temps

    @check_radius
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
        return Interpolation(expr, increment, implicit_dims, self_subs, self)

    @check_radius
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
        return Injection(field, expr, implicit_dims, self)

    def _interpolate(self, expr, increment=False, self_subs={}, implicit_dims=None):
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
        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            _expr = expr.evaluate
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            _expr = expr

        variables = list(retrieve_function_carriers(_expr))
        subdomain = _extract_subdomain(variables)

        # Implicit dimensions
        implicit_dims = self._augment_implicit_dims(implicit_dims, variables)

        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interp_idx(variables, implicit_dims=implicit_dims,
                                           subdomain=subdomain)

        # Accumulate point-wise contributions into a temporary
        rhs = Symbol(name='sum', dtype=self.sfunction.dtype)
        summands = [Eq(rhs, 0., implicit_dims=implicit_dims)]
        # Substitute coordinate base symbols into the interpolation coefficients
        weights = self._weights(subdomain=subdomain)
        summands.extend([Inc(rhs, (weights * _expr).xreplace(idx_subs),
                             implicit_dims=implicit_dims)])

        # Write/Incr `self`
        lhs = self.sfunction.subs(self_subs)
        ecls = Inc if increment else Eq
        last = [ecls(lhs, rhs, implicit_dims=implicit_dims)]

        return temps + summands + last

    def _inject(self, field, expr, implicit_dims=None):
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
        # Make iterable to support inject((u, v), expr=expr)
        # or inject((u, v), expr=(expr1, expr2))
        fields, exprs = as_tuple(field), as_tuple(expr)

        # Provide either one expr per field or on expr for all fields
        if len(fields) > 1:
            if len(exprs) == 1:
                exprs = tuple(exprs[0] for _ in fields)
            else:
                assert len(exprs) == len(fields)

        subdomain = _extract_subdomain(fields)

        # Derivatives must be evaluated before the introduction of indirect accesses
        try:
            _exprs = tuple(e.evaluate for e in exprs)
        except AttributeError:
            # E.g., a generic SymPy expression or a number
            _exprs = exprs

        variables = list(v for e in _exprs for v in retrieve_function_carriers(e))

        # Implicit dimensions
        implicit_dims = self._augment_implicit_dims(implicit_dims, variables)
        # Move all temporaries inside inner loop to improve parallelism
        # Can only be done for inject as interpolation need a temporary
        # summing temp that wouldn't allow collapsing
        implicit_dims = implicit_dims + tuple(r.parent for r in
                                              self._rdim(subdomain=subdomain))

        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interp_idx(fields, implicit_dims=implicit_dims,
                                           pos_only=variables, subdomain=subdomain)

        # Substitute coordinate base symbols into the interpolation coefficients
        eqns = [Inc(_field.xreplace(idx_subs),
                    (self._weights(subdomain=subdomain) * _expr).xreplace(idx_subs),
                    implicit_dims=implicit_dims)
                for (_field, _expr) in zip(fields, _exprs)]

        return temps + eqns


class LinearInterpolator(WeightedInterpolator):
    """
    Concrete implementation of WeightedInterpolator implementing a Linear interpolation
    scheme, i.e. Bilinear for 2D and Trilinear for 3D problems.

    Parameters
    ----------
    sfunction: The SparseFunction that this Interpolator operates on.
    """

    _name = 'linear'

    @memoized_meth
    def _weights(self, subdomain=None):
        rdim = self._rdim(subdomain=subdomain)
        c = [(1 - p) * (1 - r) + p * r
             for (p, d, r) in zip(self._point_symbols, self._gdims, rdim)]
        return Mul(*c)

    @cached_property
    def _point_symbols(self):
        """Symbol for coordinate value in each Dimension of the point."""
        dtype = self.sfunction.coordinates.dtype
        return DimensionTuple(*(Symbol(name=f'p{d}', dtype=dtype)
                                for d in self.grid.dimensions),
                              getters=self.grid.dimensions)

    def _coeff_temps(self, implicit_dims):
        # Positions
        pmap = self.sfunction._position_map
        poseq = [Eq(self._point_symbols[d], pos - floor(pos),
                    implicit_dims=implicit_dims)
                 for (d, pos) in zip(self._gdims, pmap.keys())]
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

    _name = 'precomp'

    def _positions(self, implicit_dims):
        if self.sfunction.gridpoints_data is None:
            return super()._positions(implicit_dims)
        else:
            # No position temp as we have directly the gridpoints
            return[Eq(p, k, implicit_dims=implicit_dims)
                   for (k, p) in self.sfunction._position_map.items()]

    @property
    def interpolation_coeffs(self):
        return self.sfunction.interpolation_coeffs

    @memoized_meth
    def _weights(self, subdomain=None):
        ddim, cdim = self.interpolation_coeffs.dimensions[1:]
        mappers = [{ddim: ri, cdim: rd-rd.parent.symbolic_min}
                   for (ri, rd) in enumerate(self._rdim(subdomain=subdomain))]
        return Mul(*[self.interpolation_coeffs.subs(mapper)
                     for mapper in mappers])


class SincInterpolator(PrecomputedInterpolator):
    """
    Hicks windowed sinc interpolation scheme.

    Arbitrary source and receiver positioning in finite‐difference schemes
    using Kaiser windowed sinc functions

    https://library.seg.org/doi/10.1190/1.1451454

    """

    _name = 'sinc'

    # Table 1
    _b_table = {2: 2.94, 3: 4.53,
                4: 4.14, 5: 5.26, 6: 6.40,
                7: 7.51, 8: 8.56, 9: 9.56, 10: 10.64}

    def __init__(self, sfunction):
        if i0 is np.i0:
            warning("""
Using `numpy.i0`. We (and numpy) recommend to install scipy to improve the performance
of the SincInterpolator that uses i0 (Bessel function).
""")
        super().__init__(sfunction)

    @cached_property
    def interpolation_coeffs(self):
        coeffs = []
        shape = (self.sfunction.npoint, 2 * self.r)
        for r in self._cdim:
            dimensions = (self.sfunction._sparse_dim, r)
            sf = SubFunction(name=f"wsinc{r.name}", dtype=self.sfunction.dtype,
                             shape=shape, dimensions=dimensions,
                             space_order=0, alias=self.sfunction.alias,
                             parent=None)
            coeffs.append(sf)
        return tuple(coeffs)

    @memoized_meth
    def _weights(self, subdomain=None):
        rdims = self._rdim(subdomain=subdomain)
        return Mul(*[w._subs(rd, rd-rd.parent.symbolic_min)
                     for (rd, w) in zip(rdims, self.interpolation_coeffs)])

    def _arg_defaults(self, coords=None, sfunc=None):
        args = {}
        b = self._b_table[self.r]
        b0 = i0(b)
        if coords is None or sfunc is None:
            raise ValueError("No coordinates or sparse function provided")
        # Coords to indices
        coords = coords / np.array(sfunc.grid.spacing)
        coords = coords - np.floor(coords)

        # Precompute sinc
        for j in range(len(self._gdims)):
            data = np.zeros((coords.shape[0], 2*self.r), dtype=sfunc.dtype)
            for ri in range(2*self.r):
                rpos = ri - self.r + 1 - coords[:, j]
                num = i0(b*np.sqrt(1 - (rpos/self.r)**2))
                data[:, ri] = num / b0 * np.sinc(rpos)
            args[self.interpolation_coeffs[j].name] = data

        return args
