from abc import ABC, abstractmethod
from contextlib import suppress
from functools import cached_property, wraps
from itertools import groupby

import numpy as np
import sympy

try:
    from scipy.special import i0
except ImportError:
    from numpy import i0

from devito.finite_differences.differentiable import Mul
from devito.finite_differences.elementary import floor
from devito.logger import warning
from devito.symbolics import INT, retrieve_function_carriers, retrieve_functions
from devito.tools import (
    Pickable, as_fp64_decimal, as_list, as_tuple, filter_ordered, flatten, memoized_meth
)
from devito.types import CustomDimension, Eq, Evaluable, Inc, SubFunction, Symbol
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


def check_coords(func):
    @wraps(func)
    def wrapper(interp, *args, **kwargs):
        inputs = args + as_tuple(kwargs.get('expr', ()))

        # SubFunction of the SparseFunction use to create the interpolator
        sfunc = interp.sfunction

        # SubFunctions found in the arguments of the interpolation/injection operation
        a_sfuncs = {f for f in retrieve_functions(inputs)
                    if f.is_SparseFunction} - {sfunc}
        if not a_sfuncs:
            # Only uses the the interpolator's SparseFunction, so no need to check
            return func(interp, *args, **kwargs)

        # Check that it uses the same coordinates as the interpolator's SparseFunction
        subfuncs = {getattr(sfunc, s, None) for s in sfunc._sub_functions}
        for f in a_sfuncs:
            for s in f._sub_functions:
                if getattr(f, s, None) not in subfuncs:
                    raise ValueError(f"Interpolation/injection with {sfunc}"
                                     f"requires {f} "
                                     f"to use the same {s} as {sfunc}")

        return func(interp, *args, **kwargs)
    return wrapper


def _extract_subdomain(variables):
    """
    Check if any of the variables provided are defined on a SubDomain
    and extract it if this is the case.
    """
    sdms = set()
    for v in variables:
        with suppress(AttributeError):
            if v.grid.is_SubDomain:
                sdms.add(v.grid)

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

    __str__ = __repr__


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

    __str__ = __repr__


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
    def _weights(self, subdomain=None, shifts=None):
        raise NotImplementedError

    @property
    def _gdims(self):
        return self.grid.dimensions

    @property
    def _cdim(self):
        """Base CustomDimensions used to construct _rdim"""
        dims = [self.sfunction._crdim(d) for d in self._gdims]
        return dims

    def _field_shifts(self, field):
        """
        Per-grid-Dimension half-cell shift induced by `field`'s staggering
        (e.g. `h_x/2` for a field staggered in `x`). Returns None for
        unstaggered fields. SubDomain-induced origin offsets are deliberately
        ignored — they are not staggering.
        """
        staggered = field.staggered
        if not staggered or staggered.on_node:
            return ()
        return tuple((d.spacing / 2) if s else 0
                     for d, s in zip(field.dimensions, staggered, strict=True)
                     if d.is_Space)

    @memoized_meth
    def _rdim(self, subdomain=None, shifts=None):
        # If the interpolation operation is limited to a SubDomain,
        # use the SubDimensions of that SubDomain
        if subdomain:
            gdims = tuple(subdomain.dimension_map[d] for d in self._gdims)
        else:
            gdims = self._gdims

        # Make radius dimension conditional to avoid OOB
        rdims = []
        pos = self.sfunction._position_map(shifts=shifts).values()

        for (d, rd, p) in zip(gdims, self._cdim, pos, strict=True):
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

            rdims.append(self.sfunction._cond_rdim(d.root, cond))

        return DimensionTuple(*rdims, getters=gdims)

    def _augment_implicit_dims(self, implicit_dims, extras=None):
        if extras is not None:
            # If variables are defined on a SubDomain of the Grid, then omit the
            # dimensions of that SubDomain from any extra dimensions found
            edims = []
            for v in extras:
                with suppress(AttributeError):
                    if v.grid.is_SubDomain:
                        edims.extend([d for d in v.grid.dimensions
                                      if d.is_Sub and d.root in self._gdims])

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

    def _coeff_temps(self, implicit_dims, shifts=None):
        return []

    def _positions(self, implicit_dims, shifts=None):
        return [Eq(v, INT(floor(k)), implicit_dims=implicit_dims)
                for k, v in self.sfunction._position_map(shifts=shifts).items()]

    def _interp_idx(self, variables, implicit_dims=None, subdomain=None,
                    shifts=None):
        """
        Generate interpolation indices for the DiscreteFunctions in `variables`.

        `shifts` is a per-Dimension physical offset for the target field's
        origin: it only affects the integer position symbol via the position
        map (`pos = floor((c - o - shift)/h)`). The index substitution itself
        is unchanged — any staggered offset in a field's own symbolic access is
        absorbed by Devito's normal indexing.
        """
        pos = self.sfunction._position_map(shifts=shifts).values()

        # Temporaries for the position
        temps = self._positions(implicit_dims, shifts=shifts)

        # Coefficient symbol expression
        temps.extend(self._coeff_temps(implicit_dims, shifts=shifts))

        # Substitution mapper for variables
        mapper = self._rdim(subdomain=subdomain, shifts=shifts).getters

        # Index substitution to make in variables
        subs = {
            ki: c + p
            for ((k, c), p) in zip(mapper.items(), pos, strict=True)
            for ki in {k, k.root}
        }

        idx_subs = {v: v.subs(subs) for v in variables}

        return idx_subs, temps

    @check_radius
    @check_coords
    def interpolate(self, expr, increment=False, self_subs=None, implicit_dims=None):
        """
        Generate equations interpolating an arbitrary expression into `self`.

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
        if self_subs is None:
            self_subs = {}
        return Interpolation(expr, increment, implicit_dims, self_subs, self)

    @check_radius
    @check_coords
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

    def _interpolate(self, expr, increment=False, self_subs=None, implicit_dims=None):
        """
        Generate equations interpolating an arbitrary expression into `self`.

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
        with suppress(AttributeError):
            expr = expr._eval_at(self.sfunction).evaluate

        if self_subs is None:
            self_subs = {}

        variables = list(retrieve_function_carriers(expr))
        subdomain = _extract_subdomain(variables)

        # Implicit dimensions
        implicit_dims = self._augment_implicit_dims(implicit_dims, variables)

        # List of indirection indices for all adjacent grid points
        idx_subs, temps = self._interp_idx(variables, implicit_dims=implicit_dims,
                                           subdomain=subdomain)

        # Accumulate point-wise contributions into a temporary
        rhs = Symbol(name=f'sum{self.sfunction.name}', dtype=self.sfunction.dtype)
        summands = [Eq(rhs, 0., implicit_dims=implicit_dims)]
        # Substitute coordinate base symbols into the interpolation coefficients
        weights = self._weights(subdomain=subdomain)
        summands.extend([Inc(rhs, (weights * expr).xreplace(idx_subs),
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

        # Derivatives must be evaluated before the introduction of indirect
        # accesses. Variables are sampled at their own grid location; the
        # position map for the target field carries the staggering so the
        # field's stencil neighbors land on the right indices.
        with suppress(AttributeError):
            exprs = tuple(e._eval_at(f).evaluate
                          for e, f in zip(exprs, fields, strict=True))

        eqns = []
        temps = []
        # We need to create one set of equations (temps and and coeffs) per staggering
        # field in which we inject as the reference index depends on the field's origin
        for _, g in groupby(zip(fields, exprs, strict=True), lambda f: f[0].staggered):
            g_fields, g_exprs = zip(*g, strict=True)
            variables = list(v for e in g_exprs for v in retrieve_function_carriers(e))

            implicit_dims = self._augment_implicit_dims(implicit_dims, variables)

            # All fields in a single injection share the same staggering by
            # construction (they are written together at the same indices), so
            # derive shifts from the first field.
            shifts = self._field_shifts(g_fields[0])

            # Move all temporaries inside inner loop to improve parallelism
            # Can only be done for inject as interpolation needs a summing temp
            # that wouldn't allow collapsing
            implicit_dims = implicit_dims + tuple(r.parent for r in
                                                  self._rdim(subdomain=subdomain,
                                                             shifts=shifts))

            # List of indirection indices for all adjacent grid points
            idx_subs, _temps = self._interp_idx(list(g_fields) + variables,
                                                implicit_dims=implicit_dims,
                                                subdomain=subdomain, shifts=shifts)

            w = self._weights(subdomain=subdomain, shifts=shifts)
            temps.extend(_temps)
            eqns.extend([Inc(f.xreplace(idx_subs), (w * e).xreplace(idx_subs),
                             implicit_dims=implicit_dims)
                         for f, e in zip(g_fields, g_exprs, strict=True)])

        return filter_ordered(temps) + eqns


def _shift_tag(shifts):
    """Suffix used to distinguish per-staggering table names ("_s10", ...)."""
    if not shifts or not any(shifts):
        return ''
    return '_s' + ''.join('1' if s else '0' for s in shifts)


def _shift_values(shifts, grid, spacing):
    """Physical half-cell offsets for each grid dim, as fp64."""
    if not shifts:
        return np.zeros(grid.dim, dtype=np.float64)
    subs = {d.spacing: float(h)
            for d, h in zip(grid.dimensions, spacing, strict=True)}
    return np.array([float(sympy.sympify(s).xreplace(subs)) for s in shifts])


class _HostTable(SubFunction):
    """SubFunction populated on the host by the parent interpolator's
    `_arg_defaults` from the already-scattered coordinates, exactly the way
    sinc's precomputed weights are populated.

    `parent` links the table to its SparseFunction so the base
    `SubFunction._arg_values` routing triggers `SparseFunction._arg_defaults`
    -> `LinearInterpolator._arg_defaults`. The link is *not* preserved by
    pickle (parent is not in `__rkwargs__`) so decoupled workers don't ship
    the sfunction back through every table."""

    def _arg_apply(self, *args, **kwargs):
        return


class Gridpoints(_HostTable):
    """int32 cell indices per sparse point, shape ``(npoint, ndim)``."""


class Coeffs(_HostTable):
    """Per-dim `(1 - frac, frac)` interpolation weights, shape ``(npoint, 2)``."""


def _resolved_geometry(grid, kwargs):
    """Fp64 (spacing, origin) tuple honoring runtime `h_x`/`o_x`/... overrides."""
    spacing = np.array([as_fp64_decimal(kwargs.get(s.name, v)) for s, v
                        in zip(grid.spacing_symbols, grid.spacing, strict=True)])
    origin = np.array([as_fp64_decimal(kwargs.get(o.name, v)) for o, v
                       in zip(grid.origin_symbols, grid.origin, strict=True)])
    return spacing, origin


def _positions_fp64(coords, grid, shifts, spacing, origin):
    """Fp64 fractional grid position `(c - o - shift)/h` for each sparse point,
    computed in double precision to avoid boundary-crossing rounding errors
    that fp32 `floor((c - o)/h)` can produce for coordinates that sit on an
    exact cell boundary."""
    c64 = np.asarray(coords, dtype=np.float64)
    return (c64 - origin - _shift_values(shifts, grid, spacing)) / spacing


def _cell_indices(coords, grid, shifts, spacing, origin):
    """Int32 base cell index per sparse point, one entry per grid Dimension."""
    return np.floor(_positions_fp64(coords, grid, shifts, spacing,
                                    origin)).astype(np.int32)


def _linear_weights(coords, grid, shifts, j, dtype, spacing, origin):
    """`(1 - frac, frac)` linear interpolation weights along dim `j`; `frac`
    is the fractional cell offset from the base index returned by
    `_cell_indices`."""
    pos = _positions_fp64(coords, grid, shifts, spacing, origin)
    frac = pos[:, j] - np.floor(pos[:, j])
    data = np.empty((pos.shape[0], 2), dtype=dtype)
    data[:, 0] = 1.0 - frac
    data[:, 1] = frac
    return data


class LinearInterpolator(WeightedInterpolator):
    """
    Linear (bilinear/trilinear) interpolator.

    Gridpoints and per-dim `(1-frac, frac)` weights are precomputed on the
    host in fp64 (see `_arg_defaults`) and passed to the kernel as int32/fp
    SubFunctions. The generated C only indexes those tables and never sees
    `(c-o)/h` or `floor` on fp32.
    """

    _name = 'linear'

    def __init__(self, sfunction, shifts=()):
        super().__init__(sfunction)
        # Every shift set the interpolator has been asked to produce tables
        # for. Persisted with the parent SparseFunction via `__rkwargs__` so
        # a pickled/rebuilt interpolator (decoupled workers) knows which
        # tables to emit in `_arg_defaults`.
        self._shifts_used = set(tuple(s) if s else None for s in shifts)

    @cached_property
    def _coeff_dtype(self):
        # Weights are real even for complex-valued sparse fields.
        dtype = np.dtype(self.sfunction.dtype)
        if np.issubdtype(dtype, np.complexfloating):
            return np.finfo(dtype).dtype.type
        return dtype.type

    @memoized_meth
    def _generate_coeffs(self, key):
        """Create the ``(gridpoints, coeffs_per_dim)`` SubFunction tuple for
        a given shift set. ``key`` is either ``None`` (plain, non-staggered)
        or a tuple of per-dim shifts. Mirrors sinc's ``interpolation_coeffs``
        cached_property but keyed on ``shifts``."""
        # Record that the caller has emitted tables for this shift set, so
        # `_arg_defaults` can regenerate them on the fly.
        self._shifts_used.add(key)

        shifts = as_list(key)
        tag = _shift_tag(shifts)
        sfname = self.sfunction.name
        sfdim = self.sfunction._sparse_dim

        # Gridpoints: `(npoint, ndim)` int32 base cell index per sparse point.
        gp_name = f'{sfname}_gp{tag}'
        ddim = CustomDimension(f'{gp_name}d', 0, self.grid.dim - 1,
                               self.grid.dim, sfdim)
        gp = Gridpoints(name=gp_name, dtype=np.int32,
                        shape=(self.sfunction.npoint, self.grid.dim),
                        dimensions=(sfdim, ddim), space_order=0,
                        alias=self.sfunction.alias,
                        parent=self.sfunction)

        # Per-dim linear weights: `(npoint, 2)` holding `(1 - frac, frac)`.
        coeffs = tuple(
            Coeffs(name=f'{sfname}_w{d.name}{tag}',
                   dtype=self._coeff_dtype,
                   shape=(self.sfunction.npoint, 2),
                   dimensions=(sfdim, r), space_order=0,
                   alias=self.sfunction.alias,
                   parent=self.sfunction)
            for d, r in zip(self._gdims, self._cdim, strict=True)
        )

        return gp, coeffs

    def _gridpoints(self, shifts=None):
        return self._generate_coeffs(tuple(shifts) if shifts else None)[0]

    def _coeffs(self, shifts=None):
        return self._generate_coeffs(tuple(shifts) if shifts else None)[1]

    def _positions(self, implicit_dims, shifts=None):
        gp = self._gridpoints(shifts=shifts)
        ddim = gp.dimensions[-1]
        return [Eq(p, gp._subs(ddim, di), implicit_dims=implicit_dims)
                for (di, p) in enumerate(
                    self.sfunction._pos_symbols(shifts=shifts))]

    def _coeff_temps(self, implicit_dims, shifts=None):
        return []

    @memoized_meth
    def _weights(self, subdomain=None, shifts=None):
        rdims = self._rdim(subdomain=subdomain, shifts=shifts)
        coeffs = self._coeffs(shifts=shifts)
        return Mul(*[
            w._subs(rd, rd - rd.parent.symbolic_min)
            for (rd, w) in zip(rdims, coeffs, strict=True)
        ])

    def _arg_defaults(self, coords=None, sfunc=None, origin=None):
        """Fill the gridpoints/coeffs tables from the already-scattered
        ``coords`` handed in by ``SparseFunction._arg_defaults``. Mirrors
        sinc's `_arg_defaults`: regenerates the tables from the persisted
        shift set, so a pickled/rebuilt interpolator (decoupled workers)
        still emits data for every table the operator was compiled with."""
        if coords is None or sfunc is None:
            raise ValueError("No coordinates or sparse function provided")

        # Fp64 grid geometry -- avoids fp32 rounding on cell boundaries.
        grid = sfunc.grid
        spacing = np.array([as_fp64_decimal(h) for h in grid.spacing])
        origin = np.array([as_fp64_decimal(o)
                           for o in (origin or grid.origin)])

        args = {}
        for key in self._shifts_used or {None}:
            shifts = as_list(key)
            gp, coeffs = self._generate_coeffs(key)

            # Tabulated data (int32 cell indices + fp linear weights per dim).
            args[gp.name] = _cell_indices(coords, grid, shifts, spacing, origin)
            for i, w in enumerate(coeffs):
                args[w.name] = _linear_weights(
                    coords, grid, shifts, i, w.dtype, spacing, origin
                )

            # Bounds for each table's dimensions, matching the computed data.
            for f in (gp, *coeffs):
                for d, s in zip(f.dimensions, args[f.name].shape, strict=True):
                    args.update(d._arg_defaults(_min=0, size=s))

        return args


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

    def _positions(self, implicit_dims, shifts=None):
        if self.sfunction.gridpoints_data is None:
            return super()._positions(implicit_dims, shifts=shifts)
        else:
            # No position temp as we have directly the gridpoints
            return[Eq(p, k, implicit_dims=implicit_dims)
                   for (k, p) in self.sfunction._position_map(shifts=shifts).items()]

    @property
    def interpolation_coeffs(self):
        return self.sfunction.interpolation_coeffs

    @memoized_meth
    def _weights(self, subdomain=None, shifts=None):
        ddim, cdim = self.interpolation_coeffs.dimensions[1:]
        mappers = [{ddim: ri, cdim: rd-rd.parent.symbolic_min}
                   for (ri, rd) in enumerate(self._rdim(subdomain=subdomain,
                                                        shifts=shifts))]
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
    def _weights(self, subdomain=None, shifts=None):
        rdims = self._rdim(subdomain=subdomain, shifts=shifts)
        return Mul(*[
            w._subs(rd, rd-rd.parent.symbolic_min)
            for (rd, w) in zip(rdims, self.interpolation_coeffs, strict=True)
        ])

    def _arg_defaults(self, coords=None, sfunc=None, origin=None):
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
