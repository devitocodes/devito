from abc import ABC, abstractmethod
from contextlib import suppress
from functools import cached_property, wraps

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
from devito.symbolics.extended_dtypes import DOUBLE
from devito.tools import as_tuple, filter_ordered, memoized_meth
from devito.types import (
    Eq, Inc, IncrInterpolation, Injection, Interpolation, SubFunction, Symbol
)
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


def _build_interpolation(expr, increment, implicit_dims, self_subs, interpolator):
    """
    Construct the sparse-op Eq for an interpolation: the synthetic Eq
    is ``Eq(sf[..., p_*], expr[..., rp_*])``; with ``increment`` it is
    an ``Inc``. User-supplied ``implicit_dims`` are carried as-is; the
    SparseFunction's iteration Dimensions are augmented in by
    ``lower_eq`` so the cluster pipeline sees them.
    """
    eq = interpolator._interpolate(expr=expr, increment=increment,
                                   self_subs=self_subs,
                                   implicit_dims=None)
    cls = IncrInterpolation if isinstance(eq, Inc) else Interpolation
    return cls(eq.lhs, eq.rhs, interpolator=interpolator,
               implicit_dims=implicit_dims)


def _build_injection(field, expr, implicit_dims, interpolator):
    """
    Construct the ``Injection``(s) for an injection: each synthetic Eq
    is ``Inc(field[..., x, y, ...], weights * expr[..., rp_*])``
    produced by ``interpolator._inject``. A multi-field injection
    expands into one ``Injection`` per ``(field, expr)`` pair so each
    target field is individually visible to the cluster pipeline.
    User-supplied ``implicit_dims`` are carried as-is; sparse-function
    iteration Dimensions are augmented in by ``lower_eq``.
    """
    fields, exprs = as_tuple(field), as_tuple(expr)
    if len(exprs) == 1:
        exprs = tuple(exprs[0] for _ in fields)
    eqs = []
    for (f, e) in zip(fields, exprs, strict=True):
        inc = interpolator._inject(field=f, expr=e, implicit_dims=None)
        eqs.append(Injection(inc.lhs, inc.rhs, interpolator=interpolator,
                             implicit_dims=implicit_dims))
    return eqs[0] if len(eqs) == 1 else eqs


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

        # Make the radius dimensions conditional to avoid OOB accesses.
        # ``rd ∈ [-r+1, r]`` and the access is ``field[pos + rd]``, so
        # the OOB guard on ``rd`` reads ``pos + rd ∈ [d_min - r, d_max + r]``.
        rdims = []
        pos_symbols = self.sfunction._pos_symbols(shifts=shifts)

        for d in gdims:
            # The radius CustomDimension is keyed on the grid Dimension
            # (``d.root``) so a SubDomain-restricted operation reuses
            # the same ``rp_*`` dim as the full-grid case; only the
            # bounds carry the SubDomain's symbolic_min/max.
            rd = self.sfunction._crdim(d.root)
            pos = pos_symbols[d.root]
            lb = sympy.And(pos + rd >= d.symbolic_min - self.r, evaluate=False)
            ub = sympy.And(pos + rd <= d.symbolic_max + self.r, evaluate=False)
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

    @memoized_meth
    def _raw_pos_symbols(self, shifts=None):
        """
        Per-Dimension Symbol holding the unrounded grid-relative position
        ``(coord - origin - shift)/h``. Both the integer position
        (``floor(...)``) and the linear-interp fractional part
        (``... - floor(...)``) reuse this Symbol so the divide-and-shift
        expression is emitted only once per sparse point.
        """
        dtype = self.sfunction.coordinates.dtype
        symbols = []
        for d, s in zip(self.grid.dimensions,
                        shifts or (0,) * len(self.grid.dimensions),
                        strict=True):
            suffix = '_s1' if s != 0 else ''
            symbols.append(Symbol(name=f'rpos{d}{suffix}', dtype=dtype))
        return DimensionTuple(*symbols, getters=self.grid.dimensions)

    def _positions(self, implicit_dims, shifts=None):
        # The ``(coord - origin)/h`` subtract is the only step that can lose
        # precision to catastrophic cancellation when ``coord`` and ``origin``
        # are large and close to each other (e.g. an origin-shifted survey).
        # Promote ``origin`` and ``h`` to float64 so the subtract and divide
        # happen in double precision in C (one cast operand promotes the
        # whole expression); the result narrows to the field dtype on store
        # to ``rpos*`` so downstream ``floor`` / fractional math stays in
        # the field dtype.
        rposs = self._raw_pos_symbols(shifts=shifts)
        subs = {o: DOUBLE(o) for o in self.grid.origin_symbols}
        subs.update({d.spacing: DOUBLE(d.spacing) for d in self._gdims})
        return [Eq(rposs[d], k.xreplace(subs), implicit_dims=implicit_dims)
                for d, k in zip(self._gdims,
                                self.sfunction._position_map(shifts=shifts),
                                strict=True)] + \
               [Eq(v, INT(floor(rposs[d])), implicit_dims=implicit_dims)
                for d, v in zip(self._gdims,
                                self.sfunction._position_map(shifts=shifts).values(),
                                strict=True)]

    def sparse_temps(self, rhs, implicit_dims, field=None):
        """
        Position/coefficient temps for a sparse op with right-hand side
        ``rhs``. For an injection, ``field`` drives the per-Dimension
        shifts so the temps' lhs (``pos*`` symbols) match the rhs of a
        staggered injection; for an interpolation, ``field`` is None
        and no shifts are applied.
        """
        if field is not None:
            extras = [field] + list(retrieve_function_carriers(rhs))
            shifts = self._field_shifts(field)
        else:
            extras = list(retrieve_function_carriers(rhs)) or None
            shifts = None

        implicit_dims = self._augment_implicit_dims(implicit_dims, extras=extras)
        return list(self._positions(implicit_dims, shifts=shifts)) + \
            list(self._coeff_temps(implicit_dims, shifts=shifts))

    def _interp_idx(self, variables, subdomain=None, shifts=None):
        """
        Generate the indirect-access index substitutions for the
        DiscreteFunctions in ``variables``. Each grid Dimension ``x``
        is replaced with ``pos_x + rd_x``, where ``rd_x`` is the
        StencilDimension for the radius and ``pos_x`` is the per-point
        position offset; together they realise the radius-neighbourhood
        access ``field[pos_x + rd_x, pos_y + rd_y]``.

        ``shifts`` is a per-Dimension physical offset for the target
        field's origin (e.g. ``h_x/2`` for a field staggered in ``x``);
        it only affects the integer position symbol via the position
        map (``pos = floor((c - o - shift)/h)``).
        """
        pos = self.sfunction._position_map(shifts=shifts).values()

        # Substitution mapper for variables
        mapper = self._rdim(subdomain=subdomain, shifts=shifts).getters

        # Index substitution to make in variables
        subs = {
            ki: p + c
            for ((k, c), p) in zip(mapper.items(), pos, strict=True)
            for ki in {k, k.root}
        }

        return {v: v.subs(subs) for v in variables}

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
        return _build_interpolation(expr, increment, implicit_dims, self_subs, self)

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
        return _build_injection(field, expr, implicit_dims, self)

    def _interpolate(self, expr, increment=False, self_subs=None, implicit_dims=None):
        """
        Build the synthetic single Eq for an interpolation:

            Eq(sf[..., p_*], expr[..., rp_*])         # or Inc when increment=True

        The grid Dimensions inside ``expr`` are substituted for the
        radius ConditionalDimensions ``rp_*``, whose parent is the
        original grid Dimension. The cluster scheduler therefore derives
        an IterationSpace ``(..., p_*, rp_*)``; the IET pass
        ``lower_sparse_ops`` later wraps that nest in an
        ElementalFunction and inserts the position/weight temps and
        accumulator inside it.
        """
        # Derivatives must be evaluated before the introduction of indirect accesses.
        # CSE will pick up any shared subexpression.
        try:
            _expr = expr._eval_at(self.sfunction).evaluate
        except AttributeError:
            _expr = expr

        if self_subs is None:
            self_subs = {}

        variables = list(retrieve_function_carriers(_expr))
        subdomain = _extract_subdomain(variables)

        implicit_dims = self._augment_implicit_dims(implicit_dims, variables)
        idx_subs = self._interp_idx(variables, subdomain=subdomain)

        lhs = self.sfunction.subs(self_subs)
        rhs = _expr.xreplace(idx_subs)

        ecls = Inc if increment else Eq
        return ecls(lhs, rhs, implicit_dims=implicit_dims)

    def _inject(self, field, expr, implicit_dims=None):
        """
        Build the synthetic single Inc for an injection:

            Inc(field[..., pos_x + rd_x, ...], weights(rd_*) * expr[..., pos + rd_*])

        Both lhs and rhs share the radius-indexed access pattern so the
        cluster scheduler derives the same ``(..., p_*, rd_*)``
        IterationSpace whether the operation is interpolation or
        injection. The IET pass ``lower_sparse_ops`` wraps the resulting
        nest in an ElementalFunction.
        """
        # Derivatives must be evaluated before the introduction of indirect
        # accesses. Variables are sampled at their own grid location; the
        # position map for the target field carries the staggering so the
        # field's stencil neighbors land on the right indices.
        try:
            _expr = expr._eval_at(field).evaluate
        except AttributeError:
            _expr = expr

        subdomain = _extract_subdomain((field,))

        variables = list(retrieve_function_carriers(_expr))

        implicit_dims = self._augment_implicit_dims(implicit_dims,
                                                    (field,) + tuple(variables))

        # The reference index depends on the field's origin (staggering).
        shifts = self._field_shifts(field)

        idx_subs = self._interp_idx((field,) + tuple(variables),
                                    subdomain=subdomain, shifts=shifts)

        weights = self._weights(subdomain=subdomain, shifts=shifts)
        lhs = field.xreplace(idx_subs)
        rhs = (weights * _expr).xreplace(idx_subs)

        return Inc(lhs, rhs, implicit_dims=implicit_dims)


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
    def _weights(self, subdomain=None, shifts=None):
        rdim = self._rdim(subdomain=subdomain, shifts=shifts)
        c = [(1 - p) * (1 - r) + p * r
             for (p, d, r) in zip(self._point_symbols(shifts), self._gdims, rdim,
                                  strict=True)]
        return Mul(*c)

    @memoized_meth
    def _point_symbols(self, shifts=None):
        """Symbol for coordinate value in each Dimension of the point."""
        dtype = self.sfunction.coordinates.dtype
        symbols = []
        for d in self.grid.dimensions:
            if shifts and shifts[self.grid.dimensions.index(d)] != 0:
                symbols.append(Symbol(name=f'p{d}_s1', dtype=dtype))
            else:
                symbols.append(Symbol(name=f'p{d}', dtype=dtype))
        return DimensionTuple(*symbols, getters=self.grid.dimensions)

    def _coeff_temps(self, implicit_dims, shifts=None):
        # The fractional part of the unrounded position; reuse the
        # ``rpos*`` Symbols emitted by ``_positions`` rather than the full
        # ``(c - o)/h`` expression so the divide is computed only once.
        rposs = self._raw_pos_symbols(shifts=shifts)
        psyms = self._point_symbols(shifts)
        return [Eq(psyms[d], rposs[d] - floor(rposs[d]),
                   implicit_dims=implicit_dims)
                for d in self._gdims]


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
        # The radius CustomDim spans ``[-r+1, r]``; the coefficients
        # table is indexed from 0 to ``2r-1``, so shift by ``r-1``.
        offset = self.r - 1
        mappers = [{ddim: ri, cdim: rd + offset}
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
        for d in self._gdims:
            rd = self.sfunction._crdim(d)
            dimensions = (self.sfunction._sparse_dim, rd)
            sf = SubFunction(name=f"wsinc{rd.name}", dtype=self.sfunction.dtype,
                             shape=shape, dimensions=dimensions,
                             space_order=0, alias=self.sfunction.alias,
                             parent=self.sfunction)
            coeffs.append(sf)
        return tuple(coeffs)

    @memoized_meth
    def _weights(self, subdomain=None, shifts=None):
        rdims = self._rdim(subdomain=subdomain, shifts=shifts)
        return Mul(*[
            w._subs(rd, rd-rd.parent.symbolic_min)
            for (rd, w) in zip(rdims, self.interpolation_coeffs, strict=True)
        ])

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
