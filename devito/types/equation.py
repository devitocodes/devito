"""User API to specify equations."""
from functools import cached_property

import sympy

from devito.deprecations import deprecations
from devito.tools import Pickable, as_tuple, frozendict
from devito.types.lazy import Evaluable

__all__ = ['Eq', 'Inc', 'ReduceMax', 'ReduceMin', 'ReduceMinMax', 'SparseEq']


class Eq(sympy.Eq, Evaluable, Pickable):

    """
    An equal relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional, default=0
        The right-hand side.
    subdomain : SubDomain, optional, default=None
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.
    coefficients : Substitutions, optional, default=None
        Can be used to replace symbolic finite difference weights with user
        defined weights.
    implicit_dims : Dimension or list of Dimension, optional, default=None
        An ordered list of Dimensions that do not explicitly appear in either the
        left-hand side or in the right-hand side, but that should be honored when
        constructing an Operator.

    Examples
    --------
    >>> from devito import Grid, Function, Eq
    >>> grid = Grid(shape=(4, 4))
    >>> f = Function(name='f', grid=grid)
    >>> Eq(f, f + 1)
    Eq(f(x, y), f(x, y) + 1)

    Any SymPy expressions may be used in the right-hand side.

    >>> from devito import sin
    >>> Eq(f, sin(f.dx)**2)
    Eq(f(x, y), sin(Derivative(f(x, y), x))**2)

    Notes
    -----
    An Eq can be thought of as an assignment in an imperative programming language
    (e.g., ``a[i] = b[i]*c``).
    """

    is_Reduction = False
    is_SparseOperation = False

    __rargs__ = ('lhs', 'rhs')
    __rkwargs__ = ('subdomain', 'coefficients', 'implicit_dims')

    def __new__(cls, lhs, rhs=0, subdomain=None, coefficients=None,
                implicit_dims=None, **kwargs):
        if coefficients is not None:
            _ = deprecations.coeff_warn
        kwargs['evaluate'] = False
        # Backward compatibility
        rhs = cls._apply_coeffs(rhs, coefficients)
        lhs = cls._apply_coeffs(lhs, coefficients)

        obj = sympy.Eq.__new__(cls, lhs, rhs, **kwargs)

        obj._subdomain = subdomain
        obj._substitutions = coefficients
        obj._implicit_dims = as_tuple(implicit_dims)

        return obj

    @classmethod
    def _apply_coeffs(cls, expr, coefficients):
        """
        This processes legacy API of Substitution/Coefficients applying the weights
        to the target Derivatives.
        """
        from devito.symbolics import retrieve_derivatives
        if coefficients is None:
            return expr
        mapper = {}
        for coeff in coefficients.coefficients:
            derivs = [d for d in retrieve_derivatives(expr)
                      if coeff.dimension in d.dims and
                      coeff.function in d.expr._functions and
                      coeff.deriv_order == d.deriv_order.get(coeff.dimension, None)]
            if not derivs:
                continue
            mapper.update({d: d._rebuild(weights=coeff.weights) for d in derivs})
        if not mapper:
            return expr

        return expr.subs(mapper)

    def _evaluate(self, **kwargs):
        """
        Evaluate the Equation or system of Equations.

        The RHS of the Equation is evaluated at the indices of the LHS if required.
        """
        try:
            lhs = self.lhs._evaluate(**kwargs)
            rhs = self.rhs._eval_at(self.lhs, **kwargs)._evaluate(**kwargs)
        except AttributeError:
            lhs, rhs = self._evaluate_args(**kwargs)
        eq = self.func(lhs, rhs, subdomain=self.subdomain,
                       coefficients=self.substitutions,
                       implicit_dims=self._implicit_dims)

        return eq

    @property
    def _flatten(self):
        """
        Flatten vectorial/tensorial Equation into list of scalar Equations.
        """
        if self.lhs.is_Matrix:
            # Maps the Equations to retrieve the rhs from relevant lhs
            try:
                eqs = dict(zip(self.lhs, self.rhs, strict=True))
            except TypeError:
                # Same rhs for all lhs
                assert not self.rhs.is_Matrix
                eqs = {i: self.rhs for i in self.lhs}
            # Get the relevant equations from the lhs structure. .values removes
            # the symmetric duplicates and off-diagonal zeros.
            lhss = self.lhs.values()
            return [self.func(l, eqs[l], subdomain=self.subdomain,
                              coefficients=self.substitutions,
                              implicit_dims=self._implicit_dims)
                    for l in lhss]
        else:
            return [self]

    @cached_property
    def subdomain(self):
        """The SubDomain in which the Eq is defined."""
        if self._subdomain is not None:
            return self._subdomain

        from devito.symbolics.search import retrieve_functions  # noqa

        funcs = retrieve_functions(self)
        subdomains = {f.grid for f in funcs} - {None}
        subdomains = {g for g in subdomains if g.is_SubDomain}

        if len(subdomains) == 0:
            return None
        elif len(subdomains) == 1:
            return subdomains.pop()
        else:
            raise ValueError("Multiple `SubDomain`s detected. Provide a `SubDomain`"
                             " explicitly (i.e., via `Eq(..., subdomain=...)`)"
                             " to unambiguously define the `Eq`'s iteration domain.")

    @property
    def substitutions(self):
        return self._substitutions

    coefficients = substitutions

    @property
    def implicit_dims(self):
        return self._implicit_dims

    @property
    def conditionals(self):
        return frozendict()

    func = Evaluable._rebuild

    def xreplace(self, rules):
        return self.func(self.lhs.xreplace(rules), self.rhs.xreplace(rules))

    def __str__(self):
        return f"{self.__class__.__name__}({self.lhs}, {self.rhs})"

    __repr__ = __str__


class Reduction(Eq):

    """
    An Eq in which the right-hand side represents a reduction operation, whose
    result is stored in to the left-hand side.
    """

    is_Reduction = True

    def __str__(self):
        return f"{self.__class__.__name__}({self.lhs}, {self.rhs})"

    __repr__ = __str__


class Inc(Reduction):

    """
    An increment Reduction.

    Examples
    --------
    Inc may be used to express tensor contractions. Below, a summation along
    the user-defined Dimension `i`.

    >>> from devito import Grid, Dimension, Function, Inc
    >>> grid = Grid(shape=(4, 4))
    >>> x, y = grid.dimensions
    >>> i = Dimension(name='i')
    >>> f = Function(name='f', grid=grid)
    >>> g = Function(name='g', shape=(10, 4, 4), dimensions=(i, x, y))
    >>> Inc(f, g)
    Inc(f(x, y), g(i, x, y))

    Notes
    -----
    An Inc can be thought of as the augmented assignment '+=' in an imperative
    programming language (e.g., ``a[i] += c``).
    """

    pass


class ReduceMax(Reduction):
    pass


class ReduceMin(Reduction):
    pass


class ReduceMinMax(Reduction):

    """
    A coupled min/max Reduction.

    The left-hand side must have room for two components, one for the minimum and
    one for the maximum; the behaviour is otherwise undefined.
    The right-hand side is the expression to be reduced.
    """

    def __new__(cls, lhs, rhs=0, **kwargs):
        if not lhs.function.is_AbstractFunction:
            raise ValueError(
                f"The left-hand side of a {cls.__name__} must be a "
                "Function of size at least 2"
            )

        return super().__new__(cls, lhs, rhs=rhs, **kwargs)


class SparseEq(Eq):

    """
    An Eq representing a sparse-grid operation (interpolate or inject).

    SparseEq carries the symbolic shape of a normal Eq (lhs/rhs) so it
    composes naturally with the rest of the Devito DSL, plus the
    operation payload (kind, interpolator, source expr, ...) needed to
    expand it into a sequence of grid-level equations.

    `_evaluate` returns `[self]`, leaving the SparseEq opaque for the
    cluster/IET pipeline. The actual expansion happens in the IET pass
    `lower_sparse_ops`, which calls `operation()` and wraps the result
    in an ElementalFunction via `rcompile`.

    Parameters
    ----------
    lhs : Function or SparseFunction
        Synthetic left-hand side: the SparseFunction for an
        interpolation, the grid Function (or first of a tuple of
        fields) for an injection.
    rhs : expr-like
        Synthetic right-hand side: the user expression for an
        interpolation, `field + expr` for an injection.
    interpolator : Interpolator
        The Interpolator that knows how to expand this operation.
    kind : str
        Either ``'interpolate'`` or ``'inject'``.
    expr : expr-like
        The unevaluated source expression (carried separately from
        `rhs` because it may contain Derivatives that should not be
        indexified by `lower_exprs`).
    field : Function or tuple of Functions, optional
        Target field(s) for an injection.
    increment : bool, optional
        For an interpolation, emit increments rather than assignments.
    self_subs : dict, optional
        Time/sparse-index substitutions to apply to the sink of an
        interpolation.
    implicit_dims : Dimension or list of Dimension, optional
        Dimensions that don't appear in lhs/rhs but should be honoured
        when scheduling the operation (typically a SteppingDimension
        pinning the operation inside a parent time loop).
    """

    is_SparseOperation = True

    __rkwargs__ = Eq.__rkwargs__ + (
        'interpolator', 'kind', 'expr', 'field', 'increment', 'self_subs'
    )

    def __new__(cls, lhs, rhs=0, *, interpolator=None, kind=None, expr=None,
                field=None, increment=False, self_subs=None,
                implicit_dims=None, **kwargs):
        obj = super().__new__(cls, lhs, rhs=rhs, implicit_dims=implicit_dims,
                              **kwargs)
        obj._interpolator = interpolator
        obj._kind = kind
        obj._expr = expr
        obj._field = field
        obj._increment = increment
        obj._self_subs = self_subs or {}
        return obj

    @property
    def interpolator(self):
        return self._interpolator

    @property
    def kind(self):
        return self._kind

    @property
    def expr(self):
        return self._expr

    @property
    def field(self):
        return self._field

    @property
    def increment(self):
        return self._increment

    @property
    def self_subs(self):
        return self._self_subs

    def _evaluate(self, **kwargs):
        # Stay atomic at expression lowering time; the IET pass
        # `lower_sparse_ops` invokes `operation()` later
        return [self]

    def operation(self):
        """
        Expand the sparse operation into its grid-level Eq sequence by
        dispatching to the interpolator.
        """
        if self._kind == 'interpolate':
            return self._interpolator._interpolate(
                expr=self._expr, increment=self._increment,
                self_subs=self._self_subs, implicit_dims=self.implicit_dims
            )
        if self._kind == 'inject':
            return self._interpolator._inject(
                expr=self._expr, field=self._field,
                implicit_dims=self.implicit_dims
            )
        raise ValueError(f"Unknown SparseEq kind: {self._kind!r}")

    def func(self, *args, **kwargs):
        # `func` is called by sympy machinery (uxreplace, xreplace, ...)
        # with `(new_lhs, new_rhs)` to rebuild the relational. Side-step
        # the standard reconstruction so the operation payload survives.
        if len(args) == 2 and not kwargs:
            new = sympy.Eq.__new__(type(self), *args, evaluate=False)
            new.__dict__.update(self.__dict__)
            return new
        return self._rebuild(*args, **kwargs)

    def __add__(self, other):
        # Allow `list_of_eqs + sparse_eq` and `sparse_eq + sparse_eq` to
        # produce a flat list — handy when composing user expression
        # bundles passed to `Operator(...)`.
        from devito.tools import flatten
        return flatten([self, other])

    def __radd__(self, other):
        from devito.tools import flatten
        return flatten([other, self])

    def __repr__(self):
        sf = self._interpolator.sfunction if self._interpolator else '?'
        if self._kind == 'interpolate':
            return f"Interpolation({self._expr!r} into {sf!r})"
        if self._kind == 'inject':
            return f"Injection({self._expr!r} into {self._field!r})"
        return super().__repr__()

    __str__ = __repr__
