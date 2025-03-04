"""User API to specify equations."""
import sympy

from functools import cached_property

from devito.deprecations import deprecations
from devito.tools import as_tuple, frozendict, Pickable
from devito.types.lazy import Evaluable

__all__ = ['Eq', 'Inc', 'ReduceMax', 'ReduceMin']


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

    __rargs__ = ('lhs', 'rhs')
    __rkwargs__ = ('subdomain', 'coefficients', 'implicit_dims')

    def __new__(cls, lhs, rhs=0, subdomain=None, coefficients=None, implicit_dims=None,
                **kwargs):
        if coefficients is not None:
            deprecations.coeff_warn
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
            rhs = self.rhs._eval_at(self.lhs)._evaluate(**kwargs)
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
                eqs = dict(zip(self.lhs, self.rhs))
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
        return "%s(%s, %s)" % (self.__class__.__name__, self.lhs, self.rhs)

    __repr__ = __str__


class Reduction(Eq):

    """
    An Eq in which the right-hand side represents a reduction operation, whose
    result is stored in to the left-hand side.
    """

    is_Reduction = True

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.lhs, self.rhs)

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
