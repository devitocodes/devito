"""User API to specify equations and relations."""

import sympy

from cached_property import cached_property

from devito.finite_differences import Evaluable, default_rules
from devito.tools import as_tuple

__all__ = ['Eq', 'Inc', 'solve', 'Le', 'Lt', 'Ge',
           'Gt', 'Ne']


class Eq(sympy.Eq, Evaluable):

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
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.
    coefficients : Substitutions, optional
        Can be used to replace symbolic finite difference weights with user
        defined weights.
    implicit_dims : Dimension or list of Dimension, optional
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

    >>> from sympy import sin
    >>> Eq(f, sin(f.dx)**2)
    Eq(f(x, y), sin(Derivative(f(x, y), x))**2)

    Notes
    -----
    An Eq can be thought of as an assignment in an imperative programming language
    (e.g., ``a[i] = b[i]*c``).
    """

    is_Increment = False

    def __new__(cls, lhs, rhs=0, subdomain=None, coefficients=None, implicit_dims=None,
                **kwargs):
        kwargs['evaluate'] = False
        obj = sympy.Eq.__new__(cls, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        obj._substitutions = coefficients
        obj._implicit_dims = as_tuple(implicit_dims)

        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Eq is defined."""
        return self._subdomain

    @cached_property
    def evaluate(self):
        """
        Evaluate the Equation or system of Equations.

        The RHS of the Equation is evaluated at the indices of the LHS if required.
        """
        try:
            lhs, rhs = self.lhs.evaluate, self.rhs._eval_at(self.lhs).evaluate
        except AttributeError:
            lhs, rhs = self._evaluate_args()
        eq = self.func(lhs, rhs, subdomain=self.subdomain,
                       coefficients=self.substitutions,
                       implicit_dims=self._implicit_dims)
        if eq._uses_symbolic_coefficients:
            # NOTE: As Coefficients.py is expanded we will not want
            # all rules to be expunged during this procress.
            rules = default_rules(eq, eq._symbolic_functions)
            try:
                eq = eq.xreplace({**eq.substitutions.rules, **rules})
            except AttributeError:
                if bool(rules):
                    eq = eq.xreplace(rules)
        return eq

    @property
    def _flatten(self):
        """
        Flatten vectorial/tensorial Equation into list of scalar Equations.
        """
        if self.lhs.is_Matrix:
            # Maps the Equations to retrieve the rhs from relevant lhs
            eqs = dict(zip(as_tuple(self.lhs), as_tuple(self.rhs)))
            # Get the relevant equations from the lhs structure. .values removes
            # the symmetric duplicates and off-diagonal zeros.
            lhss = self.lhs.values()
            return [self.func(l, eqs[l], subdomain=self.subdomain,
                              coefficients=self.substitutions,
                              implicit_dims=self._implicit_dims)
                    for l in lhss]
        else:
            return [self]

    @property
    def substitutions(self):
        return self._substitutions

    @property
    def implicit_dims(self):
        return self._implicit_dims

    @cached_property
    def _uses_symbolic_coefficients(self):
        return bool(self._symbolic_functions)

    @cached_property
    def _symbolic_functions(self):
        try:
            return self.lhs._symbolic_functions.union(self.rhs._symbolic_functions)
        except AttributeError:
            pass
        try:
            return self.lhs._symbolic_functions
        except AttributeError:
            pass
        try:
            return self.rhs._symbolic_functions
        except AttributeError:
            return frozenset()
        else:
            TypeError('Failed to retrieve symbolic functions')

    def xreplace(self, rules):
        return self.func(self.lhs.xreplace(rules), rhs=self.rhs.xreplace(rules),
                         subdomain=self._subdomain, coefficients=self._substitutions,
                         implicit_dims=self._implicit_dims)

    def __str__(self):
        return "%s(%s, %s)" % (self.__class__.__name__, self.lhs, self.rhs)

    __repr__ = __str__


class Ne(sympy.Ne):
    """
    A Not-Equal ("!=") relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Ne.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Le is defined."""
        return self._subdomain


class Le(sympy.Le):
    """
    A less-than ("<=") relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Le.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Le is defined."""
        return self._subdomain


class Lt(sympy.Lt):
    """
    A strict-less-than ("<") relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Lt.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Lt is defined."""
        return self._subdomain


class Ge(sympy.Ge):
    """
    A greater-than (">=") relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Ge.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Ge is defined."""
        return self._subdomain


class Gt(sympy.Gt):
    """
    A strict-greater-than (">") relation between two objects, the left-hand side and the
    right-hand side.

    The left-hand side may be a Function or a SparseFunction. The right-hand
    side may be any arbitrary expressions with numbers, Dimensions, Constants,
    Functions and SparseFunctions as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like, optional
        The right-hand side. Defaults to 0.
    subdomain : SubDomain, optional
        To restrict the evalaution of the relation to a particular sub-region in the
        computational domain.

    Examples
    --------

    Notes
    -----
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Gt.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Gt is defined."""
        return self._subdomain


class Inc(Eq):

    """
    An increment relation between two objects, the left-hand side and the
    right-hand side.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr-like
        The right-hand side.
    subdomain : SubDomain, optional
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.
    coefficients : Substitutions, optional
        Can be used to replace symbolic finite difference weights with user
        defined weights.
    implicit_dims : Dimension or list of Dimension, optional
        An ordered list of Dimensions that do not explicitly appear in either the
        left-hand side or in the right-hand side, but that should be honored when
        constructing an Operator.

    Examples
    --------
    Inc may be used to express tensor contractions. Below, a summation along
    the user-defined Dimension ``i``.

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

    is_Increment = True

    def __str__(self):
        return "Inc(%s, %s)" % (self.lhs, self.rhs)

    __repr__ = __str__


def solve(eq, target, **kwargs):
    """
    Algebraically rearrange an Eq w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    Parameters
    ----------
    eq : expr-like
        The equation to be rearranged.
    target : symbol
        The symbol w.r.t. which the equation is rearranged. May be a `Function`
        or any other symbolic object.
    **kwargs
        Symbolic optimizations applied while rearranging the equation. For more
        information. refer to ``sympy.solve.__doc__``.
    """
    # Enforce certain parameters to values that are known to guarantee a quick
    # turnaround time
    kwargs['rational'] = False  # Avoid float indices
    kwargs['simplify'] = False  # Do not attempt premature optimisation
    kwargs['manual'] = True  # Force sympy to solve one line at a time for VectorFunction
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs if eq.rhs != 0 else eq.lhs
    sol = sympy.solve(eq.evaluate, target.evaluate, **kwargs)[0]

    # We need to rebuild the vector/tensor as sympy.solve outputs a tuple of solutions
    from devito.types import TensorFunction
    if isinstance(target, TensorFunction):
        return target.new_from_mat(sol)
    else:
        return sol
