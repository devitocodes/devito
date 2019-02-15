"""User API to specify equations."""

import sympy

from cached_property import cached_property

from devito.finite_differences import default_rules

__all__ = ['Eq', 'Inc', 'solve']


class Eq(sympy.Eq):

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
    rhs : expr-like
        The right-hand side.
    subdomain : SubDomain, optional
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.
    coefficients : Substitutions, optional
        Can be used to replace symbolic finite difference weights with user
        defined weights.

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
    Eq(f(x, y), sin(f(x, y)/h_x - f(x + h_x, y)/h_x)**2)

    Notes
    -----
    An Eq can be thought of as an assignment in an imperative programming language
    (e.g., ``a[i] = b[i]*c``).
    """

    is_Increment = False

    def __new__(cls, *args, **kwargs):
        kwargs['evaluate'] = False
        subdomain = kwargs.pop('subdomain', None)
        substitutions = kwargs.pop('coefficients', None)
        obj = sympy.Eq.__new__(cls, *args, **kwargs)
        obj._subdomain = subdomain
        obj._substitutions = substitutions
        if obj._uses_symbolic_coefficients:
            # NOTE: As Coefficients.py is expanded we will not want
            # all rules to be expunged during this procress.
            rules = default_rules(obj, obj._symbolic_functions)
            try:
                obj = obj.xreplace({**substitutions.rules, **rules})
            except AttributeError:
                if bool(rules):
                    obj = obj.xreplace(rules)
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Eq is defined."""
        return self._subdomain

    @property
    def substitutions(self):
        return self._substitutions

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
        """"""
        return self.func(self.lhs.xreplace(rules), self.rhs.xreplace(rules),
                         subdomain=self._subdomain)


class Inc(Eq):

    """
    An increment relation between two objects, the left-hand side and the
    right-hand side.

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
    return sympy.solve(eq, target, **kwargs)[0]
