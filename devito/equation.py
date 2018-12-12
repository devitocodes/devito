"""User API to specify equations."""

import sympy

__all__ = ['Eq', 'Inc', 'solve']


class Eq(sympy.Eq):

    """
    An equal relation between two objects, the left-hand side and the right-hand side.

    The left-hand side may be a :class:`Function` or a :class:`SparseFunction`. The
    right-hand side may be any arbitrary expressions with numbers, :class:`Dimension`,
    :class:`Constant`, :class:`Function` and :class:`SparseFunction` as operands.

    Parameters
    ----------
    lhs : Function or SparseFunction
        The left-hand side.
    rhs : expr
        The right-hand side.
    subdomain : SubDomain, optional
        To restrict the computation of the Eq to a particular sub-region in the
        computational domain.

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
    An `Eq` can be thought of as an assignment in an imperative programming language
    (e.g., ``a[i] = b[i]*c``).
    """

    is_Increment = False

    def __new__(cls, *args, **kwargs):
        kwargs['evaluate'] = False
        subdomain = kwargs.pop('subdomain', None)
        obj = sympy.Eq.__new__(cls, *args, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The Eq SubDomain."""
        return self._subdomain

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
    `Inc` may be used to express tensor contractions. Below, a summation along
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
    An `Inc` can be thought of as the augmented assignment '+=' in an imperative
    programming language (e.g., ``a[i] += c``).
    """

    is_Increment = True

    def __str__(self):
        return "Inc(%s, %s)" % (self.lhs, self.rhs)

    __repr__ = __str__


def solve(eq, target, **kwargs):
    """
    Algebraically rearrange an :class:`Eq` w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    Parameters
    ----------
    eq : expr
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
