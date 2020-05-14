"""User API to specify relationals."""

import sympy


__all__ = ['Le', 'Lt', 'Ge', 'Gt']


class Le(sympy.Le):
    """
    A less-than or equal ("<=") relation between two objects, the left-hand side and the
    right-hand side. Can be used to build conditionals but not directly to
    construct an Operator.

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
    Le may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator
    >>> from devito.types import Le
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', shape=grid.shape, dimensions=grid.dimensions)
    >>> x, y = grid.dimensions
    >>> cond = Le(g, 1)
    >>> ci = ConditionalDimension(name='ci', parent=y, condition=cond)
    >>> f = Function(name='f', shape=grid.shape, dimensions=(x, ci))
    >>> op = Operator(Eq(g, f))
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
    right-hand side. Can be used to build conditionals but not directly to
    construct an Operator.

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
    Lt may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator
    >>> from devito.types import Lt
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', shape=grid.shape, dimensions=grid.dimensions)
    >>> x, y = grid.dimensions
    >>> cond = Lt(g, 1)
    >>> ci = ConditionalDimension(name='ci', parent=y, condition=cond)
    >>> f = Function(name='f', shape=grid.shape, dimensions=(x, ci))
    >>> op = Operator(Eq(g, f))
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
    right-hand side. Can be used to build conditionals but not directly to
    construct an Operator.

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
    Le may be used to express a relation (e.g. in a Subdomain).

    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator
    >>> from devito.types import Ge
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', shape=grid.shape, dimensions=grid.dimensions)
    >>> x, y = grid.dimensions
    >>> cond = Ge(g, 1)
    >>> ci = ConditionalDimension(name='ci', parent=y, condition=cond)
    >>> f = Function(name='f', shape=grid.shape, dimensions=(x, ci))
    >>> op = Operator(Eq(g, f))
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
    right-hand side. Can be used to build conditionals but not directly to
    construct an Operator.

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
    Gt may be used to express a relation.
    >>> from devito import Grid, Function, ConditionalDimension, Eq, Operator
    >>> from devito.types import Gt
    >>> grid = Grid(shape=(8, 8))
    >>> g = Function(name='g', shape=grid.shape, dimensions=grid.dimensions)
    >>> x, y = grid.dimensions
    >>> cond = Gt(g, 1)
    >>> ci = ConditionalDimension(name='ci', parent=y, condition=cond)
    >>> f = Function(name='f', shape=grid.shape, dimensions=(x, ci))
    >>> op = Operator(Eq(g, f))
    """

    def __init__(self, lhs, rhs=0, subdomain=None, **kwargs):
        obj = sympy.Gt.__new__(self, lhs, rhs, **kwargs)
        obj._subdomain = subdomain
        return obj

    @property
    def subdomain(self):
        """The SubDomain in which the Gt is defined."""
        return self._subdomain
