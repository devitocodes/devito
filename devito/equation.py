"""User API to specify equations."""

import sympy

__all__ = ['Eq', 'Inc', 'solve']


class Eq(sympy.Eq):

    """
    A :class:`sympy.Eq` that accepts the additional keyword parameter ``subdomain``.

    The ``subdomain``, an object of type :class:`SubDomain`, can be used to
    restrict the execution of the equation to a particular subdomain.
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
        return self._subdomain

    def xreplace(self, rules):
        return self.func(self.lhs.xreplace(rules), self.rhs.xreplace(rules),
                         subdomain=self._subdomain)


class Inc(Eq):

    """
    A :class:`Eq` performing a linear increment.
    """

    is_Increment = True

    def __str__(self):
        return "Inc(%s, %s)" % (self.lhs, self.rhs)

    __repr__ = __str__


def solve(eq, target, **kwargs):
    """
    solve(expr, target, **kwargs)

    Algebraically rearrange an equation w.r.t. a given symbol.

    This is a wrapper around ``sympy.solve``.

    :param eq: The :class:`sympy.Eq` to be rearranged.
    :param target: The symbol w.r.t. which the equation is rearranged.
    :param kwargs: (Optional) Symbolic optimizations applied while rearranging
                   the equation. For more information. refer to
                   ``sympy.solve.__doc__``.
    """
    # Enforce certain parameters to values that are known to guarantee a quick
    # turnaround time
    kwargs['rational'] = False  # Avoid float indices
    kwargs['simplify'] = False  # Do not attempt premature optimisation
    return sympy.solve(eq, target, **kwargs)[0]
