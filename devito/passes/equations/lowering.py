from collections import namedtuple
from functools import singledispatch

import sympy

from devito.symbolics import search
from devito.tools import as_mapper, split

__all__ = ['rewrite_exprs']


def rewrite_exprs(expressions):
    """
    Apply fundamental mathematical properties to rewrite the given expressions
    to maximize the impact of the subsequent performance-related passes.

    Currently, the rewrite rules applied are:

        * Factorization of `Derivative`s exploiting linearity of finite differences.
          This helps CIRE with creating fewer temporaries while catching larger
          redundant sub-expressions.
    """
    processed = _rewrite_exprs_linearity(expressions)

    return processed


def _rewrite_exprs_linearity(expressions):
    Term = namedtuple('Term', 'other deriv func')
    Term.__new__.__defaults__ = (None, None, None)

    # `D0(a) + D1(b) == D(a + b)` <=> `D0` and `D1`'s metadata match, i.e. they
    # are the same type of derivative
    key = lambda e: e._metadata

    @singledispatch
    def _is_const_coeff(c, deriv):
        """True if coefficient definitely constant w.r.t. derivative, False otherwise."""
        return False

    @_is_const_coeff.register(sympy.Number)
    def _(c, deriv):
        return True

    @_is_const_coeff.register(sympy.Symbol)
    def _(c, deriv):
        try:
            return c.is_const
        except AttributeError:
            # Retrocompatibility -- if a sympy.Symbol, there's no `is_const` to query
            # We conservatively return False
            return False

    @_is_const_coeff.register(sympy.Function)
    def _(c, deriv):
        return not set(deriv.dims) & c.free_symbols

    @_is_const_coeff.register(sympy.Expr)
    def _(c, deriv):
        return all(_is_const_coeff(a) for a in c.args)

    def _doit(expr):
        args = []
        terms = []
        changed = False
        for a in expr.args:
            ax, term, flag = _doit(a)
            args.append(ax)
            terms.append(term)
            changed |= flag
        if changed:
            expr = expr.func(*args)
        return _doit_handle(expr, terms)

    @singledispatch
    def _doit_handle(expr, terms):
        #print('IN FALLBACK'); from IPython import embed; embed()
        return expr, Term(expr), False

    @_doit_handle.register(sympy.Derivative)
    def _(expr, terms):
        #print('IN DERIVATIVE'); from IPython import embed; embed()
        return expr, Term(sympy.S.One, expr), False

    @_doit_handle.register(sympy.Mul)
    def _(expr, terms):
        derivs, others = split(terms, lambda i: i.deriv is not None)
        #print('IN MUL'); from IPython import embed; embed()
        if len(derivs) == 1:
            # Linear => propagate found Derivative upstream
            deriv = derivs[0].deriv
            other = expr.func(*[i.other for i in others])  # De-nest terms
            return expr, Term(other, deriv, expr.func), False
        else:
            return expr, Term(expr), False

    @_doit_handle.register(sympy.Add)
    def _(expr, terms):
        derivs, others = split(terms, lambda i: i.deriv is not None)
        if not derivs:
            return expr, Term(expr), False

        # key: same type of derivative, same coefficient
        # E.g., 2*a*f.dx + 2*a*g.dx -> 2*a*(f + g).dx
        mapper = as_mapper(derivs, lambda i: key(i.deriv))
        if len(mapper) == len(derivs):
            # No derivatives to group unfortunately
            return expr, Term(expr), False

        processed = []
        for v in mapper.values():
            fact, nonfact = split(v, lambda i: _is_const_coeff(i.other, i.deriv))
            if fact:
                # Finally factorize derivative arguments
                func = fact[0].deriv._new_from_self
                exprs = []
                for i in fact:
                    if i.func:
                        exprs.append(i.func(i.other, i.deriv.expr))
                    else:
                        assert i.other == 1
                        exprs.append(i.deriv.expr)
                fact = [Term(func(expr=expr.func(*exprs)))]

            for i in fact + nonfact:
                if i.func:
                    processed.append(i.func(i.other, i.deriv))
                else:
                    processed.append(i.other)

        others = [i.other for i in others]
        expr = expr.func(*(processed + others))

        return expr, Term(expr), True

    processed = [_doit(e) for e in expressions]
    processed = list(zip(*processed))[0]

    return processed
