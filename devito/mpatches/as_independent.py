"""Monkeypatch for as_independent required for Devito Derivative. """

from packaging.version import Version

import sympy
from sympy.core.add import _unevaluated_Add
from sympy.core.expr import Expr
from sympy.core.mul import _unevaluated_Mul
from sympy.core.symbol import Symbol
from sympy.core.singleton import S

"""
Copy of upstream sympy methods, without docstrings, comments or typehints
Imports are moved to the top
"""
if Version(sympy.__version__) < Version('1.15.0.dev0'):
    def _sift_true_false(seq, keyfunc):
        true = []
        false = []
        for i in seq:
            if keyfunc(i):
                true.append(i)
            else:
                false.append(i)
        return true, false

    def as_independent(self, *deps, as_Add=None, strict=True):
        if self is S.Zero:
            return (self, self)

        if as_Add is None:
            as_Add = self.is_Add

        syms, other = _sift_true_false(deps, lambda d: isinstance(d, Symbol))
        syms_set = set(syms)

        if other:
            def has(e):
                return e.has_xfree(syms_set) or e.has(*other)
        else:
            def has(e):
                return e.has_xfree(syms_set)

        if as_Add:
            if not self.is_Add:
                if has(self):
                    return (S.Zero, self)
                else:
                    return (self, S.Zero)

            depend, indep = _sift_true_false(self.args, has)
            return (self.func(*indep), _unevaluated_Add(*depend))

        else:
            if not self.is_Mul:
                if has(self):
                    return (S.One, self)
                else:
                    return (self, S.One)

            args, nc = self.args_cnc()
            depend, indep = _sift_true_false(args, has)

            for i, n in enumerate(nc):
                if has(n):
                    depend.extend(nc[i:])
                    break
                indep.append(n)

            return self.func(*indep), _unevaluated_Mul(*depend)

    # Monkeypatch the method
    Expr.as_independent = as_independent
