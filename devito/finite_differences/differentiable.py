from collections import ChainMap
from itertools import product
from functools import singledispatch, cached_property

import numpy as np
import sympy
from sympy.core.add import _addsort
from sympy.core.mul import _keep_coeff, _mulsort
from sympy.core.decorators import call_highest_priority
from sympy.core.evalf import evalf_table
try:
    from sympy.core.core import ordering_of_classes
except ImportError:
    # Moved in 1.13
    from sympy.core.basic import ordering_of_classes

from devito.finite_differences.tools import make_shift_x0, coeff_priority
from devito.logger import warning
from devito.tools import (as_tuple, filter_ordered, flatten, frozendict,
                          infer_dtype, extract_dtype, is_integer, split, is_number)
from devito.types import Array, DimensionTuple, Evaluable, StencilDimension
from devito.types.basic import AbstractFunction

__all__ = ['Differentiable', 'DiffDerivative', 'IndexDerivative', 'EvalDerivative',
           'Weights', 'Real', 'Imag', 'Conj']


class Differentiable(sympy.Expr, Evaluable):

    """
    A Differentiable is an algebric expression involving Functions, which can
    be derived w.r.t. one or more Dimensions.
    """

    # Set the operator priority higher than SymPy (10.0) to force the overridden
    # operators to be used
    _op_priority = sympy.Expr._op_priority + 1.

    __rkwargs__ = ('space_order', 'time_order', 'indices')

    @cached_property
    def _functions(self):
        return frozenset().union(*[i._functions for i in self._args_diff])

    @cached_property
    def _args_diff(self):
        ret = [i for i in self.args if isinstance(i, Differentiable)]
        ret.extend([i.function for i in self.args if i.is_Indexed])
        return tuple(ret)

    @cached_property
    def space_order(self):
        # Default 100 is for "infinitely" differentiable
        return min([getattr(i, 'space_order', 100) or 100 for i in self._args_diff],
                   default=100)

    @cached_property
    def time_order(self):
        # Default 100 is for "infinitely" differentiable
        return min([getattr(i, 'time_order', 100) or 100 for i in self._args_diff],
                   default=100)

    @cached_property
    def grid(self):
        grids = {getattr(i, 'grid', None) for i in self._args_diff} - {None}
        grids = {g.root for g in grids}
        if len(grids) > 1:
            warning("Expression contains multiple grids, returning first found")
        try:
            return grids.pop()
        except KeyError:
            return None

    @cached_property
    def dtype(self):
        dtypes = {f.dtype for f in self._functions} - {None}
        return infer_dtype(dtypes)

    @cached_property
    def indices(self):
        return tuple(filter_ordered(flatten(getattr(i, 'indices', ())
                                            for i in self._args_diff)))

    @cached_property
    def dimensions(self):
        return tuple(filter_ordered(flatten(getattr(i, 'dimensions', ())
                                            for i in self._args_diff)))

    @cached_property
    def root_dimensions(self):
        """Tuple of root Dimensions of the physical space Dimensions."""
        return tuple(d.root for d in self.dimensions if d.is_Space)

    @property
    def indices_ref(self):
        """The reference indices of the object (indices at first creation)."""
        if len(self._args_diff) == 1:
            return self._args_diff[0].indices_ref
        elif len(self._args_diff) == 0:
            return DimensionTuple(*self.dimensions, getters=self.dimensions)
        return highest_priority(self).indices_ref

    @cached_property
    def staggered(self):
        return tuple(filter_ordered(flatten(getattr(i, 'staggered', ())
                                            for i in self._args_diff)))

    @cached_property
    def is_Staggered(self):
        return any([getattr(i, 'is_Staggered', False) for i in self._args_diff])

    @cached_property
    def is_TimeDependent(self):
        return any(i.is_Time for i in self.dimensions)

    @cached_property
    def _fd(self):
        # Filter out all args with fd order too high
        fd_args = []
        for f in self._args_diff:
            try:
                if f.space_order <= self.space_order and \
                        (not f.is_TimeDependent or f.time_order <= self.time_order):
                    fd_args.append(f)
            except AttributeError:
                pass
        return dict(ChainMap(*[getattr(i, '_fd', {}) for i in fd_args]))

    @cached_property
    def _symbolic_functions(self):
        return frozenset([i for i in self._functions if i.coefficients == 'symbolic'])

    @cached_property
    def function(self):
        if len(self._functions) == 1:
            return set(self._functions).pop()
        else:
            return None

    @cached_property
    def _uses_symbolic_coefficients(self):
        return bool(self._symbolic_functions)

    @cached_property
    def coefficients(self):
        coefficients = {f.coefficients for f in self._functions}
        # If there is multiple ones, we have to revert to the highest priority
        # i.e we have to remove symbolic
        key = lambda x: coeff_priority.get(x, -1)
        return sorted(coefficients, key=key, reverse=True)[0]

    def _eval_at(self, func):
        if not func.is_Staggered:
            # Cartesian grid, do no waste time
            return self
        return self.func(*[getattr(a, '_eval_at', lambda x: a)(func) for a in self.args])

    def _subs(self, old, new, **hints):
        if old == self:
            return new
        if old == new:
            return self
        args = list(self.args)
        for i, arg in enumerate(args):
            try:
                args[i] = arg._subs(old, new, **hints)
            except AttributeError:
                continue
        return self.func(*args, evaluate=False)

    @property
    def _eval_deriv(self):
        return self.func(*[getattr(a, '_eval_deriv', a) for a in self.args])

    @property
    def _fd_priority(self):
        return .75 if self.is_TimeDependent else .5

    def __hash__(self):
        return super().__hash__()

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        if name in self._fd:
            return self._fd[name][0](self)
        raise AttributeError(f"{self.__class__!r} object has no attribute {name!r}")

    # Override SymPy arithmetic operators
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return Add(self, other)

    @call_highest_priority('__add__')
    def __iadd__(self, other):
        return Add(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return Add(other, self)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return Add(self, -other)

    @call_highest_priority('__sub__')
    def __isub__(self, other):
        return Add(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return Add(other, -self)

    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return Mul(self, other)

    @call_highest_priority('__mul__')
    def __imul__(self, other):
        return Mul(self, other)

    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return Mul(other, self)

    def __pow__(self, other):
        return Pow(self, other)

    def __rpow__(self, other):
        return Pow(other, self)

    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return Mul(self, Pow(other, sympy.S.NegativeOne))

    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        return Mul(other, Pow(self, sympy.S.NegativeOne))

    __truediv__ = __div__
    __rtruediv__ = __rdiv__

    def __floordiv__(self, other):
        from .elementary import floor
        return floor(self / other)

    def __rfloordiv__(self, other):
        from .elementary import floor
        return floor(other / self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __neg__(self):
        return Mul(sympy.S.NegativeOne, self)

    def __eq__(self, other):
        ret = super().__eq__(other)
        if ret is NotImplemented or not ret:
            # Non comparable or not equal as sympy objects
            return False

        return all(getattr(self, i, None) == getattr(other, i, None)
                   for i in self.__rkwargs__)

    def _hashable_content(self):
        # SymPy computes the hash of all Basic objects as:
        # `hash((type(self).__name__,) + self._hashable_content())`
        # However, our subclasses will be named after the main SymPy classes,
        # for example sympy.Add -> differentiable.Add, so we need to override
        # the hashable content to specify it's our own subclasses
        return super()._hashable_content() + ('differentiable',)

    @property
    def name(self):
        return "".join(f.name for f in self._functions)

    def shift(self, dim, shift):
        """
        Shift  expression by `shift` along the Dimension `dim`.
        For example u.shift(x, x.spacing) = u(x + h_x).
        """
        return self._subs(dim, dim + shift)

    @property
    def laplace(self):
        """
        Generates a symbolic expression for the Laplacian, the second
        derivative w.r.t all spatial Dimensions.
        """
        return self.laplacian()

    def laplacian(self, shift=None, order=None, method='FD', **kwargs):
        """
        Laplacian of the Differentiable with shifted derivatives and custom
        FD order.

        Each second derivative is left-right (i.e D^T D with D the first derivative ):
        `(self.dx(x0=dim+shift*dim.spacing,
                  fd_order=order)).dx(x0=dim-shift*dim.spacing, fd_order=order)`

        Parameters
        ----------
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite differences.
        """
        w = kwargs.get('weights', kwargs.get('w'))
        order = order or self.space_order
        space_dims = self.root_dimensions
        shift_x0 = make_shift_x0(shift, (len(space_dims),))
        derivs = tuple(f'd{d.name}2' for d in space_dims)
        return Add(*[getattr(self, d)(x0=shift_x0(shift, space_dims[i], None, i),
                                      method=method, fd_order=order, w=w)
                     for i, d in enumerate(derivs)])

    def div(self, shift=None, order=None, method='FD', **kwargs):
        """
        Divergence of the input Function.

        Parameters
        ----------
        func : Function or TensorFunction
            Function to take the divergence of
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite difference coefficients.
        """
        w = kwargs.get('weights', kwargs.get('w'))
        space_dims = self.root_dimensions
        shift_x0 = make_shift_x0(shift, (len(space_dims),))
        order = order or self.space_order
        return Add(*[getattr(self, f'd{d.name}')(x0=shift_x0(shift, d, None, i),
                                                 fd_order=order, method=method, w=w)
                     for i, d in enumerate(space_dims)])

    def grad(self, shift=None, order=None, method='FD', **kwargs):
        """
        Gradient of the input Function.

        Parameters
        ----------
        func : Function or TensorFunction
            Function to take the gradient of
        shift: Number, optional, default=None
            Shift for the center point of the derivative in number of gridpoints
        order: int, optional, default=None
            Discretization order for the finite differences.
            Uses `func.space_order` when not specified
        method: str, optional, default='FD'
            Discretization method. Options are 'FD' (default) and
            'RSFD' (rotated staggered grid finite-difference).
        weights/w: list, tuple, or dict, optional, default=None
            Custom weights for the finite
        """
        from devito.types.tensor import VectorFunction, VectorTimeFunction
        space_dims = self.root_dimensions
        shift_x0 = make_shift_x0(shift, (len(space_dims),))
        order = order or self.space_order
        w = kwargs.get('weights', kwargs.get('w'))
        comps = [getattr(self, f'd{d.name}')(x0=shift_x0(shift, d, None, i),
                                             fd_order=order, method=method, w=w)
                 for i, d in enumerate(space_dims)]
        vec_func = VectorTimeFunction if self.is_TimeDependent else VectorFunction
        return vec_func(name=f'grad_{self.name}', time_order=self.time_order,
                        space_order=self.space_order, components=comps, grid=self.grid)

    def biharmonic(self, weight=1):
        """
        Generates a symbolic expression for the weighted biharmonic operator w.r.t.
        all spatial Dimensions Laplace(weight * Laplace (self))
        """
        space_dims = self.root_dimensions
        derivs = tuple(f'd{d.name}2' for d in space_dims)
        return Add(*[getattr(self.laplace * weight, d) for d in derivs])

    def diff(self, *symbols, **assumptions):
        """
        Like ``sympy.diff``, but return a ``devito.Derivative`` instead of a
        ``sympy.Derivative``.
        """
        from devito.finite_differences.derivative import Derivative
        return Derivative(self, *symbols, **assumptions)

    def has(self, *pattern):
        """
        Unlike generic SymPy use cases, in Devito the majority of calls to `has`
        occur through the finite difference routines passing `sympy.core.symbol.Symbol`
        as `pattern`. Since the generic `_has` can be prohibitively expensive,
        we here quickly handle this special case, while using the superclass' `has`
        as fallback.
        """
        for p in pattern:
            # Following sympy convention, return True if any is found
            if isinstance(p, type) and issubclass(p, sympy.Symbol):
                # Symbols (and subclasses) are the leaves of an expression, and they
                # are promptly available via `free_symbols`. So this is super quick
                if any(isinstance(i, p) for i in self.free_symbols):
                    return True
        return super().has(*pattern)

    def has_free(self, *patterns):
        """
        Return True if self has object(s) `patterns` as a free expression,
        False otherwise.

        Notes
        -----
        This is overridden in SymPy 1.10, but not in previous versions.
        """
        try:
            return super().has_free(*patterns)
        except AttributeError:
            return all(i in self.free_symbols for i in patterns)


def highest_priority(DiffOp):
    # We want to get the object with highest priority
    # We also need to make sure that the object with the largest
    # set of dimensions is used when multiple ones with the same
    # priority appear
    prio = lambda x: (getattr(x, '_fd_priority', 0), len(x.dimensions))
    return sorted(DiffOp._args_diff, key=prio, reverse=True)[0]


class DifferentiableOp(Differentiable):

    __sympy_class__ = None

    def __new__(cls, *args, **kwargs):
        # Do not re-evaluate if any of the args is an EvalDerivative,
        # since the integrity of these objects must be preserved
        if any(isinstance(i, EvalDerivative) for i in args):
            kwargs['evaluate'] = False

        obj = cls.__base__.__new__(cls, *args, **kwargs)

        # Unfortunately SymPy may build new sympy.core objects (e.g., sympy.Add),
        # so here we have to rebuild them as devito.core objects
        if kwargs.get('evaluate', True):
            obj = diffify(obj)

        return obj

    def subs(self, *args, **kwargs):
        return self.func(*[getattr(a, 'subs', lambda x: a)(*args, **kwargs)
                           for a in self.args], evaluate=False)

    _subs = Differentiable._subs

    @property
    def _gather_for_diff(self):
        return self

    # Bypass useless expensive SymPy _eval_ methods, for which we either already
    # know or don't care about the answer, because it'd have ~zero impact on our
    # average expressions

    def _eval_is_even(self):
        return None

    def _eval_is_odd(self):
        return None

    def _eval_is_integer(self):
        return None

    def _eval_is_negative(self):
        return None

    def _eval_is_extended_negative(self):
        return None

    def _eval_is_positive(self):
        return None

    def _eval_is_extended_positive(self):
        return None

    def _eval_is_zero(self):
        return None


class DifferentiableFunction(DifferentiableOp):

    def __new__(cls, *args, **kwargs):
        return cls.__sympy_class__.__new__(cls, *args, **kwargs)

    def _eval_at(self, func):
        return self


class Add(DifferentiableOp, sympy.Add):
    __sympy_class__ = sympy.Add

    def __new__(cls, *args, **kwargs):
        # Here, often we get `evaluate=False` to prevent SymPy evaluation (e.g.,
        # when `cls==EvalDerivative`), but in all cases we at least apply a small
        # set of basic simplifications

        # (a+b)+c -> a+b+c (flattening)
        # TODO: use symbolics.flatten_args; not using it to avoid a circular import
        nested, others = split(args, lambda e: isinstance(e, Add))
        args = flatten(e.args for e in nested) + list(others)

        # a+0 -> a
        args = [i for i in args if i != 0]

        # Reorder for homogeneity with pure SymPy types
        _addsort(args)

        return super().__new__(cls, *args, **kwargs)


class Mul(DifferentiableOp, sympy.Mul):
    __sympy_class__ = sympy.Mul

    def __new__(cls, *args, **kwargs):
        # A Mul, being a DifferentiableOp, may not trigger evaluation upon
        # construction (e.g., when an EvalDerivative is present among its
        # arguments), so here we apply a small set of basic simplifications
        # to avoid generating functional, but ugly, code

        # (a*b)*c -> a*b*c (flattening)
        # TODO: use symbolics.flatten_args; not using it to avoid a circular import
        nested, others = split(args, lambda e: isinstance(e, Mul))
        args = flatten(e.args for e in nested) + list(others)

        # Gather all numbers and simplify
        nums, others = split(args, lambda e: is_number(e))
        scalar = sympy.Mul(*nums)

        # a*0 -> 0
        if scalar == 0:
            return sympy.S.Zero

        # a*1 -> a
        if scalar - 1 == 0:
            args = others
        else:
            args = [scalar] + others

        # Reorder for homogeneity with pure SymPy types
        _mulsort(args)

        # `sympy.Mul.flatten(coeff, Add)` flattens out nested Adds within Add,
        # which would destroy `EvalDerivative`s if present. So here we perform
        # a similar thing, but cautiously construct an evaluated Add, which
        # will preserve the integrity of `EvalDerivative`s, if any
        try:
            a, b = args
            if a.is_Rational:
                r, b = b.as_coeff_Mul()
                if r is sympy.S.One and type(b) is Add:
                    return Add(*[_keep_coeff(a, bi) for bi in b.args], evaluate=False)
        except (AttributeError, ValueError):
            pass

        return super().__new__(cls, *args, **kwargs)

    @property
    def _gather_for_diff(self):
        """
        We handle Mul arguments by hand in case of staggered inputs
        such as `f(x)*g(x + h_x/2)` that will be transformed into
        f(x + h_x/2)*g(x + h_x/2) and priority  of indexing is applied
        to have single indices as in this example.
        The priority is from least to most:
            - param
            - NODE
            - staggered
        """

        if len(set(f.staggered for f in self._args_diff)) == 1:
            return self

        func_args = highest_priority(self)
        new_args = []
        ref_inds = func_args.indices_ref.getters

        for f in self.args:
            if f not in self._args_diff:
                new_args.append(f)
            elif f is func_args or isinstance(f, DifferentiableFunction):
                new_args.append(f)
            else:
                ind_f = f.indices_ref.getters
                mapper = {ind_f.get(d, d): ref_inds.get(d, d)
                          for d in self.dimensions
                          if ind_f.get(d, d) is not ref_inds.get(d, d)}
                if mapper:
                    new_args.append(f.subs(mapper))
                else:
                    new_args.append(f)

        return self.func(*new_args, evaluate=False)


class Pow(DifferentiableOp, sympy.Pow):
    _fd_priority = 0
    __sympy_class__ = sympy.Pow


class Mod(DifferentiableOp, sympy.Mod):
    __sympy_class__ = sympy.Mod


class SafeInv(Differentiable, sympy.core.function.Application):
    _fd_priority = 0

    @property
    def base(self):
        return self.args[1]

    @property
    def val(self):
        return self.args[0]

    def __str__(self):
        return Pow(self.args[0], -1).__str__()

    __repr__ = __str__


class ComplexPart(Differentiable, sympy.core.function.Application):
    """Abstract class for `Real`, `Imag`, or `Conj` of an expression"""
    _name = None

    def __new__(cls, *args, **kwargs):
        if len(args) != 1:
            raise ValueError(f"{cls.__name__} expects exactly one arg;"
                             f" {len(args)} were supplied instead.")

        return super().__new__(cls, *args, **kwargs)

    def __str__(self):
        return f"{self.__class__.__name__}({self.args[0]})"

    __repr__ = __str__


class RealComplexPart(ComplexPart):

    @cached_property
    def dtype(self):
        dtype = extract_dtype(self)
        return dtype(0).real.__class__


class Real(RealComplexPart):
    """Get the real part of an expression"""
    _name = 'real'


class Imag(RealComplexPart):
    """Get the imaginary part of an expression"""
    _name = 'imag'


class Conj(ComplexPart):
    """Get the complex conjugate of an expression"""
    _name = 'conj'


class IndexSum(sympy.Expr, Evaluable):

    """
    Represent the summation over a multiindex, that is a collection of
    Dimensions, of an indexed expression.
    """

    __rargs__ = ('expr', 'dimensions')

    is_commutative = True

    def __new__(cls, expr, dimensions, **kwargs):
        dimensions = as_tuple(dimensions)
        if not dimensions:
            return expr
        for d in dimensions:
            try:
                if d.is_Dimension and is_integer(d.symbolic_size):
                    continue
            except AttributeError:
                pass
            raise ValueError("Expected Dimension with numeric size, "
                             "got `%s` instead" % d)

        # TODO: `has_free` only available with SymPy v>=1.10
        # We should start using `not expr.has_free(*dimensions)` once we drop
        # support for SymPy 1.8<=v<1.0
        if not all(d in expr.free_symbols for d in dimensions):
            raise ValueError("All Dimensions `%s` must appear in `expr` "
                             "as free variables" % str(dimensions))

        for i in expr.find(IndexSum):
            for d in dimensions:
                if d in i.dimensions:
                    raise ValueError("Dimension `%s` already appears in a "
                                     "nested tensor contraction" % d)

        obj = sympy.Expr.__new__(cls, expr)
        obj._expr = expr
        obj._dimensions = dimensions

        return obj

    def __repr__(self):
        return "%s(%s, (%s))" % (self.__class__.__name__, self.expr,
                                 ', '.join(d.name for d in self.dimensions))

    __str__ = __repr__

    def _sympystr(self, printer):
        return str(self)

    _latex = _sympystr

    def _hashable_content(self):
        return super()._hashable_content() + (self.dimensions,)

    @property
    def expr(self):
        return self._expr

    @property
    def dimensions(self):
        return self._dimensions

    def _evaluate(self, **kwargs):
        expr = self.expr._evaluate(**kwargs)

        if not kwargs.get('expand', True):
            return self._rebuild(expr)

        values = product(*[list(d.range) for d in self.dimensions])
        terms = []
        for i in values:
            mapper = dict(zip(self.dimensions, i))
            terms.append(expr.xreplace(mapper))
        return sum(terms)

    @property
    def free_symbols(self):
        return super().free_symbols - set(self.dimensions)

    func = DifferentiableOp._rebuild


class Weights(Array):

    """
    The weights (or coefficients) of a finite-difference expansion.
    """

    def __init_finalize__(self, *args, **kwargs):
        dimensions = as_tuple(kwargs.get('dimensions'))
        weights = kwargs.get('initvalue')

        assert len(dimensions) == 1
        d = dimensions[0]
        assert isinstance(d, StencilDimension) and d.symbolic_size == len(weights)
        assert isinstance(weights, (list, tuple, np.ndarray))

        # Normalize `weights`
        from devito.symbolics import pow_to_mul  # noqa, sigh
        weights = tuple(pow_to_mul(sympy.sympify(i)) for i in weights)

        kwargs['scope'] = kwargs.get('scope', 'stack')
        kwargs['initvalue'] = weights

        super().__init_finalize__(*args, **kwargs)

    @classmethod
    def class_key(cls):
        # Ensure Weights appear before any other AbstractFunction
        p, v, _ = Array.class_key()
        return p, v - 1, cls.__name__

    def __eq__(self, other):
        return (isinstance(other, Weights) and
                self.name == other.name and
                self.dimension == other.dimension and
                self.indices == other.indices and
                self.weights == other.weights)

    __hash__ = sympy.Basic.__hash__

    def _hashable_content(self):
        return (self.name, self.dimension, str(self.weights), self.scope)

    @property
    def dimension(self):
        return self.dimensions[0]

    weights = Array.initvalue

    def _xreplace(self, rule):
        if self in rule:
            return rule[self], True
        elif not rule:
            return self, False
        else:
            try:
                weights, flags = zip(*[i._xreplace(rule) for i in self.weights])
                if any(flags):
                    return self.func(initvalue=weights, function=None), True
            except AttributeError:
                # `float` weights
                pass
            return super()._xreplace(rule)

    @cached_property
    def _npweights(self):
        # NOTE: `self.weights` cannot just be an array or SymPy will fail
        # internally at `__eq__` since numpy arrays requite .all, not ==,
        # for equality comparison
        return np.array(self.weights)

    def value(self, idx):
        try:
            v = self.weights[idx]
        except TypeError:
            # E.g., `idx` is a tuple
            v = self._npweights[idx]
        if v.is_Number or v.is_Indexed:
            return sympy.sympify(v)
        else:
            return self[idx]


class IndexDerivative(IndexSum):

    __rargs__ = ('expr', 'mapper')

    def __new__(cls, expr, mapper, **kwargs):
        dimensions = as_tuple(set(mapper.values()))

        # Detect the Weights among the arguments
        weightss = []
        for a in expr.args:
            try:
                f = a.function
            except AttributeError:
                continue
            if isinstance(f, Weights):
                weightss.append(a)

        # Sanity check
        if not (expr.is_Mul and len(weightss) == 1):
            raise ValueError("Expect `expr*weights`, got `%s` instead" % str(expr))
        weights = weightss.pop()

        obj = super().__new__(cls, expr, dimensions)
        obj._weights = weights
        obj._mapper = frozendict(mapper)

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + (self.mapper,)

    def compare(self, other):
        if self is other:
            return 0
        n1 = self.__class__
        n2 = other.__class__
        if n1.__name__ == n2.__name__:
            return (self.weights.compare(other.weights) or
                    self.base.compare(other.base))
        else:
            return super().compare(other)

    @cached_property
    def base(self):
        return self.expr.func(*[a for a in self.expr.args if a is not self.weights])

    @property
    def weights(self):
        return self._weights

    @property
    def mapper(self):
        return self._mapper

    @property
    def depth(self):
        iderivs = self.expr.find(IndexDerivative)
        return 1 + max([i.depth for i in iderivs], default=0)

    def _evaluate(self, **kwargs):
        expr = super()._evaluate(**kwargs)

        if not kwargs.get('expand', True):
            return expr

        w = self.weights
        f = w.function
        d = w.dimension
        mapper = {w.subs(d, i): f.weights[n] for n, i in enumerate(d.range)}
        expr = expr.xreplace(mapper)

        return EvalDerivative(*expr.args, base=self.base)


class DiffDerivative(IndexDerivative, DifferentiableOp):
    pass


# SymPy args ordering is the same for Derivatives and IndexDerivatives
for i in ('DiffDerivative', 'IndexDerivative'):
    ordering_of_classes.insert(ordering_of_classes.index('Derivative') + 1, i)


class EvalDerivative(DifferentiableOp, sympy.Add):

    is_commutative = True

    __rkwargs__ = ('base',)

    def __new__(cls, *args, base=None, **kwargs):
        kwargs['evaluate'] = False

        # a+0 -> a
        args = [i for i in args if i != 0]

        # Reorder for homogeneity with pure SymPy types
        _addsort(args)

        obj = super().__new__(cls, *args, **kwargs)

        try:
            obj.base = base
        except AttributeError:
            # This might happen if e.g. one attempts a (re)construction with
            # one sole argument. The (re)constructed EvalDerivative degenerates
            # to an object of different type, in classic SymPy style. That's fine
            assert len(args) <= 1
            assert not obj.is_Add
            return obj

        return obj

    func = DifferentiableOp._rebuild

    # Since obj.base = base, then Differentiable.__eq__ leads to infinite recursion
    # as it checks obj.base == other.base
    __eq__ = sympy.Add.__eq__
    __hash__ = sympy.Add.__hash__

    def _new_rawargs(self, *args, **kwargs):
        kwargs.pop('is_commutative', None)
        return self.func(*args, **kwargs)


class diffify:

    """
    Helper class based on single dispatch to reconstruct all nodes in a sympy
    tree such they are all of type Differentiable.

    Notes
    -----
    The name "diffify" stems from SymPy's "simpify", which has an analogous task --
    converting all arguments into SymPy core objects.
    """

    def __new__(cls, obj):
        args = [diffify._doit(i) for i in obj.args]
        obj = diffify._doit(obj, args)
        return obj

    def _doit(obj, args=None):
        cls = diffify._cls(obj)
        args = args or obj.args

        if cls is obj.__class__:
            # Try to just update the args if possible (Add, Mul)
            try:
                return obj._new_rawargs(*args, is_commutative=obj.is_commutative)
            # Or just return the object (Float, Symbol, Function, ...)
            except AttributeError:
                return obj

        # Create object directly from args, avoid any rebuild
        return cls(*args, evaluate=False)

    @singledispatch
    def _cls(obj):
        return obj.__class__

    @_cls.register(sympy.Add)
    def _(obj):
        return Add

    @_cls.register(sympy.Mul)
    def _(obj):
        return Mul

    @_cls.register(sympy.Pow)
    def _(obj):
        return Pow

    @_cls.register(sympy.Mod)
    def _(obj):
        return Mod

    @_cls.register(Add)
    @_cls.register(Mul)
    @_cls.register(Pow)
    @_cls.register(Mod)
    @_cls.register(EvalDerivative)
    def _(obj):
        return obj.__class__


def diff2sympy(expr):
    """
    Translate a Differentiable expression into a SymPy expression.
    """

    def _diff2sympy(obj):
        flag = False
        args = []
        for a in obj.args:
            ax, af = _diff2sympy(a)
            args.append(ax)
            flag |= af

        # Handle special objects
        if isinstance(obj, DiffDerivative):
            return IndexDerivative(*args, obj.mapper), True

        # Handle generic objects such as arithmetic operations
        try:
            return obj.__sympy_class__(*args, evaluate=False), True
        except AttributeError:
            # Not of type DifferentiableOp
            pass
        except TypeError:
            # Won't lower (e.g., EvalDerivative)
            pass
        if flag:
            try:
                return obj.func(*args, evaluate=False), True
            except TypeError:
                # In case of indices using other Function, evaluate
                # may not be a supported argument.
                return obj.func(*args), True
        else:
            return obj, False

    return _diff2sympy(expr)[0]


# Make sure `sympy.evalf` knows how to evaluate the inherited classes
# Without these, `evalf` would rely on a much slower, much more generic, and
# thus much more time-inefficient fallback routine. This would hit us
# pretty badly when taking derivatives (see `finite_difference.py`), where
# `evalf` is used systematically
evalf_table[Add] = evalf_table[sympy.Add]
evalf_table[Mul] = evalf_table[sympy.Mul]
evalf_table[Pow] = evalf_table[sympy.Pow]


# Interpolation for finite differences
@singledispatch
def interp_for_fd(expr, x0, **kwargs):
    return expr


@interp_for_fd.register(sympy.Derivative)
def _(expr, x0, **kwargs):
    x0_expr = {d: v for d, v in x0.items() if d not in expr.dims}
    return expr.func(interp_for_fd(expr.expr, x0_expr, **kwargs))


@interp_for_fd.register(sympy.Expr)
def _(expr, x0, **kwargs):
    if expr.args:
        return expr.func(*[interp_for_fd(i, x0, **kwargs) for i in expr.args])
    else:
        return expr


@interp_for_fd.register(AbstractFunction)
def _(expr, x0, **kwargs):
    from devito.finite_differences.derivative import Derivative
    x0_expr = {d: v for d, v in x0.items() if v is not expr.indices_ref[d]}
    if x0_expr:
        dims = tuple((d, 0) for d in x0_expr)
        fd_o = tuple([2]*len(dims))
        return Derivative(expr, *dims, fd_order=fd_o, x0=x0_expr)
    else:
        return expr
