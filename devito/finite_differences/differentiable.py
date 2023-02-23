from collections import ChainMap
from itertools import product
from functools import singledispatch

from cached_property import cached_property
import numpy as np
import sympy
from sympy.core.add import _addsort
from sympy.core.mul import _keep_coeff, _mulsort
from sympy.core.decorators import call_highest_priority
from sympy.core.evalf import evalf_table

from devito.finite_differences.tools import make_shift_x0
from devito.logger import warning
from devito.tools import (as_tuple, filter_ordered, flatten, frozendict,
                          infer_dtype, is_integer, split)
from devito.types import (Array, DimensionTuple, Evaluable, Indexed, Spacing,
                          StencilDimension)

__all__ = ['Differentiable', 'IndexDerivative', 'EvalDerivative', 'Weights']


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
        if len(grids) > 1:
            warning("Expression contains multiple grids, returning first found")
        try:
            return grids.pop()
        except KeyError:
            return None

    @cached_property
    def dtype(self):
        dtypes = {f.dtype for f in self.find(Indexed)} - {None}
        return infer_dtype(dtypes)

    @cached_property
    def indices(self):
        return tuple(filter_ordered(flatten(getattr(i, 'indices', ())
                                            for i in self._args_diff)))

    @cached_property
    def dimensions(self):
        return tuple(filter_ordered(flatten(getattr(i, 'dimensions', ())
                                            for i in self._args_diff)))

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
        return dict(ChainMap(*[getattr(i, '_fd', {}) for i in self._args_diff]))

    @cached_property
    def _symbolic_functions(self):
        return frozenset([i for i in self._functions if i.coefficients == 'symbolic'])

    @cached_property
    def _uses_symbolic_coefficients(self):
        return bool(self._symbolic_functions)

    def _eval_at(self, func):
        if not func.is_Staggered:
            # Cartesian grid, do no waste time
            return self
        return self.func(*[getattr(a, '_eval_at', lambda x: a)(func) for a in self.args])

    def _subs(self, old, new, **hints):
        if old is self:
            return new
        if old is new:
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
        return super(Differentiable, self).__hash__()

    def __getattr__(self, name):
        """
        Try calling a dynamically created FD shortcut.

        Notes
        -----
        This method acts as a fallback for __getattribute__
        """
        if name in self._fd:
            return self._fd[name][0](self)
        raise AttributeError("%r object has no attribute %r" % (self.__class__, name))

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
        return super(Differentiable, self).__eq__(other) and\
            all(getattr(self, i, None) == getattr(other, i, None)
                for i in self.__rkwargs__)

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
        space_dims = [d for d in self.dimensions if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
        return Add(*[getattr(self, d) for d in derivs])

    def div(self, shift=None):
        space_dims = [d for d in self.dimensions if d.is_Space]
        shift_x0 = make_shift_x0(shift, (len(space_dims),))
        return Add(*[getattr(self, 'd%s' % d.name)(x0=shift_x0(shift, d, None, i))
                     for i, d in enumerate(space_dims)])

    def grad(self, shift=None):
        from devito.types.tensor import VectorFunction, VectorTimeFunction
        space_dims = [d for d in self.dimensions if d.is_Space]
        shift_x0 = make_shift_x0(shift, (len(space_dims),))
        comps = [getattr(self, 'd%s' % d.name)(x0=shift_x0(shift, d, None, i))
                 for i, d in enumerate(space_dims)]
        vec_func = VectorTimeFunction if self.is_TimeDependent else VectorFunction
        return vec_func(name='grad_%s' % self.name, time_order=self.time_order,
                        space_order=self.space_order, components=comps, grid=self.grid)

    def biharmonic(self, weight=1):
        """
        Generates a symbolic expression for the weighted biharmonic operator w.r.t.
        all spatial Dimensions Laplace(weight * Laplace (self))
        """
        space_dims = [d for d in self.dimensions if d.is_Space]
        derivs = tuple('d%s2' % d.name for d in space_dims)
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
    prio = lambda x: getattr(x, '_fd_priority', 0)
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
        nested, others = split(args, lambda e: isinstance(e, Mul))
        args = flatten(e.args for e in nested) + list(others)

        # a*0 -> 0
        if any(i == 0 for i in args):
            return sympy.S.Zero

        # a*1 -> a
        args = [i for i in args if i != 1]

        # a*-1 -> a*-1
        # a*-1*-1 -> a
        # a*-1*-1*-1 -> a*-1
        nminus = len([i for i in args if i == sympy.S.NegativeOne])
        args = [i for i in args if i != sympy.S.NegativeOne]
        if nminus % 2 == 1:
            args.append(sympy.S.NegativeOne)

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
        except ValueError:
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
        ref_inds = func_args.indices_ref._getters

        for f in self.args:
            if f not in self._args_diff:
                new_args.append(f)
            elif f is func_args or isinstance(f, DifferentiableFunction):
                new_args.append(f)
            else:
                ind_f = f.indices_ref._getters
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


class IndexSum(DifferentiableOp):

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
            return self.func(expr, self.dimensions)

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

        try:
            self._spacings = set().union(*[i.find(Spacing) for i in weights])
        except AttributeError:
            self._spacing = set()

        kwargs['scope'] = 'constant'

        super().__init_finalize__(*args, **kwargs)

    def __eq__(self, other):
        return (isinstance(other, Weights) and
                self.dimension is other.dimension and
                self.name == other.name and
                self.indices == other.indices and
                self.weights == other.weights)

    __hash__ = sympy.Basic.__hash__

    def _hashable_content(self):
        return super()._hashable_content() + (self.name,) + tuple(self.weights)

    @property
    def dimension(self):
        return self.dimensions[0]

    @property
    def spacings(self):
        return self._spacings

    weights = Array.initvalue


class IndexDerivative(IndexSum):

    __rargs__ = ('expr', 'mapper')

    def __new__(cls, expr, mapper, **kwargs):
        dimensions = as_tuple(mapper.values())

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

        return expr


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

    def _new_rawargs(self, *args, **kwargs):
        kwargs.pop('is_commutative', None)
        return self.func(*args, **kwargs)


class diffify(object):

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
