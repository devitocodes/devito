from collections import OrderedDict
from collections.abc import Iterable
from functools import singledispatch

from sympy import Pow, Add, Mul, Min, Max
from sympy.core.add import _addsort
from sympy.core.mul import _mulsort

from devito.symbolics.extended_sympy import rfunc
from devito.symbolics.queries import q_leaf
from devito.symbolics.search import retrieve_indexed, retrieve_functions
from devito.tools import as_list, as_tuple, flatten, split, transitive_closure
from devito.types.basic import Basic
from devito.types.array import ComponentAccess
from devito.types.equation import Eq
from devito.types.relational import Le, Lt, Gt, Ge

__all__ = ['xreplace_indices', 'pow_to_mul', 'indexify', 'subs_op_args',
           'uxreplace', 'Uxmapper', 'reuse_if_untouched', 'evalrel']


def uxreplace(expr, rule):
    """
    An alternative to SymPy's ``xreplace`` for when the caller can guarantee
    that no re-evaluations are necessary or when re-evaluations should indeed
    be avoided at all costs (e.g., to prevent SymPy from unpicking Devito
    transformations, such as factorization).

    The canonical ordering of the arguments is however guaranteed; where this is
    not possible, a re-evaluation will be enforced.

    By avoiding re-evaluations, this function is typically much quicker than
    SymPy's xreplace.

    A further feature of ``uxreplace`` consists of enabling the substitution
    of compound nodes. Consider the expression `a*b*c*d`; if one wants to replace
    `b*c` with say `e*f`, then the following mapper may be passed:
    `{a*b*c*d: {b: e*f, c: None}}`. This way, only the `b` and `c` pertaining to
    `a*b*c*d` will be affected, and in particular `c` will be dropped, while `b`
    will be replaced by `e*f`, thus obtaining `a*d*e*f`.
    """
    return _uxreplace(expr, rule)[0]


def _uxreplace(expr, rule):
    if expr in rule:
        v = rule[expr]
        if not isinstance(v, dict):
            return v, True
        args, eargs = split(expr.args, lambda i: i in v)
        args = [v[i] for i in args if v[i] is not None]
        changed = True
    elif expr.__class__ in rule:
        # Exact type -> type substitution
        cls = rule[expr.__class__]
        expr = cls(*expr.args)
        args, eargs = [], expr.args
        changed = True
    else:
        try:
            args, eargs = [], expr.args
        except AttributeError:
            # E.g., unsympified `int`
            args, eargs = [], []
        changed = False

    if rule:
        for a in eargs:
            try:
                ax, flag = _uxreplace(a, rule)
                args.append(ax)
                changed |= flag
            except AttributeError:
                # E.g., un-sympified numbers
                args.append(a)
        if changed:
            return _uxreplace_handle(expr, args), True

    return expr, False


@singledispatch
def _uxreplace_handle(expr, args):
    return expr.func(*args)


@_uxreplace_handle.register(Min)
@_uxreplace_handle.register(Max)
@_uxreplace_handle.register(Pow)
def _(expr, args):
    evaluate = all(i.is_Number for i in args)
    return expr.func(*args, evaluate=evaluate)


@_uxreplace_handle.register(Add)
def _(expr, args):
    if all(i.is_commutative for i in args):
        _addsort(args)
        _eval_numbers(expr, args)
        return expr.func(*args, evaluate=False)
    else:
        return expr._new_rawargs(*args)


@_uxreplace_handle.register(Mul)
def _(expr, args):
    if all(i.is_commutative for i in args):
        _mulsort(args)
        _eval_numbers(expr, args)
        return expr.func(*args, evaluate=False)
    else:
        return expr._new_rawargs(*args)


@_uxreplace_handle.register(ComponentAccess)
@_uxreplace_handle.register(Eq)
def _(expr, args):
    # Handler for all other Reconstructable objects
    kwargs = {i: getattr(expr, i) for i in expr.__rkwargs__}
    return expr.func(*args, **kwargs)


def _eval_numbers(expr, args):
    """
    Helper function for in-place reduction of the expr arguments.
    """
    numbers, others = split(args, lambda i: i.is_Number)
    if len(numbers) > 1:
        args[:] = [expr.func(*numbers)] + others


class Uxmapper(dict):

    """
    A helper mapper for uxreplace. Any dict can be passed as subs to uxreplace,
    but with a Uxmapper it is also possible to easily express compound replacements,
    such as sub-expressions within n-ary operations (e.g., `a*b` inside the 4-way
    Mul `a*c*b*d`).
    """

    class Uxsubmap(dict):
        @property
        def free_symbols(self):
            return {v for v in self.values() if v is not None}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extracted = OrderedDict()

    def add(self, expr, make, terms=None):
        """
        Without ``terms``: add ``expr`` to the mapper binding it to the symbol
        generated with the callback ``make``.
        With ``terms``: add the compound sub-expression made of ``terms`` to the
        mapper. ``terms`` is a list of one or more items in ``expr.args``.
        """
        if expr in self:
            return

        if not terms:
            self[expr] = self.extracted[expr] = make()
            return

        terms = as_list(terms)

        base = terms.pop(0)
        if terms:
            k = expr.func(base, *terms)
            try:
                symbol = self.extracted[k]
            except KeyError:
                symbol = self.extracted.setdefault(k, make())
            self[expr] = self.Uxsubmap.fromkeys(terms)
            self[expr][base] = symbol
        else:
            self[base] = self.extracted[base] = make()


def xreplace_indices(exprs, mapper, key=None):
    """
    Replace array indices in expressions.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions to which the replacement is applied.
    mapper : dict
        The substitution rules.
    key : list of symbols or callable
        An escape hatch to rule out some objects from the replacement.
        If a list, apply the replacement to the symbols in ``key`` only. If a
        callable, apply the replacement to a symbol S if and only if ``key(S)``
        gives True.
    """
    handle = flatten(retrieve_indexed(i) for i in as_tuple(exprs))
    if isinstance(key, Iterable):
        handle = [i for i in handle if i.base.label in key]
    elif callable(key):
        handle = [i for i in handle if key(i)]
    mapper = dict(zip(handle, [i.xreplace(mapper) for i in handle]))
    replaced = [uxreplace(i, mapper) for i in as_tuple(exprs)]
    return replaced if isinstance(exprs, Iterable) else replaced[0]


def pow_to_mul(expr):
    if q_leaf(expr) or isinstance(expr, Basic):
        return expr
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        try:
            int(exp)
        except TypeError:
            # E.g., a Symbol, or possibly a generic expression
            return expr
        if exp > 10 or exp < -10 or int(exp) != exp or exp == 0:
            # Large and non-integer powers remain untouched
            return expr
        elif exp == -1:
            # Reciprocals also remain untouched, but we traverse the base
            # looking for other Pows
            return expr.func(pow_to_mul(base), exp, evaluate=False)
        elif exp > 0:
            return Mul(*[base]*int(exp), evaluate=False)
        else:
            # SymPy represents 1/x as Pow(x,-1). Also, it represents
            # 2/x as Mul(2, Pow(x, -1)). So we shouldn't end up here,
            # but just in case SymPy changes its internal conventions...
            posexpr = Mul(*[base]*(-int(exp)), evaluate=False)
            return Pow(posexpr, -1, evaluate=False)
    else:
        args = [pow_to_mul(i) for i in expr.args]

        # Some SymPy versions will evaluate the two-args case
        # `(negative integer, mul)` despite the `evaluate=False`. For example,
        # `Mul(-2, a*a, evaluate=False)` gets evaluated to `-2/a**2`. By swapping
        # the args, the issue disappears...
        try:
            a0, a1 = args
            if a0.is_Number and a0 < 0 and not q_leaf(a1):
                args = [a1, a0]
        except ValueError:
            pass

        return expr.func(*args, evaluate=False)


def indexify(expr):
    """
    Given a SymPy expression, return a new SymPy expression in which all
    AbstractFunction objects have been converted into Indexed objects.
    """
    mapper = {}
    for i in retrieve_functions(expr):
        try:
            if i.is_AbstractFunction:
                mapper[i] = i.indexify()
        except AttributeError:
            pass
    return expr.xreplace(mapper)


def subs_op_args(expr, args):
    """
    Substitute Operator argument values into an expression. `args` is
    expected to be as produced by `Operator.arguments` -- it can only
    contain string keys, with values that are not themselves expressions
    which will be substituted into.
    """
    return expr.subs({i.name: args[i.name] for i in expr.free_symbols if i.name in args})


def reuse_if_untouched(expr, args, evaluate=False):
    """
    Reconstruct `expr` iff any of the provided `args` is different than
    the corresponding arg in `expr.args`.
    """
    if all(a is b for a, b in zip(expr.args, args)):
        return expr
    else:
        return expr.func(*args, evaluate=evaluate)


def evalrel(func=min, input=None, assumptions=None):
    """
    The purpose of this function is two-fold: (i) to reduce the `input` candidates of a
    for a Min/Max expression based on the given `assumptions` and (ii) return the nested
    Min/Max expression of the reduced-size input.

    Parameters
    ----------
    func : builtin function or method
        min or max. Defines the operation to simplify. Defaults to `min`.
    input : list
        A list of the candidate symbols to be simplified. Defaults to None.
    assumptions : list
        A list of assumptions formed as relationals between candidates, assumed to be
        True. Defaults to None.

    Examples
    --------
    Assuming no values are known for `a`, `b`, `c`, `d` but we know that `d<=a` and
    `c>=b` we can safely drop `a` and `c` from the candidate list.

    >>> from devito import Symbol
    >>> a = Symbol('a')
    >>> b = Symbol('b')
    >>> c = Symbol('c')
    >>> d = Symbol('d')
    >>> evalrel(max, [a, b, c, d], [Le(d, a), Ge(c, b)])
    Max(a, c)
    """
    sfunc = (Min if func is min else Max)  # Choose SymPy's Min/Max

    # Form relationals so that RHS has more arguments than LHS:
    # i.e. (a + d >= b) ---> (b <= a + d)
    assumptions = [a.reversed if len(a.lhs.args) > len(a.rhs.args) else a
                   for a in as_list(assumptions)]

    # Break-down relations if possible
    processed = []
    for a in as_list(assumptions):
        if isinstance(a, (Ge, Gt)) and a.rhs.is_Add and a.lhs.is_positive:
            if all(i.is_positive for i in a.rhs.args):
                # If `c >= a + b, {a, b, c} >= 0` then add 'c>=a, c>=b'
                processed.extend(Ge(a.lhs, i) for i in a.rhs.args)
            elif len(a.rhs.args) == 2:
                # If `c >= a + b, a>=0, b<=0` then add 'c>=b, c<=a'
                processed.extend(Ge(a.lhs, i) if not i.is_positive else Le(a.lhs, i)
                                 for i in a.rhs.args)
            else:
                processed.append(a)
        else:
            processed.append(a)

    # Apply assumptions to fill a subs mapper
    # e.g. When looking for 'max' and Gt(a, b), mapper is filled with {b: a} so that `b`
    # is subsituted by `a`
    mapper = {}
    for a in processed:
        if set(a.args).issubset(input):
            # If a.args=(a, b) and input=(a, b, c), condition is True,
            # if a.args=(a, d) and input=(a, b, c), condition is False
            assert len(a.args) == 2
            a0, a1 = a.args
            if ((isinstance(a, (Ge, Gt)) and func is max) or
               (isinstance(a, (Le, Lt)) and func is min)):
                mapper[a1] = a0
            elif ((isinstance(a, (Le, Lt)) and func is max) or
                  (isinstance(a, (Ge, Gt)) and func is min)):
                mapper[a0] = a1

    # Collapse graph paths
    mapper = transitive_closure(mapper)
    input = [i.subs(mapper) for i in input]

    # Explore simplification opportunities that may have emerged and generate Min/Max
    # expression
    try:
        exp = sfunc(*input)  # Can it be evaluated or simplified?
        if exp.is_Function:
            # Use the new args only if evaluation managed to reduce the number
            # of candidates.
            input = min(input, exp.args, key=len)
        else:
            # Since we are here, exp is a simplified expression
            return exp
    except TypeError:
        pass
    return rfunc(func, *input)
