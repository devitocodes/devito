from collections import OrderedDict, namedtuple
from collections.abc import Iterable

import sympy
from sympy import Number, Indexed, Symbol, LM, LC

from devito.symbolics.extended_sympy import Add, Mul, Pow, Eq, FrozenExpr
from devito.symbolics.search import retrieve_indexed, retrieve_functions
from devito.tools import as_tuple, flatten, split

__all__ = ['freeze', 'unfreeze', 'evaluate', 'yreplace', 'xreplace_indices',
           'pow_to_mul', 'as_symbol', 'indexify', 'split_affine']


def freeze(expr):
    """
    Reconstruct ``expr`` turning all sympy.Mul and sympy.Add
    into FrozenExpr equivalents.
    """
    if expr.is_Atom or expr.is_Indexed:
        return expr
    elif expr.is_Add:
        rebuilt_args = [freeze(e) for e in expr.args]
        return Add(*rebuilt_args, evaluate=False)
    elif expr.is_Mul:
        rebuilt_args = [freeze(e) for e in expr.args]
        return Mul(*rebuilt_args, evaluate=False)
    elif expr.is_Pow:
        rebuilt_args = [freeze(e) for e in expr.args]
        return Pow(*rebuilt_args, evaluate=False)
    elif expr.is_Equality:
        rebuilt_args = [freeze(e) for e in expr.args]
        if isinstance(expr, FrozenExpr):
            # Avoid dropping metadata associated with /expr/
            return expr.func(*rebuilt_args)
        else:
            return Eq(*rebuilt_args, evaluate=False)
    else:
        return expr.func(*[freeze(e) for e in expr.args])


def unfreeze(expr):
    """
    Reconstruct ``expr`` turning all FrozenExpr subtrees into their
    SymPy equivalents.
    """
    if expr.is_Atom or expr.is_Indexed:
        return expr
    func = expr.func.__base__ if isinstance(expr, FrozenExpr) else expr.func
    return func(*[unfreeze(e) for e in expr.args])


def evaluate(expr, **subs):
    """
    Numerically evaluate a SymPy expression. Subtrees of type FrozenExpr
    are forcibly evaluated.
    """
    expr = unfreeze(expr)
    return expr.subs(subs)


def yreplace(exprs, make, rule=None, costmodel=lambda e: True, repeat=False, eager=False):
    """
    Unlike SymPy's ``xreplace``, which performs structural replacement based on a mapper,
    ``yreplace`` applies replacements using two callbacks:

        * The "matching rule" -- a boolean function telling whether an expression
          honors a certain property.
        * The "cost model" -- a boolean function telling whether an expression exceeds
          a certain (e.g., operation count) cost.

    Parameters
    ----------
    exprs : expr-like or list of expr-like
        One or more expressions searched for replacements.
    make : dict or callable
        Either a mapper of substitution rules (just like in ``xreplace``), or
        or a callable returning unique symbols each time it is called.
    rule : callable, optional
        The matching rule (see above). Unnecessary if ``make`` is a dict.
    costmodel : callable, optional
        The cost model (see above).
    repeat : bool, optional
        If True, repeatedly apply ``xreplace`` until no more replacements are
        possible. Defaults to False.
    eager : bool, optional
        If True, replaces an expression ``e`` as soon as the condition
        ``rule(e) and costmodel(e)`` is True. Otherwise, the search continues
        for larger, more expensive expressions. Defaults to False.

    Notes
    -----
    In general, there is no relationship between the set of expressions for which
    the matching rule gives True and the set of expressions passing the cost test.
    For example, in the expression `a + b` all of `a`, `b` and `a+b` may satisfy
    the matching rule, whereas only `a+b` satisfy the cost test. Likewise, an
    expression may pass the cost test, but not satisfy the matching rule.
    """
    found = OrderedDict()
    rebuilt = []

    # Define `replace()` based on the user-provided `make`
    if isinstance(make, dict):
        rule = rule if rule is not None else (lambda i: i in make)
        replace = lambda i: make[i]
    else:
        assert callable(make) and callable(rule)

        def replace(expr):
            temporary = found.get(expr)
            if not temporary:
                temporary = make()
                found[expr] = temporary
            return temporary

    def run(expr):
        if expr.is_Atom or expr.is_Indexed:
            return expr, rule(expr)
        elif expr.is_Pow:
            base, flag = run(expr.base)
            if flag:
                if costmodel(base):
                    return expr.func(replace(base), expr.exp, evaluate=False), False
                elif costmodel(expr):
                    return replace(expr), False
                else:
                    # If `rule(expr)`, keep searching for larger expressions
                    return expr.func(base, expr.exp, evaluate=False), rule(expr)
            else:
                return expr.func(base, expr.exp, evaluate=False), False
        else:
            children = [run(a) for a in expr.args]
            matching = [a for a, flag in children if flag]
            other = [a for a, _ in children if a not in matching]

            if not matching:
                return expr.func(*other, evaluate=False), False

            if eager is False:
                matched = expr.func(*matching, evaluate=False)
                if len(matching) == len(children) and rule(expr):
                    # Keep searching for larger expressions
                    return matched, True
                elif rule(matched) and costmodel(matched):
                    # E.g.: a*b*c*d -> a*r0
                    rebuilt = expr.func(*(other + [replace(matched)]), evaluate=False)
                    return rebuilt, False
                else:
                    # E.g.: a*b*c*d -> a*r0*r1*r2
                    replaced = [replace(e) for e in matching if costmodel(e)]
                    unreplaced = [e for e in matching if not costmodel(e)]
                    rebuilt = expr.func(*(other + replaced + unreplaced), evaluate=False)
                    return rebuilt, False
            else:
                replaceable, unreplaced = split(matching, lambda e: costmodel(e))
                if replaceable:
                    # E.g.: a*b*c*d -> a*r0*r1*r2
                    replaced = [replace(e) for e in replaceable]
                    rebuilt = expr.func(*(other + replaced + unreplaced), evaluate=False)
                    return rebuilt, False
                matched = expr.func(*matching, evaluate=False)
                if rule(matched) and costmodel(matched):
                    if len(matching) == len(children):
                        # E.g.: a*b*c*d -> r0
                        return replace(matched), False
                    else:
                        # E.g.: a*b*c*d -> a*r0
                        rebuilt = expr.func(*(other + [replace(matched)]), evaluate=False)
                        return rebuilt, False
                elif len(matching) == len(children) and rule(expr):
                    # Keep searching for larger expressions
                    return matched, True
                else:
                    # E.g.: a*b*c*d; a,b,a*b replaceable but not satisfying the cost
                    # model, hence giving up as c,d,c*d aren't replaceable
                    return expr.func(*(matching + other), evaluate=False), False

    # Process the provided expressions
    built = []
    for expr in as_tuple(exprs):
        assert expr.is_Equality
        root = expr.rhs

        while True:
            ret, flag = run(root)

            # The whole RHS may need to be replaced
            if flag and costmodel(ret):
                ret = replace(root)

            if repeat and ret != root:
                root = ret
            else:
                rebuilt.append(expr.func(expr.lhs, ret, evaluate=False))
                break

        # Construct Eqs for temporaries
        built.extend(expr.func(v, k) for k, v in list(found.items())[len(built):])

    return built + rebuilt, built


def xreplace_indices(exprs, mapper, key=None, only_rhs=False):
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
    only_rhs : bool, optional
        If True, apply the replacement to Eq right-hand sides only.
    """
    get = lambda i: i.rhs if only_rhs is True else i
    handle = flatten(retrieve_indexed(get(i)) for i in as_tuple(exprs))
    if isinstance(key, Iterable):
        handle = [i for i in handle if i.base.label in key]
    elif callable(key):
        handle = [i for i in handle if key(i)]
    mapper = dict(zip(handle, [i.xreplace(mapper) for i in handle]))
    replaced = [i.xreplace(mapper) for i in as_tuple(exprs)]
    return replaced if isinstance(exprs, Iterable) else replaced[0]


def pow_to_mul(expr):
    if expr.is_Atom or expr.is_Indexed:
        return expr
    elif expr.is_Pow:
        base, exp = expr.as_base_exp()
        if exp > 10 or exp < -10 or int(exp) != exp or exp == 0:
            # Large and non-integer powers remain untouched
            return expr
        elif exp == -1:
            # Reciprocals also remain untouched, but we traverse the base
            # looking for other Pows
            return expr.func(pow_to_mul(base), exp, evaluate=False)
        elif exp > 0:
            return sympy.Mul(*[base]*exp, evaluate=False)
        else:
            # SymPy represents 1/x as Pow(x,-1). Also, it represents
            # 2/x as Mul(2, Pow(x, -1)). So we shouldn't end up here,
            # but just in case SymPy changes its internal conventions...
            posexpr = sympy.Mul(*[base]*(-exp), evaluate=False)
            return sympy.Pow(posexpr, -1, evaluate=False)
    else:
        return expr.func(*[pow_to_mul(i) for i in expr.args], evaluate=False)


def as_symbol(expr):
    """Cast to sympy.Symbol."""
    from devito.types import Dimension
    try:
        if expr.is_Symbol:
            return expr
    except AttributeError:
        pass
    try:
        return Number(expr)
    except (TypeError, ValueError):
        pass
    if isinstance(expr, str):
        return Symbol(expr)
    elif isinstance(expr, Dimension):
        return Symbol(expr.name)
    elif expr.is_Symbol:
        return expr
    elif isinstance(expr, Indexed):
        return expr.base.label
    else:
        raise TypeError("Cannot extract symbol from type %s" % type(expr))


AffineFunction = namedtuple("AffineFunction", "var, coeff, shift")


def split_affine(expr):
    """
    Split an affine scalar function into its three components, namely variable,
    coefficient, and translation from origin.

    Raises
    ------
    ValueError
        If ``expr`` is non affine.
    """
    if expr.is_Number:
        return AffineFunction(None, None, expr)

    # Handle super-quickly the calls like `split_affine(x+1)`, which are
    # the majority.
    if expr.is_Add and len(expr.args) == 2:
        if expr.args[0].is_Number and expr.args[1].is_Symbol:
            # SymPy deterministically orders arguments -- first numbers, then symbols
            return AffineFunction(expr.args[1], 1, expr.args[0])

    # Fallback
    poly = expr.as_poly()
    if not (poly.is_univariate and poly.is_linear) or not LM(poly).is_Symbol:
        raise ValueError
    return AffineFunction(LM(poly), LC(poly), poly.TC())


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
