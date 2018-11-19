from collections import Iterable, OrderedDict, namedtuple

import sympy
from sympy import Number, Indexed, Symbol, LM, LC

from devito.symbolics.extended_sympy import Add, Mul, Pow, Eq, FrozenExpr
from devito.symbolics.search import retrieve_indexed, retrieve_functions
from devito.tools import as_tuple, flatten

__all__ = ['freeze', 'unfreeze', 'evaluate', 'xreplace_constrained', 'xreplace_indices',
           'pow_to_mul', 'as_symbol', 'indexify', 'split_affine']


def freeze(expr):
    """
    Reconstruct ``expr`` turning all :class:`sympy.Mul` and :class:`sympy.Add`
    into :class:`FrozenExpr` equivalents.
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
    Reconstruct ``expr`` turning all :class:`FrozenExpr` subtrees into their
    SymPy equivalents.
    """
    if expr.is_Atom or expr.is_Indexed:
        return expr
    func = expr.func.__base__ if isinstance(expr, FrozenExpr) else expr.func
    return func(*[unfreeze(e) for e in expr.args])


def evaluate(expr, **subs):
    """
    Numerically evaluate a SymPy expression. Subtrees of type :class:`FrozenExpr`
    are forcibly evaluated.
    """
    expr = unfreeze(expr)
    return expr.subs(subs)


def xreplace_constrained(exprs, make, rule=None, costmodel=lambda e: True, repeat=False):
    """
    Unlike ``xreplace``, which replaces all objects specified in a mapper,
    this function replaces all objects satisfying two criteria: ::

        * The "matching rule" -- a function returning True if a node within ``expr``
            satisfies a given property, and as such should be replaced;
        * A "cost model" -- a function triggering replacement only if a certain
            cost (e.g., operation count) is exceeded. This function is optional.

    Note that there is not necessarily a relationship between the set of nodes
    for which the matching rule returns True and those nodes passing the cost
    model check. It might happen for example that, given the expression ``a + b``,
    all of ``a``, ``b``, and ``a + b`` satisfy the matching rule, but only
    ``a + b`` satisfies the cost model.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param make: Either a mapper M: K -> V, indicating how to replace an expression
                 in K with a symbol in V, or a function with internal state that,
                 when called, returns unique symbols.
    :param rule: The matching rule (a lambda function). May be left unspecified if
                 ``make`` is a mapper.
    :param costmodel: The cost model (a lambda function, optional).
    :param repeat: Repeatedly apply ``xreplace`` until no more replacements are
                   possible (optional, defaults to False).
    """
    found = OrderedDict()
    rebuilt = []

    # Define /replace()/ based on the user-provided /make/
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
            if flag and costmodel(base):
                return expr.func(replace(base), expr.exp, evaluate=False), False
            else:
                return expr.func(base, expr.exp, evaluate=False), flag
        else:
            children = [run(a) for a in expr.args]
            matching = [a for a, flag in children if flag]
            other = [a for a, _ in children if a not in matching]
            if matching:
                matched = expr.func(*matching, evaluate=False)
                if len(matching) == len(children) and rule(expr):
                    # Go look for longer expressions first
                    return matched, True
                elif rule(matched) and costmodel(matched):
                    # Replace what I can replace, then give up
                    rebuilt = expr.func(*(other + [replace(matched)]), evaluate=False)
                    return rebuilt, False
                else:
                    # Replace flagged children, then give up
                    replaced = [replace(e) for e in matching if costmodel(e)]
                    unreplaced = [e for e in matching if not costmodel(e)]
                    rebuilt = expr.func(*(other + replaced + unreplaced), evaluate=False)
                    return rebuilt, False
            return expr.func(*other, evaluate=False), False

    # Process the provided expressions
    for expr in as_tuple(exprs):
        assert expr.is_Equality
        root = expr.rhs

        while True:
            ret, flag = run(root)
            if isinstance(make, dict) and root.is_Atom and flag:
                rebuilt.append(expr.func(expr.lhs, replace(root), evaluate=False))
                break
            elif repeat and ret != root:
                root = ret
            else:
                rebuilt.append(expr.func(expr.lhs, ret, evaluate=False))
                break

    # Post-process the output
    found = [Eq(v, k) for k, v in found.items()]

    return found + rebuilt, found


def xreplace_indices(exprs, mapper, key=None, only_rhs=False):
    """
    Replace indices in SymPy equations.

    :param exprs: The target SymPy expression, or a collection of SymPy expressions.
    :param mapper: A dictionary containing the index substitution rules.
    :param key: (Optional) either an iterable or a function. In the former case,
                all objects whose name does not appear in ``key`` are ruled out.
                Likewise, if a function, all objects for which ``key(obj)`` gives
                False are ruled out.
    :param only_rhs: (Optional) apply the substitution rules to right-hand sides only.
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
        if exp > 10 or exp < -10 or int(exp) != exp or exp == 0 or exp == -1:
            # Large and non-integer powers remain untouched, as do reciprocals
            return expr
        elif exp > 0:
            return sympy.Mul(*[base]*exp, evaluate=False)
        else:
            # sympy represents 1/x as Pow(x,-1)
            posexpr = sympy.Mul(*[base]*(-exp), evaluate=False)
            return sympy.Pow(posexpr, -1, evaluate=False)
    else:
        return expr.func(*[pow_to_mul(i) for i in expr.args], evaluate=False)


def as_symbol(expr):
    """
    Extract the "main" symbol from a SymPy object.
    """
    from devito.dimension import Dimension
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
    split_affine(expr)

    Split an affine scalar function into its three components, namely variable,
    coefficient, and translation from origin.

    :raises ValueError: If ``expr`` is non affine.
    """
    if expr.is_Number:
        return AffineFunction(None, None, expr)
    poly = expr.as_poly()
    if not (poly.is_univariate and poly.is_linear) or not LM(poly).is_Symbol:
        raise ValueError
    return AffineFunction(LM(poly), LC(poly), poly.TC())


def indexify(expr):
    """
    Given a SymPy expression, return a new SymPy expression in which all
    :class:`AbstractFunction` objects have been converted into :class:`Indexed`
    objects.
    """
    mapper = {}
    for i in retrieve_functions(expr):
        try:
            if i.is_AbstractFunction:
                mapper[i] = i.indexify()
        except AttributeError:
            pass
    return expr.xreplace(mapper)
