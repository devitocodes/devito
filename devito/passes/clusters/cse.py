from collections import Counter, OrderedDict, defaultdict, namedtuple
from functools import singledispatch

import sympy
from sympy import Add, Function, Indexed, Mul, Pow
try:
    from sympy.core.core import ordering_of_classes
except ImportError:
    # Moved in 1.13
    from sympy.core.basic import ordering_of_classes

from devito.finite_differences.differentiable import IndexDerivative
from devito.ir import Cluster, Scope, cluster_pass
from devito.passes.clusters.utils import makeit_ssa
from devito.symbolics import estimate_cost, q_leaf, search
from devito.symbolics.manipulation import Uxmapper, _uxreplace
from devito.tools import as_list, frozendict
from devito.types import Eq, Symbol, Temp

__all__ = ['cse']


Candidate = namedtuple('Candidate', 'expr conditionals sources')
Candidate.__new__.__defaults__ = (None, None, None)


class CTemp(Temp):

    """
    A cluster-level Temp, similar to Temp, ensured to have different priority
    """
    ordering_of_classes.insert(ordering_of_classes.index('Temp') + 1, 'CTemp')


@cluster_pass
def cse(cluster, sregistry=None, options=None, **kwargs):
    """
    Common sub-expressions elimination (CSE).
    """
    make = lambda: CTemp(name=sregistry.make_name(), dtype=cluster.dtype)
    exprs = _cse(cluster, make,
                 min_cost=options['cse-min-cost'],
                 mode=options['cse-algo'])

    return cluster.rebuild(exprs=exprs)


def _cse(maybe_exprs, make, min_cost=1, mode='default'):
    """
    Main common sub-expressions elimination routine.

    Note: the output is guaranteed to be topologically sorted.

    Parameters
    ----------
    maybe_exprs : expr-like or list of expr-like  or Cluster
        One or more expressions to which CSE is applied.
    make : callable
        Build symbols to store temporary, redundant values.
    mode : str, optional
        The CSE algorithm applied. Accepted: ['default', 'tuplets', 'advanced'].
    """
    assert mode in ('default', 'tuplets', 'advanced')

    # Note: not defaulting to SymPy's CSE() function for three reasons:
    # - it also captures array index access functions
    #   (e.g., i+1 in A[i+1] and B[i+1]);
    # - it sometimes "captures too much", losing factorization opportunities;
    # - very slow

    # Accept Clusters, Eqs or even just exprs
    if isinstance(maybe_exprs, Cluster):
        exprs = list(maybe_exprs.exprs)
        scope = maybe_exprs.scope
    else:
        maybe_exprs = as_list(maybe_exprs)
        if all(e.is_Equality for e in maybe_exprs):
            exprs = maybe_exprs
            scope = Scope(maybe_exprs)
        else:
            exprs = [Eq(make(), e) for e in maybe_exprs]
            scope = Scope([])

    # Some sub-expressions aren't really "common" -- that's the case of Dimension-
    # independent data dependences. For example:
    #
    # ... = ... a[i] + 1 ...
    # a[i] = ...
    # ... = ... a[i] + 1 ...
    #
    # `a[i] + 1` will be excluded, as there's a flow Dimension-independent data
    # dependence involving `a`
    d_flow = {i.source.access for i in scope.d_flow.independent()}
    d_anti = {i.source.access for i in scope.d_anti.independent()}
    exclude = d_flow & d_anti

    if mode in ('default', 'advanced'):
        exprs = _cse_default(exprs, exclude, make, min_cost)
    if mode in ('tuplets', 'advanced'):
        exprs = _cse_tuplets(exprs, exclude, make)

    # Drop useless temporaries (e.g., r0=r1)
    processed = _compact_temporaries(exprs, exclude)

    return processed


def _cse_default(exprs, exclude, make, min_cost):
    """
    The default common sub-expressions elimination algorithm.
    """
    while True:
        # Detect redundancies
        counted = count(exprs).items()
        targets = OrderedDict([(k, estimate_cost(k.expr, True))
                               for k, v in counted if v > 1])

        # Rule out Dimension-independent data dependencies
        targets = OrderedDict([(k, v) for k, v in targets.items()
                               if not k.expr.free_symbols & exclude])
        if not targets or max(targets.values()) < min_cost:
            break

        # Create temporaries
        hit = max(targets.values())
        chosen = [(k, make()) for k, v in targets.items() if v == hit]

        # Apply replacements
        exprs, scheduled = _inject_temporaries(exprs, chosen, exclude)

        # Update `exclude` for the same reasons as above -- to rule out CSE across
        # Dimension-independent data dependences
        exclude.update(scheduled)

    return exprs


def _cse_tuplets(exprs, exclude, make):
    """
    The tuplets-based common sub-expressions elimination algorithm.

    This algo relies on SymPy's canonical ordering of operands. It extracts
    sub-expressions of decreasing size that may or may not be redundant.

    Unlike the default algorithm, this one looks inside the individual operations.
    However, it does so speculatively, as it doesn't attempt to estimate the cost
    of the extracted sub-expressions, which would be an hard problem to solve.

    Another simplification is that we only explore operations whose operands are
    leaves, i.e., symbols or indexed objects.

    Examples
    --------
    Given the expression `a*b*c*d + c*d + a*b*c + a*b*e`, the following
    sub-expressions are extracted: `r0 = a*b, r1 = r0*c`, which leads to the
    following optimized expression: `r1*d + c*d + r1 + r0*e`.
    """
    key = lambda candidate: len(candidate.expr.args)

    while True:
        mapper = defaultdict(list)
        for e in exprs:
            try:
                cond = e.conditionals
            except AttributeError:
                cond = None

            for op in (Add, Mul):
                for i in search(e, op):
                    # The args are in canonical order (thanks to SymPy); let's pick
                    # the largest sub-expression that is not `i` itself
                    args = i.args[:-1]

                    terms = [a for a in args if q_leaf(a)]

                    if len(terms) > 1:
                        mapper[Candidate(op(*terms), cond)].append(i)

        #mapper = {k: v for k, v in mapper.items()
        #          if not k.expr.free_symbols & exclude}

        if not mapper:
            break

        # Create temporaries of decreasing size
        hit = max(mapper, key=key)
        chosen = [(Candidate(i.expr, i.conditionals, sources), make())
                  for i, sources in mapper.items() if key(i) == key(hit)]

        # Apply replacements
        exprs, _ = _inject_temporaries(exprs, chosen, exclude)

    return exprs


def _inject_temporaries(exprs, chosen, exclude):
    """
    Insert temporaries into the expression list such that they appear right
    before the first expression that contains them.
    """
    scheduled = []
    processed = []
    for e in exprs:
        pe = e
        for k, v in chosen:
            if k.conditionals != e.conditionals:
                continue

            if k.sources:
                # Perform compound-based replacement, see uxreplace.__doc__
                args = list(k.expr.args)
                pivot = args.pop(0)
                compound = {pivot: v, **{a: None for a in args}}
                subs = {i: compound for i in k.sources}
            else:
                subs = {k.expr: v}

            pe, changed = _uxreplace(pe, subs)
            if changed and v not in scheduled:
                processed.append(pe.func(v, k.expr, operation=None))
                scheduled.append(v)
        processed.append(pe)

    return processed, scheduled


def _compact_temporaries(exprs, exclude):
    """
    Drop temporaries consisting of isolated symbols.
    """
    # First of all, convert to SSA
    exprs = makeit_ssa(exprs)

    # Drop candidates are all exprs in the form `t0 = s` where `s` is a symbol
    # Note: only CSE-captured Temps, which are by construction local objects, may
    # safely be compacted; a generic Symbol could instead be accessed in a subsequent
    # Cluster, for example: `for (i = ...) { a = b; for (j = a ...) ...`
    mapper = {e.lhs: e.rhs for e in exprs
              if isinstance(e.lhs, CTemp) and q_leaf(e.rhs) and e.lhs not in exclude}

    processed = []
    for e in exprs:
        if e.lhs not in mapper:
            # The temporary is retained, and substitutions may be applied
            expr, changed = e, True
            while changed:
                expr, changed = _uxreplace(expr, mapper)
            processed.append(expr)

    return processed


@singledispatch
def count(expr):
    """
    Construct a mapper `expr -> #occurrences` for each sub-expression in `expr`.
    """
    mapper = Counter()
    for a in expr.args:
        mapper.update(count(a))
    return mapper


@count.register(list)
@count.register(tuple)
def _(exprs):
    mapper = Counter()
    for e in exprs:
        mapper.update(count(e))

    return mapper


@count.register(sympy.Eq)
def _(expr):
    mapper = count(expr.rhs)
    try:
        cond = expr.conditionals
    except AttributeError:
        cond = frozendict()
    return {Candidate(e, cond): v for e, v in mapper.items()}


@count.register(Indexed)
@count.register(Symbol)
def _(expr):
    """
    Handler for objects preventing CSE to propagate through their arguments.
    """
    return Counter()


@count.register(IndexDerivative)
def _(expr):
    """
    Handler for symbol-binding objects. There can be many of them and therefore
    they should be detected as common subexpressions, but it's either pointless
    or forbidden to look inside them.
    """
    return Counter([expr])


@count.register(Add)
@count.register(Mul)
@count.register(Pow)
@count.register(Function)
def _(expr):
    mapper = Counter()
    for a in expr.args:
        mapper.update(count(a))

    mapper[expr] += 1

    return mapper
