from collections import defaultdict, Counter
from functools import cached_property, singledispatch

import numpy as np
import sympy
from sympy import Add, Function, Indexed, Mul, Pow
try:
    from sympy.core.core import ordering_of_classes
except ImportError:
    # Moved in 1.13
    from sympy.core.basic import ordering_of_classes

from devito.finite_differences.differentiable import IndexDerivative
from devito.ir import Cluster, Scope, cluster_pass
from devito.symbolics import estimate_cost, q_leaf, q_terminal
from devito.symbolics.search import search
from devito.symbolics.manipulation import _uxreplace
from devito.tools import DAG, as_list, as_tuple, frozendict, extract_dtype
from devito.types import Eq, Symbol, Temp

__all__ = ['cse']


class CTemp(Temp):

    """
    A cluster-level Temp, similar to Temp, ensured to have different priority
    """

    ordering_of_classes.insert(ordering_of_classes.index('Temp') + 1, 'CTemp')


def retrieve_ctemps(exprs, mode='all'):
    """Shorthand to retrieve the CTemps in `exprs`"""
    return search(exprs, lambda expr: isinstance(expr, CTemp), mode, 'dfs')


@cluster_pass
def cse(cluster, sregistry=None, options=None, **kwargs):
    """
    Perform common sub-expressions elimination (CSE) on a Cluster.

    Three algorithms are available, 'basic', 'smartsort', and 'advanced'.

    The 'basic' algorithm searches for common sub-expressions across the
    operations in a given Cluster. However, it does not look for sub-expressions
    that are subsets of operands of a given n-ary operation. For example, given
    the expression `a*b*c*d + c*d + a*b*c + a*b*e`, it would capture `a*b*c`,
    but not `a*b`.

    The 'smartsort' algorithm is an extension of the 'basic' algorithm. It
    performs a final topological sorting of the expressions to maximize the
    proximity of the common sub-expressions to their uses.

    The 'advanced' algorithm also extracts subsets of operands from a
    given n-ary operation, e.g. `a*b` in `a*b*c*d`. In particular, for a given
    operation `op(a1, a2, ..., an)` it searches for `n-2` additional
    sub-expressions of increasing size, namely `a1*a2`, `a1*a2*a3`, etc.
    This algorithm heuristically relies on SymPy's canonical ordering of operands
    to maximize the likelihood of finding common sub-expressions.
    This algorithm also performs a final topological sorting of the expressions,
    like the 'smartsort' algorithm.

    Parameters
    ----------
    cluster : Cluster
        The input Cluster.
    sregistry : SymbolRegistry
        The symbol registry to use for creating temporaries.
    options : dict
        The optimization options.
        Accepted: ['cse-min-cost', 'cse-algo'].
        * 'cse-min-cost': int. The minimum cost of a common sub-expression to be
          considered for CSE. Default is 1.
        * 'cse-algo': str. The CSE algorithm to apply. Accepted: ['basic',
          'smartsort', 'advanced']. Default is 'basic'.
    """
    min_cost = options['cse-min-cost']
    mode = options['cse-algo']
    try:
        dtype = np.promote_types(options['scalar-min-type'], cluster.dtype).type
    except TypeError:
        dtype = cluster.dtype

    if cluster.is_fence:
        return cluster

    make_dtype = lambda e: np.promote_types(e.dtype, dtype).type
    make = lambda e: CTemp(name=sregistry.make_name(), dtype=make_dtype(e))

    exprs = _cse(cluster, make, min_cost=min_cost, mode=mode)

    return cluster.rebuild(exprs=exprs)


def _cse(maybe_exprs, make, min_cost=1, mode='basic'):
    """
    Carry out the bulk of the CSE process.

    Parameters
    ----------
    maybe_exprs : expr-like or list of expr-like  or Cluster
        One or more expressions to which CSE is applied.
    make : callable
        Build symbols to store temporary, redundant values.
    mode : str, optional
        The CSE algorithm applied. Accepted: ['basic', 'smartsort', 'advanced'].

    Notes
    -----
    We're not using SymPy's CSE for three reasons:

        * It also captures array index access functions (e.g., i+1 in A[i+1]);
        * It sometimes "captures too much", losing factorization opportunities;
        * It tends to be very slow.
    """
    assert mode in ('basic', 'smartsort', 'advanced')

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
            exprs = [Eq(make(e), e) for e in maybe_exprs]
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

    # Perform CSE
    key = lambda c: c.cost

    scheduled = {}
    while True:
        # Detect redundancies
        candidates = catch(exprs, mode)

        # Rule out Dimension-independent data dependencies
        candidates = [c for c in candidates if not c.expr.free_symbols & exclude]

        if not candidates:
            break

        # Start with the largest
        cost = key(max(candidates, key=key))
        if cost < min_cost:
            break
        candidates = [c for c in candidates if c.cost == cost]

        # Apply replacements
        chosen = [(c, scheduled.get(c.key) or make(c)) for c in candidates]
        exprs = _inject(exprs, chosen, scheduled)

    # Drop useless temporaries (e.g., r0=r1)
    processed = _compact(exprs, exclude)

    # Ensure topo-sorting ('basic' doesn't require it)
    if mode in ('smartsort', 'advanced'):
        processed = _toposort(processed)

    return processed


def _inject(exprs, chosen, scheduled):
    """
    Insert temporaries into the expression list.

    The resulting expression list may not be topologically sorted. The caller
    is responsible for ensuring that.
    """
    processed = []
    for e in exprs:
        pe = e
        for k, v in chosen:
            if k.conditionals != e.conditionals:
                continue

            if e.lhs is v:
                # This happens when `k.expr` wasn't substituted in a previous
                # iteration because `k.sources` (whose construction
                # is based on heuristics to avoid a combinatorial explosion)
                # didn't include all of the `k.expr` occurrences across `exprs`,
                # in particular those as part of a middle-term in a n-ary operation
                # (e.g., `b*c` in `a*b*c*d`)
                assert k.expr == e.rhs
                continue

            subs = k.as_subs(v)

            pe, changed = _uxreplace(pe, subs)

            if changed and k.key not in scheduled:
                processed.append(pe.func(v, k.expr, operation=None))
                scheduled[k.key] = v

        processed.append(pe)

    return processed


def _compact(exprs, exclude):
    """
    Drop useless temporaries:

        * Temporaries of the form `t0 = s`, where `s` is a leaf;
        * Temporaries of the form `t0 = expr` such that `t0` is accessed only once.

    Notes
    -----
    Only CSE-captured Temps, namely CTemps, can safely be optimized; a
    generic Symbol could instead be accessed in a subsequent Cluster, e.g.
    `for (i = ...) { a = b; for (j = a ...) ... }`. Hence, this routine
    only targets CTemps.
    """
    candidates = [e for e in exprs
                  if isinstance(e.lhs, CTemp) and e.lhs not in exclude]

    mapper = {e.lhs: e.rhs for e in candidates if q_leaf(e.rhs)}

    # Find all the CTemps in expression right-hand-sides without removing duplicates
    ctemps = retrieve_ctemps(e.rhs for e in exprs)

    # If there are ctemps in the expressions, then add any that only appear once to
    # the mapper
    if ctemps:
        ctemp_count = Counter(ctemps)
        mapper.update({e.lhs: e.rhs for e in candidates
                       if ctemp_count[e.lhs] == 1})

    processed = []
    for e in exprs:
        if e.lhs not in mapper:
            # The temporary is retained, and substitutions may be applied
            expr, changed = e, True
            while changed:
                expr, changed = _uxreplace(expr, mapper)
            processed.append(expr)

    return processed


def _toposort(exprs):
    """
    Ensure the expression list is topologically sorted.
    """
    if not any(isinstance(e.lhs, CTemp) for e in exprs):
        # No CSE temps, no need to topological sort
        return exprs

    dag = DAG(exprs)

    for e0 in exprs:
        if not isinstance(e0.lhs, CTemp):
            continue

        for e1 in exprs:
            if e0.lhs in e1.rhs.free_symbols:
                dag.add_edge(e0, e1, force_add=True)

    def choose_element(queue, scheduled):
        tmps = [i for i in queue if isinstance(i.lhs, CTemp)]
        if tmps:
            # Try to honor temporary names as much as possible
            first = sorted(tmps, key=lambda i: i.lhs.name).pop(0)
            queue.remove(first)
        else:
            first = sorted(queue, key=lambda i: exprs.index(i)).pop(0)
            queue.remove(first)
        return first

    processed = dag.topological_sort(choose_element)

    return processed


class Candidate(tuple):

    def __new__(cls, expr, conditionals=None, sources=()):
        conditionals = frozendict(conditionals or {})
        sources = as_tuple(sources)
        return tuple.__new__(cls, (expr, conditionals, sources))

    @property
    def expr(self):
        return self[0]

    @property
    def dtype(self):
        return extract_dtype(self.expr)

    @property
    def conditionals(self):
        return self[1]

    @property
    def sources(self):
        return self[2]

    @property
    def key(self):
        return (self.expr, self.conditionals)

    @cached_property
    def cost(self):
        if len(self.sources) == 1:
            return 0
        else:
            return estimate_cost(self.expr)

    def as_subs(self, v):
        subs = {self.expr: v}

        # Also add in subs for compound-based replacement
        # E.g., `a*b*c*d` -> `r0*c*d`
        for i in self.sources:
            if self.expr == i:
                continue

            args = [v]
            queue = list(self.expr.args)
            for a in i.args:
                try:
                    queue.remove(a)
                except ValueError:
                    args.append(a)
            assert not queue
            subs[i] = self.expr.func(*args)

        return subs


def catch(exprs, mode):
    """
    Return all common sub-expressions in `exprs` as Candidates.
    """
    mapper = _catch(exprs)

    candidates = []
    for k, v in mapper.items():
        if mode in ('basic', 'smartsort'):
            sources = [i for i in v if i == k.expr]
        else:
            sources = v

        if len(sources) > 1:
            candidates.append(Candidate(k.expr, k.conditionals, sources))

    return candidates


@singledispatch
def _catch(expr):
    """
    Construct a mapper `(expr, cond) -> [occurrences]` for each sub-expression
    in `expr`.

    For example, given `expr = a*b*c`, the output would be:
    `{(a*b*c, None): [a*b*c], (a*b, None): [a*b*c]}`.
    """
    mapper = defaultdict(list)
    for a in expr.args:
        for k, v in _catch(a).items():
            mapper[k].extend(v)
    return mapper


@_catch.register(list)
@_catch.register(tuple)
def _(exprs):
    mapper = defaultdict(list)
    for e in exprs:
        for k, v in _catch(e).items():
            mapper[k].extend(v)
    return mapper


@_catch.register(sympy.Eq)
def _(expr):
    mapper = _catch(expr.rhs)
    try:
        cond = expr.conditionals
    except AttributeError:
        cond = frozendict()
    return {Candidate(c.expr, cond): v for c, v in mapper.items()}


@_catch.register(Indexed)
@_catch.register(Symbol)
def _(expr):
    """
    Handler for objects preventing CSE to propagate through their arguments.
    """
    return {}


@_catch.register(IndexDerivative)
def _(expr):
    """
    Handler for symbol-binding objects. There can be many of them and therefore
    they should be detected as common subexpressions, but it's either pointless
    or forbidden to look inside them.
    """
    return {Candidate(expr): [expr]}


@_catch.register(Add)
@_catch.register(Mul)
def _(expr):
    mapper = _catch(expr.args)

    mapper[Candidate(expr)].append(expr)

    for n in range(2, len(expr.args)):
        terms = expr.args[:n]

        # Heuristic: let the factorizer handle the rest
        terms = [a for a in terms if q_terminal(a)]

        v = expr.func(*terms, evaluate=False)
        mapper[Candidate(v)].append(expr)

    return mapper


@_catch.register(Pow)
@_catch.register(Function)
def _(expr):
    mapper = _catch(expr.args)

    mapper[Candidate(expr)].append(expr)

    return mapper
