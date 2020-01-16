from collections import OrderedDict

from sympy import Add, Mul, collect, collect_const

from devito.ir import DummyEq, Cluster, Scope
from devito.symbolics import (count, estimate_cost, q_xop, q_leaf, retrieve_scalars,
                              retrieve_terminals, yreplace)
from devito.tools import ReducerMap
from devito.types import Dimension, Symbol

__all__ = ['collect_nested', 'common_subexprs_elimination', 'make_is_time_invariant']


def collect_nested(expr):
    """
    Collect numeric coefficients, trascendental functions, and symbolic powers,
    across all levels of the expression tree.

    The collection gives precedence to (in order of importance):

        1) Trascendental functions,
        2) Symbolic powers,
        3) Numeric coefficients.

    Parameters
    ----------
    expr : expr-like
        The expression to be factorized.
    """

    def run(expr):
        # Return semantic (rebuilt expression, factorization candidates)

        if expr.is_Number:
            return expr, {'coeffs': expr}
        elif expr.is_Function:
            return expr, {'funcs': expr}
        elif expr.is_Pow:
            return expr, {'pows': expr}
        elif expr.is_Symbol or expr.is_Indexed or expr.is_Atom:
            return expr, {}
        elif expr.is_Add:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            candidates = ReducerMap.fromdicts(*candidates)

            funcs = candidates.getall('funcs', [])
            pows = candidates.getall('pows', [])
            coeffs = candidates.getall('coeffs', [])

            # Functions/Pows are collected first, coefficients afterwards
            # Note: below we use sets, but SymPy will ensure determinism
            args = set(args)
            w_funcs = {i for i in args if any(j in funcs for j in i.args)}
            args -= w_funcs
            w_pows = {i for i in args if any(j in pows for j in i.args)}
            args -= w_pows
            w_coeffs = {i for i in args if any(j in coeffs for j in i.args)}
            args -= w_coeffs

            # Collect common funcs
            w_funcs = collect(expr.func(*w_funcs), funcs, evaluate=False)
            try:
                w_funcs = Add(*[Mul(k, collect_const(v)) for k, v in w_funcs.items()])
            except AttributeError:
                assert w_funcs == 0

            # Collect common pows
            w_pows = collect(expr.func(*w_pows), pows, evaluate=False)
            try:
                w_pows = Add(*[Mul(k, collect_const(v)) for k, v in w_pows.items()])
            except AttributeError:
                assert w_pows == 0

            # Collect common temporaries (r0, r1, ...)
            w_coeffs = collect(expr.func(*w_coeffs), tuple(retrieve_scalars(expr)),
                               evaluate=False)
            try:
                w_coeffs = Add(*[Mul(k, collect_const(v)) for k, v in w_coeffs.items()])
            except AttributeError:
                assert w_coeffs == 0

            # Collect common coefficients
            w_coeffs = collect_const(w_coeffs)

            rebuilt = Add(w_funcs, w_pows, w_coeffs, *args)

            return rebuilt, {}
        elif expr.is_Mul:
            args, candidates = zip(*[run(arg) for arg in expr.args])

            # Always collect coefficients
            rebuilt = collect_const(expr.func(*args))
            try:
                if rebuilt.args:
                    # Note: Mul(*()) -> 1, and since sympy.S.Zero.args == (),
                    # the `if` prevents turning 0 into 1
                    rebuilt = Mul(*rebuilt.args)
            except AttributeError:
                pass

            return rebuilt, ReducerMap.fromdicts(*candidates)
        elif expr.is_Equality:
            args, candidates = zip(*[run(expr.lhs), run(expr.rhs)])
            return expr.func(*args, evaluate=False), ReducerMap.fromdicts(*candidates)
        else:
            args, candidates = zip(*[run(arg) for arg in expr.args])
            return expr.func(*args), ReducerMap.fromdicts(*candidates)

    return run(expr)[0]


def make_is_time_invariant(context):
    """
    Given an ordered list of expressions, returns a callable that finds out whether
    a given expression is time invariant or not.
    """
    mapper = OrderedDict([(i.lhs, i) for i in _makeit_ssa(context)])

    def is_time_invariant(mapper, expr):
        if any(isinstance(i, Dimension) and i.is_Time for i in expr.free_symbols):
            return False

        queue = [expr.rhs if expr.is_Equality else expr]
        seen = set()
        while queue:
            item = queue.pop()
            nodes = set()
            for i in retrieve_terminals(item):
                if i in seen:
                    # Already inspected, nothing more can be inferred
                    continue
                elif any(isinstance(j, Dimension) and j.is_Time for j in i.free_symbols):
                    # Definitely not time-invariant
                    return False
                elif i in mapper:
                    # Go on with the search
                    nodes.add(i)
                elif isinstance(i, Dimension):
                    # Go on with the search, as `i` is not a time dimension
                    pass
                elif not i.function.is_DiscreteFunction:
                    # It didn't come from the outside and it's not in `mapper`, so
                    # cannot determine if time-invariant; assume time-varying then
                    return False
                seen.add(i)
            queue.extend([mapper[i].rhs for i in nodes])
        return True

    callback = lambda i: is_time_invariant(mapper, i)

    return callback


def common_subexprs_elimination(maybe_exprs, make, mode='default'):
    """
    Perform common sub-expressions elimination, or CSE.

    Note: the output is guaranteed to be topologically sorted.

    Parameters
    ----------
    maybe_exprs : expr-like or list of expr-like or Cluster
        One or more expressions to which CSE is applied.
    make : callable
        Build symbols to store temporary, redundant values.
    mode : str, optional
        The CSE algorithm applied. Accepted: ['default'].
    """

    # Note: not defaulting to SymPy's CSE() function for three reasons:
    # - it also captures array index access functions (eg, i+1 in A[i+1] and B[i+1]);
    # - it sometimes "captures too much", losing factorization opportunities;
    # - very slow
    # TODO: a second "sympy" mode will be provided, relying on SymPy's CSE() but
    # also ensuring some sort of post-processing
    assert mode == 'default'  # Only supported mode ATM

    # Just for flexibility, accept either Clusters or exprs
    if isinstance(maybe_exprs, Cluster):
        cluster = maybe_exprs
        processed = list(cluster.exprs)
        scope = cluster.scope
    else:
        processed = list(maybe_exprs)
        scope = Scope(maybe_exprs)

    # Some sub-expressions aren't really "common" -- that's the case of Dimension-
    # independent data dependences. For example:
    #
    # ... = ... a[i] + 1 ...
    # a[i] = ...
    # ... = ... a[i] + 1 ...
    #
    # `a[i] + 1` will be excluded, as there's a flow Dimension-independent data
    # dependence involving `a`
    exclude = {i.source.indexed for i in scope.d_flow.independent()}

    mapped = []
    while True:
        # Detect redundancies
        counted = count(mapped + processed, q_xop).items()
        targets = OrderedDict([(k, estimate_cost(k, True)) for k, v in counted if v > 1])

        # Rule out Dimension-independent data dependencies
        targets = OrderedDict([(k, v) for k, v in targets.items()
                               if not k.free_symbols & exclude])

        if not targets:
            break

        # Create temporaries
        hit = max(targets.values())
        picked = [k for k, v in targets.items() if v == hit]
        mapper = OrderedDict([(e, make()) for i, e in enumerate(picked)])

        # Apply replacements
        processed = [e.xreplace(mapper) for e in processed]
        mapped = [e.xreplace(mapper) for e in mapped]
        mapped = [DummyEq(v, k) for k, v in reversed(list(mapper.items()))] + mapped

        # Prepare for the next round
        for k in picked:
            targets.pop(k)
    processed = mapped + processed

    # At this point we may have useless temporaries (e.g., r0=r1). Let's drop them
    processed = _compact_temporaries(processed)

    return processed


# Private functions


def _makeit_ssa(exprs):
    """
    Convert an iterable of Eqs into Static Single Assignment (SSA) form.
    """
    # Identify recurring LHSs
    seen = {}
    for i, e in enumerate(exprs):
        seen.setdefault(e.lhs, []).append(i)
    # Optimization: don't waste time reconstructing stuff if already in SSA form
    if all(len(i) == 1 for i in seen.values()):
        return exprs
    # SSA conversion
    c = 0
    mapper = {}
    processed = []
    for i, e in enumerate(exprs):
        where = seen[e.lhs]
        rhs = e.rhs.xreplace(mapper)
        if len(where) > 1:
            needssa = e.is_Scalar or where[-1] != i
            lhs = Symbol(name='ssa%d' % c, dtype=e.dtype) if needssa else e.lhs
            if e.is_Increment:
                # Turn AugmentedAssignment into Assignment
                processed.append(e.func(lhs, mapper[e.lhs] + rhs, is_Increment=False))
            else:
                processed.append(e.func(lhs, rhs))
            mapper[e.lhs] = lhs
            c += 1
        else:
            processed.append(e.func(e.lhs, rhs))
    return processed


def _compact_temporaries(exprs):
    """
    Drop temporaries consisting of isolated symbols.
    """
    # First of all, convert to SSA
    exprs = _makeit_ssa(exprs)

    # What's gonna be dropped
    mapper = {e.lhs: e.rhs for e in exprs
              if e.lhs.is_Symbol and (q_leaf(e.rhs) or e.rhs.is_Function)}

    processed = []
    for e in exprs:
        if e.lhs not in mapper:
            # The temporary is retained, and substitutions may be applied
            handle, _ = yreplace(e, mapper, repeat=True)
            assert len(handle) == 1
            processed.extend(handle)

    return processed
