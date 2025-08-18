from collections import defaultdict

from sympy import S
from itertools import combinations

from devito.ir.iet import (Call, Expression, HaloSpot, Iteration, FindNodes,
                           FindWithin, MapNodes, MapHaloSpots, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import PARALLEL, Scope
from devito.mpi.reduction_scheme import DistReduce
from devito.mpi.routines import HaloExchangeBuilder, ReductionBuilder
from devito.passes.iet.engine import iet_pass
from devito.tools import generator

__all__ = ['mpiize']


@iet_pass
def optimize_halospots(iet, **kwargs):
    """
    Optimize the HaloSpots in ``iet``. HaloSpots may be dropped, hoisted,
    merged and moved around in order to improve the halo exchange performance.
    """
    iet = _drop_reduction_halospots(iet)
    iet = _hoist_redundant_from_conditionals(iet)
    iet = _merge_halospots(iet)
    iet = _hoist_invariant(iet)
    iet = _drop_if_unwritten(iet, **kwargs)
    iet = _mark_overlappable(iet)

    return iet, {}


def _drop_reduction_halospots(iet):
    """
    Remove HaloSpots that are used to compute Increments (in which case, a halo
    exchange is actually unnecessary)
    """
    mapper = defaultdict(set)

    # If all HaloSpot reads pertain to reductions, then the HaloSpot is useless
    for hs, expressions in MapNodes(HaloSpot, Expression).visit(iet).items():
        scope = Scope(i.expr for i in expressions)
        for k, v in hs.fmapper.items():
            f = v.bundle or k
            if f not in scope.reads:
                continue
            v = scope.reads[f]
            if all(i.is_reduction for i in v):
                mapper[hs].add(k)

    # Transform the IET introducing the "reduced" HaloSpots
    mapper = {hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(mapper[hs]))
              for hs in FindNodes(HaloSpot).visit(iet)}
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def _hoist_redundant_from_conditionals(iet):
    """
    Hoist redundant HaloSpots from Conditionals. The idea is that, by doing so,
    the subsequent passes (hoist, merge) will be able to optimize them away.

    Examples
    --------

    for time                          for time
      if(cond)                          if(cond)
        haloupd u[t0]                     haloupd u[t0]
        haloupd v[t0]                   haloupd v[t0]
      W v[t1]- R u[t0],v[t0]   --->     W v[t1]- R u[t0],v[t0]
      haloupd v[t0]                     haloupd v[t0]
      R v[t0]                           R v[t0]

    Note that in the above example, the `haloupd v[t0]` in the if branch is
    redundant, as it also appears later on in the Iteration body, so it is
    hoisted out of the Conditional.
    """
    cond_mapper = _make_cond_mapper(iet)
    iter_mapper = _filter_iter_mapper(iet)

    mapper = HaloSpotMapper()
    for it, halo_spots in iter_mapper.items():
        scope = Scope(e.expr for e in FindNodes(Expression).visit(it))

        for hs0 in halo_spots:
            conditions = cond_mapper[hs0]
            if not conditions:
                continue
            condition = conditions[-1]  # Take the innermost Conditional

            for f in hs0.fmapper:
                hsf0 = hs0.halo_scheme.project(f)

                # Find candidate for subsequent merging
                for hs1 in halo_spots:
                    if hs0 is hs1 or cond_mapper[hs1]:
                        continue

                    hsf1 = hs1.halo_scheme.project(f)
                    if not _is_mergeable(hsf1, hsf0, scope) or \
                       not hsf1.issubset(hsf0):
                        continue

                    break
                else:
                    # No candidate found, skip
                    continue

                mapper.drop(hs0, f)
                mapper.add(condition, hsf0)

    iet = mapper.apply(iet)

    return iet


def _merge_halospots(iet):
    """
    Merge HaloSpots on the same IET level when data dependences allow it. This
    has two effects: anticipating communication over computation, and (potentially)
    avoiding redundant halo exchanges.

    Examples
    --------
    In the following example, we have two HaloSpots that both require a halo
    exchange for the same Function `v` at `t0`. Since `v[t0]` is not written
    to in the Iteration nest, we can merge the second HaloSpot into the first
    one, thus avoiding a redundant halo exchange.

    for time                        for time
      haloupd v[t0]                   haloupd v[t0], h
      W v[t1]- R v[t0]      --->      W v[t1]- R v[t0]
      haloupd v[t0], h
      W g[t1]- R v[t0], h             W g[t1]- R v[t0], h
    """
    cond_mapper = _make_cond_mapper(iet)
    iter_mapper = _filter_iter_mapper(iet)

    mapper = HaloSpotMapper()
    for it, halo_spots in iter_mapper.items():
        for hs0, hs1 in combinations(halo_spots, r=2):
            if _check_control_flow(hs0, hs1, cond_mapper):
                continue

            scope = _derive_scope(it, hs0, hs1)

            for f in hs1.fmapper:
                hsf0 = mapper.get(hs0).halo_scheme
                hsf1 = mapper.get(hs1).halo_scheme.project(f)
                if not _is_mergeable(hsf0, hsf1, scope):
                    continue

                # All good -- `hsf1` can be merged within `hs0`
                mapper.add(hs0, hsf1)

                # If the `loc_indices` differ, we rely on hoisting to optimize
                # `hsf1` out of `it`, otherwise we just drop it
                if not _semantical_eq_loc_indices(hsf0, hsf1):
                    continue

                mapper.drop(hs1, f)

    iet = mapper.apply(iet)

    return iet


def _hoist_invariant(iet):
    """
    Hoist iteration-carried HaloSpots out of Iterations.

    Examples
    --------
    There is one typical case in which hoisting is possible, i.e., when a HaloSpot
    is iteration-carried, and it is a subset of another HaloSpot within the same
    Iteration. In this case, we can hoist the former out of the Iteration
    containing the latter, as follows:

                                haloupd v[t0]
    for time                    for time
      haloupd v[t0]
      W v[t1]- R v[t0]   --->     W v[t1]- R v[t0]
      haloupd v[t1]               haloupd v[t1]
      R v[t1]                     R v[t1]
    """
    cond_mapper = _make_cond_mapper(iet)
    iter_mapper = _filter_iter_mapper(iet)

    mapper = HaloSpotMapper()
    for it, halo_spots in iter_mapper.items():
        for hs0, hs1 in combinations(halo_spots, r=2):
            if _check_control_flow(hs0, hs1, cond_mapper):
                continue

            scope = _derive_scope(it, hs0, hs1)

            for f in hs1.fmapper:
                hsf0 = hs0.halo_scheme.project(f)
                if hsf0.is_void:
                    continue
                hsf1 = hs1.halo_scheme.project(f)

                # Ensure there's another HaloScheme that could cover for
                # us should we get hoisted while still satisfying the
                # data dependences
                if hsf1.issubset(hsf0) and _is_iter_carried(hsf1, scope):
                    hs, hsf = hs1, hsf1
                elif hsf0.issubset(hsf1) and hs0 is halo_spots[0]:
                    # Special case
                    hs, hsf = hs0, hsf0
                else:
                    # No hoisting possible, skip
                    continue

                # At this point, we must infer valid loc_indices
                hse = hsf.fmapper[f]
                loc_indices = {}
                for d, v in hse.loc_indices.items():
                    if v in it.uindices:
                        loc_indices[d] = v.symbolic_min.subs(it.dim, it.start)
                    else:
                        loc_indices[d] = v
                hhs = hsf.drop(f).add(f, hse._rebuild(loc_indices=loc_indices))

                mapper.drop(hs, f)
                mapper.add(it, hhs)

    iet = mapper.apply(iet)

    return iet


def _drop_if_unwritten(iet, options=None, **kwargs):
    """
    Drop HaloSpots for unwritten Functions.

    Notes
    -----
    This may be relaxed if Devito were to be used within existing legacy codes,
    which would call the generated library directly.
    """
    drop_unwritten = options['dist-drop-unwritten']
    if not callable(drop_unwritten):
        key = lambda f: drop_unwritten
    else:
        key = drop_unwritten

    # Analysis
    writes = {i.write for i in FindNodes(Expression).visit(iet)}
    mapper = {}
    for hs in FindNodes(HaloSpot).visit(iet):
        for f, v in hs.fmapper.items():
            if not writes.intersection({f, v.bundle}) and key(f):
                mapper[hs] = mapper.get(hs, hs.halo_scheme).drop(f)

    # Post-process analysis
    mapper = {i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
              for i, hs in mapper.items()}

    # Transform the IET dropping the halo exchanges for unwritten Functions
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


class OverlappableHaloSpot(HaloSpot):
    """A HaloSpot allowing computation/communication overlap."""
    pass


def _mark_overlappable(iet):
    """
    Detect the HaloSpots allowing computation/communication overlap and turn
    them into OverlappableHaloSpots.
    """
    # Analysis
    found = []
    for hs in FindNodes(HaloSpot).visit(iet):
        expressions = FindNodes(Expression).visit(hs)
        if not expressions:
            continue

        scope = Scope(i.expr for i in expressions)

        # Comp/comm overlaps is legal only if the OWNED regions can grow
        # arbitrarly, which means all of the dependences must be carried
        # along a non-halo Dimension
        for dep in scope.d_all_gen():
            if dep.function in hs.functions:
                cause = dep.cause & hs.dimensions
                if any(dep.distance_mapper[d] is S.Infinity for d in cause):
                    # E.g., dependences across PARALLEL iterations
                    # for x
                    #   for y
                    #     ... = ... f[x, y-1] ...
                    #   for y
                    #     f[x, y] = ...
                    test = False
                    break
        else:
            test = True

        # Heuristic: avoid comp/comm overlap for sparse Iteration nests
        if test:
            for i in FindNodes(Iteration).visit(hs):
                if i.dim._defines & set(hs.halo_scheme.distributed_aindices) and \
                   not i.is_Affine:
                    test = False
                    break

        if test:
            found.append(hs)

    # Transform the IET replacing HaloSpots with OverlappableHaloSpots
    mapper = {hs: OverlappableHaloSpot(**hs.args) for hs in found}
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


@iet_pass
def make_halo_exchanges(iet, mpimode=None, fallback='basic', **kwargs):
    """
    Lower HaloSpots into halo exchanges for distributed-memory parallelism.
    """
    # To produce unique object names
    generators = {'msg': generator(), 'comm': generator(), 'comp': generator()}

    sync_heb = HaloExchangeBuilder(fallback, generators, **kwargs)
    user_heb = HaloExchangeBuilder(mpimode, generators, **kwargs)
    mapper = {}
    for hs in FindNodes(HaloSpot).visit(iet):
        heb = user_heb if isinstance(hs, OverlappableHaloSpot) else sync_heb
        mapper[hs] = heb.make(hs)

    efuncs = sync_heb.efuncs + user_heb.efuncs
    iet = Transformer(mapper, nested=True).visit(iet)

    # Must drop the PARALLEL tag from the Iterations within which halo
    # exchanges are performed
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        for i in reversed(tree):
            if i in mapper:
                # Already seen this subtree, skip
                break
            if FindNodes(Call).visit(i):
                mapper.update({n: n._rebuild(properties=set(n.properties)-{PARALLEL})
                               for n in tree[:tree.index(i)+1]})
                break
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {'includes': ['mpi.h'], 'efuncs': efuncs}


@iet_pass
def make_reductions(iet, mpimode=None, **kwargs):
    rb = ReductionBuilder()

    mapper = {}
    for e in FindNodes(Expression).visit(iet):
        if not isinstance(e.expr.rhs, DistReduce):
            continue
        elif mpimode:
            mapper[e] = rb.make(e.expr.rhs)
        else:
            mapper[e] = None
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet, {}


def mpiize(graph, **kwargs):
    """
    Perform three IET passes:

        * Optimization of halo exchanges
        * Injection of code for halo exchanges
        * Injection of code for reductions
    """
    options = kwargs['options']

    if options['opt-comms']:
        optimize_halospots(graph, **kwargs)

    mpimode = options['mpi']
    if mpimode:
        make_halo_exchanges(graph, mpimode=mpimode, **kwargs)

    make_reductions(graph, mpimode=mpimode, **kwargs)


# *** Utilities


class HaloSpotMapper(dict):

    def get(self, hs):
        return super().get(hs, hs)

    def drop(self, hs, functions):
        """
        Drop `functions` from the HaloSpot `hs`.
        """
        v = self.get(hs)
        hss = v.halo_scheme.drop(functions)
        self[hs] = hs._rebuild(halo_scheme=hss)

    def add(self, node, hss):
        """
        Add the HaloScheme `hss` to `node`:

            * If `node` is a HaloSpot, then `hss` is added to its
              existing HaloSchemes;
            * Otherwise, a HaloSpot is created wrapping `node`, and `hss`
              is added to it.
        """
        v = self.get(node)
        if isinstance(v, HaloSpot):
            hss = v.halo_scheme.merge(hss)
            hs = v._rebuild(halo_scheme=hss)
        else:
            hs = HaloSpot(v._rebuild(), hss)
        self[node] = hs

    def apply(self, iet):
        """
        Transform `iet` using the HaloSpotMapper.
        """
        mapper = {i: i.body if hs.is_void else hs for i, hs in self.items()}
        iet = Transformer(mapper, nested=True).visit(iet)
        return iet


def _filter_iter_mapper(iet):
    """
    Given an IET, return a mapper from Iterations to the HaloSpots.
    Additionally, filter out Iterations that are not of interest.
    """
    iter_mapper = {}
    for k, v in MapNodes(Iteration, HaloSpot, 'immediate').visit(iet).items():
        filtered_hs = [hs for hs in v if not hs.halo_scheme.is_void]
        if k is not None and len(filtered_hs) > 1:
            iter_mapper[k] = filtered_hs

    return iter_mapper


def _make_cond_mapper(iet):
    """
    Return a mapper from HaloSpots to the Conditionals that contain them.
    """
    mapper = MapHaloSpots().visit(iet)
    return {hs: tuple(i for i in v if i.is_Conditional) for hs, v in mapper.items()}


def _derive_scope(it, hs0, hs1):
    """
    Derive a Scope within the Iteration `it` that starts at the HaloSpot `hs0`
    and ends at the HaloSpot `hs1`.
    """
    expressions = FindWithin(Expression, hs0, stop=hs1).visit(it)
    return Scope(e.expr for e in expressions)


def _check_control_flow(hs0, hs1, cond_mapper):
    """
    If there are Conditionals involved, both `hs0` and `hs1` must be
    within the same Conditional, otherwise we would break control flow
    """
    cond0 = cond_mapper.get(hs0)
    cond1 = cond_mapper.get(hs1)

    return cond0 != cond1


def _is_iter_carried(hsf, scope):
    """
    True if the provided HaloScheme `hsf` is iteration-carried, i.e., it induces
    a halo exchange that requires values from the previous iteration(s); False
    otherwise.
    """

    def rule0(dep):
        # E.g., `dep=W<f,[t1, x]> -> R<f,[t0, x-1]>`, `d=t` => OK
        return not any(dep.distance_mapper[d] is S.Infinity for d in dep.cause)

    def rule1(dep, loc_indices):
        # E.g., `dep=W<f,[t1, x+1]> -> R<f,[t1, xl+1]>`, `loc_indices={t: t0}` => OK
        return any(dep.distance_mapper[d] == 0 and
                   dep.source[d] is not v and
                   dep.sink[d] is not v
                   for d, v in loc_indices.items())

    for f, v in hsf.fmapper.items():
        for dep in scope.d_flow.project(f):
            if not rule0(dep) and not rule1(dep, v.loc_indices):
                return False

    return True


def _is_mergeable(hsf0, hsf1, scope):
    """
    True if `hsf1` can be merged into `hsf0`, i.e., if they are compatible
    and the data dependences would be satisfied, False otherwise.
    """
    # If `hsf1` is empty there's nothing to merge
    if hsf1.is_void:
        return False

    # Ensure `hsf0` and `hsf1` are compatible
    if hsf0.dimensions != hsf1.dimensions or \
       not hsf0.functions & hsf1.functions:
        return False

    # Finally, check the data dependences would be satisfied
    return _is_iter_carried(hsf1, scope)


def _semantical_eq_loc_indices(hsf0, hsf1):
    if hsf0.loc_indices != hsf1.loc_indices:
        return False

    for v0, v1 in zip(hsf0.loc_values, hsf1.loc_values):
        if v0 is v1:
            continue

        # Special case: they might be syntactically different, but semantically
        # equivalent, e.g., `t0` and `t1` with same modulus
        try:
            if v0.modulo == v1.modulo == 1:
                continue
        except AttributeError:
            return False

        return False

    return True
