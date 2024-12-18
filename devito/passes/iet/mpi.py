from collections import defaultdict

from sympy import S
from itertools import combinations

from devito.ir.iet import (Call, Expression, HaloSpot, Iteration, FindNodes,
                           MapNodes, MapHaloSpots, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import PARALLEL, Scope
from devito.ir.support.guards import GuardFactorEq
from devito.mpi.halo_scheme import HaloScheme
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
    iet = _hoist_invariant(iet)
    iet = _merge_halospots(iet)
    iet = _drop_if_unwritten(iet, **kwargs)
    iet = _mark_overlappable(iet)

    return iet, {}


def _drop_reduction_halospots(iet):
    """
    Remove HaloSpots that are used to compute Increments
    (in which case, a halo exchange is actually unnecessary)
    """
    mapper = defaultdict(set)

    # If all HaloSpot reads pertain to reductions, then the HaloSpot is useless
    for hs, expressions in MapNodes(HaloSpot, Expression).visit(iet).items():
        scope = Scope([i.expr for i in expressions])
        for f, v in scope.reads.items():
            if f in hs.fmapper and all(i.is_reduction for i in v):
                mapper[hs].add(f)

    # Transform the IET introducing the "reduced" HaloSpots
    mapper = {hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(mapper[hs]))
              for hs in FindNodes(HaloSpot).visit(iet)}
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def _hoist_invariant(iet):
    """
    Hoist HaloSpots from inner to outer Iterations where all data dependencies
    would be honored. This pass avoids redundant halo exchanges when the same
    data is redundantly exchanged within the same Iteration tree level.

    Example:
                                   haloupd v[t0]
    for time                       for time
      W v[t1]- R v[t0]               W v[t1]- R v[t0]
      haloupd v[t1]                  haloupd v[t1]
      R v[t1]                        R v[t1]
      haloupd v[t0]                  R v[t0]
      R v[t0]

    """

    # Precompute scopes to save time
    scopes = {i: Scope([e.expr for e in v]) for i, v in MapNodes().visit(iet).items()}

    # Analysis
    hsmapper = {}
    imapper = defaultdict(list)

    cond_mapper = _make_cond_mapper(iet)
    iter_mapper = _filter_iter_mapper(iet)

    for it, halo_spots in iter_mapper.items():
        for hs0, hs1 in combinations(halo_spots, r=2):

            if _check_control_flow(hs0, hs1, cond_mapper):
                continue

            # If there are overlapping loc_indices, skip
            hs0_mdims = hs0.halo_scheme.loc_values
            hs1_mdims = hs1.halo_scheme.loc_values
            if hs0_mdims.intersection(hs1_mdims):
                continue

            for f, v in hs1.fmapper.items():
                if f not in hs0.functions:
                    continue

                for dep in scopes[it].d_flow.project(f):
                    if not any(r(dep, hs1, v.loc_indices) for r in rules):
                        break
                else:
                    # `hs1`` can be hoisted out of `it`, but we need to infer valid
                    # loc_indices
                    hse = hs1.halo_scheme.fmapper[f]
                    loc_indices = {}

                    for d, v in hse.loc_indices.items():
                        if v in it.uindices:
                            loc_indices[d] = v.symbolic_min.subs(it.dim, it.start)
                        else:
                            loc_indices[d] = v

                    hse = hse._rebuild(loc_indices=loc_indices)
                    hs1.halo_scheme.fmapper[f] = hse

                    hsmapper[hs1] = hsmapper.get(hs1, hs1.halo_scheme).drop(f)
                    imapper[it].append(hs1.halo_scheme.project(f))

    mapper = {i: HaloSpot(i._rebuild(), HaloScheme.union(hss))
              for i, hss in imapper.items()}
    mapper.update({i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
                   for i, hs in hsmapper.items()})
    # Transform the IET hoisting/dropping HaloSpots as according to the analysis
    iet = Transformer(mapper, nested=True).visit(iet)
    return iet


def _merge_halospots(iet):
    """
    Merge HaloSpots on the same Iteration tree level where all data dependencies
    would be honored. Avoids redundant halo exchanges when the same data is
    redundantly exchanged within the same Iteration tree level as well as to initiate
    multiple halo exchanges at once.

    Example:

    for time                       for time
      haloupd v[t0]                  haloupd v[t0], h
      W v[t1]- R v[t0]               W v[t1]- R v[t0]
      haloupd v[t0], h
      W g[t1]- R v[t0], h            W g[t1]- R v[t0], h

    """

    # Analysis
    mapper = {}
    cond_mapper = _make_cond_mapper(iet)
    iter_mapper = _filter_iter_mapper(iet)

    for it, halo_spots in iter_mapper.items():
        scope = Scope([e.expr for e in FindNodes(Expression).visit(it)])

        hs0 = halo_spots[0]

        for hs1 in halo_spots[1:]:

            if _check_control_flow(hs0, hs1, cond_mapper):
                continue

            for f, v in hs1.fmapper.items():
                for dep in scope.d_flow.project(f):
                    if not any(r(dep, hs1, v.loc_indices) for r in rules):
                        break
                else:
                    # hs1 is merged with hs0
                    hs = hs1.halo_scheme.project(f)
                    mapper[hs0] = HaloScheme.union([mapper.get(hs0, hs0.halo_scheme), hs])
                    mapper[hs1] = mapper.get(hs1, hs1.halo_scheme).drop(f)

    # Post-process analysis
    mapper = {i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
              for i, hs in mapper.items()}

    # Transform the IET merging/dropping HaloSpots as according to the analysis
    iet = Transformer(mapper, nested=True).visit(iet)

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
        for f in hs.fmapper:
            if f not in writes and key(f):
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

        scope = Scope([i.expr for i in expressions])

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
def make_halo_exchanges(iet, mpimode=None, **kwargs):
    """
    Lower HaloSpots into halo exchanges for distributed-memory parallelism.
    """
    # To produce unique object names
    generators = {'msg': generator(), 'comm': generator(), 'comp': generator()}

    sync_heb = HaloExchangeBuilder('basic', generators, **kwargs)
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
    cond_mapper = {}
    for hs, v in MapHaloSpots().visit(iet).items():
        conditionals = {i for i in v if i.is_Conditional and
                        not isinstance(i.condition, GuardFactorEq)}
        cond_mapper[hs] = conditionals

    return cond_mapper


def _check_control_flow(hs0, hs1, cond_mapper):
    """
    If there are Conditionals involved, both `hs0` and `hs1` must be
    within the same Conditional, otherwise we would break control flow
    """
    cond0 = cond_mapper.get(hs0)
    cond1 = cond_mapper.get(hs1)

    return cond0 != cond1


# Code motion rules -- if the retval is True, then it means the input `dep` is not
# a stopper to moving the HaloSpot `hs` around

def _rule0(dep, hs, loc_indices):
    # E.g., `dep=W<f,[t1, x]> -> R<f,[t0, x-1]>` => True
    return not any(d in hs.dimensions or dep.distance_mapper[d] is S.Infinity
                   for d in dep.cause)


def _rule1(dep, hs, loc_indices):
    # E.g., `dep=W<f,[t1, x+1]> -> R<f,[t1, xl+1]>` and `loc_indices={t: t0}` => True
    return any(dep.distance_mapper[d] == 0 and dep.source[d] is not v
               for d, v in loc_indices.items())


rules = (_rule0, _rule1)
