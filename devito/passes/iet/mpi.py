from collections import defaultdict

from sympy import S

from devito.ir.iet import (Call, Expression, HaloSpot, Iteration, FindNodes,
                           MapNodes, MapHaloSpots, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import PARALLEL, Scope
from devito.mpi.halo_scheme import HaloScheme
from devito.mpi.routines import HaloExchangeBuilder
from devito.passes.iet.engine import iet_pass
from devito.tools import generator

__all__ = ['mpiize']


@iet_pass
def optimize_halospots(iet, **kwargs):
    """
    Optimize the HaloSpots in ``iet``. HaloSpots may be dropped, merged and moved
    around in order to improve the halo exchange performance.
    """
    iet = _drop_halospots(iet)
    iet = _hoist_halospots(iet)
    iet = _merge_halospots(iet)
    iet = _drop_if_unwritten(iet, **kwargs)
    iet = _mark_overlappable(iet)

    return iet, {}


def _drop_halospots(iet):
    """
    Remove HaloSpots that:

        * Would be used to compute Increments (in which case, a halo exchange
          is actually unnecessary)
    """
    mapper = defaultdict(set)

    # If all HaloSpot reads pertain to reductions, then the HaloSpot is useless
    for hs, expressions in MapNodes(HaloSpot, Expression).visit(iet).items():
        scope = Scope([i.expr for i in expressions])
        for f, v in scope.reads.items():
            if f in hs.fmapper and all(i.is_reduction for i in v):
                mapper[hs].add(f)

    # Transform the IET introducing the "reduced" HaloSpots
    subs = {hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(mapper[hs]))
            for hs in FindNodes(HaloSpot).visit(iet)}
    iet = Transformer(subs, nested=True).visit(iet)

    return iet


def _hoist_halospots(iet):
    """
    Hoist HaloSpots from inner to outer Iterations where all data dependencies
    would be honored.
    """

    # Hoisting rules -- if the retval is True, then it means the input `dep` is not
    # a stopper to halo hoisting

    def rule0(dep, candidates, loc_dims):
        # E.g., `dep=W<f,[x]> -> R<f,[x-1]>` and `candidates=({time}, {x})` => False
        # E.g., `dep=W<f,[t1, x, y]> -> R<f,[t0, x-1, y+1]>`, `dep.cause={t,time}` and
        #       `candidates=({x},)` => True
        return (all(i & set(dep.distance_mapper) for i in candidates) and
                not any(i & dep.cause for i in candidates) and
                not any(i & loc_dims for i in candidates))

    def rule1(dep, candidates, loc_dims):
        # A reduction isn't a stopper to hoisting
        return dep.write is not None and dep.write.is_reduction

    rules = [rule0, rule1]

    # Precompute scopes to save time
    scopes = {i: Scope([e.expr for e in v]) for i, v in MapNodes().visit(iet).items()}

    # Analysis
    hsmapper = {}
    imapper = defaultdict(list)
    for iters, halo_spots in MapNodes(Iteration, HaloSpot, 'groupby').visit(iet).items():
        for hs in halo_spots:
            hsmapper[hs] = hs.halo_scheme

            for f, v in hs.fmapper.items():
                loc_dims = frozenset().union([q for d in v.loc_indices
                                              for q in d._defines])

                for n, i in enumerate(iters):
                    if i not in scopes:
                        continue

                    candidates = [i.dim._defines for i in iters[n:]]

                    all_candidates = set().union(*candidates)
                    reads = scopes[i].getreads(f)
                    if any(set(a.ispace.dimensions) & all_candidates
                           for a in reads):
                        continue

                    for dep in scopes[i].d_flow.project(f):
                        if not any(r(dep, candidates, loc_dims) for r in rules):
                            break
                    else:
                        hsmapper[hs] = hsmapper[hs].drop(f)
                        imapper[i].append(hs.halo_scheme.project(f))
                        break

    # Post-process analysis
    mapper = {i: HaloSpot(i._rebuild(), HaloScheme.union(hss))
              for i, hss in imapper.items()}
    mapper.update({i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
                   for i, hs in hsmapper.items()})

    # Transform the IET hoisting/dropping HaloSpots as according to the analysis
    iet = Transformer(mapper, nested=True).visit(iet)

    # Clean up: de-nest HaloSpots if necessary
    mapper = {}
    for hs in FindNodes(HaloSpot).visit(iet):
        if hs.body.is_HaloSpot:
            halo_scheme = HaloScheme.union([hs.halo_scheme, hs.body.halo_scheme])
            mapper[hs] = hs._rebuild(halo_scheme=halo_scheme, body=hs.body.body)
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def _merge_halospots(iet):
    """
    Merge HaloSpots on the same Iteration tree level where all data dependencies
    would be honored.
    """

    # Merge rules -- if the retval is True, then it means the input `dep` is not
    # a stopper to halo merging

    def rule0(dep, hs, loc_indices):
        # E.g., `dep=W<f,[t1, x]> -> R<f,[t0, x-1]>` => True
        return not any(d in hs.dimensions or dep.distance_mapper[d] is S.Infinity
                       for d in dep.cause)

    def rule1(dep, hs, loc_indices):
        # TODO This is apparently never hit, but feeling uncomfortable to remove it
        return (dep.is_regular and
                dep.read is not None and
                all(not any(dep.read.touched_halo(d.root)) for d in dep.cause))

    def rule2(dep, hs, loc_indices):
        # E.g., `dep=W<f,[t1, x+1]> -> R<f,[t1, xl+1]>` and `loc_indices={t: t0}` => True
        return any(dep.distance_mapper[d] == 0 and dep.source[d] is not v
                   for d, v in loc_indices.items())

    rules = [rule0, rule1, rule2]

    # Analysis
    cond_mapper = MapHaloSpots().visit(iet)
    cond_mapper = {hs: {i for i in v if i.is_Conditional}
                   for hs, v in cond_mapper.items()}

    iter_mapper = MapNodes(Iteration, HaloSpot, 'immediate').visit(iet)

    mapper = {}
    for i, halo_spots in iter_mapper.items():
        if i is None or len(halo_spots) <= 1:
            continue

        scope = Scope([e.expr for e in FindNodes(Expression).visit(i)])

        hs0 = halo_spots[0]
        mapper[hs0] = hs0.halo_scheme

        for hs1 in halo_spots[1:]:
            mapper[hs1] = hs1.halo_scheme

            # If there are Conditionals involved, both `hs0` and `hs1` must be
            # within the same Conditional, otherwise we would break the control
            # flow semantics
            if cond_mapper.get(hs0) != cond_mapper.get(hs1):
                continue

            for f, v in hs1.fmapper.items():
                for dep in scope.d_flow.project(f):
                    if not any(r(dep, hs1, v.loc_indices) for r in rules):
                        break
                else:
                    try:
                        hs = hs1.halo_scheme.project(f)
                        mapper[hs0] = HaloScheme.union([mapper[hs0], hs])
                        mapper[hs1] = mapper[hs1].drop(f)
                    except ValueError:
                        # `hs1.loc_indices=<frozendict {t: t1}` and
                        # `hs0.loc_indices=<frozendict {t: t0}`
                        pass

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
def make_mpi(iet, mpimode=None, **kwargs):
    """
    Inject MPI Callables and Calls implementing halo exchanges for
    distributed-memory parallelism.
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


def mpiize(graph, **kwargs):
    """
    Perform two IET passes:

        * Optimization of communications
        * Injection of MPI code

    The former is implemented by manipulating HaloSpots.

    The latter resorts to creating MPI Callables and replacing HaloSpots with Calls
    to MPI Callables.
    """
    options = kwargs['options']

    if options['opt-comms']:
        optimize_halospots(graph, **kwargs)

    mpimode = options['mpi']
    if mpimode:
        make_mpi(graph, mpimode=mpimode, **kwargs)
