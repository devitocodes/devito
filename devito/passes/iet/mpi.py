from collections import defaultdict

from devito.ir.iet import (Call, Expression, HaloSpot, Iteration, List, FindAdjacent,
                           FindNodes, MapNodes, Transformer, retrieve_iteration_tree)
from devito.ir.support import PARALLEL, Scope
from devito.logger import perf_adv
from devito.mpi import HaloExchangeBuilder, HaloScheme
from devito.passes.iet.engine import iet_pass
from devito.tools import filter_sorted, generator

__all__ = ['optimize_halospots', 'mpiize']


@iet_pass
def optimize_halospots(iet):
    """
    Optimize the HaloSpots in ``iet``. HaloSpots may be dropped, merged and moved
    around in order to improve the halo exchange performance.
    """
    iet = _drop_halospots(iet)
    iet = _hoist_halospots(iet)
    iet = _merge_halospots(iet)
    iet = _aggregate_in_halospots(iet)
    iet = _drop_if_unwritten(iet)

    return iet, {}


def _drop_halospots(iet):
    """
    Remove HaloSpots that:

        * Embed SEQUENTIAL Iterations
        * Would be used to compute Increments (in which case, a halo exchange
          is actually unnecessary)
    """
    mapper = defaultdict(set)

    # If a HaloSpot Dimension turns out to be SEQUENTIAL, then the HaloSpot is useless
    for hs, iterations in MapNodes(HaloSpot, Iteration).visit(iet).items():
        if any(i.is_Sequential for i in iterations if i.dim.root in hs.dimensions):
            mapper[hs].update(set(hs.functions))

    # If all HaloSpot reads pertain to increments, then the HaloSpot is useless
    for hs, expressions in MapNodes(HaloSpot, Expression).visit(iet).items():
        for f in hs.fmapper:
            scope = Scope([i.expr for i in expressions])
            if all(i.is_increment for i in scope.reads.get(f, [])):
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
    # Precompute scopes to save time
    scopes = {i: Scope([e.expr for e in v]) for i, v in MapNodes().visit(iet).items()}

    # Analysis
    hsmapper = {}
    imapper = defaultdict(list)
    for iters, halo_spots in MapNodes(Iteration, HaloSpot, 'groupby').visit(iet).items():
        for hs in halo_spots:
            hsmapper[hs] = hs.halo_scheme

            for f in hs.fmapper:
                for n, i in enumerate(iters):
                    maybe_hoistable = set().union(*[i.dim._defines for i in iters[n:]])
                    d_flow = scopes[i].d_flow.project(f)

                    if all(not (dep.cause & maybe_hoistable) or dep.write.is_increment
                           for dep in d_flow):
                        hsmapper[hs] = hsmapper[hs].drop(f)
                        imapper[i].append(hs.halo_scheme.project(f))
                        break

    # Post-process analysis
    mapper = {i: HaloSpot(HaloScheme.union(hss), i._rebuild())
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
    # Analysis
    mapper = {}
    for i, halo_spots in MapNodes(Iteration, HaloSpot, 'immediate').visit(iet).items():
        if i is None or len(halo_spots) <= 1:
            continue

        scope = Scope([e.expr for e in FindNodes(Expression).visit(i)])

        hs0 = halo_spots[0]
        mapper[hs0] = hs0.halo_scheme

        for hs in halo_spots[1:]:
            mapper[hs] = hs.halo_scheme

            for f in hs.fmapper:
                test = True
                for dep in scope.d_flow.project(f):
                    if not (dep.cause & set(hs.dimensions)):
                        continue
                    if all(not any(dep.read.touched_halo(c.root)) for c in dep.cause):
                        continue
                    test = False
                    break
                if test:
                    mapper[hs0] = HaloScheme.union([mapper[hs0],
                                                    hs.halo_scheme.project(f)])
                    mapper[hs] = mapper[hs].drop(f)

    # Post-process analysis
    mapper = {i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
              for i, hs in mapper.items()}

    # Transform the IET merging/dropping HaloSpots as according to the analysis
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


def _aggregate_in_halospots(iet):
    """
    Try to move HaloSpot-free Iteration nests within HaloSpot
    subtrees, to overlap as much computation as possible. The HaloSpot-free
    Iteration nests must be fully affine, otherwise we wouldn't be able to
    honour the data dependences along the halo

    <HaloSpot(u,v)>            HaloSpot(u,v)
      <A>             ---->      <A>
    <B>              affine?     <B>

    Here, <B> doesn't require any halo exchange, but it might still need the
    output of <A>; thus, if we do computation/communication overlap over <A>
    *and* want to embed <B> within the HaloSpot, then <B>'s iteration space
    will have to be split as well. For this, <B> must be affine.
    """
    #TODO: we can probably drop this one now
    # Analysis
    mapper = {}
    for v in FindAdjacent((HaloSpot, Iteration)).visit(iet).values():
        for g in v:
            root = None

            for i in g:
                if i.is_HaloSpot:
                    root = i
                    mapper[root] = [root.body]
                elif root and all(j.is_Affine for j in FindNodes(Iteration).visit(i)):
                    mapper[root].append(i)
                    mapper[i] = None
                else:
                    root = None

    # Post-process analysis
    mapper = {k: k._rebuild(body=List(body=v)) if v else v for k, v in mapper.items()}

    # Transform the IET aggregating Iterations inside HaloSpots
    iet = Transformer(mapper).visit(iet)

    return iet


def _drop_if_unwritten(iet):
    """
    Drop HaloSpots for unwritten Functions.

    Notes
    -----
    This may be relaxed if Devito+MPI were to be used within existing
    legacy codes, which would call the generated library directly.
    """
    # Analysis
    writes = {i.write for i in FindNodes(Expression).visit(iet)}
    mapper = {}
    for hs in FindNodes(HaloSpot).visit(iet):
        for f in hs.fmapper:
            if f not in writes:
                mapper[hs] = mapper.get(hs, hs.halo_scheme).drop(f)

    # Post-process analysis
    mapper = {i: i.body if hs.is_void else i._rebuild(halo_scheme=hs)
              for i, hs in mapper.items()}

    # Transform the IET dropping the halo exchanges for unwritten Functions
    iet = Transformer(mapper, nested=True).visit(iet)

    return iet


@iet_pass
def mpiize(iet, **kwargs):
    """
    Add MPI routines performing halo exchanges to emit distributed-memory
    parallel code.
    """
    mode = kwargs.pop('mode')

    # To produce unique object names
    generators = {'msg': generator(), 'comm': generator(), 'comp': generator()}
    sync_heb = HaloExchangeBuilder('basic', **generators)
    user_heb = HaloExchangeBuilder(mode, **generators)
    mapper = {}
    for hs in FindNodes(HaloSpot).visit(iet):
        # Not all HaloExchangeBuilders are guaranteed to work unless all of the
        # inner Iterations are PARALLEL
        iterations = FindNodes(Iteration).visit(hs)
        heb = user_heb if all(i.is_Parallel for i in iterations) else sync_heb

        mapper[hs] = heb.make(hs)

    efuncs = sync_heb.efuncs + user_heb.efuncs
    objs = filter_sorted(sync_heb.objs + user_heb.objs)
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

    return iet, {'includes': ['mpi.h'], 'efuncs': efuncs, 'args': objs}
