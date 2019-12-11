from devito.ir.iet import (Call, HaloSpot, Iteration, List, PARALLEL, FindAdjacent,
                           FindNodes, MapNodes, Transformer, retrieve_iteration_tree)
from devito.logger import perf_adv
from devito.mpi import HaloExchangeBuilder, HaloScheme
from devito.targets.common.engine import target_pass
from devito.tools import generator

__all__ = ['optimize_halospots', 'mpiize']


@target_pass
def optimize_halospots(iet):
    """
    Optimize the HaloSpots in ``iet``.

    * Remove all ``useless`` HaloSpots;
    * Merge all ``hoistable`` HaloSpots with their root HaloSpot, thus
      removing redundant communications and anticipating communications
      that will be required by later Iterations.
    """
    # Drop `useless` HaloSpots
    mapper = {hs: hs._rebuild(halo_scheme=hs.halo_scheme.drop(hs.useless))
              for hs in FindNodes(HaloSpot).visit(iet)}
    iet = Transformer(mapper, nested=True).visit(iet)

    # Handle `hoistable` HaloSpots
    # First, we merge `hoistable` HaloSpots together, to anticipate communications
    mapper = {}
    for tree in retrieve_iteration_tree(iet):
        halo_spots = FindNodes(HaloSpot).visit(tree.root)
        if not halo_spots:
            continue
        root = halo_spots[0]
        if root in mapper:
            continue
        hss = [root.halo_scheme]
        hss.extend([hs.halo_scheme.project(hs.hoistable) for hs in halo_spots[1:]])
        try:
            mapper[root] = root._rebuild(halo_scheme=HaloScheme.union(hss))
        except ValueError:
            # HaloSpots have non-matching `loc_indices` and therefore can't be merged
            perf_adv("Found hoistable HaloSpots with disjoint loc_indices, "
                     "skipping optimization")
            continue
        for hs in halo_spots[1:]:
            halo_scheme = hs.halo_scheme.drop(hs.hoistable)
            if halo_scheme.is_void:
                mapper[hs] = hs.body
            else:
                mapper[hs] = hs._rebuild(halo_scheme=halo_scheme)
    iet = Transformer(mapper, nested=True).visit(iet)

    # Then, we make sure the halo exchanges get performed *before*
    # the first distributed Dimension. Again, we do this to anticipate
    # communications, which hopefully has a pay off in performance
    #
    # <Iteration x>                    <HaloSpot(u)>, in y
    #   <HaloSpot(u)>, in y    ---->   <Iteration x>
    #   <Iteration y>                    <Iteration y>
    mapper = {}
    for i, halo_spots in MapNodes(Iteration, HaloSpot).visit(iet).items():
        hoistable = [hs for hs in halo_spots if hs.hoistable]
        if not hoistable:
            continue
        elif len(hoistable) > 1:
            # We should never end up here, but for now we can't prove it formally
            perf_adv("Found multiple hoistable HaloSpots, skipping optimization")
            continue
        hs = hoistable.pop()
        if hs in mapper:
            continue
        if i.dim.root in hs.dimensions:
            halo_scheme = hs.halo_scheme.drop(hs.hoistable)
            if halo_scheme.is_void:
                mapper[hs] = hs.body
            else:
                mapper[hs] = hs._rebuild(halo_scheme=halo_scheme)

            halo_scheme = hs.halo_scheme.project(hs.hoistable)
            mapper[i] = hs._rebuild(halo_scheme=halo_scheme, body=i._rebuild())
    iet = Transformer(mapper, nested=True).visit(iet)

    # Finally, we try to move HaloSpot-free Iteration nests within HaloSpot
    # subtrees, to overlap as much computation as possible. The HaloSpot-free
    # Iteration nests must be fully affine, otherwise we wouldn't be able to
    # honour the data dependences along the halo
    #
    # <HaloSpot(u,v)>            HaloSpot(u,v)
    #   <A>             ---->      <A>
    # <B>              affine?     <B>
    #
    # Here, <B> doesn't require any halo exchange, but it might still need the
    # output of <A>; thus, if we do computation/communication overlap over <A>
    # *and* want to embed <B> within the HaloSpot, then <B>'s iteration space
    # will have to be split as well. For this, <B> must be affine.
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
    mapper = {k: k._rebuild(body=List(body=v)) if v else v for k, v in mapper.items()}
    iet = Transformer(mapper).visit(iet)

    return iet, {}


@target_pass
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
        heb = user_heb if hs.is_Overlappable else sync_heb
        mapper[hs] = heb.make(hs)
    efuncs = sync_heb.efuncs + user_heb.efuncs
    objs = sync_heb.objs + user_heb.objs
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
