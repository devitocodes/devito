from collections import OrderedDict
from functools import cmp_to_key

from devito.ir.iet import (Iteration, HaloSpot, MapNodes, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import (TILABLE, WRAPPABLE, ROUNDABLE, AFFINE,
                               OVERLAPPABLE, hoistable, useless, Forward, Scope)
from devito.tools import as_tuple

__all__ = ['iet_analyze']


class Analysis(object):

    def __init__(self, iet):
        self.iet = iet
        self.properties = OrderedDict()

        self.trees = retrieve_iteration_tree(iet, mode='superset')
        self.scopes = OrderedDict([(k, Scope([i.expr for i in v]))
                                   for k, v in MapNodes().visit(iet).items()])

    def update(self, properties):
        for k, v in properties.items():
            self.properties.setdefault(k, []).append(v)


def propertizer(func):
    def wrapper(arg):
        analysis = Analysis(arg) if not isinstance(arg, Analysis) else arg
        func(analysis)
        return analysis
    return wrapper


def iet_analyze(iet):
    """
    Analyze an Iteration/Expression tree and decorate it with metadata describing
    relevant computational properties (e.g., if an Iteration is parallelizable or not).
    This function performs actual data dependence analysis.
    """
    # Analyze Iterations
    analysis = mark_iteration_wrappable(iet)
    analysis = mark_iteration_roundable(analysis)
    analysis = mark_iteration_affine(analysis)
    analysis = mark_iteration_tilable(analysis)

    # Analyze HaloSpots
    analysis = mark_halospot_useless(analysis)
    analysis = mark_halospot_hoistable(analysis)
    analysis = mark_halospot_overlappable(analysis)

    # Decorate the Iteration/Expression tree with the found properties
    mapper = OrderedDict()
    for k, v in list(analysis.properties.items()):
        args = k.args
        properties = as_tuple(args.pop('properties')) + as_tuple(v)
        mapper[k] = k._rebuild(properties=properties, **args)
    processed = Transformer(mapper, nested=True).visit(iet)

    return processed


@propertizer
def mark_iteration_tilable(analysis):
    """
    Update ``analysis`` detecting the TILABLE Iterations within ``analysis.iet``.
    """
    for tree in analysis.trees:
        for n, i in enumerate(tree):
            # An Iteration is TILABLE only if it's PARALLEL and AFFINE
            if not i.is_Parallel:
                continue
            if AFFINE not in analysis.properties.get(i, []):
                continue

            # In addition, we use the heuristic that we do not consider
            # TILEABLE an Iteration that is not embedded in at least one
            # SEQUENTIAL loop. This is to rule out tiling when the Iteration
            # nest is not likely to be computationally intensive, thus
            # improving code size and readability
            if not any(j.is_Sequential for j in tree[:n]):
                continue

            # Likewise, it won't be marked TILABLE if at least one Iteration
            # is over a local SubDimension
            if any(j.dim.is_Sub and j.dim.local for j in tree):
                continue

            # If it induces dynamic bounds in any of the inner Iterations,
            # then it's ruled out too
            if any(d.is_lex_non_stmt for d in analysis.scopes[i].d_all):
                continue

            analysis.update({i: TILABLE})


@propertizer
def mark_iteration_wrappable(analysis):
    """
    Update ``analysis`` detecting the WRAPPABLE Iterations within ``analysis.iet``.
    """
    for i, scope in analysis.scopes.items():
        if not i.dim.is_Time:
            continue

        accesses = [a for a in scope.accesses if a.function.is_TimeFunction]

        # If not using modulo-buffered iteration, then `i` is surely not WRAPPABLE
        if not accesses or any(not a.function._time_buffering_default for a in accesses):
            continue

        stepping = {a.function.time_dim for a in accesses}
        if len(stepping) > 1:
            # E.g., with ConditionalDimensions we may have `stepping={t, tsub}`
            continue
        stepping = stepping.pop()

        # All accesses must be affine in `stepping`
        if any(not a.affine_if_present(stepping._defines) for a in accesses):
            continue

        # Pick the `back` and `front` slots accessed
        try:
            compareto = cmp_to_key(lambda a0, a1: a0.distance(a1, stepping))
            accesses = sorted(accesses, key=compareto)
            back, front = accesses[0][stepping], accesses[-1][stepping]
        except TypeError:
            continue

        # Check we're not accessing (read, write) always the same slot
        if back == front:
            continue

        accesses_back = [a for a in accesses if a[stepping] == back]

        # There must be NO writes to the `back` timeslot
        if any(a.is_write for a in accesses_back):
            continue

        # There must be NO further accesses to the `back` timeslot after
        # any earlier timeslot is written
        # Note: potentially, this can be relaxed by replacing "any earlier timeslot"
        # with the `front timeslot`
        if not all(all(d.sink is not a or d.source.lex_ge(a) for d in scope.d_flow)
                   for a in accesses_back):
            continue

        analysis.update({i: WRAPPABLE})


@propertizer
def mark_iteration_roundable(analysis):
    """
    Update ``analysis`` detecting the ROUNDABLE Iterations within ``analysis.iet``.
    """
    properties = OrderedDict()
    for tree in analysis.trees:
        innermost = tree.inner
        if not innermost.is_Parallel:
            continue

        # All non-scalar writes must be over Arrays, that is temporaries, otherwise
        # we would end up overwriting user data
        writes = [w for w in analysis.scopes[innermost].writes if w.is_Tensor]
        if any(not w.is_Array for w in writes):
            continue

        # All accessed Functions must have enough room in the PADDING region
        # so that the `innermost` trip count can safely be rounded up
        # Note: autopadding guarantees that the padding size along the
        # Fastest Varying Dimension is a multiple of the SIMD vector length
        functions = [f for f in analysis.scopes[innermost].functions if f.is_Tensor]
        if any(not f._honors_autopadding for f in functions):
            continue

        # Mixed data types (e.g., float and double) is unsupported
        if len({f.dtype for f in functions}) > 1:
            continue

        # The iteration direction must be Forward -- ROUNDABLE is for rounding *up*
        if innermost.direction is not Forward:
            continue

        properties[innermost] = ROUNDABLE

    analysis.update(properties)


@propertizer
def mark_iteration_affine(analysis):
    """
    Update ``analysis`` detecting the AFFINE Iterations within ``analysis.iet``.
    """
    properties = OrderedDict()
    for tree in analysis.trees:
        for i in tree:
            if i in properties:
                continue
            arrays = [a for a in analysis.scopes[i].accesses if not a.is_scalar]
            if all(a.is_regular and a.affine_if_present(i.dim._defines) for a in arrays):
                properties[i] = AFFINE

    analysis.update(properties)


@propertizer
def mark_halospot_useless(analysis):
    """
    Update ``analysis`` detecting the ``useless`` HaloSpots within ``analysis.iet``.
    """
    properties = OrderedDict()

    # If a HaloSpot Dimension turns out to be SEQUENTIAL, then the HaloSpot is useless
    for hs, iterations in MapNodes(HaloSpot, Iteration).visit(analysis.iet).items():
        if any(i.is_Sequential for i in iterations if i.dim.root in hs.dimensions):
            properties[hs] = useless(hs.functions)
            continue

    # If a Function is never written to, or if all HaloSpot reads pertain to an increment
    # expression, then the HaloSpot is useless
    for tree in analysis.trees:
        scope = analysis.scopes[tree.root]

        for hs, v in MapNodes(HaloSpot).visit(tree.root).items():
            if hs in properties:
                continue

            found = []
            for f in hs.fmapper:
                test0 = not scope.writes.get(f)
                test1 = (all(i.is_Expression for i in v) and
                         all(r.is_increment for r in Scope([i.expr for i in v]).reads[f]))
                if test0 or test1:
                    found.append(f)

            if found:
                properties[hs] = useless(tuple(found))

    analysis.update(properties)


@propertizer
def mark_halospot_hoistable(analysis):
    """
    Update ``analysis`` detecting the ``hoistable`` HaloSpots within ``analysis.iet``.
    """
    properties = OrderedDict()
    for i, halo_spots in MapNodes(Iteration, HaloSpot).visit(analysis.iet).items():
        for hs in halo_spots:
            if hs in properties:
                # Already went through this HaloSpot
                continue

            found = []
            scope = analysis.scopes[i]
            for f, hse in hs.fmapper.items():
                # The sufficient condition for `f`'s halo-update to be
                # `hoistable` is that there are no `hs.dimensions`-induced
                # flow-dependences touching the halo
                test = True
                for dep in scope.d_flow.project(f):
                    test = not (dep.cause & set(hs.dimensions))
                    if test:
                        continue

                    test = dep.write.is_increment
                    if test:
                        continue

                    test = all(not any(dep.read.touched_halo(c.root)) for c in dep.cause)
                    if test:
                        continue

                    # `dep` is indeed a flow-dependence touching the halo of distributed
                    # Dimension, so we must assume it's non-hoistable
                    break

                if test:
                    found.append(f)

            if found:
                properties[hs] = hoistable(tuple(found))

    analysis.update(properties)


@propertizer
def mark_halospot_overlappable(analysis):
    """
    Update ``analysis`` detecting the OVERLAPPABLE HaloSpots within ``analysis.iet``.
    """
    properties = OrderedDict()
    for hs, iterations in MapNodes(HaloSpot, Iteration).visit(analysis.iet).items():
        # To be OVERLAPPABLE, all inner Iterations must be PARALLEL
        if all(i.is_Parallel for i in iterations):
            properties[hs] = OVERLAPPABLE

    analysis.update(properties)
