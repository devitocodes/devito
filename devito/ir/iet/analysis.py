from collections import OrderedDict

from devito.ir.iet import (Iteration, HaloSpot, MapNodes, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import OVERLAPPABLE, hoistable, useless, Scope
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
    analysis = mark_halospot_useless(iet)
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
