from collections import OrderedDict

from devito.ir.iet import (Iteration, HaloSpot, FindNodes, MapNodes, Transformer,
                           retrieve_iteration_tree)
from devito.ir.support import OVERLAPPABLE, Scope
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
    analysis = mark_halospot_overlappable(iet)

    # Decorate the Iteration/Expression tree with the found properties
    mapper = OrderedDict()
    for k, v in list(analysis.properties.items()):
        args = k.args
        properties = as_tuple(args.pop('properties')) + as_tuple(v)
        mapper[k] = k._rebuild(properties=properties, **args)
    processed = Transformer(mapper, nested=True).visit(iet)

    return processed


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
