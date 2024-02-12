from collections import defaultdict

from .cluster import Cluster
from devito.tools import as_tuple, flatten
from devito.types import CriticalRegion, Eq, Symbol

__all__ = ['make_critical_sequence', 'bind_critical_regions',
           'in_critical_sequence']


def make_critical_sequence(ispace, sequence, **kwargs):
    """
    Create a critical sequence, i.e. a sequence of Clusters wrapped by
    CriticalRegions.
    """
    sequence = as_tuple(sequence)
    assert len(sequence) >= 1

    processed = []

    # Opening
    expr = Eq(Symbol(name='⋈'), CriticalRegion(True))
    processed.append(Cluster(exprs=expr, ispace=ispace, **kwargs))

    processed.extend(sequence)

    # Closing
    expr = Eq(Symbol(name='⋈'), CriticalRegion(False))
    processed.append(Cluster(exprs=expr, ispace=ispace, **kwargs))

    return processed


def bind_critical_regions(clusters):
    """
    A mapper from CriticalRegions to the critical sequences they open.
    """
    critical_region = False
    mapper = defaultdict(list)
    for c in clusters:
        if c.is_critical_region:
            critical_region = not critical_region and c
        elif critical_region:
            mapper[critical_region].append(c)
    return mapper


def in_critical_sequence(cluster, clusters):
    """
    True if `cluster` is part of a critical sequence, False otherwise.
    """
    mapper = bind_critical_regions(clusters)
    return cluster in flatten(mapper.values())
