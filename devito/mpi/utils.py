from collections import OrderedDict
from itertools import product

from devito.types import OWNED, HALO, LEFT, RIGHT

__all__ = ['get_views', 'derive_halo_updates']


def get_views(f, fixed):
    """
    Return a mapper ``(dimension, side, region) -> (size, offset)`` for a
    :class:`TensorFunction`.
    """
    mapper = OrderedDict()
    for dimension, side, region in product(f.dimensions, [LEFT, RIGHT], [OWNED, HALO]):
        if dimension in fixed:
            continue
        sizes = []
        offsets = []
        for d, i in zip(f.dimensions, f.symbolic_shape):
            if d in fixed:
                sizes.append(1)
                offsets.append(fixed[d])
            elif dimension is d:
                offset, extent = f._get_region(region, dimension, side, True)
                sizes.append(extent)
                offsets.append(offset)
            else:
                sizes.append(i)
                offsets.append(0)
        mapper[(dimension, side, region)] = (sizes, offsets)
    return mapper


def derive_halo_updates(dspace):
    """
    Given a :class:`DataSpace`, return two mappers: ::

        * ``dimension -> [(function, side, amount), ...]}``
        * ``function -> [(dimension, side, amount), ...]

    describing what halo exchanges are required.
    """
    dmapper = {}
    fmapper = {}
    for k, v in dspace.parts.items():
        if k.grid is None:
            continue
        for i in v:
            if i.dim not in k.grid.distributor.dimensions:
                continue
            lsize = k._offset_domain[i.dim].left - i.lower
            if lsize > 0:
                dmapper.setdefault(i.dim, []).append((k, LEFT, lsize))
                fmapper.setdefault(k, []).append((i.dim, LEFT, lsize))
            rsize = i.upper - k._offset_domain[i.dim].right
            if rsize > 0:
                dmapper.setdefault(i.dim, []).append((k, RIGHT, rsize))
                fmapper.setdefault(k, []).append((i.dim, RIGHT, rsize))
    return dmapper, fmapper
