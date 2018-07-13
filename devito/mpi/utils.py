from collections import OrderedDict
from itertools import product

from devito.ir.support import Forward
from devito.types import OWNED, HALO, LEFT, RIGHT

__all__ = ['get_views', 'derive_halo_scheme']


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


def derive_halo_scheme(dspace, directions):
    """
    Given a :class:`DataSpace`, return three mappers: ::

        * ``dimension -> [(function, side, amount), ...]}``
        * ``function -> [(dimension, side, amount), ...]
        * ``dimension -> dimension``.

    The first two mappers describe what halo exchanges are required. The last
    mapper tells how to access dimensions that need no halo exchange.
    """
    dmapper = {}
    fmapper = {}
    fixed = {}
    for f, v in dspace.parts.items():
        if not f.is_TensorFunction or f.grid is None:
            continue
        for d in f.dimensions:
            r = d.root
            if v[r].is_Null:
                continue
            elif d in f.grid.distributor.dimensions:
                lsize = f._offset_domain[d].left - v[r].lower
                if lsize > 0:
                    dmapper.setdefault(d, []).append((f, LEFT, lsize))
                    fmapper.setdefault(f, []).append((d, LEFT, lsize))
                rsize = v[r].upper - f._offset_domain[d].right
                if rsize > 0:
                    dmapper.setdefault(d, []).append((f, RIGHT, rsize))
                    fmapper.setdefault(f, []).append((d, RIGHT, rsize))
            elif r in directions:
                last = (dspace[r].upper - 1) if v[r] is Forward else (dspace[r].lower + 1)
                fixed[d] = d + last
    return dmapper, fmapper, fixed
