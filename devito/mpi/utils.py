from collections import OrderedDict
from itertools import product

from devito.types import OWNED, HALO, LEFT, RIGHT

__all__ = ['get_views']


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
