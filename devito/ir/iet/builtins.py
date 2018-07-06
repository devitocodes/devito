from itertools import product

from sympy import S
import numpy as np

from devito.dimension import DefaultDimension, IncrDimension
from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import (ArrayCast, Call, Callable, Conditional, Expression,
                                 Iteration, List)
from devito.ir.iet.utils import derive_parameters
from devito.types import Array, Scalar, OWNED, HALO
from devito.tools import is_integer

__all__ = ['copy', 'halo_exchange']


def copy(src, fixed):
    """
    Construct a :class:`Callable` copying an arbitrary convex region of ``src``
    into a contiguous :class:`Array`.
    """
    src_indices = []
    dst_indices = []
    dst_shape = []
    dst_dimensions = []
    for d in src.dimensions:
        dst_d = IncrDimension(d, S.Zero, S.One, name='dst_%s' % d)
        dst_dimensions.append(dst_d)
        if d in fixed:
            src_indices.append(fixed[d])
            dst_indices.append(0)
            dst_shape.append(1)
        else:
            src_indices.append(d + Scalar(name='o%s' % d, dtype=np.int32))
            dst_indices.append(dst_d)
            dst_shape.append(dst_d)
    dst = Array(name='dst', shape=dst_shape, dimensions=dst_dimensions)

    # FIXME: somehow the halo/padding shouldn't appear here !!!!! or should they??

    iet = Expression(DummyEq(dst[dst_indices], src[src_indices]))
    for sd, dd, s in reversed(list(zip(src.dimensions, dst.dimensions, dst.shape))):
        if is_integer(s) or sd in fixed:
            continue
        iet = Iteration(iet, sd, s, uindices=dd)
    iet = List(body=[ArrayCast(src), ArrayCast(dst), iet])
    parameters = derive_parameters(iet)
    return Callable('copy', iet, 'void', parameters, ('static',))


def halo_exchange(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    assert f.is_Function

    # Construct send/recv buffers
    buffers = {}
    for d0, side, region in product(f.dimensions, [LEFT, RIGHT], [OWNED, HALO]):
        if d0 in fixed:
            continue
        dimensions = []
        halo = []
        offsets = []
        for d1 in f.dimensions:
            if d1 in fixed:
                dimensions.append(DefaultDimension(name='h%s' % d1, default_value=1))
                halo.append((0, 0))
                offsets.append(fixed[d1])
            elif d0 is d1:
                offset, extent = f._get_region(region, d0, side, True)
                dimensions.append(DefaultDimension(name='h%s' % d1, default_value=extent))
                halo.append((0, 0))
                offsets.append(offset)
            else:
                dimensions.append(d1)
                halo.append(f._extent_halo[d0])
                offsets.append(0)
        array = Array(name='B%s%s' % (d0, side.name[0]), dimensions=dimensions, halo=halo)
        buffers[(d0, side, region)] = (array, offsets)

    # Construct Callable
    for (d, side, region), (array, offsets) in buffers.items():
        mask = Scalar(name='m%s%s' % (d, side.name[0]), dtype=np.int32)
        args = [array.name] + list(array.shape)
        args.append(f.name)
        args.extend([i for i, d in zip(offsets, f.dimensions) if d not in fixed])
        args.extend([d.symbolic_size for d in f.dimensions if d not in fixed])
        # TODO: x_size or x_size + 1
        # ANSWER: PROBABLY X_SIZE + 1 -- WE HAVE TO CHANGE COPY() TO USE src, NOT f, and
        # so to use arbitrary sizes...
        from IPython import embed; embed()
        call = Call('copy', array.name, 1)
        cond = Conditional(mask, 1)
