from collections import OrderedDict
from functools import reduce
from itertools import product
from operator import mul
from ctypes import c_void_p

from devito.dimension import Dimension
from devito.distributed import LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet.nodes import (ArrayCast, Call, Callable, Conditional,
                                 Expression, Iteration, List, PointerCast)
from devito.ir.iet.scheduler import iet_insert_C_decls
from devito.ir.iet.utils import derive_parameters
from devito.symbolics import FieldFromPointer, Macro
from devito.types import Array, Symbol, LocalObject, OWNED, HALO
from devito.tools import numpy_to_mpitypes

__all__ = ['copy', 'sendrecv', 'update_halo']


def copy(f, swap=False):
    """
    Construct a :class:`Callable` capable of copying: ::

        * an arbitrary convex region of ``f`` into a contiguous :class:`Array`, OR
        * if ``swap=True``, a contiguous :class:`Array` into an arbitrary convex
          region of ``f``.
    """
    src_indices, dst_indices = [], []
    src_dimensions, dst_dimensions = [], []
    for d in f.dimensions:
        dst_dimensions.append(Dimension(name='dst_%s' % d.root))
        src_dimensions.append(Dimension(name='src_%s' % d.root))
        src_indices.append(d.root + Symbol(name='o%s' % d.root))
        dst_indices.append(d.root)
    src = Array(name='src', dimensions=src_dimensions, dtype=f.dtype)
    dst = Array(name='dst', dimensions=dst_dimensions, dtype=f.dtype)

    if swap is False:
        eq = DummyEq(dst[dst_indices], src[src_indices])
        name = 'gather'
    else:
        eq = DummyEq(src[src_indices], dst[dst_indices])
        name = 'scatter'

    iet = Expression(eq)
    for d, dd in reversed(list(zip(f.dimensions, dst.dimensions))):
        iet = Iteration(iet, d.root, dd.symbolic_size)
    iet = List(body=[ArrayCast(src), ArrayCast(dst), iet])
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable(name, iet, 'void', parameters, ('static',))


def sendrecv(f):
    """Construct an IET performing a halo exchange along arbitrary
    dimension and side."""
    assert f.is_Function
    assert f.grid is not None

    comm = f.grid.distributor._C_comm

    buf_dimensions = [Dimension(name='buf_%s' % d.root) for d in f.dimensions]
    bufg = Array(name='bufg', dimensions=buf_dimensions, dtype=f.dtype, scope='stack')
    bufs = Array(name='bufs', dimensions=buf_dimensions, dtype=f.dtype, scope='stack')

    dat_dimensions = [Dimension(name='dat_%s' % d.root) for d in f.dimensions]
    dat = Array(name='dat', dimensions=dat_dimensions, dtype=f.dtype,
                scope='external')

    ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
    ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

    params = [bufg] + list(bufg.symbolic_shape) + ofsg + [dat] + list(dat.symbolic_shape)
    gather = Call('gather', params)
    params = [bufs] + list(bufs.symbolic_shape) + ofss + [dat] + list(dat.symbolic_shape)
    scatter = Call('scatter', params)

    fromrank = Symbol(name='fromrank')
    torank = Symbol(name='torank')

    MPI_Request = type('MPI_Request', (c_void_p,), {})
    rrecv = LocalObject(name='rrecv', dtype=MPI_Request)
    rsend = LocalObject(name='rsend', dtype=MPI_Request)

    count = reduce(mul, bufs.symbolic_shape, 1)
    recv = Call('MPI_Irecv', [bufs, count, Macro(numpy_to_mpitypes(f.dtype)),
                              fromrank, Macro('MPI_ANY_TAG'), comm, rrecv])
    send = Call('MPI_Isend', [bufg, count, Macro(numpy_to_mpitypes(f.dtype)),
                              torank, Macro('MPI_ANY_TAG'), comm, rsend])

    waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
    waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

    iet = List(body=[recv, gather, send, waitrecv, waitsend, scatter])
    iet = iet_insert_C_decls(iet)
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable('sendrecv', iet, 'void', parameters, ('static',))


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


def update_halo(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    # Requirements
    assert f.is_Function
    assert f.grid is not None

    distributor = f.grid.distributor
    nb = distributor._C_neighbours.obj
    comm = distributor._C_comm

    mapper = get_views(f, fixed)

    body = []
    for d in f.dimensions:
        if d in fixed:
            continue

        rpeer = FieldFromPointer("%sright" % d, nb.name)
        lpeer = FieldFromPointer("%sleft" % d, nb.name)

        # Sending to left, receiving from right
        lsizes, loffsets = mapper[(d, LEFT, OWNED)]
        rsizes, roffsets = mapper[(d, RIGHT, HALO)]
        assert lsizes == rsizes
        sizes = lsizes
        parameters = ([f] + sizes + [comm] + list(f.symbolic_shape) + [rpeer] +
                      loffsets + roffsets + [lpeer])
        call = Call('sendrecv', parameters)
        body.append(Conditional(Symbol(name='m%sl' % d), call))

        # Sending to right, receiving from left
        rsizes, roffsets = mapper[(d, RIGHT, OWNED)]
        lsizes, loffsets = mapper[(d, LEFT, HALO)]
        assert rsizes == lsizes
        sizes = rsizes
        parameters = ([f] + sizes + [comm] + list(f.symbolic_shape) + [lpeer] +
                      roffsets + loffsets + [rpeer])
        call = Call('sendrecv', parameters)
        body.append(Conditional(Symbol(name='m%sr' % d), call))

    iet = List(body=([PointerCast(comm), PointerCast(nb)] + body))
    parameters = derive_parameters(iet, drop_locals=True)
    return Callable('halo_exchange', iet, 'void', parameters, ('static',))
