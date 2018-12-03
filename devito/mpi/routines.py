from functools import reduce
from operator import mul
from ctypes import c_void_p
from itertools import product

from devito.data import OWNED, HALO, NOPAD, LEFT, RIGHT
from devito.dimension import Dimension
from devito.ir.equations import DummyEq
from devito.ir.iet import (ArrayCastSymbolic, Call, Callable, Conditional, Expression,
                           Iteration, List, iet_insert_C_decls)
from devito.symbolics import CondNe, FieldFromPointer, Macro
from devito.types import Array, Symbol, LocalObject
from devito.tools import dtype_to_mpitype

__all__ = ['copy', 'sendrecv', 'update_halo']


def copy(f, fixed, swap=False):
    """
    Construct a :class:`Callable` capable of copying: ::

        * an arbitrary convex region of ``f`` into a contiguous :class:`Array`, OR
        * if ``swap=True``, a contiguous :class:`Array` into an arbitrary convex
          region of ``f``.
    """
    buf_dims = []
    buf_indices = []
    for d in f.dimensions:
        if d not in fixed:
            buf_dims.append(Dimension(name='buf_%s' % d.root))
            buf_indices.append(d.root)
    buf = Array(name='buf', dimensions=buf_dims, dtype=f.dtype)

    f_offsets = []
    f_indices = []
    for d in f.dimensions:
        offset = Symbol(name='o%s' % d.root)
        f_offsets.append(offset)
        f_indices.append(offset + (d.root if d not in fixed else 0))

    # Use `dummy_f`, instead of the actual `f`, so that we don't regenerate
    # code for Functions that only differ from `f` in the name
    dummy_f = f.__class__.__base__(name='f', grid=f.grid, shape=f.shape_global,
                                   dimensions=f.dimensions)

    if swap is False:
        eq = DummyEq(buf[buf_indices], dummy_f[f_indices])
        name = 'gather_%s' % f.name
    else:
        eq = DummyEq(dummy_f[f_indices], buf[buf_indices])
        name = 'scatter_%s' % f.name

    iet = Expression(eq)
    for i, d in reversed(list(zip(buf_indices, buf_dims))):
        iet = Iteration(iet, i, d.symbolic_size - 1)  # -1 as Iteration generates <=
    iet = List(body=[ArrayCastSymbolic(dummy_f), ArrayCastSymbolic(buf), iet])
    parameters = [buf] + list(buf.shape) + [dummy_f] + f_offsets
    return Callable(name, iet, 'void', parameters, ('static',))


def sendrecv(f, fixed):
    """Construct an IET performing a halo exchange along arbitrary
    dimension and side."""
    assert f.is_Function
    assert f.grid is not None

    comm = f.grid.distributor._obj_comm

    buf_dims = [Dimension(name='buf_%s' % d.root) for d in f.dimensions if d not in fixed]
    bufg = Array(name='bufg', dimensions=buf_dims, dtype=f.dtype, scope='heap')
    bufs = Array(name='bufs', dimensions=buf_dims, dtype=f.dtype, scope='heap')

    # Use `dummy_f`, instead of the actual `f`, so that we don't regenerate
    # code for Functions that only differ from `f` in the name
    dummy_f = f.__class__.__base__(name='f', grid=f.grid, shape=f.shape_global,
                                   dimensions=f.dimensions)

    ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
    ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

    fromrank = Symbol(name='fromrank')
    torank = Symbol(name='torank')

    parameters = [bufg] + list(bufg.shape) + [dummy_f] + ofsg
    gather = Call('gather_%s' % f.name, parameters)
    parameters = [bufs] + list(bufs.shape) + [dummy_f] + ofss
    scatter = Call('scatter_%s' % f.name, parameters)

    # The scatter must be guarded as we must not alter the halo values along
    # the domain boundary, where the sender is actually MPI.PROC_NULL
    scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

    srecv = MPIStatusObject(name='srecv')
    rrecv = MPIRequestObject(name='rrecv')
    rsend = MPIRequestObject(name='rsend')

    count = reduce(mul, bufs.shape, 1)
    recv = Call('MPI_Irecv', [bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                              fromrank, '13', comm, rrecv])
    send = Call('MPI_Isend', [bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                              torank, '13', comm, rsend])

    waitrecv = Call('MPI_Wait', [rrecv, srecv])
    waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

    iet = List(body=[recv, gather, send, waitsend, waitrecv, scatter])
    iet = List(body=iet_insert_C_decls(iet))
    parameters = [dummy_f] + list(bufs.shape) + ofsg + ofss + [fromrank, torank, comm]
    return Callable('sendrecv_%s' % f.name, iet, 'void', parameters, ('static',))


def update_halo(f, fixed):
    """
    Construct an IET performing a halo exchange for a :class:`TensorFunction`.
    """
    # Requirements
    assert f.is_Function
    assert f.grid is not None

    distributor = f.grid.distributor
    nb = distributor._obj_neighbours
    comm = distributor._obj_comm

    fixed = {d: Symbol(name="o%s" % d.root) for d in fixed}

    # Build a mapper `(dim, side, region) -> (size, ofs)` for `f`. `size` and
    # `ofs` are symbolic objects. This mapper tells what data values should be
    # sent (OWNED) or received (HALO) given dimension and side
    mapper = {}
    for d0, side, region in product(f.dimensions, (LEFT, RIGHT), (OWNED, HALO)):
        if d0 in fixed:
            continue
        sizes = []
        offsets = []
        for d1 in f.dimensions:
            if d1 in fixed:
                offsets.append(fixed[d1])
            else:
                meta = f._C_get_field(region if d0 is d1 else NOPAD, d1, side)
                offsets.append(meta.offset)
                sizes.append(meta.extent)
        mapper[(d0, side, region)] = (sizes, offsets)

    body = []
    masks = []
    for d in f.dimensions:
        if d in fixed:
            continue

        rpeer = FieldFromPointer("%sright" % d, nb)
        lpeer = FieldFromPointer("%sleft" % d, nb)

        # Sending to left, receiving from right
        lsizes, loffsets = mapper[(d, LEFT, OWNED)]
        rsizes, roffsets = mapper[(d, RIGHT, HALO)]
        sizes = lsizes
        parameters = ([f] + sizes + loffsets + roffsets + [rpeer, lpeer, comm])
        call = Call('sendrecv_%s' % f.name, parameters)
        mask = Symbol(name='m%sl' % d)
        body.append(Conditional(mask, call))
        masks.append(mask)

        # Sending to right, receiving from left
        rsizes, roffsets = mapper[(d, RIGHT, OWNED)]
        lsizes, loffsets = mapper[(d, LEFT, HALO)]
        sizes = rsizes
        parameters = ([f] + sizes + roffsets + loffsets + [lpeer, rpeer, comm])
        call = Call('sendrecv_%s' % f.name, parameters)
        mask = Symbol(name='m%sr' % d)
        body.append(Conditional(mask, call))
        masks.append(mask)

    iet = List(body=body)
    parameters = [f] + masks + [comm, nb] + list(fixed.values())
    return Callable('halo_exchange_%s' % f.name, iet, 'void', parameters, ('static',))


class MPIStatusObject(LocalObject):

    dtype = type('MPI_Status', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']


class MPIRequestObject(LocalObject):

    dtype = type('MPI_Request', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']
