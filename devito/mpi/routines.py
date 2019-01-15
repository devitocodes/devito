import abc
from collections import OrderedDict
from ctypes import c_void_p
from functools import reduce
from itertools import product
from operator import mul

from sympy import Integer

from devito.data import OWNED, HALO, NOPAD, LEFT, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet import (ArrayCast, Call, Callable, Conditional, Expression,
                           Iteration, List, iet_insert_C_decls, PARALLEL)
from devito.symbolics import CondNe, FieldFromPointer, Macro
from devito.tools import dtype_to_mpitype
from devito.types import Array, Dimension, Symbol, LocalObject

__all__ = ['HaloExchangeBuilder']


class HaloExchangeBuilder(object):

    """
    Build IET-based routines to implement MPI halo exchange.
    """

    def __new__(cls, threaded):
        obj = object.__new__(BasicHaloExchangeBuilder)
        obj.__init__(threaded)
        return obj

    def __init__(self, threaded):
        self._threaded = threaded

    @abc.abstractmethod
    def make(self, halo_spots):
        """
        Construct Callables and Calls implementing a halo exchange for the
        provided HaloSpots.

        For each (unique) HaloSpot, three Callables are built:

            * ``update_halo``, to be called when a halo exchange is necessary,
            * ``sendrecv``, called multiple times by ``update_halo``.
            * ``copy``, called twice by ``sendrecv``, to implement, for example,
              data gathering prior to an MPI_Send, and data scattering following
              an MPI recv.
        """
        callables = []
        calls = OrderedDict()
        mapper = {}
        for hs in halo_spots:
            for f, v in hs.fmapper.items():
                # Sanity check
                assert f.is_Function
                assert f.grid is not None

                # Have we already generated code for the same exact type of
                # halo exchange (i.e., same Function, same mask, etc.) ?
                key = (f, v)
                if key in mapper:
                    calls.setdefault(hs, []).append(mapper[key])
                    continue

                # Callables construction
                progressid = len(mapper)
                gather, extra = self._make_copy(f, v.loc_indices, progressid)
                scatter, _ = self._make_copy(f, v.loc_indices, progressid, swap=True)
                sendrecv = self._make_sendrecv(f, v.loc_indices, progressid, extra)
                haloupdate = self._make_haloupdate(f, v.loc_indices, hs.mask[f],
                                                   progressid, extra)
                callables.extend([gather, scatter, sendrecv, haloupdate])

                # `haloupdate` Call construction
                comm = f.grid.distributor._obj_comm
                nb = f.grid.distributor._obj_neighborhood
                loc_indices = list(v.loc_indices.values())
                args = [f, comm, nb] + loc_indices + extra
                call = Call('haloupdate%d' % progressid, args)
                calls.setdefault(hs, []).append(call)

                # Track the newly built halo exchange pattern
                mapper[key] = call

        return callables, calls

    @abc.abstractmethod
    def _make_haloupdate(self, f, fixed, halos, progressid, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange.
        """
        return

    @abc.abstractmethod
    def _make_sendrecv(self, f, fixed, progressid, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange
        along given Dimension and DataSide.
        """
        return

    def _make_copy(self, f, fixed, progressid, swap=False):
        """
        Construct a Callable performing a copy of:

            * an arbitrary convex region of ``f`` into a contiguous Array, OR
            * if ``swap=True``, a contiguous Array into an arbitrary convex
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
            name = 'gather%d' % progressid
        else:
            eq = DummyEq(dummy_f[f_indices], buf[buf_indices])
            name = 'scatter%d' % progressid

        iet = Expression(eq)
        for i, d in reversed(list(zip(buf_indices, buf_dims))):
            # The -1 below is because an Iteration, by default, generates <=
            iet = Iteration(iet, i, d.symbolic_size - 1, properties=PARALLEL)
        iet = List(body=[ArrayCast(dummy_f), ArrayCast(buf), iet])

        # Optimize the memory copy with the DLE
        from devito.dle import transform
        state = transform(iet, 'simd', {'openmp': self._threaded})

        parameters = [buf] + list(buf.shape) + [dummy_f] + f_offsets + state.input
        return Callable(name, state.nodes, 'void', parameters, ('static',)), state.input


class BasicHaloExchangeBuilder(HaloExchangeBuilder):

    """
    Build basic routines for MPI halo exchanges. No optimisations are performed.

    The only constraint is that the built ``update_halo`` Callable is called prior
    to executing the code region requiring up-to-date halos.
    """

    def _make_sendrecv(self, f, fixed, progressid, extra=None):
        extra = extra or []
        comm = f.grid.distributor._obj_comm

        buf_dims = [Dimension(name='buf_%s' % d.root) for d in f.dimensions
                    if d not in fixed]
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

        args = [bufg] + list(bufg.shape) + [dummy_f] + ofsg + extra
        gather = Call('gather%d' % progressid, args)
        args = [bufs] + list(bufs.shape) + [dummy_f] + ofss + extra
        scatter = Call('scatter%d' % progressid, args)

        # The scatter must be guarded as we must not alter the halo values along
        # the domain boundary, where the sender is actually MPI.PROC_NULL
        scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

        srecv = MPIStatusObject(name='srecv')
        rrecv = MPIRequestObject(name='rrecv')
        rsend = MPIRequestObject(name='rsend')

        count = reduce(mul, bufs.shape, 1)
        recv = Call('MPI_Irecv', [bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                                  fromrank, Integer(13), comm, rrecv])
        send = Call('MPI_Isend', [bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                                  torank, Integer(13), comm, rsend])

        waitrecv = Call('MPI_Wait', [rrecv, srecv])
        waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

        iet = List(body=[recv, gather, send, waitsend, waitrecv, scatter])
        iet = List(body=iet_insert_C_decls(iet))
        parameters = ([dummy_f] + list(bufs.shape) + ofsg + ofss +
                      [fromrank, torank, comm] + extra)
        return Callable('sendrecv%d' % progressid, iet, 'void', parameters, ('static',))

    def _make_haloupdate(self, f, fixed, mask, progressid, extra=None):
        extra = extra or []
        distributor = f.grid.distributor
        nb = distributor._obj_neighborhood
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
                    sizes.append(meta.size)
            mapper[(d0, side, region)] = (sizes, offsets)

        body = []
        for d in f.dimensions:
            if d in fixed:
                continue

            rpeer = FieldFromPointer("%sright" % d, nb)
            lpeer = FieldFromPointer("%sleft" % d, nb)

            if mask[(d, LEFT)]:
                # Sending to left, receiving from right
                lsizes, loffsets = mapper[(d, LEFT, OWNED)]
                rsizes, roffsets = mapper[(d, RIGHT, HALO)]
                args = [f] + lsizes + loffsets + roffsets + [rpeer, lpeer, comm] + extra
                body.append(Call('sendrecv%d' % progressid, args))

            if mask[(d, RIGHT)]:
                # Sending to right, receiving from left
                rsizes, roffsets = mapper[(d, RIGHT, OWNED)]
                lsizes, loffsets = mapper[(d, LEFT, HALO)]
                args = [f] + rsizes + roffsets + loffsets + [lpeer, rpeer, comm] + extra
                body.append(Call('sendrecv%d' % progressid, args))

        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values()) + extra
        return Callable('haloupdate%d' % progressid, iet, 'void', parameters, ('static',))


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
