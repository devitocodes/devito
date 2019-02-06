import abc
from collections import OrderedDict
from ctypes import POINTER, c_void_p
from functools import reduce
from itertools import product
from operator import mul

from sympy import Integer

from devito.data import OWNED, HALO, NOPAD, LEFT, CENTER, RIGHT, default_allocator
from devito.ir.equations import DummyEq
from devito.ir.iet import (ArrayCast, Call, Callable, Conditional, Expression,
                           Iteration, List, iet_insert_C_decls, PARALLEL, EFuncNode)
from devito.symbolics import CondNe, FieldFromPointer, Macro
from devito.tools import dtype_to_mpitype, dtype_to_ctype, flatten
from devito.types import Array, Dimension, Symbol, LocalObject, CompositeObject

__all__ = ['HaloExchangeBuilder']


class HaloExchangeBuilder(object):

    """
    Build IET-based routines to implement MPI halo exchange.
    """

    def __new__(cls, mode='basic'):
        if mode is True or mode == 'basic':
            obj = object.__new__(BasicHaloExchangeBuilder)
        elif mode == 'diag':
            obj = object.__new__(DiagHaloExchangeBuilder)
        elif mode == 'overlap':
            obj = object.__new__(OverlapHaloExchangeBuilder)
        else:
            assert False, "unexpected value `mode=%s`" % mode
        return obj

    def make(self, halo_spots):
        """
        Turn the input HaloSpots into Callables/Calls implementing
        distributed-memory halo exchange.

        For each HaloSpot, at least three Callables are constructed:

            * ``update_halo``, to be called to trigger the halo exchange,
            * ``sendrecv``, called from within ``update_halo``.
            * ``copy``, called from within ``sendrecv``, to implement, for example,
              data gathering prior to an MPI_Send, and data scattering following
              an MPI recv.

        Additional Callables may be constructed in the case of asynchronous
        halo exchange.

        In any case, the HaloExchangeBuilder creator only needs to worry about
        placing the ``update_halo`` Calls.
        """
        calls = {}
        msgs = {}
        generated = OrderedDict()
        for hs in halo_spots:
            begin_exchange = []
            wait_exchange = []
            remainder = []
            for f, v in hs.fmapper.items():
                # Sanity check
                assert f.is_Function
                assert f.grid is not None

                # 0) Build data structures to be passed along the MPI Call stack
                # --------------------------------------------------------------
                if (f, v) not in msgs:
                    key = len(msgs)
                    msgs[(f, v)] = self._make_msg(f, v, key)

                # 1) Callables/Calls for send/recv
                # --------------------------------
                # Note: to construct the halo exchange Callables, use the generic `df`,
                # instead of `f`, so that we don't need to regenerate code for Functions
                # that are symbolically identical to `f` except for the name
                df = f.__class__.__base__(name='a', grid=f.grid, shape=f.shape_global,
                                          dimensions=f.dimensions)
                # `gather`, `scatter`, `sendrecv` and `haloupdate` are generic by
                # construction -- they only need to be generated once for each unique
                # pair (`ndim`, `halos`)
                if (f.ndim, v) not in generated:
                    key = len(generated)
                    haloupdate = self._make_haloupdate(df, v.loc_indices, v.halos, key)
                    sendrecv = self._make_sendrecv(df, v.loc_indices)
                    gather = self._make_copy(df, v.loc_indices)
                    scatter = self._make_copy(df, v.loc_indices, swap=True)
                    # Arrange the newly constructed Callables in a suitable data
                    # structure to capture the call tree. This may be useful to
                    # the HaloExchangeBuilder user
                    haloupdate = EFuncNode(haloupdate)
                    sendrecv = EFuncNode(sendrecv, haloupdate)
                    gather = EFuncNode(gather, sendrecv)
                    scatter = EFuncNode(scatter, sendrecv)
                    generated[(f.ndim, v)] = haloupdate
                # `haloupdate` Call construction
                comm = f.grid.distributor._obj_comm
                nb = f.grid.distributor._obj_neighborhood
                loc_indices = list(v.loc_indices.values())
                args = [f, comm, nb] + loc_indices
                begin_exchange.append(Call(generated[(f.ndim, v)].name, args))

                # 2) Callables/Calls for wait (no-op in case of synchronous halo exchange)
                # ------------------------------------------------------------------------
                # TODO

                # 3) Callables/Calls for remainder computation (no-op in case of
                # synchronous halo exchange)
                # --------------------------------------------------------------
                # TODO

            calls[hs] = List(body=begin_exchange + [hs.body] + wait_exchange + remainder)

        return flatten(generated.values()), calls

    @abc.abstractmethod
    def _make_msg(self, f, halos, key):
        """
        Construct data structures carrying information about the HaloSpot
        ``hs``, to propagate information across the MPI Call stack.
        """
        return

    @abc.abstractmethod
    def _make_copy(self, f, fixed, swap=False):
        """
        Construct a Callable performing a copy of:

            * an arbitrary convex region of ``f`` into a contiguous Array, OR
            * if ``swap=True``, a contiguous Array into an arbitrary convex
              region of ``f``.
        """
        return

    @abc.abstractmethod
    def _make_sendrecv(self, f, fixed, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange
        along given Dimension and DataSide.
        """
        return

    @abc.abstractmethod
    def _make_haloupdate(self, f, fixed, halos, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange.
        """
        return

    @abc.abstractmethod
    def _make_halowait(self, f):
        """
        Construct a Callable performing, for a given DiscreteFunction, a wait on
        one or more asynchronous calls.
        """
        return

    @abc.abstractmethod
    def _make_halocomp(self, body):
        """
        Construct a Callable performing computation over the OWNED region, that is
        the region requiring up-to-date halo values.
        """
        return


class BasicHaloExchangeBuilder(HaloExchangeBuilder):

    """
    A HaloExchangeBuilder making use of synchronous MPI routines only.
    """

    def _make_msg(self, f, halos, key):
        return

    def _make_copy(self, f, fixed, swap=False):
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

        if swap is False:
            eq = DummyEq(buf[buf_indices], f[f_indices])
            name = 'gather%dd' % f.ndim
        else:
            eq = DummyEq(f[f_indices], buf[buf_indices])
            name = 'scatter%dd' % f.ndim

        iet = Expression(eq)
        for i, d in reversed(list(zip(buf_indices, buf_dims))):
            # The -1 below is because an Iteration, by default, generates <=
            iet = Iteration(iet, i, d.symbolic_size - 1, properties=PARALLEL)
        iet = List(body=[ArrayCast(f), ArrayCast(buf), iet])

        parameters = [buf] + list(buf.shape) + [f] + f_offsets
        return Callable(name, iet, 'void', parameters, ('static',))

    def _make_sendrecv(self, f, fixed):
        comm = f.grid.distributor._obj_comm

        buf_dims = [Dimension(name='buf_%s' % d.root) for d in f.dimensions
                    if d not in fixed]
        bufg = Array(name='bufg', dimensions=buf_dims, dtype=f.dtype, scope='heap')
        bufs = Array(name='bufs', dimensions=buf_dims, dtype=f.dtype, scope='heap')

        ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
        ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

        fromrank = Symbol(name='fromrank')
        torank = Symbol(name='torank')

        args = [bufg] + list(bufg.shape) + [f] + ofsg
        gather = Call('gather%dd' % f.ndim, args)
        args = [bufs] + list(bufs.shape) + [f] + ofss
        scatter = Call('scatter%dd' % f.ndim, args)

        # The `gather` is unnecessary if sending to MPI.PROC_NULL
        gather = Conditional(CondNe(torank, Macro('MPI_PROC_NULL')), gather)
        # The `scatter` must be guarded as we must not alter the halo values along
        # the domain boundary, where the sender is actually MPI.PROC_NULL
        scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

        count = reduce(mul, bufs.shape, 1)
        rrecv = MPIRequestObject(name='rrecv')
        rsend = MPIRequestObject(name='rsend')
        recv = Call('MPI_Irecv', [bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                                  fromrank, Integer(13), comm, rrecv])
        send = Call('MPI_Isend', [bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                                  torank, Integer(13), comm, rsend])

        waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
        waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

        iet = List(body=[recv, gather, send, waitsend, waitrecv, scatter])
        iet = List(body=iet_insert_C_decls(iet))
        parameters = ([f] + list(bufs.shape) + ofsg + ofss + [fromrank, torank, comm])
        return Callable('sendrecv%dd' % f.ndim, iet, 'void', parameters, ('static',))

    def _make_haloupdate(self, f, fixed, halos, key=None):
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

            name = ''.join('r' if i is d else 'c' for i in distributor.dimensions)
            rpeer = FieldFromPointer(name, nb)
            name = ''.join('l' if i is d else 'c' for i in distributor.dimensions)
            lpeer = FieldFromPointer(name, nb)

            if (d, LEFT) in halos:
                # Sending to left, receiving from right
                lsizes, loffsets = mapper[(d, LEFT, OWNED)]
                rsizes, roffsets = mapper[(d, RIGHT, HALO)]
                args = [f] + lsizes + loffsets + roffsets + [rpeer, lpeer, comm]
                body.append(Call('sendrecv%dd' % f.ndim, args))

            if (d, RIGHT) in halos:
                # Sending to right, receiving from left
                rsizes, roffsets = mapper[(d, RIGHT, OWNED)]
                lsizes, loffsets = mapper[(d, LEFT, HALO)]
                args = [f] + rsizes + roffsets + loffsets + [lpeer, rpeer, comm]
                body.append(Call('sendrecv%dd' % f.ndim, args))

        name = 'haloupdate%dd%s' % (f.ndim, key)
        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return Callable(name, iet, 'void', parameters, ('static',))

    def _make_halowait(self, f):
        return

    def _make_halocomp(self, body):
        return


class DiagHaloExchangeBuilder(BasicHaloExchangeBuilder):

    """
    Similar to a BasicHaloExchangeBuilder, but communications to diagonal
    neighbours are performed explicitly.
    """

    def _make_haloupdate(self, f, fixed, halos, key=None):
        halos = [i.side for i in halos]
        distributor = f.grid.distributor
        nb = distributor._obj_neighborhood
        comm = distributor._obj_comm

        fixed = {d: Symbol(name="o%s" % d.root) for d in fixed}
        all_tosides = list(product([LEFT, CENTER, RIGHT], repeat=distributor.ndim))
        all_fromsides = list(reversed(all_tosides))

        # Build an iterable `[((tosides, size, ofs), (fromsides, size, ofs)), ...]`
        # for `f`. `size` and `ofs` are symbolic objects. This mapper tells what data
        # values should be sent (OWNED) or received (HALO) given the dimension sides
        candidates = []
        for i in zip(all_tosides, all_fromsides):
            handle = []
            for sides, region in zip(i, [OWNED, HALO]):
                sizes = []
                offsets = []
                mapper = dict(zip(distributor.dimensions, sides))
                for d in f.dimensions:
                    if d in fixed:
                        offsets.append(fixed[d])
                    else:
                        meta = f._C_get_field(region, d, mapper[d])
                        offsets.append(meta.offset)
                        sizes.append(meta.size)
                handle.append((sides, sizes, offsets))
            candidates.append(tuple(handle))

        body = []
        for (tosides, tosizes, tooffs), (fromsides, fromsizes, fromoffs) in candidates:
            if tosides not in halos:
                # Ignore useless halo exchanges
                continue

            name = ''.join(i.name[0] for i in tosides)
            topeer = FieldFromPointer(name, nb)
            name = ''.join(i.name[0] for i in fromsides)
            frompeer = FieldFromPointer(name, nb)

            args = [f] + tosizes + tooffs + fromoffs + [frompeer, topeer, comm]
            body.append(Call('sendrecv%dd' % f.ndim, args))

        name = 'haloupdate%dd%s' % (f.ndim, key)
        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return Callable(name, iet, 'void', parameters, ('static',))


class OverlapHaloExchangeBuilder(DiagHaloExchangeBuilder):

    """
    A DiagHaloExchangeBuilder making use of asynchronous MPI routines to implement
    computation-communication overlap.
    """

    def _make_msg(self, f, halos, key):
        # TODO: `halos` not used yet, but will be exploited for optimization
        # in later versions (or perhaps in newer subclasses). By knowing the halos
        # we could constrain the buffer size and the amount of data that is
        # sent over to the neighbours
        return MPIMsg('msg%d' % key, f)


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


c_mpirequest_p = type('MPI_Request', (c_void_p,), {})


class MPIMsg(CompositeObject):

    _C_field_bufs = 'bufs'
    _C_field_bufg = 'bufg'
    _C_field_rrecv = 'rrecv'
    _C_field_rsend = 'rsend'

    def __init__(self, name, function):
        self._function = function
        fields = [
            (MPIMsg._C_field_bufs, POINTER(dtype_to_ctype(function.dtype))),
            (MPIMsg._C_field_bufg, POINTER(dtype_to_ctype(function.dtype))),
            (MPIMsg._C_field_rrecv, c_mpirequest_p),
            (MPIMsg._C_field_rsend, c_mpirequest_p),
        ]
        super(MPIMsg, self).__init__(name, 'msg', fields)

    @property
    def function(self):
        return self._function

    def _arg_values(self, **kwargs):
        value = self._arg_defaults()
        # Allocate the send/recv buffers
        from IPython import embed; embed()

    def _arg_apply(self):
        # Deallocate the buffers
        pass

    # Pickling support
    _pickle_args = ['name', 'function_name']
