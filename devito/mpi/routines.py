import abc
from collections import OrderedDict
from ctypes import POINTER, c_void_p, c_int, sizeof
from functools import reduce
from itertools import product
from operator import mul

from cached_property import cached_property
from sympy import Integer

from devito.data import CORE, OWNED, HALO, NOPAD, LEFT, CENTER, RIGHT, default_allocator
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Callable, Conditional, Expression, ExpressionBundle,
                           Iteration, LocalExpression, List, Prodder, PARALLEL,
                           make_efunc, FindNodes, Transformer)
from devito.mpi import MPI
from devito.symbolics import (Byref, CondNe, FieldFromPointer, FieldFromComposite,
                              IndexedPointer, Macro)
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
        elif mode == 'overlap2':
            obj = object.__new__(Overlap2HaloExchangeBuilder)
        elif mode == 'full':
            obj = object.__new__(FullHaloExchangeBuilder)
        else:
            assert False, "unexpected value `mode=%s`" % mode
        obj._cache = OrderedDict()
        obj._regions = OrderedDict()
        obj._msgs = OrderedDict()
        obj._efuncs = []
        return obj

    @property
    def efuncs(self):
        return self._efuncs

    @property
    def msgs(self):
        return [i for i in self._msgs.values() if i is not None]

    @property
    def regions(self):
        return [i for i in self._regions.values() if i is not None]

    @property
    def objs(self):
        return self.msgs + self.regions

    def make(self, hs, key):
        """
        Construct Callables and Calls implementing distributed-memory halo
        exchange for the HaloSpot ``hs``.
        """
        # Sanity check
        assert all(f.is_Function and f.grid is not None for f in hs.fmapper)

        # Callables for send/recv/wait
        for f, hse in hs.fmapper.items():
            msg = self._make_msg(f, hse, key='%d_%d' % (key, len(self.msgs)))
            msg = self._msgs.setdefault((f, hse), msg)
            if (f.ndim, hse) not in self._cache:
                df = f.__class__.__base__(name='a', grid=f.grid, shape=f.shape_global,
                                          dimensions=f.dimensions)
                self._cache[(f.ndim, hse)] = self._make_all(df, hse, key, msg)

        msgs = [self._msgs[(f, hse)] for f, hse in hs.fmapper.items()]

        # Callable for poking the asynchronous progress engine
        poke = self._make_poke(hs, key, msgs)
        if poke is not None:
            self._efuncs.append(poke)

        # Callable for compute over the CORE region
        callpoke = self._call_poke(poke)
        compute = self._make_compute(hs, key, msgs, callpoke)
        if compute is not None:
            self._efuncs.append(compute)

        # Callable for compute over the OWNED region
        region = self._make_region(hs, key)
        region = self._regions.setdefault(hs, region)
        callcompute = self._call_compute(hs, compute, msgs)
        remainder = self._make_remainder(hs, key, callcompute, region)
        if remainder is not None:
            self._efuncs.append(remainder)

        # Now build up the HaloSpot body, with explicit Calls to the constructed Callables
        body = [callcompute]
        for f, hse in hs.fmapper.items():
            msg = self._msgs[(f, hse)]
            haloupdate, halowait = self._cache[(f.ndim, hse)]
            body.insert(0, self._call_haloupdate(haloupdate.name, f, hse, msg))
            if halowait is not None:
                body.append(self._call_halowait(halowait.name, f, hse, msg))
        if remainder is not None:
            body.append(self._call_remainder(remainder))

        return List(body=body)

    @abc.abstractmethod
    def _make_region(self, hs, key):
        """
        Construct an MPIRegion describing the HaloSpot's OWNED DataRegion.
        """
        return

    @abc.abstractmethod
    def _make_msg(self, f, hse, key):
        """
        Construct an MPIMsg, to propagate information such as buffers, sizes,
        offsets, ..., across the MPI Call stack.
        """
        return

    @abc.abstractmethod
    def _make_all(self, f, hse, key, msg):
        """
        Construct the Callables required to perform a halo update given a
        DiscreteFunction and a set of halo requirements.
        """
        return

    @abc.abstractmethod
    def _make_copy(self, f, hse, key, swap=False):
        """
        Construct a Callable performing a copy of:

            * an arbitrary convex region of ``f`` into a contiguous Array, OR
            * if ``swap=True``, a contiguous Array into an arbitrary convex
              region of ``f``.
        """
        return

    @abc.abstractmethod
    def _make_sendrecv(self, f, hse, key, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange
        along given Dimension and DataSide.
        """
        return

    @abc.abstractmethod
    def _call_sendrecv(self, name, *args, **kwargs):
        """
        Construct a Call to ``sendrecv``, the Callable produced by
        :meth:`_make_sendrecv`.
        """
        return

    @abc.abstractmethod
    def _make_haloupdate(self, f, hse, key, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a halo exchange.
        """
        return

    @abc.abstractmethod
    def _call_haloupdate(self, name, f, hse, *args):
        """
        Construct a Call to ``haloupdate``, the Callable produced by
        :meth:`_make_haloupdate`.
        """
        return

    @abc.abstractmethod
    def _make_compute(self, hs, key, *args):
        """
        Construct a Callable performing computation over the CORE region, that is
        the region that does *not* require up-to-date halo values. The Callable
        body will essentially coincide with the HaloSpot body.
        """
        return

    @abc.abstractmethod
    def _call_compute(self, hs, *args):
        """
        Construct a Call to ``compute``, the Callable produced by :meth:`_make_compute`.
        """
        return

    @abc.abstractmethod
    def _make_poke(self, hs, key, msgs):
        """
        Construct a Callable poking the MPI engine for asynchronous progress (e.g.,
        by calling MPI_Test)
        """
        return

    @abc.abstractmethod
    def _call_poke(self, poke):
        """
        Construct a Call to ``poke``, the Callable produced by :meth:`_make_poke`.
        """
        return

    @abc.abstractmethod
    def _make_wait(self, f, hse, key, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a wait on
        a halo exchange along given Dimension and DataSide.
        """
        return

    @abc.abstractmethod
    def _make_halowait(self, f, hse, key, **kwargs):
        """
        Construct a Callable performing, for a given DiscreteFunction, a wait on
        a halo exchange.
        """
        return

    @abc.abstractmethod
    def _call_halowait(self, name, f, hse, *args):
        """
        Construct a Call to ``halowait``, the Callable produced by :meth:`_make_halowait`.
        """
        return

    @abc.abstractmethod
    def _make_remainder(self, hs, key, callcompute, *args):
        """
        Construct a Callable performing computation over the OWNED region, that is
        the region requiring up-to-date halo values.
        """
        return

    @abc.abstractmethod
    def _call_remainder(self, remainder):
        """
        Construct a Call to ``remainder``, the Callable produced by
        :meth:`_make_remainder`.
        """
        return


class BasicHaloExchangeBuilder(HaloExchangeBuilder):

    """
    A HaloExchangeBuilder making use of synchronous MPI routines only.
    """

    def _make_msg(self, f, hse, key):
        return

    def _make_all(self, f, hse, key, msg):
        haloupdate = self._make_haloupdate(f, hse, key, msg=msg)
        sendrecv = self._make_sendrecv(f, hse, key, msg=msg)
        gather = self._make_copy(f, hse, key)
        scatter = self._make_copy(f, hse, key, swap=True)

        self._efuncs.extend([haloupdate, sendrecv, gather, scatter])

        halowait = self._make_halowait(f, hse, key, msg=msg)
        assert halowait is None

        return haloupdate, halowait

    def _make_copy(self, f, hse, key='', swap=False):
        buf_dims = []
        buf_indices = []
        for d in f.dimensions:
            if d not in hse.loc_indices:
                buf_dims.append(Dimension(name='buf_%s' % d.root))
                buf_indices.append(d.root)
        buf = Array(name='buf', dimensions=buf_dims, dtype=f.dtype)

        f_offsets = []
        f_indices = []
        for d in f.dimensions:
            offset = Symbol(name='o%s' % d.root)
            f_offsets.append(offset)
            f_indices.append(offset + (d.root if d not in hse.loc_indices else 0))

        if swap is False:
            eq = DummyEq(buf[buf_indices], f[f_indices])
            name = 'gather%s' % key
        else:
            eq = DummyEq(f[f_indices], buf[buf_indices])
            name = 'scatter%s' % key

        iet = Expression(eq)
        for i, d in reversed(list(zip(buf_indices, buf_dims))):
            # The -1 below is because an Iteration, by default, generates <=
            iet = Iteration(iet, i, d.symbolic_size - 1)
        iet = iet._rebuild(properties=PARALLEL)

        parameters = [buf] + list(buf.shape) + [f] + f_offsets
        return Callable(name, iet, 'void', parameters, ('static',))

    def _make_sendrecv(self, f, hse, key='', **kwargs):
        comm = f.grid.distributor._obj_comm

        buf_dims = [Dimension(name='buf_%s' % d.root) for d in f.dimensions
                    if d not in hse.loc_indices]
        bufg = Array(name='bufg', dimensions=buf_dims, dtype=f.dtype, scope='heap')
        bufs = Array(name='bufs', dimensions=buf_dims, dtype=f.dtype, scope='heap')

        ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]
        ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

        fromrank = Symbol(name='fromrank')
        torank = Symbol(name='torank')

        gather = Call('gather%s' % key, [bufg] + list(bufg.shape) + [f] + ofsg)
        scatter = Call('scatter%s' % key, [bufs] + list(bufs.shape) + [f] + ofss)

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
        parameters = ([f] + list(bufs.shape) + ofsg + ofss + [fromrank, torank, comm])
        return Callable('sendrecv%s' % key, iet, 'void', parameters, ('static',))

    def _call_sendrecv(self, name, *args, **kwargs):
        return Call(name, flatten(args))

    def _make_haloupdate(self, f, hse, key='', **kwargs):
        distributor = f.grid.distributor
        nb = distributor._obj_neighborhood
        comm = distributor._obj_comm

        fixed = {d: Symbol(name="o%s" % d.root) for d in hse.loc_indices}

        # Build a mapper `(dim, side, region) -> (size, ofs)` for `f`. `size` and
        # `ofs` are symbolic objects. This mapper tells what data values should be
        # sent (OWNED) or received (HALO) given dimension and side
        mapper = {}
        for d0, side, region in product(f.dimensions, (LEFT, RIGHT), (OWNED, HALO)):
            if d0 in fixed:
                continue
            sizes = []
            ofs = []
            for d1 in f.dimensions:
                if d1 in fixed:
                    ofs.append(fixed[d1])
                else:
                    meta = f._C_get_field(region if d0 is d1 else NOPAD, d1, side)
                    ofs.append(meta.offset)
                    sizes.append(meta.size)
            mapper[(d0, side, region)] = (sizes, ofs)

        body = []
        for d in f.dimensions:
            if d in fixed:
                continue

            name = ''.join('r' if i is d else 'c' for i in distributor.dimensions)
            rpeer = FieldFromPointer(name, nb)
            name = ''.join('l' if i is d else 'c' for i in distributor.dimensions)
            lpeer = FieldFromPointer(name, nb)

            if (d, LEFT) in hse.halos:
                # Sending to left, receiving from right
                lsizes, lofs = mapper[(d, LEFT, OWNED)]
                rsizes, rofs = mapper[(d, RIGHT, HALO)]
                args = [f, lsizes, lofs, rofs, rpeer, lpeer, comm]
                body.append(self._call_sendrecv('sendrecv%s' % key, *args, **kwargs))

            if (d, RIGHT) in hse.halos:
                # Sending to right, receiving from left
                rsizes, rofs = mapper[(d, RIGHT, OWNED)]
                lsizes, lofs = mapper[(d, LEFT, HALO)]
                args = [f, rsizes, rofs, lofs, lpeer, rpeer, comm]
                body.append(self._call_sendrecv('sendrecv%s' % key, *args, **kwargs))

        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return Callable('haloupdate%s' % key, iet, 'void', parameters, ('static',))

    def _call_haloupdate(self, name, f, hse, *args):
        comm = f.grid.distributor._obj_comm
        nb = f.grid.distributor._obj_neighborhood
        args = [f, comm, nb] + list(hse.loc_indices.values())
        return Call(name, flatten(args))

    def _make_compute(self, *args):
        return

    def _make_poke(self, *args):
        return

    def _call_poke(self, *args):
        return

    def _call_compute(self, hs, *args):
        return hs.body

    def _make_halowait(self, *args, **kwargs):
        return

    def _call_halowait(self, *args, **kwargs):
        return

    def _make_remainder(self, *args):
        return

    def _call_remainder(self, *args):
        return


class DiagHaloExchangeBuilder(BasicHaloExchangeBuilder):

    """
    Similar to a BasicHaloExchangeBuilder, but communications to diagonal
    neighbours are performed explicitly.
    """

    def _make_haloupdate(self, f, hse, key='', **kwargs):
        distributor = f.grid.distributor
        nb = distributor._obj_neighborhood
        comm = distributor._obj_comm

        fixed = {d: Symbol(name="o%s" % d.root) for d in hse.loc_indices}

        # Only retain the halos required by the Diag scheme
        # Note: `sorted` is only for deterministic code generation
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))

        body = []
        for dims, tosides in halos:
            mapper = OrderedDict(zip(dims, tosides))

            sizes = [f._C_get_field(OWNED, d, s).size for d, s in mapper.items()]

            torank = FieldFromPointer(''.join(i.name[0] for i in mapper.values()), nb)
            ofsg = [fixed.get(d, f._C_get_field(OWNED, d, mapper.get(d)).offset)
                    for d in f.dimensions]

            mapper = OrderedDict(zip(dims, [i.flip() for i in tosides]))
            fromrank = FieldFromPointer(''.join(i.name[0] for i in mapper.values()), nb)
            ofss = [fixed.get(d, f._C_get_field(HALO, d, mapper.get(d)).offset)
                    for d in f.dimensions]

            kwargs['haloid'] = len(body)

            body.append(self._call_sendrecv('sendrecv%s' % key, f, sizes, ofsg, ofss,
                                            fromrank, torank, comm, **kwargs))

        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return Callable('haloupdate%s' % key, iet, 'void', parameters, ('static',))


class OverlapHaloExchangeBuilder(DiagHaloExchangeBuilder):

    """
    A DiagHaloExchangeBuilder making use of asynchronous MPI routines to implement
    computation-communication overlap.
    """

    def _make_msg(self, f, hse, key):
        # Only retain the halos required by the Diag scheme
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))
        return MPIMsg('msg%s' % key, f, halos)

    def _make_all(self, f, hse, key, msg):
        # Callables for asynchronous send/recv
        haloupdate = self._make_haloupdate(f, hse, key, msg=msg)
        sendrecv = self._make_sendrecv(f, hse, key, msg=msg)
        gather = self._make_copy(f, hse, key)

        # Callables for wait
        halowait = self._make_halowait(f, hse, key, msg=msg)
        wait = self._make_wait(f, hse, key, msg=msg)
        scatter = self._make_copy(f, hse, key, swap=True)

        self._efuncs.extend([haloupdate, sendrecv, gather, halowait, wait, scatter])

        return haloupdate, halowait

    def _make_sendrecv(self, f, hse, key='', msg=None):
        comm = f.grid.distributor._obj_comm

        bufg = FieldFromPointer(msg._C_field_bufg, msg)
        bufs = FieldFromPointer(msg._C_field_bufs, msg)

        ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]

        fromrank = Symbol(name='fromrank')
        torank = Symbol(name='torank')

        sizes = [FieldFromPointer('%s[%d]' % (msg._C_field_sizes, i), msg)
                 for i in range(len(f._dist_dimensions))]

        # The `gather` is unnecessary if sending to MPI.PROC_NULL
        gather = Call('gather%s' % key, [bufg] + sizes + [f] + ofsg)
        gather = Conditional(CondNe(torank, Macro('MPI_PROC_NULL')), gather)

        count = reduce(mul, sizes, 1)
        rrecv = Byref(FieldFromPointer(msg._C_field_rrecv, msg))
        rsend = Byref(FieldFromPointer(msg._C_field_rsend, msg))
        recv = Call('MPI_Irecv', [bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                                  fromrank, Integer(13), comm, rrecv])
        send = Call('MPI_Isend', [bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                                  torank, Integer(13), comm, rsend])

        iet = List(body=[recv, gather, send])
        parameters = ([f] + ofsg + [fromrank, torank, comm, msg])
        return Callable('sendrecv%s' % key, iet, 'void', parameters, ('static',))

    def _call_sendrecv(self, name, *args, msg=None, haloid=None):
        # Drop `sizes` as this HaloExchangeBuilder conveys them through `msg`
        # Drop `ofss` as this HaloExchangeBuilder only needs them in `wait()`,
        # to collect and scatter the result of an MPI_Irecv
        f, _, ofsg, _, fromrank, torank, comm = args
        msg = Byref(IndexedPointer(msg, haloid))
        return Call(name, [f] + ofsg + [fromrank, torank, comm, msg])

    def _make_haloupdate(self, f, hse, key='', msg=None):
        iet = super(OverlapHaloExchangeBuilder, self)._make_haloupdate(f, hse, key,
                                                                       msg=msg)
        iet = iet._rebuild(parameters=iet.parameters + (msg,))
        return iet

    def _call_haloupdate(self, name, f, hse, msg):
        call = super(OverlapHaloExchangeBuilder, self)._call_haloupdate(name, f, hse)
        call = call._rebuild(arguments=call.arguments + (msg,))
        return call

    def _make_compute(self, hs, key, *args):
        if hs.body.is_Call:
            return None
        else:
            return make_efunc('compute%s' % key, hs.body, hs.dimensions)

    def _call_compute(self, hs, compute, *args):
        if compute is None:
            assert hs.body.is_Call
            return hs.body._rebuild(dynamic_args_mapper=hs.omapper, incr=True)
        else:
            return compute.make_call(dynamic_args_mapper=hs.omapper, incr=True)

    def _make_wait(self, f, hse, key='', msg=None):
        bufs = FieldFromPointer(msg._C_field_bufs, msg)

        ofss = [Symbol(name='os%s' % d.root) for d in f.dimensions]

        fromrank = Symbol(name='fromrank')

        sizes = [FieldFromPointer('%s[%d]' % (msg._C_field_sizes, i), msg)
                 for i in range(len(f._dist_dimensions))]
        scatter = Call('scatter%s' % key, [bufs] + sizes + [f] + ofss)

        # The `scatter` must be guarded as we must not alter the halo values along
        # the domain boundary, where the sender is actually MPI.PROC_NULL
        scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

        rrecv = Byref(FieldFromPointer(msg._C_field_rrecv, msg))
        waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
        rsend = Byref(FieldFromPointer(msg._C_field_rsend, msg))
        waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

        iet = List(body=[waitsend, waitrecv, scatter])
        parameters = ([f] + ofss + [fromrank, msg])
        return Callable('wait%s' % key, iet, 'void', parameters, ('static',))

    def _make_halowait(self, f, hse, key='', msg=None):
        nb = f.grid.distributor._obj_neighborhood

        fixed = {d: Symbol(name="o%s" % d.root) for d in hse.loc_indices}

        # Only retain the halos required by the Diag scheme
        # Note: `sorted` is only for deterministic code generation
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))

        body = []
        for dims, tosides in halos:
            mapper = OrderedDict(zip(dims, [i.flip() for i in tosides]))
            fromrank = FieldFromPointer(''.join(i.name[0] for i in mapper.values()), nb)
            ofss = [fixed.get(d, f._C_get_field(HALO, d, mapper.get(d)).offset)
                    for d in f.dimensions]

            msgi = Byref(IndexedPointer(msg, len(body)))

            body.append(Call('wait%s' % key, [f] + ofss + [fromrank, msgi]))

        iet = List(body=body)
        parameters = [f] + list(fixed.values()) + [nb, msg]
        return Callable('halowait%s' % key, iet, 'void', parameters, ('static',))

    def _call_halowait(self, name, f, hse, msg):
        nb = f.grid.distributor._obj_neighborhood
        return Call(name, [f] + list(hse.loc_indices.values()) + [nb, msg])

    def _make_remainder(self, hs, key, callcompute, *args):
        assert callcompute.is_Call

        items = []
        mapper = OrderedDict()
        for d, (left, right) in hs.omapper.items():
            defleft, defright = callcompute.dynamic_defaults[d]
            dmapper = OrderedDict()
            dmapper[(d, CORE, CENTER)] = (defleft, defright)
            dmapper[(d, OWNED, LEFT)] = (defleft - left, defleft)
            dmapper[(d, OWNED, RIGHT)] = (defright, defright - right)
            mapper.update(dmapper)
            items.append(list(dmapper))

        body = []
        for i in product(*items):
            if all(r is CORE for _, r, _ in i):
                continue
            dynamic_args_mapper = {d: mapper[(d, r, s)] for d, r, s in i}
            body.append(callcompute._rebuild(dynamic_args_mapper=dynamic_args_mapper,
                                             incr=False))

        return make_efunc('remainder%s' % key, body)

    def _call_remainder(self, remainder):
        return remainder.make_call()


class Overlap2HaloExchangeBuilder(OverlapHaloExchangeBuilder):

    """
    A OverlapHaloExchangeBuilder with reduced Call overhead and increased code
    readability, achieved by supplying more values via Python-land-produced
    structs, which replace explicit Call arguments.
    """

    def _make_region(self, hs, key):
        return MPIRegion('reg%s' % key, hs.omapper)

    def _make_msg(self, f, hse, key):
        # Only retain the halos required by the Diag scheme
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))
        return MPIMsgEnriched('msg%s' % key, f, halos)

    def _make_all(self, f, hse, key, msg):
        # Callables for asynchronous send/recv
        haloupdate = self._make_haloupdate(f, hse, key, msg=msg)
        gather = self._make_copy(f, hse, key)

        # Callables for wait
        halowait = self._make_halowait(f, hse, key, msg=msg)
        scatter = self._make_copy(f, hse, key, swap=True)

        self._efuncs.extend([haloupdate, gather, halowait, scatter])

        return haloupdate, halowait

    def _make_haloupdate(self, f, hse, key='', msg=None):
        comm = f.grid.distributor._obj_comm

        fixed = {d: Symbol(name="o%s" % d.root) for d in hse.loc_indices}

        dim = Dimension(name='i')

        msgi = IndexedPointer(msg, dim)

        bufg = FieldFromComposite(msg._C_field_bufg, msgi)
        bufs = FieldFromComposite(msg._C_field_bufs, msgi)

        fromrank = FieldFromComposite(msg._C_field_from, msgi)
        torank = FieldFromComposite(msg._C_field_to, msgi)

        sizes = [FieldFromComposite('%s[%d]' % (msg._C_field_sizes, i), msgi)
                 for i in range(len(f._dist_dimensions))]
        ofsg = [FieldFromComposite('%s[%d]' % (msg._C_field_ofsg, i), msgi)
                for i in range(len(f._dist_dimensions))]
        ofsg = [fixed.get(d) or ofsg.pop(0) for d in f.dimensions]

        # The `gather` is unnecessary if sending to MPI.PROC_NULL
        gather = Call('gather%s' % key, [bufg] + sizes + [f] + ofsg)
        gather = Conditional(CondNe(torank, Macro('MPI_PROC_NULL')), gather)

        # Make Irecv/Isend
        count = reduce(mul, sizes, 1)
        rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
        rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
        recv = Call('MPI_Irecv', [bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                                  fromrank, Integer(13), comm, rrecv])
        send = Call('MPI_Isend', [bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                                  torank, Integer(13), comm, rsend])

        # The -1 below is because an Iteration, by default, generates <=
        iet = Iteration([recv, gather, send], dim, msg.npeers - 1)
        parameters = ([f, comm, msg]) + list(fixed.values())
        return Callable('haloupdate%s' % key, iet, 'void', parameters, ('static',))

    def _call_haloupdate(self, name, f, hse, msg):
        comm = f.grid.distributor._obj_comm
        return Call(name, [f, comm, msg] + list(hse.loc_indices.values()))

    def _make_sendrecv(self, *args):
        return

    def _call_sendrecv(self, *args):
        return

    def _make_halowait(self, f, hse, key='', msg=None):
        fixed = {d: Symbol(name="o%s" % d.root) for d in hse.loc_indices}

        dim = Dimension(name='i')

        msgi = IndexedPointer(msg, dim)

        bufs = FieldFromComposite(msg._C_field_bufs, msgi)

        fromrank = FieldFromComposite(msg._C_field_from, msgi)

        sizes = [FieldFromComposite('%s[%d]' % (msg._C_field_sizes, i), msgi)
                 for i in range(len(f._dist_dimensions))]
        ofss = [FieldFromComposite('%s[%d]' % (msg._C_field_ofss, i), msgi)
                for i in range(len(f._dist_dimensions))]
        ofss = [fixed.get(d) or ofss.pop(0) for d in f.dimensions]

        # The `scatter` must be guarded as we must not alter the halo values along
        # the domain boundary, where the sender is actually MPI.PROC_NULL
        scatter = Call('scatter%s' % key, [bufs] + sizes + [f] + ofss)
        scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

        rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
        waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
        rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
        waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

        # The -1 below is because an Iteration, by default, generates <=
        iet = Iteration([waitsend, waitrecv, scatter], dim, msg.npeers - 1)
        parameters = ([f] + list(fixed.values()) + [msg])
        return Callable('halowait%s' % key, iet, 'void', parameters, ('static',))

    def _call_halowait(self, name, f, hse, msg):
        return Call(name, [f] + list(hse.loc_indices.values()) + [msg])

    def _make_wait(self, *args):
        return

    def _call_wait(self, *args):
        return

    def _make_remainder(self, hs, key, callcompute, region):
        assert callcompute.is_Call

        dim = Dimension(name='i')
        regioni = IndexedPointer(region, dim)
        dynamic_args_mapper = {d: (FieldFromComposite(d.min_name, regioni),
                                   FieldFromComposite(d.max_name, regioni))
                               for d in hs.dimensions}
        iet = callcompute._rebuild(dynamic_args_mapper=dynamic_args_mapper)
        # The -1 below is because an Iteration, by default, generates <=
        iet = Iteration(iet, dim, region.nregions - 1)

        return make_efunc('remainder%s' % key, iet)


class FullHaloExchangeBuilder(Overlap2HaloExchangeBuilder):

    """
    A Overlap2HaloExchangeBuilder which generates explicit Calls to MPI_Test
    poking the MPI runtime to advance communication while computing.
    """

    def _make_compute(self, hs, key, msgs, callpoke):
        if hs.body.is_Call:
            return None
        else:
            mapper = {i: List(body=[callpoke, i]) for i in
                      FindNodes(ExpressionBundle).visit(hs.body)}
            iet = Transformer(mapper).visit(hs.body)
            return make_efunc('compute%s' % key, iet, hs.dimensions)

    def _make_poke(self, hs, key, msgs):
        flag = Symbol(name='flag')
        initflag = LocalExpression(DummyEq(flag, 0))

        body = [initflag]
        for msg in msgs:
            dim = Dimension(name='i')
            msgi = IndexedPointer(msg, dim)

            rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
            rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
            testrecv = Call('MPI_Test', [rrecv, Byref(flag), Macro('MPI_STATUS_IGNORE')])
            testsend = Call('MPI_Test', [rsend, Byref(flag), Macro('MPI_STATUS_IGNORE')])

            body.append(Iteration([testsend, testrecv], dim, msg.npeers - 1))

        return make_efunc('pokempi%s' % key, body)

    def _call_poke(self, poke):
        return Prodder(poke.name, poke.parameters, single_thread=True, periodic=True)


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


class MPIMsg(CompositeObject):

    _C_field_bufs = 'bufs'
    _C_field_bufg = 'bufg'
    _C_field_sizes = 'sizes'
    _C_field_rrecv = 'rrecv'
    _C_field_rsend = 'rsend'

    if MPI._sizeof(MPI.Request) == sizeof(c_int):
        c_mpirequest_p = type('MPI_Request', (c_int,), {})
    else:
        c_mpirequest_p = type('MPI_Request', (c_void_p,), {})

    def __init__(self, name, function, halos, fields=None):
        self._function = function
        self._halos = halos
        fields = (fields or []) + [
            (MPIMsg._C_field_bufs, c_void_p),
            (MPIMsg._C_field_bufg, c_void_p),
            (MPIMsg._C_field_sizes, POINTER(c_int)),
            (MPIMsg._C_field_rrecv, MPIMsg.c_mpirequest_p),
            (MPIMsg._C_field_rsend, MPIMsg.c_mpirequest_p),
        ]
        super(MPIMsg, self).__init__(name, 'msg', fields)

        # Required for buffer allocation/deallocation before/after jumping/returning
        # to/from C-land
        self._allocator = default_allocator()
        self._memfree_args = []

    def __del__(self):
        self._C_memfree()

    def _C_memfree(self):
        # Deallocate the MPI buffers
        for i in self._memfree_args:
            self._allocator.free(*i)
        self._memfree_args[:] = []

    def __value_setup__(self, dtype, value):
        # We eventually produce an array of `struct msg` that is as big as
        # the number of peers we have to communicate with
        return (dtype._type_*self.npeers)()

    @property
    def function(self):
        return self._function

    @property
    def halos(self):
        return self._halos

    @property
    def npeers(self):
        return len(self._halos)

    def _arg_defaults(self, alias=None):
        function = alias or self.function
        for i, halo in enumerate(self.halos):
            entry = self.value[i]
            # Buffer size for this peer
            shape = []
            for dim, side in zip(*halo):
                try:
                    shape.append(getattr(function._size_owned[dim], side.name))
                except AttributeError:
                    assert side is CENTER
                    shape.append(function._size_domain[dim])
            entry.sizes = (c_int*len(shape))(*shape)
            # Allocate the send/recv buffers
            size = reduce(mul, shape)
            ctype = dtype_to_ctype(function.dtype)
            entry.bufg, bufg_memfree_args = self._allocator._alloc_C_libcall(size, ctype)
            entry.bufs, bufs_memfree_args = self._allocator._alloc_C_libcall(size, ctype)
            # The `memfree_args` will be used to deallocate the buffer upon returning
            # from C-land
            self._memfree_args.extend([bufg_memfree_args, bufs_memfree_args])

        return {self.name: self.value}

    def _arg_values(self, args=None, **kwargs):
        return self._arg_defaults(alias=kwargs.get(self.function.name, self.function))

    def _arg_apply(self, *args, **kwargs):
        self._C_memfree()

    # Pickling support
    _pickle_args = ['name', 'function', 'halos']


class MPIMsgEnriched(MPIMsg):

    _C_field_ofss = 'ofss'
    _C_field_ofsg = 'ofsg'
    _C_field_from = 'fromrank'
    _C_field_to = 'torank'

    def __init__(self, name, function, halos):
        fields = [
            (MPIMsgEnriched._C_field_ofss, POINTER(c_int)),
            (MPIMsgEnriched._C_field_ofsg, POINTER(c_int)),
            (MPIMsgEnriched._C_field_from, c_int),
            (MPIMsgEnriched._C_field_to, c_int)
        ]
        super(MPIMsgEnriched, self).__init__(name, function, halos, fields)

    def _arg_defaults(self, alias=None):
        super(MPIMsgEnriched, self)._arg_defaults(alias)

        function = alias or self.function
        neighborhood = function.grid.distributor.neighborhood
        for i, halo in enumerate(self.halos):
            entry = self.value[i]
            # `torank` peer + gather offsets
            entry.torank = neighborhood[halo.side]
            ofsg = []
            for dim, side in zip(*halo):
                try:
                    ofsg.append(getattr(function._offset_owned[dim], side.name))
                except AttributeError:
                    assert side is CENTER
                    ofsg.append(function._offset_owned[dim].left)
            entry.ofsg = (c_int*len(ofsg))(*ofsg)
            # `fromrank` peer + scatter offsets
            entry.fromrank = neighborhood[tuple(i.flip() for i in halo.side)]
            ofss = []
            for dim, side in zip(*halo):
                try:
                    ofss.append(getattr(function._offset_halo[dim], side.flip().name))
                except AttributeError:
                    assert side is CENTER
                    # Note `_offset_owned`, and not `_offset_halo`, is *not* a bug here.
                    # If it's the CENTER we need, we can't use `_offset_halo[d].left`
                    # as otherwise we would be picking the corner
                    ofss.append(function._offset_owned[dim].left)
            entry.ofss = (c_int*len(ofss))(*ofss)

        return {self.name: self.value}


class MPIRegion(CompositeObject):

    def __init__(self, name, omapper):
        self._omapper = omapper
        fields = []
        for d in list(omapper):
            fields.append((d.min_name, c_int))
            fields.append((d.max_name, c_int))
        super(MPIRegion, self).__init__(name, 'region', fields)

    def __value_setup__(self, dtype, value):
        # We eventually produce an array of `struct region` that is as big as
        # the number of OWNED sub-regions we have to compute to complete a
        # halo update
        return (dtype._type_*self.nregions)()

    @property
    def omapper(self):
        return self._omapper

    @cached_property
    def regions(self):
        items = [((d, CORE, CENTER), (d, OWNED, LEFT), (d, OWNED, RIGHT))
                 for d in self.omapper]
        items = list(product(*items))
        items = [i for i in items if not all(r is CORE for _, r, _ in i)]
        return items

    @property
    def nregions(self):
        return len(self.regions)

    def _arg_values(self, args, **kwargs):
        values = self._arg_defaults()
        for i, region in enumerate(self.regions):
            entry = values[self.name][i]
            for d, r, s in region:
                lofs, rofs = self.omapper[d]
                if r is CORE:
                    setattr(entry, d.min_name, args[d.min_name] + lofs)
                    setattr(entry, d.max_name, args[d.max_name] + rofs)
                else:
                    if s is LEFT:
                        setattr(entry, d.min_name, args[d.min_name])
                        setattr(entry, d.max_name, args[d.min_name] + lofs)
                    else:
                        setattr(entry, d.min_name, args[d.max_name] + rofs)
                        setattr(entry, d.max_name, args[d.max_name])
        return values

    # Pickling support
    _pickle_args = ['name', 'omapper']
