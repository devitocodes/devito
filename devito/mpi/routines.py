import abc
from collections import OrderedDict
from ctypes import POINTER, c_void_p, c_int, sizeof
from functools import reduce
from itertools import product
from operator import mul

from sympy import Integer

from devito.data import OWNED, HALO, NOPAD, LEFT, CENTER, RIGHT
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Callable, Conditional, Expression, ExpressionBundle,
                           AugmentedExpression, Iteration, List, Prodder, Return,
                           make_efunc, FindNodes, Transformer)
from devito.mpi import MPI
from devito.symbolics import (Byref, CondNe, FieldFromPointer, FieldFromComposite,
                              IndexedPointer, Macro, cast_mapper, subs_op_args)
from devito.tools import (dtype_to_mpitype, dtype_len, dtype_to_ctype, flatten,
                          generator)
from devito.types import Array, Dimension, Eq, Symbol, LocalObject, CompositeObject

__all__ = ['HaloExchangeBuilder', 'mpi_registry']


class HaloExchangeBuilder(object):

    """
    Build IET-based routines to implement MPI halo exchange.
    """

    def __new__(cls, mpimode, generators=None, rcompile=None, sregistry=None, **kwargs):
        obj = object.__new__(mpi_registry[mpimode])

        obj.rcompile = rcompile
        obj.sregistry = sregistry

        # Unique name generators
        generators = generators or {}
        obj._gen_msgkey = generators.get('msg', generator())
        obj._gen_commkey = generators.get('comm', generator())
        obj._gen_compkey = generators.get('comp', generator())

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
    def headers(self):
        """
        No headers needed by default
        """
        return {}

    def make(self, hs):
        """
        Construct Callables and Calls implementing distributed-memory halo
        exchange for the HaloSpot ``hs``.
        """
        # Sanity check
        assert all(f.is_Function and f.grid is not None for f in hs.fmapper)

        mapper = {}
        for f, hse in hs.fmapper.items():
            # Build an MPIMsg, a data structure to be propagated across the
            # various halo exchange routines
            try:
                msg = self._msgs[(f, hse)]
            except KeyError:
                key = self._gen_msgkey()
                msg = self._msgs.setdefault((f, hse), self._make_msg(f, hse, key))

            # Callables for send/recv/wait
            mapper[(f, hse)] = self._make_all(f, hse, msg)

        msgs = [self._msgs[(f, hse)] for f, hse in hs.fmapper.items()]

        # Callable for poking the asynchronous progress engine
        key = self._gen_compkey()
        poke = self._make_poke(hs, key, msgs)
        if isinstance(poke, Callable):
            self._efuncs.append(poke)

        # Callable for compute over the CORE region
        callpoke = self._call_poke(poke)
        compute = self._make_compute(hs, key, msgs, callpoke)
        if isinstance(compute, Callable):
            self._efuncs.append(compute)

        # Callable for compute over the OWNED region
        region = self._make_region(hs, key)
        region = self._regions.setdefault(hs, region)
        callcompute = self._call_compute(hs, compute, msgs)
        remainder = self._make_remainder(hs, key, callcompute, region)
        if isinstance(remainder, Callable):
            self._efuncs.append(remainder)

        # Now build up the HaloSpot body, with explicit Calls to the constructed Callables
        haloupdates = []
        halowaits = []
        for i, (f, hse) in enumerate(hs.fmapper.items()):
            msg = self._msgs[(f, hse)]
            haloupdate, halowait = mapper[(f, hse)]
            haloupdates.append(self._call_haloupdate(haloupdate.name, f, hse, msg))
            if halowait is not None:
                halowaits.append(self._call_halowait(halowait.name, f, hse, msg))
        body = []
        body.append(HaloUpdateList(body=haloupdates))
        if callcompute is not None:
            body.append(callcompute)
        body.append(HaloWaitList(body=halowaits))
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
    def _make_all(self, f, hse, msg):
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

    Generates:

        haloupdate()
        compute()
    """

    def _make_msg(self, f, hse, key):
        return

    def _make_all(self, f, hse, msg):
        key = self._gen_commkey()

        sendrecv = self._make_sendrecv(f, hse, key, msg=msg)
        gather = self._make_copy(f, hse, key)
        wait = self._make_wait(f, hse, key, msg=msg)
        scatter = self._make_copy(f, hse, key, swap=True)
        haloupdate = self._make_haloupdate(f, hse, key, sendrecv, msg=msg)
        halowait = self._make_halowait(f, hse, key, wait, msg=msg)

        self._efuncs.append(haloupdate)
        if halowait is not None:
            self._efuncs.append(halowait)
        if wait is not None:
            self._efuncs.append(wait)
        if sendrecv is not None:
            self._efuncs.append(sendrecv)
        self._efuncs.extend([gather, scatter])

        return haloupdate, halowait

    def _make_copy(self, f, hse, key, swap=False):
        dims = [d.root for d in f.dimensions if d not in hse.loc_indices]
        buf = Array(name='buf', dimensions=dims, dtype=f.dtype, padding=0)

        f_offsets = []
        f_indices = []
        for d, h in zip(f.dimensions, f._size_nodomain.left):
            offset = Symbol(name='o%s' % d.root, is_const=True)
            f_offsets.append(offset)
            offset_nohalo = offset - h
            f_indices.append(offset_nohalo + (d.root if d not in hse.loc_indices else 0))

        if swap is False:
            eq = Eq(buf[dims], f[f_indices])
            name = 'gather%s' % key
        else:
            eq = Eq(f[f_indices], buf[dims])
            name = 'scatter%s' % key

        eqns = []
        eqns.extend([Eq(d.symbolic_min, 0) for d in dims])
        eqns.extend([Eq(d.symbolic_max, d.symbolic_size - 1) for d in dims])
        eqns.append(eq)

        # Compile `eqns` into an IET via recursive compilation
        irs, _ = self.rcompile(eqns)

        parameters = [buf] + list(buf.shape) + [f] + f_offsets

        return CopyBuffer(name, irs.uiet, parameters)

    def _make_sendrecv(self, f, hse, key, **kwargs):
        comm = f.grid.distributor._obj_comm

        dims = [d.root for d in f.dimensions if d not in hse.loc_indices]
        bufg = Array(name='bufg', dimensions=dims, dtype=f.dtype,
                     padding=0, liveness='eager')
        bufs = Array(name='bufs', dimensions=dims, dtype=f.dtype,
                     padding=0, liveness='eager')

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

        count = reduce(mul, bufs.shape, 1)*dtype_len(f.dtype)
        rrecv = MPIRequestObject(name='rrecv', liveness='eager')
        rsend = MPIRequestObject(name='rsend', liveness='eager')
        recv = IrecvCall([bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                         fromrank, Integer(13), comm, Byref(rrecv)])
        send = IsendCall([bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                         torank, Integer(13), comm, Byref(rsend)])

        waitrecv = Call('MPI_Wait', [Byref(rrecv), Macro('MPI_STATUS_IGNORE')])
        waitsend = Call('MPI_Wait', [Byref(rsend), Macro('MPI_STATUS_IGNORE')])

        iet = List(body=[recv, gather, send, waitsend, waitrecv, scatter])
        parameters = ([f] + list(bufs.shape) + ofsg + ofss + [fromrank, torank, comm])
        return SendRecv('sendrecv%s' % key, iet, parameters, bufg, bufs)

    def _call_sendrecv(self, name, *args, **kwargs):
        return Call(name, flatten(args))

    def _make_haloupdate(self, f, hse, key, sendrecv, **kwargs):
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
                body.append(self._call_sendrecv(sendrecv.name, *args, **kwargs))

            if (d, RIGHT) in hse.halos:
                # Sending to right, receiving from left
                rsizes, rofs = mapper[(d, RIGHT, OWNED)]
                lsizes, lofs = mapper[(d, LEFT, HALO)]
                args = [f, rsizes, rofs, lofs, lpeer, rpeer, comm]
                body.append(self._call_sendrecv(sendrecv.name, *args, **kwargs))

        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return HaloUpdate('haloupdate%s' % key, iet, parameters)

    def _call_haloupdate(self, name, f, hse, *args):
        comm = f.grid.distributor._obj_comm
        nb = f.grid.distributor._obj_neighborhood
        args = [f, comm, nb] + list(hse.loc_indices.values())
        return HaloUpdateCall(name, flatten(args))

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

    Generates:

        haloupdate()
        compute()
    """

    def _make_haloupdate(self, f, hse, key, sendrecv, **kwargs):
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

            body.append(self._call_sendrecv(sendrecv.name, f, sizes, ofsg, ofss,
                                            fromrank, torank, comm, **kwargs))

        iet = List(body=body)
        parameters = [f, comm, nb] + list(fixed.values())
        return HaloUpdate('haloupdate%s' % key, iet, parameters)


class OverlapHaloExchangeBuilder(DiagHaloExchangeBuilder):

    """
    A DiagHaloExchangeBuilder making use of asynchronous MPI routines to implement
    computation-communication overlap.

    Generates:

        haloupdate()
        compute_core()
        halowait()
        remainder()
    """

    @property
    def headers(self):
        """
        Overlap Mode uses MIN/MAX that need to be defined
        """
        return {'headers': [('MIN(a,b)', ('(((a) < (b)) ? (a) : (b))')),
                            ('MAX(a,b)', ('(((a) > (b)) ? (a) : (b))'))]}

    def _make_msg(self, f, hse, key):
        # Only retain the halos required by the Diag scheme
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))
        return MPIMsg('msg%d' % key, f, halos)

    def _make_sendrecv(self, f, hse, key, msg=None):
        comm = f.grid.distributor._obj_comm

        bufg = FieldFromPointer(msg._C_field_bufg, msg)
        bufs = FieldFromPointer(msg._C_field_bufs, msg)

        ofsg = [Symbol(name='og%s' % d.root) for d in f.dimensions]

        fromrank = Symbol(name='fromrank')
        torank = Symbol(name='torank')

        sizes = [FieldFromPointer('%s[%d]' % (msg._C_field_sizes, i), msg)
                 for i in range(len(f._dist_dimensions))]

        gather = Call('gather%s' % key, [bufg] + sizes + [f] + ofsg)
        # The `gather` is unnecessary if sending to MPI.PROC_NULL
        gather = Conditional(CondNe(torank, Macro('MPI_PROC_NULL')), gather)

        count = reduce(mul, sizes, 1)*dtype_len(f.dtype)
        rrecv = Byref(FieldFromPointer(msg._C_field_rrecv, msg))
        rsend = Byref(FieldFromPointer(msg._C_field_rsend, msg))
        recv = IrecvCall([bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                         fromrank, Integer(13), comm, rrecv])
        send = IsendCall([bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                         torank, Integer(13), comm, rsend])

        iet = List(body=[recv, gather, send])
        parameters = ([f] + ofsg + [fromrank, torank, comm, msg])
        return SendRecv('sendrecv%s' % key, iet, parameters, bufg, bufs)

    def _call_sendrecv(self, name, *args, msg=None, haloid=None):
        # Drop `sizes` as this HaloExchangeBuilder conveys them through `msg`
        # Drop `ofss` as this HaloExchangeBuilder only needs them in `wait()`,
        # to collect and scatter the result of an MPI_Irecv
        f, _, ofsg, _, fromrank, torank, comm = args
        msg = Byref(IndexedPointer(msg, haloid))
        return Call(name, [f] + ofsg + [fromrank, torank, comm, msg])

    def _make_haloupdate(self, f, hse, key, sendrecv, msg=None):
        iet = super()._make_haloupdate(f, hse, key, sendrecv, msg=msg)
        iet = iet._rebuild(parameters=iet.parameters + (msg,))
        return iet

    def _call_haloupdate(self, name, f, hse, msg):
        call = super()._call_haloupdate(name, f, hse)
        call = call._rebuild(arguments=call.arguments + (msg,))
        return call

    def _make_compute(self, hs, key, *args):
        if hs.body.is_Call:
            return None
        else:
            return make_efunc('compute%d' % key, hs.body, hs.arguments)

    def _call_compute(self, hs, compute, *args):
        if compute is None:
            assert hs.body.is_Call
            return hs.body._rebuild(dynamic_args_mapper=hs.omapper.core)
        else:
            return compute.make_call(dynamic_args_mapper=hs.omapper.core)

    def _make_wait(self, f, hse, key, msg=None):
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
        return Callable('wait_%s' % key, iet, 'void', parameters, ('static',))

    def _make_halowait(self, f, hse, key, wait, msg=None):
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

            body.append(Call(wait.name, [f] + ofss + [fromrank, msgi]))

        iet = List(body=body)
        parameters = [f] + list(fixed.values()) + [nb, msg]
        return Callable('halowait%d' % key, iet, 'void', parameters, ('static',))

    def _call_halowait(self, name, f, hse, msg):
        nb = f.grid.distributor._obj_neighborhood
        return HaloWaitCall(name, [f] + list(hse.loc_indices.values()) + [nb, msg])

    def _make_remainder(self, hs, key, callcompute, *args):
        assert callcompute.is_Call
        body = [callcompute._rebuild(dynamic_args_mapper=i) for _, i in hs.omapper.owned]
        return make_efunc('remainder%d' % key, body)

    def _call_remainder(self, remainder):
        efunc = remainder.make_call()
        call = RemainderCall(efunc.name, efunc.arguments)
        return call


class Overlap2HaloExchangeBuilder(OverlapHaloExchangeBuilder):

    """
    A OverlapHaloExchangeBuilder with reduced Call overhead and increased code
    readability, achieved by supplying more values via Python-land-produced
    structs, which replace explicit Call arguments.

    Generates:

        haloupdate()
        compute_core()
        halowait()
        remainder()
    """

    def _make_region(self, hs, key):
        return MPIRegion('reg', key, hs.arguments, hs.omapper.owned)

    def _make_msg(self, f, hse, key):
        # Only retain the halos required by the Diag scheme
        halos = sorted(i for i in hse.halos if isinstance(i.dim, tuple))
        return MPIMsgEnriched('msg%d' % key, f, halos)

    def _make_sendrecv(self, *args, **kwargs):
        return

    def _call_sendrecv(self, *args):
        return

    def _make_haloupdate(self, f, hse, key, *args, msg=None):
        cast = cast_mapper[(f.dtype, '*')]
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

        gather = Call('gather%s' % key, [cast(bufg)] + sizes + [f] + ofsg)
        gather = Conditional(CondNe(torank, Macro('MPI_PROC_NULL')), gather)

        # Make Irecv/Isend
        count = reduce(mul, sizes, 1)*dtype_len(f.dtype)
        rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
        rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
        recv = IrecvCall([bufs, count, Macro(dtype_to_mpitype(f.dtype)),
                          fromrank, Integer(13), comm, rrecv])
        send = IsendCall([bufg, count, Macro(dtype_to_mpitype(f.dtype)),
                         torank, Integer(13), comm, rsend])

        # The -1 below is because an Iteration, by default, generates <=
        ncomms = Symbol(name='ncomms')
        iet = Iteration([recv, gather, send], dim, ncomms - 1)
        parameters = ([f, comm, msg, ncomms]) + list(fixed.values())
        return HaloUpdate('haloupdate%s' % key, iet, parameters)

    def _call_haloupdate(self, name, f, hse, msg):
        comm = f.grid.distributor._obj_comm
        args = [f, comm, msg, msg.npeers] + list(hse.loc_indices.values())
        return HaloUpdateCall(name, args)

    def _make_halowait(self, f, hse, key, *args, msg=None):
        cast = cast_mapper[(f.dtype, '*')]

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
        scatter = Call('scatter%s' % key, [cast(bufs)] + sizes + [f] + ofss)
        scatter = Conditional(CondNe(fromrank, Macro('MPI_PROC_NULL')), scatter)

        rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
        waitrecv = Call('MPI_Wait', [rrecv, Macro('MPI_STATUS_IGNORE')])
        rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
        waitsend = Call('MPI_Wait', [rsend, Macro('MPI_STATUS_IGNORE')])

        # The -1 below is because an Iteration, by default, generates <=
        ncomms = Symbol(name='ncomms')
        iet = Iteration([waitsend, waitrecv, scatter], dim, ncomms - 1)
        parameters = ([f] + list(fixed.values()) + [msg, ncomms])
        return Callable('halowait%d' % key, iet, 'void', parameters, ('static',))

    def _call_halowait(self, name, f, hse, msg):
        args = [f] + list(hse.loc_indices.values()) + [msg, msg.npeers]
        return HaloWaitCall(name, args)

    def _make_wait(self, *args, **kwargs):
        return

    def _call_wait(self, *args):
        return

    def _make_remainder(self, hs, key, callcompute, region):
        assert callcompute.is_Call

        dim = Dimension(name='i')
        region_i = IndexedPointer(region, dim)

        dynamic_args_mapper = {}
        for i in hs.arguments:
            if i.is_Dimension:
                dynamic_args_mapper[i] = (FieldFromComposite(i.min_name, region_i),
                                          FieldFromComposite(i.max_name, region_i))
            else:
                dynamic_args_mapper[i] = (FieldFromComposite(i.name, region_i),)

        iet = callcompute._rebuild(dynamic_args_mapper=dynamic_args_mapper)
        # The -1 below is because an Iteration, by default, generates <=
        iet = Iteration(iet, dim, region.nregions - 1)

        return make_efunc('remainder%d' % key, iet)


class Diag2HaloExchangeBuilder(Overlap2HaloExchangeBuilder):

    """
    A DiagHaloExchangeBuilder which uses preallocated buffers for comms
    as in Overlap2HaloExchange builder.

    Generates:

        haloupdate()
        halowait()
        compute()
    """

    def _make_region(self, hs, key):
        return

    def _make_compute(self, hs, key, *args):
        return

    def _call_compute(self, hs, compute, *args):
        return

    def _make_remainder(self, hs, key, callcompute, region):
        # Just a dummy value, other than a Callable, so that we enter `_call_remainder`
        return hs.body

    def _call_remainder(self, remainder):
        return remainder


class FullHaloExchangeBuilder(Overlap2HaloExchangeBuilder):

    """
    An Overlap2HaloExchangeBuilder which generates explicit Calls to MPI_Test
    poking the MPI runtime to advance communication while computing.

    Generates:

        haloupdate()
        compute_core()
        halowait()
        remainder()
    """

    def _make_compute(self, hs, key, msgs, callpoke):
        if hs.body.is_Call:
            return None
        else:
            mapper = {i: List(body=[callpoke, i]) for i in
                      FindNodes(ExpressionBundle).visit(hs.body)}
            iet = Transformer(mapper).visit(hs.body)
            return make_efunc('compute%d' % key, iet, hs.arguments)

    def _make_poke(self, hs, key, msgs):
        lflag = Symbol(name='lflag')
        gflag = Symbol(name='gflag')

        # Init flags
        body = [Expression(DummyEq(lflag, 0)),
                Expression(DummyEq(gflag, 1))]

        # For each msg, build an Iteration calling MPI_Test on all peers
        for msg in msgs:
            dim = Dimension(name='i')
            msgi = IndexedPointer(msg, dim)

            rrecv = Byref(FieldFromComposite(msg._C_field_rrecv, msgi))
            testrecv = Call('MPI_Test', [rrecv, Byref(lflag), Macro('MPI_STATUS_IGNORE')])

            rsend = Byref(FieldFromComposite(msg._C_field_rsend, msgi))
            testsend = Call('MPI_Test', [rsend, Byref(lflag), Macro('MPI_STATUS_IGNORE')])

            update = AugmentedExpression(DummyEq(gflag, lflag), operation='&')

            body.append(Iteration([testsend, update, testrecv, update],
                                  dim, msg.npeers - 1))

        body.append(Return(gflag))

        return make_efunc('pokempi%d' % key, List(body=body), retval='int')

    def _call_poke(self, poke):
        return Prodder(poke.name, poke.parameters, single_thread=True, periodic=True)


mpi_registry = {
    True: BasicHaloExchangeBuilder,
    'basic': BasicHaloExchangeBuilder,
    'diag': DiagHaloExchangeBuilder,
    'diag2': Diag2HaloExchangeBuilder,
    'overlap': OverlapHaloExchangeBuilder,
    'overlap2': Overlap2HaloExchangeBuilder,
    'full': FullHaloExchangeBuilder
}


# Callable sub-hierarchy


class MPICallable(Callable):

    def __init__(self, name, body, parameters):
        super(MPICallable, self).__init__(name, body, 'void', parameters, ('static',))


class CopyBuffer(MPICallable):
    pass


class SendRecv(MPICallable):

    def __init__(self, name, body, parameters, bufg, bufs):
        super(SendRecv, self).__init__(name, body, parameters)
        self.bufg = bufg
        self.bufs = bufs


class HaloUpdate(MPICallable):

    def __init__(self, name, body, parameters):
        super(HaloUpdate, self).__init__(name, body, parameters)


# Call sub-hierarchy

class IsendCall(Call):

    def __init__(self, arguments, **kwargs):
        super().__init__('MPI_Isend', arguments)


class IrecvCall(Call):

    def __init__(self, arguments, **kwargs):
        super().__init__('MPI_Irecv', arguments)


class MPICall(Call):
    pass


class HaloUpdateCall(MPICall):
    pass


class HaloWaitCall(MPICall):
    pass


class RemainderCall(MPICall):
    pass


class MPIList(List):
    pass


class HaloUpdateList(MPIList):
    pass


class HaloWaitList(MPIList):
    pass


# Types sub-hierarchy


class MPIStatusObject(LocalObject):

    dtype = type('MPI_Status', (c_void_p,), {})


class MPIRequestObject(LocalObject):

    dtype = type('MPI_Request', (c_void_p,), {})


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

    fields = [
        (_C_field_bufs, c_void_p),
        (_C_field_bufg, c_void_p),
        (_C_field_sizes, POINTER(c_int)),
        (_C_field_rrecv, c_mpirequest_p),
        (_C_field_rsend, c_mpirequest_p),
    ]

    __rargs__ = ('name', 'target', 'halos')

    def __init__(self, name, target, halos):
        self._target = target
        self._halos = halos

        super().__init__(name, 'msg', self.fields)

        # Required for buffer allocation/deallocation before/after jumping/returning
        # to/from C-land
        self._allocator = None
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
    def target(self):
        return self._target

    @property
    def halos(self):
        return self._halos

    @property
    def npeers(self):
        return len(self._halos)

    def _arg_defaults(self, allocator, alias=None):
        # Lazy initialization if `allocator` is necessary as the `allocator`
        # type isn't really known until an Operator is constructed
        self._allocator = allocator

        target = alias or self.target
        for i, halo in enumerate(self.halos):
            entry = self.value[i]

            # Buffer size for this peer
            shape = []
            for dim, side in zip(*halo):
                try:
                    shape.append(getattr(target._size_owned[dim], side.name))
                except AttributeError:
                    assert side is CENTER
                    shape.append(target._size_domain[dim])
            entry.sizes = (c_int*len(shape))(*shape)

            # Allocate the send/recv buffers
            size = reduce(mul, shape)*dtype_len(target.dtype)
            ctype = dtype_to_ctype(target.dtype)
            entry.bufg, bufg_memfree_args = allocator._alloc_C_libcall(size, ctype)
            entry.bufs, bufs_memfree_args = allocator._alloc_C_libcall(size, ctype)

            # The `memfree_args` will be used to deallocate the buffer upon returning
            # from C-land
            self._memfree_args.extend([bufg_memfree_args, bufs_memfree_args])

        return {self.name: self.value}

    def _arg_values(self, args=None, **kwargs):
        return self._arg_defaults(
            args.allocator,
            alias=kwargs.get(self.target.name, self.target)
        )

    def _arg_apply(self, *args, **kwargs):
        self._C_memfree()


class MPIMsgEnriched(MPIMsg):

    _C_field_ofss = 'ofss'
    _C_field_ofsg = 'ofsg'
    _C_field_from = 'fromrank'
    _C_field_to = 'torank'

    fields = MPIMsg.fields + [
        (_C_field_ofss, POINTER(c_int)),
        (_C_field_ofsg, POINTER(c_int)),
        (_C_field_from, c_int),
        (_C_field_to, c_int)
    ]

    def _arg_defaults(self, allocator, alias=None):
        super()._arg_defaults(allocator, alias)

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

    __rargs__ = ('prefix', 'key', 'arguments', 'owned')

    def __init__(self, prefix, key, arguments, owned):
        self._prefix = prefix
        self._key = key
        self._owned = owned

        # Sorting for deterministic codegen
        self._arguments = sorted(arguments, key=lambda i: i.name)

        name = "%s%d" % (prefix, key)
        pname = "region%d" % key

        fields = []
        for i in self.arguments:
            if i.is_Dimension:
                fields.append((i.min_name, c_int))
                fields.append((i.max_name, c_int))
            else:
                fields.append((i.name, c_int))

        super(MPIRegion, self).__init__(name, pname, fields)

    def __value_setup__(self, dtype, value):
        # We eventually produce an array of `struct region` that is as big as
        # the number of OWNED sub-regions we have to compute to complete a
        # halo update
        return (dtype._type_*self.nregions)()

    @property
    def arguments(self):
        return self._arguments

    @property
    def prefix(self):
        return self._prefix

    @property
    def key(self):
        return self._key

    @property
    def owned(self):
        return self._owned

    @property
    def nregions(self):
        return len(self.owned)

    def _arg_values(self, args=None, **kwargs):
        values = self._arg_defaults()
        for i, (_, mapper) in enumerate(self.owned):
            entry = values[self.name][i]
            for a in self.arguments:
                if a.is_Dimension:
                    a_m, a_M = mapper[a]
                    setattr(entry, a.min_name, subs_op_args(a_m, args))
                    setattr(entry, a.max_name, subs_op_args(a_M, args))
                else:
                    try:
                        setattr(entry, a.name, subs_op_args(mapper[a][0], args))
                    except AttributeError:
                        setattr(entry, a.name, mapper[a][0])
        return values
