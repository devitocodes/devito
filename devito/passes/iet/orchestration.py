from collections import namedtuple

import cgen as c
from sympy import Or
import numpy as np

from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, ElementalFunction, Expression, List,
                           PointerCast, SyncSpot, While, FindNodes, LocalExpression,
                           Transformer, derive_parameters, make_efunc)
from devito.ir.support import Forward
from devito.passes.iet.engine import iet_pass
from devito.symbolics import CondEq, DefFunction, FieldFromComposite, ListInitializer
from devito.tools import as_mapper, as_list, filter_ordered, filter_sorted
from devito.types import (STDThread, WaitLock, WithLock, FetchWait, FetchWaitPrefetch,
                          Delete)

__init__ = ['Orchestrator']


class Orchestrator(object):

    """
    Coordinate host and device in a heterogeneous system (e.g., CPU and GPU).
    This boils down to introducing data transfers, synchronizations, asynchronous
    computation and so on.
    """

    _Parallelizer = None
    """
    To be specified by the subclasses. This is used to generate IETs for the
    data transfers between host and device.
    """

    def __init__(self, sregistry):
        self.sregistry = sregistry

        if self._Parallelizer is None:
            raise NotImplementedError

    @property
    def _P(self):
        """Shortcut for self._Parallelizer."""
        return self._Parallelizer

    def _make_waitlock(self, iet, sync_ops, pieces):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.target.name for s in sync_ops)),
            body=While(Or(*[CondEq(s.handle, 0) for s in sync_ops])),
            footer=c.Line()
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet

    def _make_withlock(self, iet, sync_ops, pieces):
        thread = STDThread(self.sregistry.make_name(prefix='thread'))
        pieces.threads.append(thread)
        queueid = len(pieces.threads)

        threadwait = List(
            body=Conditional(DefFunction(FieldFromComposite('joinable', thread)),
                             Call(FieldFromComposite('join', thread)))
        )

        setlock = []
        actions = []
        for s in sync_ops:
            setlock.append(Expression(DummyEq(s.handle, 0)))

            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]
            actions.append(List(
                header=self._P._map_update_wait_host(s.target, imask, queueid=queueid),
                body=Expression(DummyEq(s.handle, 1))
            ))

        # Turn `iet` into an ElementalFunction so that it can be
        # executed asynchronously by `threadhost`
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = (List(body=actions, footer=c.Line()),) + iet.body
        efunc = make_efunc(name, List(body=body))
        pieces.efuncs.append(efunc)

        # Replace `iet` with a call to `efunc` plus locking
        iet = List(
            header=c.Comment("Wait for %s to be available again" % thread),
            body=[threadwait] + [List(
                header=c.Line(),
                body=setlock + [List(
                    header=[c.Line(), c.Comment("Spawn %s to perform the copy" % thread)],
                    body=Call('std::thread', efunc.make_call(is_indirect=True),
                              retobj=thread,)
                )]
            )]
        )

        # Initialize the locks
        locks = sorted({s.lock for s in sync_ops}, key=lambda i: i.name)
        for i in locks:
            values = np.ones(i.shape, dtype=np.int32).tolist()
            pieces.init.append(LocalExpression(DummyEq(i, ListInitializer(values))))

        # Final wait before jumping back to Python land
        pieces.finalize.append(List(
            header=c.Comment("Wait for completion of %s" % thread),
            body=threadwait
        ))

        return iet

    def _make_fetchwait(self, iet, sync_ops, pieces):
        # Construct fetches
        fetches = []
        for s in sync_ops:
            fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetches.append(self._P._map_to(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=fetches, body=iet)

        return iet

    def _make_fetchwaitprefetch(self, iet, sync_ops, pieces):
        thread = STDThread(self.sregistry.make_name(prefix='thread'))
        pieces.threads.append(thread)
        queueid = len(pieces.threads)

        threadwait = Call(FieldFromComposite('join', thread))

        fetches = []
        prefetches = []
        presents = []
        for s in sync_ops:
            if s.direction is Forward:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                fsize = s.function._C_get_field(FULL, s.dim).size
                fc_cond = fc + (s.size - 1) < fsize
                pfc = s.fetch + 1
                pfc_cond = pfc + (s.size - 1) < fsize
            else:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_max)
                fc_cond = fc >= 0
                pfc = s.fetch - 1
                pfc_cond = pfc >= 0

            # Construct fetch IET
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetch = List(header=self._P._map_to_wait(s.function, imask, queueid))
            fetches.append(Conditional(fc_cond, fetch))

            # Construct present clauses
            imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            presents.extend(as_list(self._P._map_present(s.function, imask)))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            prefetch = List(header=self._P._map_to_wait(s.function, imask, queueid))
            prefetches.append(Conditional(pfc_cond, prefetch))

        functions = filter_ordered(s.function for s in sync_ops)
        casts = [PointerCast(f) for f in functions]

        # Turn init IET into an efunc
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=casts + fetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        efunc = ElementalFunction(name, body, 'void', parameters)
        pieces.efuncs.append(efunc)

        # Call init IET
        efunc_call = efunc.make_call(is_indirect=True)
        pieces.init.append(List(
            header=c.Comment("Spawn %s to initialize data" % thread),
            body=Call('std::thread', efunc_call, retobj=thread),
            footer=c.Line()
        ))

        # Turn prefetch IET into an efunc
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(body=casts + prefetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        efunc = ElementalFunction(name, body, 'void', parameters)
        pieces.efuncs.append(efunc)

        # Call prefetch IET
        efunc_call = efunc.make_call(is_indirect=True)
        call = Call('std::thread', efunc_call, retobj=thread)

        # Glue together all the new IET pieces
        iet = List(
            header=[c.Line(),
                    c.Comment("Wait for %s to be available again" % thread)],
            body=[threadwait, List(
                header=[c.Line()] + presents,
                body=(iet, List(
                    header=[c.Line(), c.Comment("Spawn %s to prefetch data" % thread)],
                    body=call,
                    footer=c.Line()
                ),)
            )]
        )

        # Final wait before jumping back to Python land
        pieces.finalize.append(List(
            header=c.Comment("Wait for completion of %s" % thread),
            body=threadwait
        ))

        return iet

    def _make_delete(self, iet, sync_ops, pieces):
        # Construct deletion clauses
        deletions = []
        for s in sync_ops:
            if s.dim.is_Custom:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                imask = [(fc, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            else:
                imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            deletions.append(self._P._map_delete(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=c.Line(), body=iet, footer=[c.Line()] + deletions)

        return iet

    @iet_pass
    def process(self, iet):

        def key(s):
            # The SyncOps are to be processed in the following order
            return [WaitLock, WithLock, Delete, FetchWait, FetchWaitPrefetch].index(s)

        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            FetchWait: self._make_fetchwait,
            FetchWaitPrefetch: self._make_fetchwaitprefetch,
            Delete: self._make_delete
        }

        sync_spots = FindNodes(SyncSpot).visit(iet)

        if not sync_spots:
            return iet, {}

        pieces = namedtuple('Pieces', 'init finalize efuncs threads')([], [], [], [])

        subs = {}
        for n in sync_spots:
            mapper = as_mapper(n.sync_ops, lambda i: type(i))
            for _type in sorted(mapper, key=key):
                subs[n] = callbacks[_type](subs.get(n, n), mapper[_type], pieces)

        iet = Transformer(subs).visit(iet)

        # Add initialization and finalization code
        init = List(body=pieces.init, footer=c.Line())
        finalize = List(header=c.Line(), body=pieces.finalize)
        iet = iet._rebuild(body=(init,) + iet.body + (finalize,))

        return iet, {'efuncs': pieces.efuncs, 'includes': ['thread']}
