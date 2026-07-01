from collections import defaultdict
from functools import singledispatch

from sympy import Expr, Mod, true

from devito.ir import (
    Backward, Forward, GuardBoundNext, PrefetchUpdate, Queue, ReleaseLock, SyncArray,
    WaitLock, WithLock, normalize_syncs
)
from devito.passes.clusters.utils import in_critical_region, is_memcpy
from devito.symbolics import IntDiv, retrieve_dimensions, retrieve_terminals, uxreplace
from devito.tools import OrderedSet, is_integer, timed_pass
from devito.types import CustomDimension, Lock, VirtualDimension

__all__ = ['memcpy_prefetch', 'tasking']


def async_trigger(c, dims):
    """
    Return the Dimension in `c`'s IterationSpace that triggers the
    asynchronous execution of `c`.
    """
    if not dims:
        return None
    else:
        key = lambda d: not d._defines.intersection(dims)
        ispace = c.ispace.project(key)
        return ispace.innermost.dim


def keys(key0):
    """
    Callbacks for `tasking` and `memcpy_prefetch` given a user-defined `key`.
    """

    def task_key(c):
        return async_trigger(c, key0(c.scope.writes))

    def memcpy_key(c):
        if task_key(c):
            # Writes would take precedence over reads
            return None
        else:
            return async_trigger(c, key0(c.scope.reads))

    return task_key, memcpy_key


@timed_pass(name='tasking')
def tasking(clusters, key0, sregistry):
    """
    Turn a Cluster `c` into an asynchronous task if `c` writes to a Function `f`
    such that `key0(f) = (d_i, ..., d_n)`, where `(d_i, ..., d_n)` are the
    task Dimensions.
    """
    key, _ = keys(key0)
    return Tasking(key, sregistry).process(clusters)


class Tasking(Queue):

    """
    Carry out the bulk of `tasking`.
    """

    def __init__(self, key0, sregistry):
        self.key0 = key0
        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        dim = prefix[-1].dim

        locks = {}
        syncs = defaultdict(lambda: defaultdict(OrderedSet))
        for c0 in clusters:
            d = self.key0(c0)
            if d is not dim:
                continue
            g = c0.guards.get(d)
            # Explicit compute guards need no pipeline; memcpy clusters
            # still need WithLock for the copy-back sync
            if g is not None and not wraps_memcpy(c0):
                # An "explicit" guard is a plain reference to `d` (e.g., `d == K`)
                # without subsampling -- subsampling guards (containing `Mod`)
                # still require the standard async pipeline
                explicit = not g.has(Mod) and d in retrieve_terminals(g)
                if explicit:
                    continue
            protected = self._schedule_waitlocks(c0, d, clusters, locks, syncs)
            self._schedule_withlocks(c0, d, protected, locks, syncs)

        processed = [c.rebuild(syncs={**c.syncs, **syncs[c]}) for c in clusters]

        return processed

    def _schedule_waitlocks(self, c0, d, clusters, locks, syncs):
        # Prevent future writes to interfere with a task by waiting on a lock
        may_require_lock = {f for f in c0.scope.reads if f.is_AbstractFunction}
        # Sort for deterministic code generation
        may_require_lock = sorted(may_require_lock, key=lambda i: i.name)

        protected = defaultdict(set)
        for c1 in clusters:
            offset = int(clusters.index(c1) <= clusters.index(c0))

            for target in may_require_lock:
                try:
                    writes = c1.scope.writes[target]
                except KeyError:
                    # No read-write dependency, ignore
                    continue

                try:
                    if all(w.aindices[d].is_Stepping for w in writes) or \
                       all(w.aindices[d].is_Modulo for w in writes):
                        sz = target.shape_allocated[d]
                        assert is_integer(sz)
                        ld = CustomDimension(name='ld', symbolic_size=sz, parent=d)
                    elif all(w[d] == 0 for w in writes):
                        # Special case, degenerates to scalar lock
                        raise KeyError
                    else:
                        # Functions over non-stepping Dimensions need no lock
                        continue
                except (AttributeError, KeyError):
                    # Would degenerate to a scalar, but we rather use a lock
                    # of size 1 for simplicity
                    ld = CustomDimension(name='ld', symbolic_size=1, parent=d)

                try:
                    lock = locks[target]
                except KeyError:
                    name = self.sregistry.make_name(prefix='lock')
                    lock = locks[target] = Lock(name=name, dimensions=ld)

                for w in writes:
                    try:
                        index = w[d]
                        logical_index = index + offset
                    except TypeError:
                        assert ld.symbolic_size == 1
                        index = 0
                        logical_index = 0

                    if logical_index in protected[target]:
                        continue

                    # Critical regions preempt WaitLocks
                    c2 = in_critical_region(c1, clusters) or c1

                    syncs[c2][d].add(WaitLock(lock[index], target))
                    protected[target].add(logical_index)

        return protected

    def _schedule_withlocks(self, c0, d, protected, locks, syncs):
        for target in protected:
            lock = locks[target]

            indices = sorted({r[d] for r in c0.scope.reads[target]})
            if indices == [None]:
                # `lock` is protecting a Function which isn't defined over `d`
                # E.g., `d=time` and the protected function is `a(x, y)`
                assert lock.size == 1
                indices = [0]

            if wraps_memcpy(c0):
                e = c0.exprs[0]
                function = e.lhs.function
                findex = e.lhs.indices[d]
            else:
                # Only for backwards compatibility (e.g., tasking w/o buffering)
                function = None
                findex = None

            for i in indices:
                syncs[c0][d].update([
                    ReleaseLock(lock[i], target),
                    WithLock(lock[i], target, i, function, findex, d)
                ])


@timed_pass(name='memcpy_prefetch')
def memcpy_prefetch(clusters, key0, sregistry):
    """
    A special form of `tasking` for optimized, asynchronous memcpy-like operations.
    Unlike `tasking`, `key0` is here applied to the reads of a given Cluster.
    """
    _, key = keys(key0)
    actions = defaultdict(Actions)
    bounds = {}

    for c in clusters:
        d = key(c)
        if d is None:
            continue

        if not wraps_memcpy(c):
            continue

        if c.properties.is_prefetchable(d._defines):
            _actions_from_update_memcpy(c, d, bounds, clusters, actions, sregistry)
        elif d.is_Custom and is_integer(c.ispace[d].size):
            _actions_from_init(c, d, actions)
        else:
            # Explicit memcpy: no prefetch, just a sync device update
            _actions_from_sync(c, d, actions)

    # Attach the computed Actions
    processed = []
    for c in clusters:
        v = actions[c]

        if v.drop:
            assert not v.syncs
            continue
        elif v.syncs:
            processed.append(c.rebuild(syncs=normalize_syncs(c.syncs, v.syncs)))
        else:
            processed.append(c)

        if v.insert:
            processed.extend(v.insert)

    return processed


def _actions_from_init(c, d, actions):
    e = c.exprs[0]
    function = e.rhs.function
    target = e.lhs.function

    tindex = e.lhs.indices[d]
    findex = e.rhs.indices[d]

    size = target.shape[d]
    assert is_integer(size)

    actions[c].syncs[d].append(
        SyncArray(None, target, tindex, function, findex, d, size)
    )


def _actions_from_sync(c, d, actions):
    """Emit a SyncArray so the device buffer is updated after the host fill."""
    e = c.exprs[0]
    function = e.rhs.function
    target = e.lhs.function

    tindex = e.lhs.indices[d]
    findex = e.rhs.indices[d]

    actions[c].syncs[d].append(
        SyncArray(None, target, tindex, function, findex, d, 1)
    )


def _actions_from_update_memcpy(c, d, bounds, clusters, actions, sregistry):
    pd = d.root  # E.g., `vd -> time`
    direction = c.ispace[pd].direction

    e = c.exprs[0]
    f = e.rhs.function
    target = e.lhs.function

    fetch = e.rhs.indices[d]
    fshift = {Forward: 1, Backward: -1}.get(direction, 0)
    findex = eval_next_index(fetch, pd, fshift)

    # Maximum allowed access along d
    if f.dimensions[d].is_Conditional:
        nslot = f.dimension_shape[d]
        v = f.dimensions[d].symbolic_factor
        fd_max = bounds.setdefault(d, v * (nslot - 1))
    else:
        fd_max = bounds.setdefault(d, f.dimension_shape[d] - 1)

    # If fetching into e.g. `ub[t1]` we might need to prefetch into e.g. `ub[t0]`
    tindex0 = e.lhs.indices[d]
    if is_integer(tindex0) or isinstance(tindex0, IntDiv):
        tindex = tindex0
    else:
        assert tindex0.is_Modulo
        mapper = {(i.offset % i.modulo): i for i in c.sub_iterators[pd]}
        toffset = tindex0.offset + 1 if direction is Forward else tindex0.offset - 1
        try:
            tindex = mapper[toffset % tindex0.modulo]
        except KeyError:
            # This can happen if e.g. the underlying buffer has size K (because
            # e.g. the user has sent `async_degree=K`), but the actual computation
            # only uses K-N indices (e.g., indices={t-1,t} and K=5})
            name = sregistry.make_name(prefix='t')
            tindex = tindex0._rebuild(name, offset=toffset, origin=None)

    # We need a lock to synchronize the copy-in
    name = sregistry.make_name(prefix='lock')
    ld = CustomDimension(name='ld', symbolic_size=1, parent=d)
    lock = Lock(name=name, dimensions=ld)
    handle = lock[0]

    # Turn `c` into a prefetch Cluster `pc`
    expr = uxreplace(e, {tindex0: tindex, fetch: findex})

    ispace = c.ispace.augment({pd: tindex}) if tindex is not tindex0 else c.ispace
    # Insert a VirtualDimension nested under `pd` so that the access guard
    # (`guard0`) below can be evaluated only after the tindex bound guard
    # (`guard1`) has already been validated
    vdnext = VirtualDimension(name=f'vdnext_{d.name}', parent=pd)
    ispace = ispace.insert(pd, vdnext)

    guard0 = c.guards.get(d, true)._subs(fetch, findex)
    guard1 = GuardBoundNext(f.indices[d], e.rhs.indices[d], direction,
                            d_min=0, d_max=fd_max)

    # First guard1; then, if guard1 is valid, we can safely evaluate guard0,
    # which will then have valid indices into f
    # Check valid tindex first
    guards = c.guards.impose(d, guard1)
    # Then check valid access
    guards = guards.impose(vdnext, guard0)

    syncs = {d: [
        ReleaseLock(handle, target),
        PrefetchUpdate(handle, target, tindex, f, findex, d, 1, e.rhs)
    ]}
    syncs = {**c.syncs, **syncs}

    pc = c.rebuild(exprs=expr, ispace=ispace, guards=guards, syncs=syncs)

    # Wait before the first, prefetch after the last access to `target`, then
    # drop the memcpy `c`. `c` may be toposorted amid the readers, so scan them
    # all; count only reads over the streamed `pd`, not the buffer-init loop.
    reads = [c1 for c1 in clusters
             if c1 is not c and target in c1.scope.reads and pd in c1.ispace.itdims]
    assert reads
    first = reads[0]
    last = reads[-1]

    # Advance `last` past its loop nest so the prefetch follows it rather than
    # splitting it, e.g. severing an interpolation's store from its point loop
    nest_dims = set(last.ispace.itdims) - set(d._defines)
    for c1 in clusters[clusters.index(last)+1:]:
        if not nest_dims & set(c1.ispace.itdims):
            break
        last = c1

    actions[first].syncs[d].append(WaitLock(handle, target))
    actions[last].insert.append(pc)
    actions[c].drop = True

    return last, pc


class Actions:

    def __init__(self, drop=False, syncs=None, insert=None):
        self.drop = drop
        self.syncs = syncs or defaultdict(list)
        self.insert = insert or []


def wraps_memcpy(cluster):
    return len(cluster.exprs) == 1 and is_memcpy(cluster.exprs[0])


@singledispatch
def eval_next_index(expr, dim, dir):
    """
    Evaluate `expr` at the next iteration point along `dim` in the given
    `dir`-ection, where `dir` is `+1` for forward and `-1` for backward.

    For `IntDiv` subexpressions encoding subsampling, forward and backward
    fetches are treated asymmetrically since piecewise correction terms may
    already be applied at the current point.
    """
    return expr._subs(dim, dim + dir)


@eval_next_index.register(Expr)
def _(expr, dim, dir):
    if not expr.args:
        return expr._subs(dim, dim + dir)
    return expr.func(*[eval_next_index(a, dim, dir) for a in expr.args])


@eval_next_index.register(IntDiv)
def _(expr, dim, dir):
    """
    Handle forward and backward fetches separately to handle non-canonical index
    expressions of the form:

        t//factor + cond(t)

    where ``cond(t)`` is a piecewise correction term.

    The forward fetch advances to the next coarse-grained slot while evaluating
    the correction at the next time point:

        t//factor + cond(t)
            -> (t//factor + 1) + cond(t + 1)

    The backward fetch is not, in general, the inverse transformation obtained by
    replacing ``+1`` with ``-1``. The correction may already be applied at the
    current time point, causing the forward and backward fetches to be asymmetric.

    For example, with ``factor=2`` and ``cond(t) := (t == a)``, the index at
    ``t=a=3`` is:

        3//2 + 1 = 2

    while the previous index is:

        2//2 + 0 = 1

    A symmetric backward transformation would instead yield:

        3//2 - 1 + 0 = 0
    """
    dims = retrieve_dimensions(expr.lhs)
    assert len(dims) == 1
    ldim = dims.pop()
    if ldim._defines & dim._defines:
        if dir == 1:
            return expr + dir
        else:
            return expr._subs(dim, dim + dir)
    else:
        return expr
