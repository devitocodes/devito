from collections import defaultdict, namedtuple
from functools import cached_property
from itertools import chain

from sympy import S
import numpy as np

from devito.ir import (Cluster, Backward, Forward, GuardBound, Interval,
                       IntervalGroup, IterationSpace, Properties, Queue, Vector,
                       InitArray, lower_exprs, vmax, vmin)
from devito.exceptions import CompilationError
from devito.logger import warning
from devito.passes.clusters.utils import is_memcpy
from devito.symbolics import IntDiv, retrieve_functions, uxreplace
from devito.tools import (Stamp, as_mapper, as_tuple, filter_ordered, frozendict,
                          flatten, is_integer, timed_pass)
from devito.types import Array, CustomDimension, Eq, ModuloDimension

__all__ = ['buffering']


@timed_pass()
def buffering(clusters, key, sregistry, options, **kwargs):
    """
    Replace written Functions with Arrays. This gives the compiler more control
    over storage layout, data movement (e.g. between host and device), etc.

    Parameters
    ----------
    cluster : list of Cluster
        Input Clusters, subject of the optimization pass.
    key : callable, optional
        A callback that takes a Function as input and returns the Dimensions
        that should be buffered.
    sregistry : SymbolRegistry
        The symbol registry, to create unique names for buffers and Dimensions.
    options : dict
        The optimization options.
        Accepted: ['buf-async-degree'].
        * 'buf-async-degree': Specify the size of the buffer. By default, the
          buffer size is the minimal one, inferred from the memory accesses in
          the ``clusters`` themselves. An asynchronous degree equals to `k`
          means that the buffer will be enforced to size=`k` along the introduced
          ModuloDimensions. This might help relieving the synchronization
          overhead when asynchronous operations are used (these are however
          implemented by other passes).
        * 'buf-reuse': If True, the pass will try to reuse existing Buffers for
          different buffered Functions. By default, False.
    **kwargs
        Additional compilation options.
        Accepted: ['opt_init_onwrite', 'opt_buffer'].
        * 'opt_init_onwrite': By default, a written buffer does not trigger the
        generation of an initializing Cluster. With `opt_init_onwrite=True`,
        instead, the buffer gets initialized to zero.
        * 'opt_reuse': A callback that takes a buffering candidate `bf` as input
        and returns True if the pass can reuse pre-existing Buffers for
        buffering `bf`, which would otherwise default to False.
        * 'opt_buffer': A callback that takes a buffering candidate as input
        and returns a buffer, which would otherwise default to an Array.

    Examples
    --------
    Assume `callback(f)=[time]`, meaning that `f`, a Function defined over `time`,
    should be replaced by a buffer using a ModuloDimension in place of `time`.
    Consider the Eq below:

        Eq(u[time+1, x], u[time, x] + u[time-1, x] + 1)

    We see that `u(time, x)` is both read and written. Applying buffering boils
    down to:

        1. Prepending an Eq to initialize the buffer `ub`, i.e.

            Eq(ub[d, x], u[d, x])

        2. Appending an Eq to copy `ub` back into `u`

            Eq(u[time+1, x], ub[d+1, x])

        3. Replacing all other occurrences of `u` with `ub`

    So eventually we have three Eqs:

        Eq(ub[d, x], u[d, x]),
        Eq(ub[d+1, x], ub[d, x] + ub[d-1, x] + 1)
        Eq(u[time+1, x], ub[d+1, x])
    """
    if key is None:
        def key(f):
            if f.is_DiscreteFunction:
                return []
            else:
                return None
    assert callable(key)

    v1 = kwargs.get('opt_init_onwrite', False)
    if callable(v1):
        init_onwrite = v1
    else:
        init_onwrite = lambda f: v1

    options = dict(options)
    options.update({
        'buf-init-onwrite': init_onwrite,
        'buf-callback': kwargs.get('opt_buffer'),
        'buf-reuse': kwargs.get('opt_reuse', options['buf-reuse']),
    })

    # Escape hatch to selectively disable buffering
    if options['buf-async-degree'] == 0:
        return clusters

    # First we generate all the necessary buffers
    mapper = generate_buffers(clusters, key, sregistry, options)

    # Then we inject them into the Clusters. This involves creating the
    # initializing Clusters, and replacing the buffered Functions with the buffers
    clusters = InjectBuffers(mapper, sregistry, options).process(clusters)

    return clusters


class InjectBuffers(Queue):

    # NOTE: We need to use a Queue because with multi-buffering we will need
    # to process the `db0` and `time` Dimensions separately for `last_idx` and
    # `first_idx` to work correctly

    def __init__(self, mapper, sregistry, options):
        super().__init__()

        self.mapper = mapper

        self.sregistry = sregistry
        self.options = options

    def process(self, clusters):
        return self._process_fatd(clusters, 1)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters
        d = prefix[-1].dim

        key = lambda f, *args: f in self.mapper
        bfmap = map_buffered_functions(clusters, key)

        # A BufferDescriptor is a simple data structure storing additional
        # information about a buffer, harvested from the subset of `clusters`
        # that access it
        descriptors = {b: BufferDescriptor(f, b, bfmap[f])
                       for f, b in self.mapper.items()
                       if f in bfmap}

        # Are we inside the right `d`?
        descriptors = {b: v for b, v in descriptors.items() if d in v.itdims}

        if not descriptors:
            return clusters

        # Then we create the initializing Clusters when necessary
        init = init_buffers(descriptors, self.options)

        # Create all the necessary ModuloDimensions to step through the buffer
        mds = make_mds(descriptors, prefix, self.sregistry)
        subiters = as_mapper(mds.values(), lambda i: i.root)

        # Substitution rules to replace buffered Functions with buffers
        # E.g., `usave[time+1, x+1, y+1] -> ub0[t1, x+1, y+1]`
        subs = {}
        for b, v in descriptors.items():
            accesses = chain(*[c.scope[v.f] for c in v.clusters])
            index_mapper = {i: mds[(v.xd, i)] for i in v.indices}
            for a in accesses:
                subs[a.access] = b.indexed[[index_mapper.get(i, i) for i in a]]

        processed = []
        for c in clusters:
            # If a buffer is read but never written, then we need to add
            # an Eq to step through the next slot
            # E.g., `ub[0, x] = usave[time+2, x]`
            for b, v in descriptors.items():
                if not v.is_readonly:
                    continue
                if c is not v.firstread:
                    continue

                idxf = v.last_idx
                idxb = mds[(v.xd, idxf)]

                lhs = v.b.indexify()._subs(v.xd, idxb)
                rhs = v.f.indexify()._subs(v.dim, idxf)

                expr = Eq(lhs, rhs)
                expr = lower_exprs(expr)

                ispace = v.step_to
                ispace = ispace.augment(subiters).augment(c.sub_iterators)

                if v.xd in ispace.itdims:
                    guards = c.guards.xandg(v.xd, GuardBound(0, v.first_idx.f))
                else:
                    guards = c.guards

                properties = c.properties.sequentialize(d)
                if not isinstance(d, BufferDimension):
                    properties = properties.prefetchable(d)
                # `c` may be a HaloTouch Cluster, so with no vision of the `bdims`
                properties = properties.parallelize(v.bdims).affine(v.bdims)

                syncs = c.syncs

                processed.append(Cluster(expr, ispace, guards, properties, syncs))

            # Substitute the buffered Functions with the buffers
            exprs = [uxreplace(e, subs) for e in c.exprs]
            ispace = c.ispace.augment(subiters)
            properties = c.properties.sequentialize(d)
            processed.append(
                c.rebuild(exprs=exprs, ispace=ispace, properties=properties)
            )

            # Append the copy-back if `c` is the last-write of some buffers
            # E.g., `usave[time+1, x] = ub[t1, x]`
            for b, v in descriptors.items():
                if v.is_readonly:
                    continue
                if c is not v.lastwrite:
                    continue

                idxf = v.last_idx
                idxb = mds[(v.xd, idxf)]

                lhs = v.f.indexify()._subs(v.dim, idxf)
                rhs = v.b.indexify()._subs(v.xd, idxb)

                expr = Eq(lhs, rhs)
                expr = lower_exprs(uxreplace(expr, v.subdims_mapper))

                ispace = v.written
                ispace = ispace.augment(subiters).augment(c.sub_iterators)

                if v.xd in ispace.itdims:
                    guards = c.guards.xandg(v.xd, GuardBound(0, v.first_idx.f))
                else:
                    guards = c.guards

                properties = c.properties.sequentialize(d)

                syncs = c.syncs

                processed.append(Cluster(expr, ispace, guards, properties, syncs))

        # Lift {write,read}-only buffers into separate IterationSpaces
        if not self.options['fuse-tasks']:
            processed = self._optimize(processed, descriptors)

        if self.options['buf-reuse']:
            init, processed = self._reuse(init, processed, descriptors)

        return init + processed

    def _optimize(self, clusters, descriptors):
        for b, v in descriptors.items():
            if v.is_writeonly:
                # `b` might be written by multiple, potentially mutually
                # exclusive, equations. For example, two equations that have or
                # will have complementary guards, hence only one will be
                # executed. In such a case, we can split the equations over
                # separate IterationSpaces
                key0 = lambda: Stamp()
            elif v.is_readonly:
                # `b` is read multiple times -- this could just be the case of
                # coupled equations, so we more cautiously perform a
                # "buffer-wise" splitting of the IterationSpaces (i.e., only
                # relevant if there are at least two read-only buffers)
                stamp = Stamp()
                key0 = lambda: stamp
            else:
                continue

            processed = []
            for c in clusters:
                if b not in c.functions:
                    processed.append(c)
                    continue

                key1 = lambda d: not d._defines & v.dim._defines
                dims = c.ispace.project(key1).itdims
                ispace = c.ispace.lift(dims, key0())
                processed.append(c.rebuild(ispace=ispace))

            clusters = processed

        return clusters

    def _reuse(self, init, clusters, descriptors):
        """
        Reuse existing Buffers for buffering candidates.
        """
        buf_reuse = self.options['buf-reuse']

        if callable(buf_reuse):
            cbk = lambda v: [i for i in v if buf_reuse(descriptors[i])]
        else:
            cbk = lambda v: v

        mapper = as_mapper(descriptors, key=lambda b: b._signature)
        mapper = {k: cbk(v) for k, v in mapper.items() if cbk(v)}

        subs = {}
        drop = set()
        for reusable in mapper.values():
            retain = reusable.pop(0)
            drop.update(reusable)

            name = self.sregistry.make_name(prefix='r')
            b = retain.func(name=name)

            for i in (retain, *reusable):
                subs.update({i: b, i.indexed: b.indexed})

        processed = []
        for c in init:
            if set(c.scope.writes) & drop:
                continue

            exprs = [uxreplace(e, subs) for e in c.exprs]
            processed.append(c.rebuild(exprs=exprs))
        init = processed

        processed = []
        for c in clusters:
            exprs = [uxreplace(e, subs) for e in c.exprs]
            processed.append(c.rebuild(exprs=exprs))

        return init, processed


Map = namedtuple('Map', 'b f')


class BufferDimension(CustomDimension):
    pass


def generate_buffers(clusters, key, sregistry, options, **kwargs):
    async_degree = options['buf-async-degree']
    callback = options['buf-callback']

    # {candidate buffered Function -> [Clusters that access it]}
    bfmap = map_buffered_functions(clusters, key)

    # Sort for deterministic code generation
    bfmap = {i: bfmap[i] for i in sorted(bfmap, key=lambda i: i.name)}

    # {buffered Function -> Buffer}
    xds = {}
    mapper = {}
    for f, clusters in bfmap.items():
        exprs = flatten(c.exprs for c in clusters)

        bdims = key(f, exprs)

        dims = [d for d in f.dimensions if d not in bdims]
        if len(dims) != 1:
            raise CompilationError(f"Unsupported multi-dimensional `buffering` "
                                   f"required by `{f}`")
        dim = dims.pop()

        if is_buffering(exprs):
            # Multi-level buffering
            # NOTE: a bit rudimentary (we could go through the exprs one by one
            # instead), but it's much shorter this way
            buffers = [f for f in retrieve_functions(exprs) if f.is_Array]
            assert len(buffers) == 1, "Unexpected form of multi-level buffering"
            buffer, = buffers
            xd = buffer.indices[dim]
        else:
            size = infer_buffer_size(f, dim, clusters)

            if async_degree is not None:
                if async_degree < size:
                    warning("Ignoring provided asynchronous degree as it'd be "
                            "too small for the required buffer (provided %d, "
                            "but need at least %d for `%s`)"
                            % (async_degree, size, f.name))
                else:
                    size = async_degree

            # A special CustomDimension to use in place of `dim` in the buffer
            try:
                xd = xds[(dim, size)]
            except KeyError:
                name = sregistry.make_name(prefix='db')
                xd = xds[(dim, size)] = BufferDimension(name, 0, size-1, size, dim)

        # The buffer dimensions
        dimensions = list(f.dimensions)
        assert dim in f.dimensions
        dimensions[dimensions.index(dim)] = xd

        # Finally create the actual buffer
        cls = callback or Array
        name = sregistry.make_name(prefix='%sb' % f.name)
        mapper[f] = cls(name=name, dimensions=dimensions, dtype=f.dtype,
                        grid=f.grid, halo=f.halo, space='mapped', mapped=f, f=f)

    return mapper


def map_buffered_functions(clusters, key):
    """
    Map each candidate Function to the Clusters that access it.
    """
    bfmap = defaultdict(list)
    for c in clusters:
        for f in c.functions:
            if key(f, c.exprs):
                bfmap[f].append(c)

    return frozendict(bfmap)


class BufferDescriptor:

    def __init__(self, f, b, clusters):
        self.f = f
        self.b = b
        self.clusters = clusters

        self.xd, = b.find(BufferDimension)
        self.bdims = tuple(d for d in b.dimensions if d is not self.xd)

        self.dim = f.indices[self.xd]

        self.indices = extract_indices(f, self.dim, clusters)

        # The IterationSpace within which the buffer will be accessed
        # NOTE: The `key` is to avoid Clusters including `f` but not directly
        # using it in an expression, such as HaloTouch Clusters
        key = lambda c: any(i in c.ispace.dimensions for i in self.bdims)
        ispaces = {c.ispace for c in clusters if key(c)}

        if len(ispaces) > 1:
            # Best effort to make buffering work in the presence of multiple
            # IterationSpaces
            stamp = Stamp()
            ispaces = {i.lift(self.bdims, v=stamp) for i in ispaces}

            if len(ispaces) > 1:
                raise CompilationError("Unsupported `buffering` over different "
                                       "IterationSpaces")

        assert len(ispaces) == 1, "Unexpected form of `buffering`"
        self.ispace = ispaces.pop()

    def __repr__(self):
        return "Descriptor[%s -> %s]" % (self.f, self.b)

    @property
    def size(self):
        return self.xd.symbolic_size

    @property
    def exprs(self):
        return flatten(c.exprs for c in self.clusters)

    @property
    def itdims(self):
        """
        The Dimensions defining an IterationSpace in which the buffer
        may be accessed.
        """
        return (self.xd, self.dim.root)

    @cached_property
    def subdims_mapper(self):
        return {d.root: d for d in self.ispace.itdims if d.is_AbstractSub}

    @cached_property
    def firstread(self):
        for c in self.clusters:
            if c.scope.reads.get(self.f):
                return c
        return None

    @cached_property
    def lastwrite(self):
        for c in reversed(self.clusters):
            if c.scope.writes.get(self.f):
                return c
        return None

    @property
    def is_read(self):
        # Wild Clusters, and in particular HaloTouch Clusters, may contain mock
        # read accesses to self's buffered Function (`self.f`), which we must
        # ignore since here we're determining whether `self.f` is actually read
        return any(not c.is_wild for c in self.clusters if c.scope.reads.get(self.f))

    @property
    def is_write(self):
        return self.lastwrite is not None

    @property
    def is_readonly(self):
        return self.is_read and not self.is_write

    @property
    def is_writeonly(self):
        return self.is_write and not self.is_read

    @property
    def is_forward_buffering(self):
        return self.ispace[self.dim].direction is Forward

    @property
    def is_double_buffering(self):
        return is_buffering(self.exprs)

    @cached_property
    def write_to(self):
        """
        The `write_to` IterationSpace, that is the iteration space that must be
        iterated over in order to initialize the buffer.
        """
        ispace = self.ispace.switch(self.dim.root, self.xd, Forward)

        # We need everything, not just the SubDomain, as all points in principle
        # might be accessed through a stencil
        ispace = ispace.promote(lambda d: d.is_AbstractSub, mode='total')

        # Analogous to the above, we need to include the halo region as well
        ihalo = IntervalGroup([
            Interval(i.dim, -h.left, h.right, i.stamp)
            for i, h in zip(ispace, self.b._size_halo)
        ])

        ispace = IterationSpace.union(ispace, IterationSpace(ihalo))

        return ispace

    @cached_property
    def step_to(self):
        """
        The `step_to` IterationSpace, that is the iteration space that must be
        iterated over to update the buffer with the buffered Function values.
        """
        # May be `db0` (e.g., for double buffering) or `time`
        dim = self.ispace[self.dim].dim

        if self.is_forward_buffering:
            direction = Forward
        else:
            direction = Backward

        return self.write_to.switch(self.xd, dim, direction)

    @cached_property
    def written(self):
        """
        The `written` IterationSpace, that is the iteration space that must be
        iterated over in order to read all of the written buffer values.
        """
        return self.ispace

    @cached_property
    def last_idx(self):
        """
        The "last index" is the index that corresponds to the last slot that
        must be read from the buffered Function.

        Examples
        --------

            * `time+1` in the case of `foo(u[time-1], u[time], u[time+1])`
               with a forward-propagating `time` Dimension;
            * `time-1` in the case of `foo(u[time-1], u[time], u[time+1])`
               with a backwards-propagating `time` Dimension.
        """
        func = vmax if self.is_forward_buffering else vmin
        return func(*[Vector(i) for i in self.indices])[0]

    @cached_property
    def first_idx(self):
        """
        The two "first indices", respectively for the buffer and the buffered
        Function.

        A "first index" is the index that corresponds to the first slot in the
        buffer that must be initialized, or the first slot that must be read from
        the buffered Function.

        Examples
        --------

            * `((time_m + db0) % 2, time_m + db0)` in the case of a
              forward-propagating `time` Dimension;
            * `((time_M - 2 + db0) % 4, time_M - 2 + db0)` in the case of a
              backwards-propagating `time` Dimension.
        """
        d = self.dim

        if self.is_double_buffering:
            assert len(self.indices) == 1
            v, = self.indices
        else:
            # The buffer is initialized at `d_m(d_M) - offset`. E.g., a buffer with
            # six slots, used to replace a buffered Function accessed at `d-3`, `d`
            # and `d + 2`, will have `offset = 3`
            p, offset = offset_from_centre(d, self.indices)

            if self.is_forward_buffering:
                v = p._subs(d.root, d.root.symbolic_min) - offset + self.xd
            else:
                v = p._subs(d.root, d.root.symbolic_max) - offset + self.xd

        return Map(v % self.xd.symbolic_size, v)


def make_mds(descriptors, prefix, sregistry):
    """
    Create the ModuloDimensions to step through the buffers. This is done
    inspecting all buffers so that ModuloDimensions are reused when possible.
    """
    mds = defaultdict(int)
    for v in descriptors.values():
        size = v.xd.symbolic_size

        if size == 1:
            continue
        if v.is_double_buffering:
            continue

        p, _ = offset_from_centre(v.dim, v.indices)

        # NOTE: indices are sorted so that the semantic order (t0, t1, t2)
        # follows SymPy's index ordering (time, time-1, time+1) after modulo
        # replacement, so that associativity errors are consistent. This very
        # same strategy is also applied in clusters/algorithms/Stepper
        key = lambda i: -np.inf if i - p == 0 else (i - p)
        indices = sorted(v.indices, key=key)

        for i in indices:
            k = (v.xd, i)
            if k in mds:
                continue

            # Can I reuse an existing ModuloDimension or do I need to create
            # a new one?
            for si in prefix.sub_iterators.get(v.dim.root, []):
                if si.offset == i and si.modulo == size:
                    mds[k] = si
                    break
            else:
                name = sregistry.make_name(prefix='t')
                mds[k] = ModuloDimension(name, v.xd, i, size)

    return mds


def init_buffers(descriptors, options):
    """
    Create the initializing Clusters for the given buffers.
    """
    init_onwrite = options['buf-init-onwrite']

    init = []
    for b, v in descriptors.items():
        f = v.f

        if v.is_read:
            # Special case: avoid initialization in the case of double (or
            # multiple) buffering because it's completely unnecessary
            if v.is_double_buffering:
                continue

            lhs = b.indexify()._subs(v.xd, v.first_idx.b)
            rhs = f.indexify()._subs(v.dim, v.first_idx.f)

        elif v.is_write and init_onwrite(f):
            lhs = b.indexify()
            rhs = S.Zero

        else:
            continue

        expr = Eq(lhs, rhs)
        expr = lower_exprs(expr)

        ispace = v.write_to

        guards = {}
        guards[None] = GuardBound(v.dim.root.symbolic_min, v.dim.root.symbolic_max)
        if v.is_read:
            guards[v.xd] = GuardBound(0, v.first_idx.f)

        properties = Properties()
        properties = properties.affine(ispace.itdims)
        properties = properties.parallelize(ispace.itdims)

        syncs = {None: [InitArray(None, b)]}

        init.append(Cluster(expr, ispace, guards, properties, syncs))

    return init


def is_buffering(exprs):
    """
    True if the given N exprs represent N levels of buffering, False otherwise.
    """
    return all(is_memcpy(e) for e in as_tuple(exprs))


def extract_indices(f, dim, clusters):
    """
    Extract the indices along `dim` in the given Clusters that access `f`.
    """
    accesses = chain(*[c.scope[f] for c in clusters])
    indices = filter_ordered(i[dim] for i in accesses)

    return indices


def infer_buffer_size(f, dim, clusters):
    """
    Infer the buffer size for the buffered Function `f` by analyzing the
    accesses along `dim` within the given Clusters.
    """
    indices = extract_indices(f, dim, clusters)

    slots = [i.subs({dim: 0, dim.spacing: 1}) for i in indices]

    try:
        size = max(slots) - min(slots) + 1
    except TypeError:
        # E.g., special case `slots=[-1 + time/factor, 2 + time/factor]`
        # Resort to the fast vector-based comparison machinery (rather than
        # the slower sympy.simplify)
        slots = [Vector(i) for i in slots]
        size = int((vmax(*slots) - vmin(*slots) + 1)[0])

    return size


def offset_from_centre(d, indices):
    if d in indices:
        p = d
        offset = d - min(indices)
        assert is_integer(offset)

    elif len(indices) == 1:
        p = indices[0]
        offset = 0

    else:
        assert len(indices) > 0

        p = None
        for i in indices:
            if i in d._defines or isinstance(i, IntDiv):
                p = i
                break

        if p is None:
            # E.g., `time/factor-1` and `time/factor+1` present among the
            # indices in `index_mapper`, but not `time/factor`. We reconstruct
            # `time/factor` -- the starting pointing at time_m or time_M
            v = indices[0]
            try:
                p = v.func(*[i for i in v.args if not is_integer(i)])
                if not ((p - v).is_Integer or (p - v).is_Symbol):
                    raise ValueError
            except (IndexError, ValueError):
                raise NotImplementedError("Cannot apply buffering with nonlinear "
                                          "index functions (found `%s`)" % v)

        try:
            # Start assuming e.g. `indices = [time - 1, time + 2]`
            offset = p - min(indices)
        except TypeError:
            # Actually, e.g. `indices = [time/factor - 1, time/factor + 2]`
            offset = p - vmin(*[Vector(i) for i in indices])[0]

    return p, offset
