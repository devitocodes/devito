from collections import OrderedDict, defaultdict, namedtuple
from itertools import combinations

from cached_property import cached_property
import numpy as np

from devito.ir import (Cluster, Forward, GuardBound, Interval, IntervalGroup,
                       IterationSpace, AFFINE, PARALLEL, Queue, Vector,
                       lower_exprs, vmax, vmin)
from devito.exceptions import InvalidOperator
from devito.logger import warning
from devito.symbolics import IntDiv, retrieve_function_carriers, uxreplace
from devito.tools import (Bunch, DefaultOrderedDict, Stamp, as_tuple,
                          filter_ordered, flatten, is_integer, timed_pass)
from devito.types import Array, CustomDimension, Dimension, Eq, ModuloDimension

__all__ = ['buffering']


@timed_pass()
def buffering(clusters, callback, sregistry, options, **kwargs):
    """
    Replace written Functions with Arrays. This gives the compiler more control
    over storage layout, data movement (e.g. between host and device), etc.

    Parameters
    ----------
    cluster : list of Cluster
        Input Clusters, subject of the optimization pass.
    callback : callable, optional
        A mechanism to express what the buffering candidates are, and what
        Dimensions, if any, should be replaced by ModuloDimensions, such that
        the buffer has a smaller footprint than that of the Function it stems
        from. The callable takes a Function as input and returns either None
        or a list. If the output is None, then the input is not a buffering
        candidate. Otherwise, the output is a buffering candidate and the list
        contains the Dimensions to be replaced by new ModuloDimensions. If
        unspecified, by default all DiscreteFunctions are turned into buffers,
        but no Dimension replacement occurs.
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
    **kwargs
        Additional compilation options.
        Accepted: ['opt_init_onread', 'opt_init_onwrite', 'opt_buffer'].
        * 'opt_init_onread': By default, a read buffer always triggers the
        generation of an initializing Cluster (see example below). When the
        size of the buffer is 1, the step-through Cluster might suffice, however.
        In such a case, and with `opt_init_onread=False`, the initalizing
        Cluster is omitted.  This creates an implicit contract between the
        caller and the buffering pass, as the step-through Cluster cannot be
        further transformed or the buffer might never be initialized with the
        content of the buffered Function.
        * 'opt_init_onwrite': By default, a written buffer does not trigger the
        generation of an initializing Cluster. With `opt_init_onwrite=True`,
        instead, the buffer gets initialized to zero.
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
    if callback is None:
        def callback(f):
            if f.is_DiscreteFunction:
                return []
            else:
                return None
    assert callable(callback)

    options = {
        'buf-async-degree': options['buf-async-degree'],
        'buf-fuse-tasks': options['fuse-tasks'],
        'buf-init-onread': kwargs.get('opt_init_onread', True),
        'buf-init-onwrite': kwargs.get('opt_init_onwrite', False),
        'buf-callback': kwargs.get('opt_buffer'),
    }

    # Escape hatch to selectively disable buffering
    if options['buf-async-degree'] == 0:
        return clusters

    return Buffering(callback, sregistry, options).process(clusters)


class Buffering(Queue):

    def __init__(self, callback0, sregistry, options):
        super().__init__()

        self.callback0 = callback0
        self.sregistry = sregistry
        self.options = options

    def process(self, clusters):
        return self._process_fatd(clusters, 1, cache={})

    def callback(self, clusters, prefix, cache=None):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        # Locate all Function accesses within the provided `clusters`
        accessmap = AccessMapper(clusters)

        # Create the buffers
        buffers = BufferBatch()
        for f, accessv in accessmap.items():
            # Has a buffer already been produced for `f`?
            if f in cache:
                continue

            # Is `f` really a buffering candidate?
            dim = self.callback0(f)
            if dim is None or d not in dim._defines:
                continue

            b = cache[f] = buffers.make(f, dim, accessv, self.options, self.sregistry)

        if not buffers:
            return clusters

        try:
            pd = prefix[-2].dim
        except IndexError:
            pd = None

        # Create Eqs to initialize buffers. Note: a buffer needs to be initialized
        # only if the buffered Function is read in at least one place or in the case
        # of non-uniform SubDimensions, to avoid uninitialized values to be copied-back
        # into the buffered Function
        init_onread = self.options['buf-init-onread']
        init_onwrite = self.options['buf-init-onwrite']
        init = []
        for b in buffers:
            if b.is_read or not b.has_uniform_subdims:
                # Special case: avoid initialization if not strictly necessary
                # See docstring for more info about what this implies
                if b.size == 1 and not init_onread:
                    continue

                dims = b.function.dimensions
                lhs = b.indexed[[b.initmap.get(d, Map(d, d)).b for d in dims]]
                rhs = b.function[[b.initmap.get(d, Map(d, d)).f for d in dims]]

            elif b.is_write and init_onwrite:
                dims = b.buffer.dimensions
                lhs = b.buffer.indexify()
                rhs = 0

            else:
                continue

            expr = lower_exprs(Eq(lhs, rhs))
            ispace = b.writeto
            guards = {pd: GuardBound(d.root.symbolic_min, d.root.symbolic_max)
                      for d in b.contraction_mapper}
            properties = {d: {AFFINE, PARALLEL} for d in ispace.itdimensions}

            init.append(Cluster(expr, ispace, guards=guards, properties=properties))

        # Substitution rules to replace buffered Functions with buffers
        subs = {}
        for b in buffers:
            for a in b.accessv.accesses:
                subs[a] = b.indexed[[b.index_mapper_flat.get(i, i) for i in a.indices]]

        processed = []
        for c in clusters:
            # If a buffer is read but never written, then we need to add
            # an Eq to step through the next slot
            # E.g., `ub[0, x] = usave[time+2, x]`
            for b in buffers:
                if not b.is_readonly:
                    continue
                try:
                    c.exprs.index(b.firstread)
                except ValueError:
                    continue

                dims = b.function.dimensions
                lhs = b.indexed[[b.lastmap.get(d, Map(d, d)).b for d in dims]]
                rhs = b.function[[b.lastmap.get(d, Map(d, d)).f for d in dims]]

                expr = lower_exprs(Eq(lhs, rhs))
                ispace = b.readfrom
                properties = c.properties.sequentialize(d)

                processed.append(
                    c.rebuild(exprs=expr, ispace=ispace, properties=properties)
                )

            # Substitute buffered Functions with the newly created buffers
            exprs = [uxreplace(e, subs) for e in c.exprs]
            ispace = c.ispace
            for b in buffers:
                ispace = ispace.augment(b.sub_iterators)
            properties = c.properties.sequentialize(d)
            processed.append(
                c.rebuild(exprs=exprs, ispace=ispace, properties=properties)
            )

            # Also append the copy-back if `e` is the last-write of some buffers
            # E.g., `usave[time + 1, x] = ub[sb1, x]`
            for b in buffers:
                if b.is_readonly:
                    continue
                try:
                    c.exprs.index(b.lastwrite)
                except ValueError:
                    continue

                dims = b.function.dimensions
                lhs = b.function[[b.lastmap.get(d, Map(d, d)).f for d in dims]]
                rhs = b.indexed[[b.lastmap.get(d, Map(d, d)).b for d in dims]]

                expr = lower_exprs(uxreplace(Eq(lhs, rhs), b.subdims_mapper))
                ispace = b.written
                properties = c.properties.sequentialize(d)

                processed.append(
                    c.rebuild(exprs=expr, ispace=ispace, properties=properties)
                )

        # Lift {write,read}-only buffers into separate IterationSpaces
        if self.options['buf-fuse-tasks']:
            return init + processed
        for b in buffers:
            if b.is_writeonly:
                # `b` might be written by multiple, potentially mutually-exclusive,
                # equations. For example, two equations that have or will have
                # complementary guards, hence only one will be executed. In such a
                # case, we can split the equations over separate IterationSpaces
                key0 = lambda: Stamp()
            elif b.is_readonly:
                # `b` is read multiple times -- this could just be the case of
                # coupled equations, so we more cautiously perform a
                # "buffer-wise" splitting of the IterationSpaces (i.e., only
                # relevant if there are at least two read-only buffers)
                stamp = Stamp()
                key0 = lambda: stamp
            else:
                continue

            contracted = set().union(*[d._defines for d in b.contraction_mapper])

            processed1 = []
            for c in processed:
                if b.buffer in c.functions:
                    key1 = lambda d: d not in contracted
                    dims = c.ispace.project(key1).itdimensions
                    ispace = c.ispace.lift(dims, key0())
                    processed1.append(c.rebuild(ispace=ispace))
                else:
                    processed1.append(c)
            processed = processed1

        return init + processed


class BufferBatch(list):

    def __init__(self):
        super().__init__()

    def make(self, *args):
        """
        Create a Buffer. See Buffer.__doc__.
        """
        b = Buffer(*args)
        self.append(b)
        return b

    @property
    def functions(self):
        return {b.function for b in self}


class Buffer(object):

    """
    A buffer with metadata attached.

    Parameters
    ----------
    function : DiscreteFunction
        The object for which the buffer is created.
    d : Dimension
        The Dimension in `function` to be contracted, that is to be replaced
        with a ModuloDimension.
    accessv : AccessValue
        All accesses involving `function`.
    options : dict, optional
        The compilation options. See `buffering.__doc__`.
    sregistry : SymbolRegistry
        The symbol registry, to create unique names for buffers and Dimensions.
    """

    def __init__(self, function, d, accessv, options, sregistry):
        # Parse compilation options
        async_degree = options['buf-async-degree']
        callback = options['buf-callback']

        self.function = function
        self.accessv = accessv

        self.contraction_mapper = {}
        self.index_mapper = defaultdict(dict)
        self.sub_iterators = defaultdict(list)
        self.subdims_mapper = DefaultOrderedDict(set)

        # Create the necessary ModuloDimensions for indexing into the buffer
        # E.g., `u[time,x] + u[time+1,x] -> `ub[sb0,x] + ub[sb1,x]`, where `sb0`
        # and `sb1` are ModuloDimensions starting at `time` and `time+1` respectively
        dims = list(function.dimensions)
        assert d in function.dimensions

        # Determine the buffer size, and therefore the span of the ModuloDimension,
        # along the contracting Dimension `d`
        indices = filter_ordered(i.indices[d] for i in accessv.accesses)
        slots = [i.subs({d: 0, d.spacing: 1}) for i in indices]
        try:
            size = max(slots) - min(slots) + 1
        except TypeError:
            # E.g., special case `slots=[-1 + time/factor, 2 + time/factor]`
            # Resort to the fast vector-based comparison machinery (rather than
            # the slower sympy.simplify)
            slots = [Vector(i) for i in slots]
            size = int((vmax(*slots) - vmin(*slots) + 1)[0])

        if async_degree is not None:
            if async_degree < size:
                warning("Ignoring provided asynchronous degree as it'd be "
                        "too small for the required buffer (provided %d, "
                        "but need at least %d for `%s`)"
                        % (async_degree, size, function.name))
            else:
                size = async_degree

        # Replace `d` with a suitable CustomDimension `bd`
        try:
            bd = sregistry.get('bds', (d, size))
        except KeyError:
            name = sregistry.make_name(prefix='db')
            v = CustomDimension(name, 0, size-1, size, d)
            bd = sregistry.setdefault('bds', (d, size), v)
        self.contraction_mapper[d] = dims[dims.index(d)] = bd

        # Finally create the ModuloDimensions as children of `bd`
        if size > 1:
            # Note: indices are sorted so that the semantic order (sb0, sb1, sb2)
            # follows SymPy's index ordering (time, time-1, time+1) after modulo
            # replacement, so that associativity errors are consistent. This very
            # same strategy is also applied in clusters/algorithms/Stepper
            p, _ = offset_from_centre(d, indices)
            indices = sorted(indices,
                             key=lambda i: -np.inf if i - p == 0 else (i - p))
            for i in indices:
                try:
                    md = sregistry.get('mds', (bd, i))
                except KeyError:
                    name = sregistry.make_name(prefix='sb')
                    v = ModuloDimension(name, bd, i, size)
                    md = sregistry.setdefault('mds', (bd, i), v)
                self.index_mapper[d][i] = md
                self.sub_iterators[d.root].append(md)
        else:
            assert len(indices) == 1
            self.index_mapper[d][indices[0]] = 0

        # Track the SubDimensions used to index into `function`
        for e in accessv.mapper:
            m = {i.root: i for i in e.free_symbols
                 if isinstance(i, Dimension) and (i.is_Sub or not i.is_Derived)}
            for d, v in m.items():
                self.subdims_mapper[d].add(v)
        if any(len(v) > 1 for v in self.subdims_mapper.values()):
            # Non-uniform SubDimensions. At this point we're going to raise
            # an exception. It's either illegal or still unsupported
            for v in self.subdims_mapper.values():
                for d0, d1 in combinations(v, 2):
                    if d0.overlap(d1):
                        raise InvalidOperator("Cannot apply `buffering` to `%s` as it "
                                              "is accessed over the overlapping "
                                              " SubDimensions `<%s, %s>`" %
                                              (function, d0, d1))
            raise NotImplementedError("`buffering` does not support multiple "
                                      "non-overlapping SubDimensions yet.")
        else:
            self.subdims_mapper = {d: v.pop() for d, v in self.subdims_mapper.items()}

        # Build and sanity-check the buffer IterationIntervals
        self.itintervals_mapper = {}
        for e in accessv.mapper:
            for i in e.ispace.itintervals:
                v = self.itintervals_mapper.setdefault(i.dim, i.args)
                if v != self.itintervals_mapper[i.dim]:
                    raise NotImplementedError("Cannot apply `buffering` as the buffered "
                                              "function `%s` is accessed over multiple, "
                                              "non-compatible iteration spaces along the "
                                              "Dimension `%s`" % (function.name, i.dim))
        # Also add IterationIntervals for initialization along `x`, should `xi` be
        # the only written Dimension in the `x` hierarchy
        for d, (interval, _, _) in list(self.itintervals_mapper.items()):
            for i in d._defines:
                self.itintervals_mapper.setdefault(i, (interval.relaxed, (), Forward))

        # Finally create the actual buffer
        kwargs = {
            'name': sregistry.make_name(prefix='%sb' % function.name),
            'dimensions': dims,
            'dtype': function.dtype,
            'halo': function.halo,
            'space': 'mapped',
            'mapped': function
        }
        try:
            self.buffer = callback(function, **kwargs)
        except TypeError:
            self.buffer = Array(**kwargs)

    def __repr__(self):
        return "Buffer[%s,<%s>]" % (self.buffer.name,
                                    ','.join(str(i) for i in self.contraction_mapper))

    @property
    def size(self):
        return np.prod([v.symbolic_size for v in self.contraction_mapper.values()])

    @property
    def firstread(self):
        return self.accessv.firstread

    @property
    def lastwrite(self):
        return self.accessv.lastwrite

    @property
    def is_read(self):
        return self.firstread is not None

    @property
    def is_write(self):
        return self.lastwrite is not None

    @property
    def is_readonly(self):
        return self.firstread is not None and self.lastwrite is None

    @property
    def is_writeonly(self):
        return self.lastwrite is not None and self.firstread is None

    @property
    def has_uniform_subdims(self):
        return self.subdims_mapper is not None

    @cached_property
    def indexed(self):
        return self.buffer.indexed

    @cached_property
    def index_mapper_flat(self):
        ret = {}
        for mapper in self.index_mapper.values():
            ret.update(mapper)
        return ret

    @cached_property
    def writeto(self):
        """
        The `writeto` IterationSpace, that is the iteration space that must be
        iterated over in order to initialize the buffer.
        """
        intervals = []
        sub_iterators = {}
        directions = {}
        for d, h in zip(self.buffer.dimensions, self.buffer._size_halo):
            try:
                interval, si, direction = self.itintervals_mapper[d]
                # The initialization must comprise the halo region as well, since
                # in principle this could be accessed through a stencil
                interval = interval.translate(v0=-h.left, v1=h.right)
            except KeyError:
                # E.g., the contraction Dimension `db0`
                assert d in self.contraction_mapper.values()
                interval, si, direction = Interval(d, 0, 0), (), Forward
            intervals.append(interval)
            sub_iterators[d] = si
            directions[d] = direction

        relations = (self.buffer.dimensions,)
        intervals = IntervalGroup(intervals, relations=relations)

        return IterationSpace(intervals, sub_iterators, directions)

    @cached_property
    def written(self):
        """
        The `written` IterationSpace, that is the iteration space that must be
        iterated over in order to read all of the written buffer values.
        """
        intervals = []
        sub_iterators = {}
        directions = {}
        for dd in self.function.dimensions:
            d = dd.xreplace(self.subdims_mapper)
            try:
                interval, si, direction = self.itintervals_mapper[d]
            except KeyError:
                # E.g., d=time_sub
                assert d.is_NonlinearDerived
                d = d.root
                interval, si, direction = self.itintervals_mapper[d]
            intervals.append(interval)
            sub_iterators[d] = si + as_tuple(self.sub_iterators[d])
            directions[d] = direction

        relations = (tuple(i.dim for i in intervals),)
        intervals = IntervalGroup(intervals, relations=relations)

        return IterationSpace(intervals, sub_iterators, directions)

    @cached_property
    def readfrom(self):
        """
        The `readfrom` IterationSpace, that is the iteration space that must be
        iterated over to update the buffer with the buffered Function values.
        """
        cdims = set().union(*[d._defines
                              for d in flatten(self.contraction_mapper.items())])

        ispace0 = self.written.project(lambda d: d in cdims)
        ispace1 = self.writeto.project(lambda d: d not in cdims)

        extra = (ispace0.itdimensions + ispace1.itdimensions,)
        ispace = IterationSpace.union(ispace0, ispace1, relations=extra)

        return ispace

    @cached_property
    def lastmap(self):
        """
        A mapper from contracted Dimensions to a 2-tuple of indices representing,
        respectively, the "last" write to the buffer and the "last" read from the
        buffered Function. For example, `{time: (sb1, time+1)}`.
        """
        mapper = {}
        for d, m in self.index_mapper.items():
            try:
                func = max if self.written.directions[d.root] is Forward else min
                v = func(m)
            except TypeError:
                func = vmax if self.written.directions[d.root] is Forward else vmin
                v = func(*[Vector(i) for i in m])[0]
            mapper[d] = Map(m[v], v)

        return mapper

    @cached_property
    def initmap(self):
        """
        A mapper from contracted Dimensions to indices representing the min points
        for buffer initialization. For example, in the case of a forward-propagating
        `time` Dimension, we could have `{time: (time_m + db0) % 2, (time_m + db0)}`;
        likewise, for backwards, `{time: (time_M - 2 + db0) % 4, time_M - 2 + db0}`.
        """
        mapper = {}
        for d, bd in self.contraction_mapper.items():
            indices = list(self.index_mapper[d])

            # The buffer is initialized at `d_m(d_M) - offset`. E.g., a buffer with
            # six slots, used to replace a buffered Function accessed at `d-3`, `d`
            # and `d + 2`, will have `offset = 3`
            p, offset = offset_from_centre(d, indices)

            if self.written.directions[d.root] is Forward:
                v = p.subs(d.root, d.root.symbolic_min) - offset + bd
            else:
                v = p.subs(d.root, d.root.symbolic_max) - offset + bd

            mapper[d] = Map(v % bd.symbolic_size, v)

        return mapper


class AccessValue(object):

    """
    A simple data structure tracking the accesses performed by a given Function
    in a sequence of Clusters.

    Parameters
    ----------
    function : Function
        The target Function.
    mapper : dict
        A mapper from expressions to Indexeds, representing all accesses to
        `function` in a sequence of expressions.
    """

    def __init__(self, function, mapper):
        self.function = function
        self.mapper = mapper

    @cached_property
    def exprs(self):
        return tuple(self.mapper)

    @cached_property
    def accesses(self):
        return tuple(flatten(as_tuple(i.reads) + as_tuple(i.write)
                             for i in self.mapper.values()))

    @cached_property
    def is_read(self):
        return any(av.reads for av in self.mapper.values())

    @cached_property
    def firstread(self):
        for e, av in self.mapper.items():
            if av.reads:
                return e
        return None

    @cached_property
    def lastwrite(self):
        for e, av in reversed(self.mapper.items()):
            if av.write is not None:
                return e
        return None


AccessTuple = lambda: Bunch(reads=[], write=None)
Map = namedtuple('Map', 'b f')


class AccessMapper(OrderedDict):

    def __init__(self, clusters):
        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(AccessTuple))
        for c in clusters:
            for e in c.exprs:
                for i in retrieve_function_carriers(e.rhs):
                    mapper[i.function][e].reads.append(i)
                mapper[e.lhs.function][e].write = e.lhs

        super().__init__([(f, AccessValue(f, mapper[f])) for f in mapper])


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
                p = sum(v.args[1:])
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
