from collections import OrderedDict
from itertools import combinations

from cached_property import cached_property

from devito.exceptions import InvalidOperator
from devito.logger import warning
from devito.symbolics import retrieve_function_carriers, uxreplace
from devito.tools import (Bunch, DefaultOrderedDict, as_tuple, filter_ordered, flatten,
                          timed_pass)
from devito.types import Array, CustomDimension, Dimension, Eq, SteppingDimension

__all__ = ['buffering']


@timed_pass()
def buffering(expressions, callback=None, options=None):
    """
    Replace written Functions with Arrays. This gives the compiler more control
    over storage layout, data movement (e.g. between host and device), etc.

    Parameters
    ----------
    expressions : list of expr-like
        The expressions to which the pass is applied.
    callback : callable, optional
        A mechanism to express what the buffering candidates are, and what
        Dimensions, if any, should be replaced by SteppingDimensions, such that
        the buffer has a smaller footprint than that of the Function it stems
        from. The callable takes a Function as input and returns either None
        or a list. If the output is None, then the input is not a buffering
        candidate. Otherwise, the output is a buffering candidate and the list
        contains the Dimensions to be replaced by new SteppingDimensions. If
        unspecified, by default all DiscreteFunctions are turned into buffers,
        but no Dimension replacement occurs.
    options : dict, optional
        The optimization options.
        Accepted: ['buf-async-degree'].
        * 'buf-async-degree': Specify the size of the buffer. By default, the
          buffer size is the minimal one, inferred from the memory accesses in
          the ``expressions`` themselves. An asynchronous degree equals to `k`
          means that the buffer will be enforced to size=`k` along the introduced
          SteppingDimensions. This might help relieving the synchronization
          overhead when asynchronous operations are used (these are however
          implemented by other passes).

    Examples
    --------
    Assume `contraction_rules=[time]`, meaning that all Functions defined over
    `time` should be replaced by a buffer that uses a SteppingDimension in place
    of `time` in the same slot.
    Consider the Eq below:

        Eq(u[time+1, x], u[time, x] + u[time-1, x] + 1)

    We see that `u(time, x)` is both read and written. So it is a buffering
    candidate. Let's assume `rule(u)` is True, so we apply buffering. This
    boils down to:

        1. Introducing two Eqs to initialize the buffer `ub`, i.e.

            Eq(ub[d, x], u[d, x])
            Eq(ub[d-1, x], u[d-1, x])

        2. Introduce one Eq to copy `ub` back into `u`

            Eq(u[time+1, x], ub[d+1, x])

        3. Replace all other occurrences of `u` with `ub`

    So eventually we have four Eqs:

        Eq(ub[d, x], u[d, x]),
        Eq(ub[d-1, x], u[d-1, x])
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

    return _buffering(expressions, callback, options)


def _buffering(expressions, callback, options):
    async_degree = options['buf-async-degree']

    # Locate all Function accesses within the provided `expressions`
    accessmap = AccessMapper(expressions)

    # Create the buffers
    buffers = []
    for n, (f, accessv) in enumerate(accessmap.items()):
        dims = callback(f)
        if dims is None:
            # Not a buffer candidate
            continue
        if accessv.lastwrite is None:
            # Read-only Functions cannot be buffering candidates
            continue
        buffers.append(Buffer(f, dims, accessv, n, async_degree))

    # Create Eqs to initialize buffers. Note: a buffer needs to be initialized
    # only if the buffered Function is read in at least one place or in the case
    # of non-uniform SubDimensions, to avoid uninitialized values to be copied-back
    # into the buffered Function
    processed = [Eq(b.indexify(), b.function.subs(b.contraction_mapper))
                 for b in buffers if b.is_read or not b.has_uniform_subdims]

    # Substitution rules to replace buffered Functions with buffers
    subs = {}
    for b in buffers:
        for a in b.accessv.accesses:
            subs[a] = b.indexify(a.indices)

    # Create Eqs to copy back buffers into their buffered Functions
    for e in expressions:
        processed.append(uxreplace(e, subs))

        # We also append the copy-back if `e` is the last-write of some buffers
        for b in buffers:
            if e is b.accessv.lastwrite:
                items = list(zip(e.lhs.indices, b.function.dimensions))
                lhs = b.function[[i if i in b.index_mapper else d for i, d in items]]
                rhs = b.indexed[[b.index_mapper.get(i, d) for i, d in items]]
                if b.subdims_mapper:
                    processed.append(uxreplace(Eq(lhs, rhs), b.subdims_mapper))
                else:
                    processed.append(Eq(lhs, rhs))
                break

    return processed


class Buffer(object):

    """
    A buffer with metadata attached.

    Parameters
    ----------
    function : DiscreteFunction
        The object for which a buffer is created.
    contracted_dims : list of Dimension
        The Dimensions in `function` to be contracted, that is to be replaced
        by SteppingDimensions.
    accessv : AccessValue
        All accesses involving `function`.
    n : int
        A unique identifier for this Buffer.
    async_degree : int, optional
        Enforce a size of `async_degree` along the contracted Dimensions.
    """

    def __init__(self, function, contracted_dims, accessv, n, async_degree):
        self.function = function
        self.accessv = accessv

        contraction_mapper = {}
        index_mapper = {}
        dims = list(function.dimensions)
        for d in contracted_dims:
            assert d in function.dimensions

            # Determine the buffer size along `d`
            indices = filter_ordered(i.indices[d] for i in accessv.accesses)
            slots = [i.xreplace({d: 0, d.spacing: 1}) for i in indices]
            size = max(slots) - min(slots) + 1

            if async_degree is not None:
                if async_degree < size:
                    warning("Ignoring provided asynchronous degree as it'd be "
                            "too small for the required buffer (provided %d, "
                            "but need at least %d for `%s`)"
                            % (async_degree, size, function.name))
                else:
                    size = async_degree

            # Replace `d` with a suitable CustomDimension
            bd = CustomDimension('db%d' % n, 0, size-1, size, d)
            contraction_mapper[d] = dims[dims.index(d)] = bd

            if size > 1:
                # Create the necessary SteppingDimensions for indexing
                sd = SteppingDimension(name='sb%d' % n, parent=bd)
                index_mapper.update({i: i.xreplace({d: sd}) for i in indices})
            else:
                # Special case, no need to keep a SteppingDimension around
                index_mapper.update({i: 0 for i in indices})

        self.contraction_mapper = contraction_mapper
        self.index_mapper = index_mapper

        # Track the SubDimensions used to index into `function`
        subdims_mapper = DefaultOrderedDict(set)
        for e in accessv.mapper:
            try:
                # Case 1: implicitly via SubDomains
                m = {d.root: v for d, v in e.subdomain.dimension_map.items()}
            except AttributeError:
                # Case 2: explicitly via the lower-level SubDimension API
                m = {i.root: i for i in e.free_symbols
                     if isinstance(i, Dimension) and (i.is_Sub or not i.is_Derived)}
            for d, v in m.items():
                subdims_mapper[d].add(v)
        if any(len(v) > 1 for v in subdims_mapper.values()):
            # Non-uniform SubDimensions. At this point we're going to raise
            # an exception. It's either illegal or still unsupported
            for v in subdims_mapper.values():
                for d0, d1 in combinations(v, 2):
                    if d0.overlap(d1):
                        raise InvalidOperator("Cannot apply `buffering` to `%s` as it "
                                              "is accessed over the overlapping "
                                              " SubDimensions `<%s, %s>`" %
                                              (function, d0, d1))
            self.subdims_mapper = None
            raise NotImplementedError("`buffering` does not support multiple "
                                      "non-overlapping SubDimensions yet.")
        else:
            self.subdims_mapper = {d: v.pop() for d, v in subdims_mapper.items()}

        self.buffer = Array(name='%sb' % function.name,
                            dimensions=dims,
                            dtype=function.dtype,
                            halo=function.halo,
                            space='mapped')

    def __repr__(self):
        return "Buffer[%s,<%s:%s>]" % (self.buffer.name,
                                       ','.join(str(i) for i in self.contraction_mapper),
                                       ','.join(str(i).replace(" ", "")
                                                for i in self.index_mapper))

    @property
    def is_read(self):
        return self.accessv.is_read

    @property
    def lastwrite(self):
        return self.accessv.lastwrite

    @property
    def has_uniform_subdims(self):
        return self.subdims_mapper is not None

    @cached_property
    def indexed(self):
        return self.buffer.indexed

    def indexify(self, indices=None):
        if indices is None:
            indices = list(self.buffer.dimensions)
        else:
            indices = [self.index_mapper.get(i, i) for i in indices]
        return self.indexed[indices]


class AccessValue(object):

    """
    A simple data structure tracking the accesses performed by a given Function
    in a sequence of expressions.

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
    def lastwrite(self):
        for e, av in reversed(self.mapper.items()):
            if av.write is not None:
                return e
        return None


AccessTuple = lambda: Bunch(reads=[], write=None)


class AccessMapper(OrderedDict):

    def __init__(self, expressions):
        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(AccessTuple))
        for e in expressions:
            for i in retrieve_function_carriers(e.rhs):
                mapper[i.function][e].reads.append(i)
            mapper[e.lhs.function][e].write = e.lhs

        super().__init__([(f, AccessValue(f, mapper[f])) for f in mapper])
