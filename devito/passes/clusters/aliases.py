from collections import OrderedDict, defaultdict

from cached_property import cached_property
import numpy as np

from devito.ir import (ROUNDABLE, DataSpace, IterationInstance, Interval, IntervalGroup,
                       LabeledVector, Scope, detect_accesses, build_intervals)
from devito.passes.clusters.utils import cluster_pass, make_is_time_invariant
from devito.symbolics import (compare_ops, estimate_cost, q_constant, q_leaf,
                              q_sum_of_product, q_terminalop, retrieve_indexed,
                              uxreplace, yreplace)
from devito.tools import flatten
from devito.types import Array, Eq, ShiftedDimension, Scalar

__all__ = ['cire']


@cluster_pass
def cire(cluster, template, mode, options, platform):
    """
    Cross-iteration redundancies elimination.

    Parameters
    ----------
    cluster : Cluster
        Input Cluster, subject of the optimization pass.
    template : callable
        To build the symbols (temporaries) storing the redundant expressions.
    mode : str
        The transformation mode. Accepted: ['invariants', 'sops'].
        * 'invariants' is for sub-expressions that are invariant w.r.t. one or
          more Dimensions.
        * 'sops' stands for sums-of-products, that is redundancies are searched
          across all expressions in sum-of-product form.
    options : dict
        The optimization mode. Accepted: ['min-storage'].
        * 'min-storage': if True, the pass will try to minimize the amount of
          storage introduced for the tensor temporaries. This might also reduce
          the operation count. On the other hand, this might affect fusion and
          therefore data locality. Defaults to False (legacy).
    platform : Platform
        The underlying platform. Used to optimize the shape of the introduced
        tensor symbols.

    Examples
    --------
    1) 'invariants'. Below is an expensive sub-expression invariant w.r.t. `t`

    t0 = (cos(a[x,y,z])*sin(b[x,y,z]))*c[t,x,y,z]

    becomes

    t1[x,y,z] = cos(a[x,y,z])*sin(b[x,y,z])
    t0 = t1[x,y,z]*c[t,x,y,z]

    2) 'sops'. Below are redundant sub-expressions in sum-of-product form (in this
    case, the sum degenerates to a single product).

    t0 = 2.0*a[x,y,z]*b[x,y,z]
    t1 = 3.0*a[x,y,z+1]*b[x,y,z+1]

    becomes

    t2[x,y,z] = a[x,y,z]*b[x,y,z]
    t0 = 2.0*t2[x,y,z]
    t1 = 3.0*t2[x,y,z+1]
    """
    # Relevant options
    min_storage = options['min-storage']
    min_cost = options['cire-mincost']
    repeats = options['cire-repeats']

    # Sanity checks
    assert mode in ['invariants', 'sops']
    assert all(i >= 0 for i in repeats.values())

    # Setup callbacks
    def callbacks_invariants(context, n):
        min_cost_inv = min_cost['invariants']
        if callable(min_cost_inv):
            min_cost_inv = min_cost_inv(n)

        extractor = make_is_time_invariant(context)
        model = lambda e: estimate_cost(e, True) >= min_cost_inv
        ignore_collected = lambda g: False
        selector = lambda c, n: c >= min_cost_inv and n >= 1
        return extractor, model, ignore_collected, selector

    def callbacks_sops(context, n):
        min_cost_sops = min_cost['sops']
        if callable(min_cost_sops):
            min_cost_sops = min_cost_sops(n)

        # The `depth` determines "how big" the extracted sum-of-products will be.
        # We observe that in typical FD codes:
        #   add(mul, mul, ...) -> stems from first order derivative
        #   add(mul(add(mul, mul, ...), ...), ...) -> stems from second order derivative
        # To catch the former, we would need `depth=1`; for the latter, `depth=3`
        depth = 2*n + 1

        extractor = lambda e: q_sum_of_product(e, depth)
        model = lambda e: not (q_leaf(e) or q_terminalop(e, depth-1))
        ignore_collected = lambda g: len(g) <= 1
        selector = lambda c, n: c >= min_cost_sops and n > 1
        return extractor, model, ignore_collected, selector

    callbacks_mapper = {
        'invariants': callbacks_invariants,
        'sops': callbacks_sops
    }

    # The main CIRE loop
    processed = []
    context = cluster.exprs
    for n in reversed(range(repeats[mode])):
        # Get the callbacks
        extractor, model, ignore_collected, selector = callbacks_mapper[mode](context, n)

        # Extract potentially aliasing expressions
        exprs, extracted = extract(cluster, extractor, model, template)
        if not extracted:
            # Do not waste time
            continue

        # There can't be Dimension-dependent data dependences with any of
        # the `processed` Clusters, otherwise we would risk either OOB accesses
        # or reading from garbage uncomputed halo
        scope = Scope(exprs=flatten(c.exprs for c in processed) + extracted)
        if not all(i.is_indep() for i in scope.d_all_gen()):
            break

        # Search aliasing expressions
        aliases = collect(extracted, min_storage, ignore_collected)

        # Rule out aliasing expressions with a bad flops/memory trade-off
        chosen, others = choose(exprs, aliases, selector)
        if not chosen:
            # Do not waste time
            continue

        # Create Aliases and assign them to Clusters
        clusters, subs = process(cluster, chosen, aliases, template, platform)

        # Rebuild `cluster` so as to use the newly created Aliases
        rebuilt = rebuild(cluster, others, aliases, subs)

        # Prepare for the next round
        processed.extend(clusters)
        cluster = rebuilt
        context = flatten(c.exprs for c in processed) + list(cluster.exprs)

    processed.append(cluster)

    return processed


def extract(cluster, rule1, model, template):
    make = lambda: Scalar(name=template(), dtype=cluster.dtype).indexify()

    # Rule out symbols inducing Dimension-independent data dependences
    exclude = {i.source.indexed for i in cluster.scope.d_flow.independent()}
    rule0 = lambda e: not e.free_symbols & exclude

    # Composite extraction rule -- correctness(0) + logic(1)
    rule = lambda e: rule0(e) and rule1(e)

    return yreplace(cluster.exprs, make, rule, model, eager=True)


def collect(exprs, min_storage, ignore_collected):
    """
    Find groups of aliasing expressions.

    We shall introduce the following (loose) terminology:

        * A ``terminal`` is the leaf of a mathematical operation. Terminals
          can be numbers (n), literals (l), or Indexeds (I).
        * ``R`` is the relaxation operator := ``R(n) = n``, ``R(l) = l``,
          ``R(I) = J``, where ``J`` has the same base as ``I`` but with all
          offsets stripped away. For example, ``R(a[i+2,j-1]) = a[i,j]``.
        * A ``relaxed expression`` is an expression in which all of the
          terminals are relaxed.

    Now we define the concept of aliasing. We say that an expression A
    aliases an expression B if:

        * ``R(A) == R(B)``
        * all pairwise Indexeds in A and B access memory locations at a
          fixed constant distance along each Dimension.

    For example, consider the following expressions:

        * a[i+1] + b[i+1]
        * a[i+1] + b[j+1]
        * a[i] + c[i]
        * a[i+2] - b[i+2]
        * a[i+2] + b[i]
        * a[i-1] + b[i-1]

    Out of the expressions above, the following alias to `a[i] + b[i]`:

        * a[i+1] + b[i+1] : same operands and operations, distance along i: 1
        * a[i-1] + b[i-1] : same operands and operations, distance along i: -1

    Whereas the following do not:

        * a[i+1] + b[j+1] : because at least one index differs
        * a[i] + c[i] : because at least one of the operands differs
        * a[i+2] - b[i+2] : because at least one operation differs
        * a[i+2] + b[i] : because the distances along ``i`` differ (+2 and +0)
    """
    # Find the potential aliases
    found = []
    for expr in exprs:
        if expr.lhs.is_Indexed or expr.is_Increment:
            continue

        indexeds = retrieve_indexed(expr.rhs)

        bases = []
        offsets = []
        for i in indexeds:
            ii = IterationInstance(i)
            if ii.is_irregular:
                break

            base = []
            offset = []
            for e, ai in zip(ii, ii.aindices):
                if q_constant(e):
                    base.append(e)
                else:
                    base.append(ai)
                    offset.append((ai, e - ai))
            bases.append(tuple(base))
            offsets.append(LabeledVector(offset))

        if indexeds and len(bases) == len(indexeds):
            found.append(Candidate(expr, indexeds, bases, offsets))

    # Create groups of aliasing expressions
    mapper = OrderedDict()
    unseen = list(found)
    while unseen:
        c = unseen.pop(0)
        group = [c]
        for u in list(unseen):
            # Is the arithmetic structure of `c` and `u` equivalent ?
            if not compare_ops(c.expr, u.expr):
                continue

            # Is `c` translated w.r.t. `u` ?
            if not c.translated(u):
                continue

            group.append(u)
            unseen.remove(u)
        group = Group(group)

        # Apply callback to heuristically discard groups
        if ignore_collected(group):
            continue

        if min_storage:
            k = group.dimensions_translated
        else:
            k = group.dimensions
        mapper.setdefault(k, []).append(group)

    aliases = Aliases()
    for _groups in mapper.values():
        groups = list(_groups)

        while groups:
            # For each Dimension, determine the Minimum Intervals (MI) spanning
            # all of the Groups diameters
            # Example: x's largest_diameter=2  => [x[-2,0], x[-1,1], x[0,2]]
            # Note: Groups that cannot evaluate their diameter are dropped
            mapper = defaultdict(int)
            for g in list(groups):
                try:
                    mapper.update({d: max(mapper[d], v) for d, v in g.diameter.items()})
                except ValueError:
                    groups.remove(g)
            intervalss = {d: make_rotations_table(d, v) for d, v in mapper.items()}

            # For each Group, find a rotation that is compatible with a given MI
            mapper = {}
            for d, intervals in intervalss.items():
                for interval in list(intervals):
                    found = {g: g.find_rotation_distance(d, interval) for g in groups}
                    if all(distance is not None for distance in found.values()):
                        # `interval` is OK !
                        mapper[interval] = found
                        break

            if len(mapper) == len(intervalss):
                break

            # Try again with fewer groups
            smallest = len(min(groups, key=len))
            groups = [g for g in groups if len(g) > smallest]

        for g in groups:
            c = g.pivot
            distances = defaultdict(int, [(i.dim, v[g]) for i, v in mapper.items()])

            # Create the basis alias
            offsets = [LabeledVector([(l, v[l] + distances[l]) for l in v.labels])
                       for v in c.offsets]
            subs = {i: i.function[[l + v.fromlabel(l, 0) for l in b]]
                    for i, b, v in zip(c.indexeds, c.bases, offsets)}
            alias = uxreplace(c.expr, subs)

            # All aliased expressions
            aliaseds = [i.expr for i in g]

            # Distance of each aliased expression from the basis alias
            distances = []
            for i in g:
                distance = [o.distance(v) for o, v in zip(i.offsets, offsets)]
                distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]
                distances.append(LabeledVector([(d, v.pop()) for d, v in distance]))

            aliases.add(alias, list(mapper), aliaseds, distances)

    return aliases


def choose(exprs, aliases, selector):
    others = []
    chosen = OrderedDict()
    for e in exprs:
        naliases = len(aliases.get(e.rhs))
        cost = estimate_cost(e, True)*naliases
        if selector(cost, naliases):
            chosen[e.rhs] = e.lhs
        else:
            others.append(e)

    return chosen, others


def process(cluster, chosen, aliases, template, platform):
    clusters = []
    subs = {}
    for alias, writeto, aliaseds, distances in aliases.iter(cluster.ispace):
        if all(i not in chosen for i in aliaseds):
            continue

        # The Dimensions defining the shape of Array
        # Note: with SubDimensions, we may have the following situation:
        #
        # for zi = z_m + zi_ltkn; zi <= z_M - zi_rtkn; ...
        #   r[zi] = ...
        #
        # Instead of `r[zi - z_m - zi_ltkn]` we have just `r[zi]`, so we'll need
        # as much room as in `zi`'s parent to avoid going OOB
        # Aside from ugly generated code, the reason we do not rather shift the
        # indices is that it prevents future passes to transform the loop bounds
        # (e.g., MPI's comp/comm overlap does that)
        dimensions = [d.parent if d.is_Sub else d for d in writeto.dimensions]

        # The halo of the Array
        halo = [(abs(i.lower), abs(i.upper)) for i in writeto]

        # The memory scope of the Array
        scope = 'stack' if any(d.is_Incr for d in writeto.dimensions) else 'heap'

        # Finally create the temporary Array that will store `alias`
        array = Array(name=template(), dimensions=dimensions, halo=halo,
                      dtype=cluster.dtype, scope=scope)

        # The access Dimensions may differ from `writeto.dimensions`. This may
        # happen e.g. if ShiftedDimensions are introduced (`a[x,y]` -> `a[xs,y]`)
        adims = [aliases.index_mapper.get(d, d) for d in writeto.dimensions]

        # The expression computing `alias`
        adims = [aliases.index_mapper.get(d, d) for d in writeto.dimensions]  # x -> xs
        indices = [d - (0 if writeto[d].is_Null else writeto[d].lower) for d in adims]
        expression = Eq(array[indices], uxreplace(alias, subs))

        # Create the substitution rules so that we can use the newly created
        # temporary in place of the aliasing expressions
        for aliased, distance in zip(aliaseds, distances):
            assert all(i.dim in distance.labels for i in writeto)

            indices = [d - i.lower + distance[i.dim] for d, i in zip(adims, writeto)]
            subs[aliased] = array[indices]

            if aliased in chosen:
                subs[chosen[aliased]] = array[indices]
            else:
                # Perhaps part of a composite alias ?
                pass

        # Construct the `alias` IterationSpace
        ispace = cluster.ispace.add(writeto).augment(aliases.index_mapper)

        # Optimization: if possible, the innermost IterationInterval is
        # rounded up to a multiple of the vector length
        try:
            it = ispace.itintervals[-1]
            if ROUNDABLE in cluster.properties[it.dim]:
                vl = platform.simd_items_per_reg(cluster.dtype)
                ispace = ispace.add(Interval(it.dim, 0, it.interval.size % vl))
        except (TypeError, KeyError):
            pass

        # Construct the `alias` DataSpace
        accesses = detect_accesses(expression)
        parts = {k: IntervalGroup(build_intervals(v)).add(ispace.intervals).relaxed
                 for k, v in accesses.items() if k}
        dspace = DataSpace(cluster.dspace.intervals, parts)

        # Finally, build a new Cluster for `alias`
        built = cluster.rebuild(exprs=expression, ispace=ispace, dspace=dspace)
        clusters.append(built)

    return clusters, subs


def rebuild(cluster, others, aliases, subs):
    # Rebuild the non-aliasing expressions
    exprs = [uxreplace(e, subs) for e in others]

    # Add any new ShiftedDimension to the IterationSpace
    ispace = cluster.ispace.augment(aliases.index_mapper)

    # Rebuild the DataSpace to include the new symbols
    accesses = detect_accesses(exprs)
    parts = {k: IntervalGroup(build_intervals(v)).relaxed
             for k, v in accesses.items() if k}
    dspace = DataSpace(cluster.dspace.intervals, parts)

    return cluster.rebuild(exprs=exprs, ispace=ispace, dspace=dspace)


# Utilities


class Candidate(object):

    def __init__(self, expr, indexeds, bases, offsets):
        self.expr = expr.rhs
        self.shifts = expr.ispace.intervals
        self.indexeds = indexeds
        self.bases = bases
        self.offsets = offsets

    def __repr__(self):
        return "Candidate(expr=%s)" % self.expr

    def translated(self, other):
        """
        True if ``self`` is translated w.r.t. ``other``, False otherwise.

        Examples
        --------
        Two candidates are translated if their bases are the same and
        their offsets are pairwise translated.

        c := A[i,j] op A[i,j+1]     -> Toffsets = {i: [0,0], j: [0,1]}
        u := A[i+1,j] op A[i+1,j+1] -> Toffsets = {i: [1,1], j: [0,1]}

        Then `c` is translated w.r.t. `u` with distance `{i: 1, j: 0}`
        """
        if len(self.Toffsets) != len(other.Toffsets):
            return False
        if len(self.bases) != len(other.bases):
            return False

        # Check the bases
        if any(b0 != b1 for b0, b1 in zip(self.bases, other.bases)):
            return False

        # Check the offsets
        for (d0, o0), (d1, o1) in zip(self.Toffsets, other.Toffsets):
            if d0 is not d1:
                return False

            distance = set(o0 - o1)
            if len(distance) != 1:
                return False

        return True

    @cached_property
    def Toffsets(self):
        return LabeledVector.transpose(*self.offsets)

    @cached_property
    def dimensions(self):
        return frozenset(i for i, _ in self.Toffsets)


class Group(tuple):

    """
    A collection of aliasing expressions.
    """

    def __repr__(self):
        return "Group(%s)" % ", ".join([str(i) for i in self])

    def find_rotation_distance(self, d, interval):
        """
        The distance from the Group pivot of a rotation along Dimension ``d`` that
        can safely iterate over the ``interval``.
        """
        assert d is interval.dim

        for rotation, distance in self._pivot_legal_rotations[d]:
            # Does `rotation` cover the `interval` ?
            if rotation.union(interval) != rotation:
                continue

            # Infer the `rotation`'s min_intervals from the pivot's
            min_interval = self._pivot_min_intervals[d].translate(-distance)

            # Does the `interval` actually cover the `rotation`'s `min_interval`?
            if interval.union(min_interval) == interval:
                return distance

        return None

    @cached_property
    def Toffsets(self):
        return [LabeledVector.transpose(*i) for i in zip(*[i.offsets for i in self])]

    @cached_property
    def diameter(self):
        """
        The size of the iteration space required to evaluate all aliasing expressions
        in this Group, along each Dimension.
        """
        ret = defaultdict(int)
        for i in self.Toffsets:
            for d, v in i:
                try:
                    distance = int(max(v) - min(v))
                except TypeError:
                    # An entry in `v` has symbolic components, e.g. `x_m + 2`
                    if len(set(v)) == 1:
                        continue
                    else:
                        raise ValueError
                ret[d] = max(ret[d], distance)

        # Drop ShiftedDimensions
        for d, v in list(ret.items()):
            if d.is_Shifted:
                if v != ret.get(d.parent, v):
                    raise ValueError
                ret.pop(d)

        return ret

    @property
    def pivot(self):
        """
        A deterministically chosen Candidate for this Group.
        """
        return self[0]

    @property
    def dimensions(self):
        return self.pivot.dimensions

    @property
    def dimensions_translated(self):
        return frozenset(d for d, v in self.diameter.items() if v > 0)

    @cached_property
    def _pivot_legal_rotations(self):
        """
        All legal rotations along each Dimension for the Group pivot.
        """
        ret = {}
        for d, (maxd, mini) in self._pivot_legal_shifts.items():
            # Rotation size = mini (min-increment) - maxd (max-decrement)
            v = mini - maxd

            # Build the table of all possible rotations
            m = make_rotations_table(d, v)

            distances = []
            for rotation in m:
                # Distance of the rotation `i` from `c`
                distance = maxd - rotation.lower
                assert distance == mini - rotation.upper
                distances.append(distance)

            ret[d] = list(zip(m, distances))

        return ret

    @cached_property
    def _pivot_min_intervals(self):
        """
        The minimum Interval along each Dimension such that by evaluating the
        pivot, all Candidates are evaluated too.
        """
        c = self.pivot

        ret = defaultdict(lambda: [np.inf, -np.inf])
        for i in self:
            distance = [o.distance(v) for o, v in zip(i.offsets, c.offsets)]
            distance = [(d, set(v)) for d, v in LabeledVector.transpose(*distance)]

            for d, v in distance:
                value = v.pop()
                ret[d][0] = min(ret[d][0], value)
                ret[d][1] = max(ret[d][1], value)

        ret = {d: Interval(d, m, M) for d, (m, M) in ret.items()}

        return ret

    @cached_property
    def _pivot_legal_shifts(self):
        """
        The max decrement and min increment along each Dimension such that the
        Group pivot does not go OOB.
        """
        c = self.pivot

        ret = defaultdict(lambda: (-np.inf, np.inf))
        for i, ofs in zip(c.indexeds, c.offsets):
            f = i.function
            for l in ofs.labels:
                # `f`'s cumulative halo size along `l`
                hsize = sum(f._size_halo[l])

                # Any `ofs`'s shift due to non-[0,0] iteration space
                try:
                    lower, upper = c.shifts[l].offsets
                except AttributeError:
                    assert l.is_Shifted
                    lower, upper = (0, 0)

                try:
                    # Assume `ofs[d]` is a number (typical case)
                    maxd = min(0, max(ret[l][0], -ofs[l] - lower))
                    mini = max(0, min(ret[l][1], hsize - ofs[l] - upper))

                    ret[l] = (maxd, mini)
                except TypeError:
                    # E.g., `ofs[d] = x_m - x + 5`
                    ret[l] = (0, 0)

        return ret


class Aliases(OrderedDict):

    """
    A mapper between aliases and collections of aliased expressions.
    """

    def __init__(self):
        super(Aliases, self).__init__()
        self.index_mapper = {}

    def add(self, alias, intervals, aliaseds, distances):
        assert len(aliaseds) == len(distances)

        self[alias] = (intervals, aliaseds, distances)

        # Update the index_mapper
        for i in intervals:
            d = i.dim
            if d in self.index_mapper:
                continue
            elif d.is_Shifted:
                self.index_mapper[d.parent] = d
            elif d.is_Incr:
                # IncrDimensions must be substituted with ShiftedDimensions
                # to access the temporaries, otherwise we would go OOB
                # E.g., r[xs][ys][z] => `xs/ys` must start at 0, not at `x0_blk0`
                # as in the case of blocking
                self.index_mapper[d] = ShiftedDimension(d, "%ss" % d.name)

    def get(self, key):
        ret = super(Aliases, self).get(key)
        if ret is not None:
            assert len(ret) == 3
            return ret[1]
        for _, aliaseds, _ in self.values():
            if key in aliaseds:
                return aliaseds
        return []

    def iter(self, ispace):
        """
        The aliases can be be scheduled in any order, but we privilege the one
        that minimizes storage while maximizing fusion.
        """
        items = []
        for alias, (intervals, aliaseds, distances) in self.items():
            mapper = {i.dim: i for i in intervals}
            mapper.update({i.dim.parent: i for i in intervals
                           if i.dim.is_NonlinearDerived})

            # Becomes True as soon as a Dimension in `ispace` is found to
            # be independent of `intervals`
            flag = False
            writeto = []
            for i in ispace.intervals:
                try:
                    interval = mapper[i.dim]
                    if not flag and interval == interval.zero():
                        # Optimize away unnecessary temporary Dimensions
                        continue

                    # Adjust the Interval's stamp
                    # E.g., `i=x[0,0]<1>` and `interval=x[-4,4]<0>`. We need to
                    # use `<1>` which is the actual stamp used in the Cluster
                    # from which the aliasing expressions were extracted
                    assert i.stamp >= interval.stamp
                    interval = interval.lift(i.stamp)

                    writeto.append(interval)
                    flag = True

                except KeyError:
                    if any(i.dim in d._defines for d in mapper):
                        # E.g., `i.dim` is `x0_blk0` in `x0_blk0[0,0]<0>`
                        pass
                    else:
                        # E.g., `t[0,0]<0>` in the case of t-invariant aliases
                        flag = True

            if writeto:
                writeto = IntervalGroup(writeto, relations=ispace.relations)
            else:
                # E.g., an `alias` having 0-distance along all Dimensions
                writeto = IntervalGroup(intervals, relations=ispace.relations)

            items.append((alias, writeto, aliaseds, distances))

        queue = list(items)
        while queue:
            # Shortest write-to region first
            item = min(queue, key=lambda i: len(i[1]))
            queue.remove(item)
            yield item


def make_rotations_table(d, v):
    """
    All possible rotations of `range(v+1)`.
    """
    m = np.array([[j-i if j > i else 0 for j in range(v+1)] for i in range(v+1)])
    m = (m - m.T)[::-1, :]

    # Shift the table so that the middle rotation is at the top
    m = np.roll(m, int(-np.floor(v/2)), axis=0)

    # Turn into a more compact representation as a list of Intervals
    m = [Interval(d, min(i), max(i)) for i in m]

    return m
