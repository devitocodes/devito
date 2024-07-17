"""
Passes to gather and form implicit equations from DSL abstractions.
"""

from collections import defaultdict
from functools import singledispatch

from devito.ir import SEQUENTIAL, Queue, Forward
from devito.symbolics import retrieve_dimensions
from devito.tools import Bunch, frozendict, timed_pass
from devito.types import Eq
from devito.types.dimension import BlockDimension
from devito.types.grid import MultiSubDimension

__all__ = ['generate_implicit']


@timed_pass()
def generate_implicit(clusters):
    """
    Create and add implicit expressions from high-level abstractions.

    Implicit expressions are those not explicitly defined by the user
    but instead are requisites of some specified functionality.

    Currently, implicit expressions stem from the following:

        * MultiSubDomains attached to input equations.
    """
    clusters = LowerExplicitMSD().process(clusters)
    clusters = LowerImplicitMSD().process(clusters)

    return clusters


class LowerMSD(Queue):
    pass


class LowerExplicitMSD(LowerMSD):

    """
    An Explicit MultiSubDomain (MSD) encodes the thickness of N (N > 0)
    user-defined SubDomains.

    This pass augments the IterationSpace to iterate over the N SubDomains and
    bind the free thickness symbols to their corresponding values.

    Examples
    --------
    Given:

        Cluster([Eq(f[t1, xi_n, yi_n], f[t0, xi_n, yi_n] + 1)])

    where `xi_n` and `yi_n` are MultiSubDimensions, generate:

        Cluster([Eq(xi_n_ltkn, xi_n_m[n])
                 Eq(xi_n_rtkn, xi_n_M[n])
                 Eq(yi_n_ltkn, yi_n_m[n])
                 Eq(yi_n_rtkn, yi_n_M[n])])
        Cluster([Eq(f[t1, xi_n, yi_n], f[t0, xi_n, yi_n] + 1)])
    """

    _q_guards_in_key = True
    _q_syncs_in_key = True

    def callback(self, clusters, prefix):
        try:
            dim = prefix[-1].dim
        except IndexError:
            dim = None

        # The non-MultiSubDimension closest to a MultiSubDimension triggers
        # the pass. For example, `t` in an `t, xi_n, yi_n` iteration space
        if msdim(dim):
            return clusters

        idx = len(prefix)

        tip = None
        processed = []
        for c in clusters:
            try:
                dd = c.ispace[idx].dim
                d = msdim(dd)
            except IndexError:
                d = None
            if d is None:
                processed.append(c)
                # If no MultiSubDomain present in this cluster, then tip should be reset
                tip = None
                continue

            # Get all MultiSubDimensions in the cluster and get the dynamic thickness
            # mapper for the associated MultiSubDomain
            mapper, dims = lower_msd({msdim(i.dim) for i in c.ispace[idx:]} - {None}, c)

            if not dims:
                # An Implicit MSD
                processed.append(c)
                continue

            exprs = make_implicit_exprs(mapper)

            ispace = c.ispace.insert(dim, dims)

            # The Cluster computing the thicknesses
            ispaceN = ispace.prefix(dims)

            if tip is None or tip != ispaceN:
                properties = {i.dim: {SEQUENTIAL} for i in ispace}
                processed.append(
                    c.rebuild(exprs=exprs, ispace=ispaceN, properties=properties)
                )
                tip = ispaceN

            # The Cluster performing the actual computation, enriched with
            # the thicknesses
            processed.append(c.rebuild(ispace=ispace))

        return processed


class LowerImplicitMSD(LowerMSD):

    """
    An Implicit MultiSubDomain (MSD) encodes the thicknesses of N (N > 0)
    indirectly-defined SubDomains, that is SubDomains whose thicknesses are
    evaluated (e.g., computed on-the-fly, fetched from a Function) along a
    certain problem Dimension.

    Examples
    --------
    Given:

        Cluster([Eq(f[t1, xi, yi], f[t0, xi, yi] + 1)])

    where `xi_n` and `yi_n` are MultiSubDimensions, generate:

        Cluster([Eq(xi_ltkn, xi_n_m[time])
                 Eq(xi_rtkn, xi_n_M[time])
                 Eq(yi_ltkn, yi_n_m[time])
                 Eq(yi_rtkn, yi_n_M[time])])
        Cluster([Eq(f[t1, xi, yi], f[t0, xi, yi] + 1)])
    """

    def callback(self, clusters, prefix):
        try:
            dim = prefix[-1].dim
        except IndexError:
            return clusters

        try:
            pd = prefix[-2].dim
        except IndexError:
            pd = None

        # There could be several MultiSubDomains around, spread over different
        # Clusters. At the same time, the same MultiSubDomain might be required
        # by multiple Clusters, and each Cluster may require accessing the
        # MultiSubDomain at different iteration points along `dim`
        found = defaultdict(lambda: Bunch(clusters=[], mapper={}))
        for c in clusters:
            ispace = c.ispace.project(msdim)
            try:
                d = msdim(ispace.outermost.dim)
            except IndexError:
                continue

            # Get the dynamic thickness mapper for the given MultiSubDomain
            mapper, dims = lower_msd(ispace.itdims, c)
            if dims:
                # An Explicit MSD
                continue

            # Make sure the "implicit expressions" are scheduled in
            # the innermost loop such that the thicknesses can be computed
            edims = set(retrieve_dimensions(mapper.values(), deep=True))
            if dim not in edims or not edims.issubset(prefix.dimensions):
                continue

            found[d.functions].clusters.append(c)
            found[d.functions].mapper = reduce(found[d.functions].mapper,
                                               mapper, edims, prefix)

        # Turn the reduced mapper into a list of equations
        processed = []
        for bunch in found.values():
            exprs = make_implicit_exprs(bunch.mapper)

            # Only retain outer guards (e.g., along None) if any
            key = lambda i: i is None or i in prefix.prefix([pd])
            guards = c.guards.filter(key)
            syncs = {d: v for d, v in c.syncs.items() if key(d)}

            processed.append(
                c.rebuild(exprs=exprs, ispace=prefix, guards=guards, syncs=syncs)
            )

        processed.extend(clusters)

        return processed


def msdim(d):
    try:
        for i in d._defines:
            if i.is_MultiSub:
                return i
    except AttributeError:
        pass
    return None


@singledispatch
def _lower_msd(dim, cluster):
    # Retval: (dynamic thickness mapper, iteration dimension)
    return {}, None


@_lower_msd.register(MultiSubDimension)
def _(dim, cluster):
    i_dim = dim.implicit_dimension
    mapper = {tkn: dim.functions[i_dim, mM]
              for tkn, mM in zip(dim.tkns, dim.bounds_indices)}
    return mapper, i_dim


@_lower_msd.register(BlockDimension)
def _(dim, cluster):
    # Pull out the parent MultiSubDimension
    msd = [d for d in dim._defines if d.is_MultiSub]
    assert len(msd) == 1  # Sanity check. MultiSubDimensions shouldn't be nested.
    msd = msd.pop()
    return _lower_msd(msd, cluster)


def lower_msd(msdims, cluster):
    mapper = {}
    dims = set()
    for d in msdims:
        dmapper, ddim = _lower_msd(d, cluster)
        mapper.update(dmapper)
        dims.add(ddim)
    return frozendict(mapper), tuple(dims - {None})


def make_implicit_exprs(mapper):
    return [Eq(k, v) for k, v in mapper.items()]


def reduce(m0, m1, edims, prefix):
    if len(edims) != 1:
        raise NotImplementedError
    d, = edims

    if prefix[d].direction is Forward:
        func = max
    else:
        func = min

    key = lambda i: i.indices[d]

    mapper = {}
    for k, e in m1.items():
        candidates = {e, m0.get(k, e)}
        mapper[k] = func(candidates, key=key)

    return frozendict(mapper)
