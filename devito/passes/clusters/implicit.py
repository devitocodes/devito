"""
Passes to gather and form implicit equations from DSL abstractions.
"""

from collections import defaultdict
from functools import singledispatch

from sympy import Le, Ge
import numpy as np

from devito.ir import SEQUENTIAL, Queue, Forward
from devito.symbolics import retrieve_dimensions
from devito.tools import Bunch, frozendict, timed_pass
from devito.types import Eq, Symbol
from devito.types.dimension import BlockDimension
from devito.types.grid import MultiSubDimension

__all__ = ['generate_implicit']


@timed_pass()
def generate_implicit(clusters, sregistry):
    """
    Create and add implicit expressions from high-level abstractions.

    Implicit expressions are those not explicitly defined by the user
    but instead are requisites of some specified functionality.

    Currently, implicit expressions stem from the following:

        * MultiSubDomains attached to input equations.
    """
    clusters = LowerExplicitMSD(sregistry).process(clusters)
    clusters = LowerImplicitMSD(sregistry).process(clusters)

    return clusters


class LowerMSD(Queue):

    def __init__(self, sregistry):
        super().__init__()
        self.sregistry = sregistry


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
                continue

            # Get all MultiSubDimensions in the cluster and get the dynamic thickness
            # mapper for the associated MultiSubDomain
            mapper, dims = lower_msd({msdim(i.dim) for i in c.ispace[idx:]} - {None}, c)

            if not dims:
                # An Implicit MSD
                processed.append(c)
                continue

            exprs, thickness = make_implicit_exprs(mapper, self.sregistry)

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
            processed.append(inject_thickness(c, ispace, thickness))

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
        mapper = {}
        processed = []
        for bunch in found.values():
            exprs, thickness = make_implicit_exprs(bunch.mapper, self.sregistry)

            mapper.update({c: thickness for c in bunch.clusters})

            # Only retain outer guards (e.g., along None) if any
            key = lambda i: i is None or i in prefix.prefix([pd])
            guards = c.guards.filter(key)
            syncs = {d: v for d, v in c.syncs.items() if key(d)}

            processed.append(
                c.rebuild(exprs=exprs, ispace=prefix, guards=guards, syncs=syncs)
            )

        # Add in the dynamic thickness
        for c in clusters:
            try:
                processed.append(inject_thickness(c, c.ispace, mapper[c]))
            except KeyError:
                processed.append(c)

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
    mapper = {(dim.root, i): dim.functions[i_dim, mM]
              for i, mM in enumerate(dim.bounds_indices)}
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


def make_implicit_exprs(mapper, sregistry):
    exprs = []
    thickness = defaultdict(lambda: [None, None])
    for (d, side), v in mapper.items():
        tkn = 'l' if side == 0 else 'r'
        name = sregistry.make_name('%s_%stkn' % (d.name, tkn))
        s = Symbol(name=name, dtype=np.int32, is_const=True, nonnegative=True)

        exprs.append(Eq(s, v))
        thickness[d][side] = s

    return exprs, frozendict(thickness)


def inject_thickness(c, ispace, thickness):
    for i in ispace.itdims:
        if i.is_Block and i._depth > 1:
            # The thickness should be injected once only!
            continue
        try:
            v0, v1 = thickness[i.root]
            ispace = ispace.translate(i, v0, -v1)
        except KeyError:
            continue

    # TODO: this is outrageously hacky, but it will be purged by #2405
    guards = c.guards
    for i in ispace.itdims:
        subs = {}
        try:
            for g in guards[i].find(Le):
                d, e = g.args
                if d.root.symbolic_max in e.free_symbols:
                    _, v1 = thickness[d.root]
                    subs[e] = e - v1
            for g in guards[i].find(Ge):
                d, e = g.args
                if d.root.symbolic_min in e.free_symbols:
                    v0, _ = thickness[d.root]
                    subs[e] = e + v0
        except (AttributeError, KeyError):
            continue

        if subs:
            guards = guards.impose(i, guards[i].subs(subs))

    return c.rebuild(ispace=ispace, guards=guards)


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
