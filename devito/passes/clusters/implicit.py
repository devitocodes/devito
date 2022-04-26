"""
Passes to gather and form implicit equations from DSL abstractions.
"""

from functools import singledispatch
from math import floor

import numpy as np

from devito.ir import (Cluster, Interval, IntervalGroup, IterationSpace, Queue,
                       SEQUENTIAL)
from devito.tools import as_tuple, timed_pass
from devito.types import Dimension, Eq, Function
from devito.types.grid import MultiSubDimension, SubDomainSet

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
    clusters = LowerMultiSubDimensions(sregistry).process(clusters)

    return clusters


class LowerMultiSubDimensions(Queue):

    """
    Bind the free thickness symbols defined by MultiSubDimensions to
    suitable values.

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

    def __init__(self, sregistry):
        super().__init__()

        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        try:
            dd = prefix[-1].dim
        except IndexError:
            dd = None

        # The non-MultiSubDimension closest to a MultiSubDimension triggers
        # the pass. For example, `t` in an `t, xi_n, yi_n` iteration space
        if msdim(dd):
            return clusters

        idx = len(prefix)

        seen = set()
        tip = None

        mapper = {}
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

            n = c.ispace.index(dd)
            ispace0 = c.ispace[:n]
            ispace1 = c.ispace[n:]

            # The "implicit expressions" created for the MultiSubDomain
            try:
                v = mapper[d.msd]
            except KeyError:
                v = mapper[d.msd] = make_implicit_exprs(d.msd, ispace0, self.sregistry)
            exprs, dims, sub_iterators = v

            # The IterationSpace induced by the MultiSubDomain
            intervals = [Interval(i, 0, 0) for i in dims]
            relations = (ispace0.itdimensions + dims, dims + ispace1.itdimensions)
            ispaceN = IterationSpace(
                IntervalGroup(intervals, relations=relations),
                sub_iterators
            )

            ispace = IterationSpace.union(ispace0, ispaceN)
            properties = {i.dim: {SEQUENTIAL} for i in ispace}
            if len(ispaceN) == 0:
                # Special case: we can factorize the thickness assignments
                # once and for all at the top of the current IterationInterval
                if ispaceN not in seen:
                    # Retain the guards and the syncs along the outer Dimensions
                    retained = {None} | set(c.ispace[:n-1].dimensions)
                    guards = {d: v for d, v in c.guards.items() if d in retained}
                    syncs = {d: v for d, v in c.syncs.items() if d in retained}

                    processed.insert(
                        0, Cluster(exprs, ispace, guards, properties, syncs)
                    )
                    seen.add(ispaceN)
            else:
                nxt = self._make_tip(c, ispaceN)
                if tip is None or tip != nxt:
                    processed.append(
                        c.rebuild(exprs=exprs, ispace=ispace, properties=properties)
                    )
                    tip = nxt

            ispace = IterationSpace.union(c.ispace, ispaceN)
            processed.append(c.rebuild(ispace=ispace))

        return processed

    def _make_tip(self, c, ispaceN):
        return (c.guards, c.syncs, ispaceN)


def msdim(d):
    try:
        for i in d._defines:
            if isinstance(i, MultiSubDimension):
                return i
    except AttributeError:
        pass
    return None


@singledispatch
def make_implicit_exprs(msd, ispace, sregistry):
    # Retval: (exprs, iteration dimensions, subiterators)
    return (), (), {}


@make_implicit_exprs.register(SubDomainSet)
def _(msd, ispace, sregistry):
    n_domains = msd.n_domains
    i_dim = Dimension(name=sregistry.make_name(prefix='n'))

    # Organise the data contained in 'bounds' into a form such that the
    # associated implicit equations can easily be created.
    ret = []
    for j in range(len(msd._local_bounds)):
        index = floor(j/2)
        d = msd.dimensions[index]
        if j % 2 == 0:
            fname = "%s_%s" % (msd.name, d.min_name)
        else:
            fname = "%s_%s" % (msd.name, d.max_name)
        f = Function(name=fname, shape=(n_domains,), dimensions=(i_dim,),
                     grid=msd._grid, dtype=np.int32)

        # Check if shorthand notation has been provided:
        if isinstance(msd._local_bounds[j], int):
            bounds = np.full((n_domains,), msd._local_bounds[j], dtype=np.int32)
            f.data[:] = bounds
        else:
            f.data[:] = msd._local_bounds[j]

        ret.append(Eq(d.thickness[j % 2][0], f[i_dim]))

    return as_tuple(ret), (i_dim,), {}
