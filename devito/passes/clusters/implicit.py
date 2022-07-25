"""
Passes to gather and form implicit equations from DSL abstractions.
"""

from functools import singledispatch
from math import floor

from devito.ir import (Cluster, Interval, IntervalGroup, IterationSpace, Queue,
                       FetchUpdate, PrefetchUpdate, SEQUENTIAL)
from devito.tools import as_tuple, timed_pass
from devito.types import Eq
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

    def _hook_syncs(self, cluster, level):
        """
        The *fetchUpdate SyncOps may require their own suitably adjusted
        thickness assigments. This method pulls such SyncOps.
        """
        syncs = []
        for i in cluster.ispace[:level]:
            for s in cluster.syncs.get(i.dim, ()):
                if isinstance(s, (FetchUpdate, PrefetchUpdate)):
                    syncs.append(s)
        return tuple(syncs)

    def _make_key_hook(self, cluster, level):
        return (self._hook_syncs(cluster, level),)

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

        seen = set()
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

            n = c.ispace.index(dd)
            ispace0 = c.ispace[:n]
            ispace1 = c.ispace[n:]

            # The "implicit expressions" created for the MultiSubDomain
            exprs, dims, sub_iterators = make_implicit_exprs(d.msd, c)

            # The IterationSpace induced by the MultiSubDomain
            intervals = [Interval(i, 0, 0) for i in dims]
            relations = (ispace0.itdimensions + dims, dims + ispace1.itdimensions)
            ispaceN = IterationSpace(
                IntervalGroup(intervals, relations=relations), sub_iterators
            )

            ispace = IterationSpace.union(ispace0, ispaceN)
            properties = {i.dim: {SEQUENTIAL} for i in ispace}
            if len(ispaceN) == 0:
                # Special case: we can factorize the thickness assignments
                # once and for all at the top of the current IterationInterval,
                # and reuse them for one or more (potentially non-consecutive)
                # `clusters`
                if ispaceN not in seen:
                    # Retain the guards and the syncs along the outer Dimensions
                    retained = {None} | set(c.ispace[:n-1].dimensions)

                    # A fetch SyncOp along `dim` binds the thickness assignments
                    if self._hook_syncs(c, n):
                        retained.add(dim)

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
def make_implicit_exprs(msd, cluster):
    # Retval: (exprs, iteration dimensions, subiterators)
    return (), (), {}


@make_implicit_exprs.register(SubDomainSet)
def _(msd, *args):
    ret = []
    for j in range(len(msd._local_bounds)):
        index = floor(j/2)
        d = msd.dimensions[index]
        f = msd._functions[j]

        ret.append(Eq(d.thickness[j % 2][0], f.indexify()))

    return as_tuple(ret), (msd._implicit_dimension,), {}
