"""
Passes to gather and form implicit equations from DSL abstractions.
"""

from devito.ir import (Cluster, Interval, IntervalGroup, IterationSpace, Queue,
                       SEQUENTIAL)
from devito.symbolics import retrieve_dimensions
from devito.tools import filter_sorted, timed_pass
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
    clusters = LowerMultiSubDimensions().process(clusters)

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

            # The implicit expressions introduced by the MultiSubDomain
            exprs = d.msd._implicit_exprs

            # The implicit Dimensions and iterators induced by the MultiSubDomain
            # NOTE: `filter_sorted` is for deterministic code generation, should
            # there ever be a crazy MultiSubDomain with multiple implicit Dimensions
            dims = filter_sorted(retrieve_dimensions(exprs, deep=True))
            idims = tuple(i for i in dims if not i.is_SubIterator)
            intervals = [Interval(i, 0, 0) for i in idims]
            sub_iterators = {i.root: i for i in dims if i.is_SubIterator}

            # The local IterationSpace of the implicit Dimensions, if any
            relations = (ispace0.itdimensions + idims,
                         idims + ispace1.itdimensions)
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
                    retained = c.ispace[:n-1].dimensions
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


# Utils


def msdim(d):
    try:
        for i in d._defines:
            if isinstance(i, MultiSubDimension):
                return i
    except AttributeError:
        pass
    return None
