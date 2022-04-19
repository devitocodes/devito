"""
Passes to gather and form implicit equations from MultiSubDomains.
"""

from devito.ir import Interval, IntervalGroup, IterationSpace, Queue
from devito.symbolics import retrieve_dimensions
from devito.tools import filter_ordered, timed_pass
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

        tip = None
        processed = []
        for c in clusters:
            try:
                idx = len(prefix)
                dd = c.ispace[idx].dim
                d = msdim(dd)
            except IndexError:
                d = None
            if d is None:
                processed.append(c)
                continue

            idx = c.ispace.index(dd)
            ispace0 = c.ispace[:idx]
            ispace1 = c.ispace[idx:]

            # The implicit expressions introduced by the MultiSubDomain
            exprs = d.msd._implicit_exprs

            # The implicit Dimensions and iterators induced by the MultiSubDomain
            dims = filter_ordered(retrieve_dimensions(exprs, deep=True))
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

            if tip is None or tip != ispaceN:
                ispace = IterationSpace.union(ispace0, ispaceN)
                processed.append(c.rebuild(exprs=exprs, ispace=ispace))
                tip = ispaceN

            ispace = IterationSpace.union(c.ispace, ispaceN)
            processed.append(c.rebuild(ispace=ispace))

        return processed


# Utils


def msdim(d):
    try:
        for i in d._defines:
            if isinstance(i, MultiSubDimension):
                return i
    except AttributeError:
        pass
    return None
