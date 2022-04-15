"""
Passes to gather and form implicit equations from MultiSubDomains.
"""

from devito.ir import Interval, IntervalGroup, IterationSpace, Queue
from devito.tools import timed_pass
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
        if not prefix:
            return clusters

        d = prefix[-1].dim
        try:
            pd = prefix[-2].dim
        except IndexError:
            pd = None

        # The bulk of the pass is done by the first MultiSubDimension
        if not isinstance(d, MultiSubDimension) or isinstance(pd, MultiSubDimension):
            return clusters

        # The implicit objects induced by the MultiSubDomain
        exprs = d.msd._implicit_exprs
        idims = d.msd.implicit_dimensions

        processed = []
        for c in clusters:
            # We ultimately need to inject the implicit Dimensions, if any, in
            # between the MultiSubDimensions and the outer Dimensions. We then
            # decouple the IterationSpace into two parts -- before and after the
            # first MultiSubDimension
            ispace0 = c.ispace.project(lambda i: not isinstance(i, MultiSubDimension))
            ispace1 = c.ispace.project(lambda i: isinstance(i, MultiSubDimension))

            # The local IterationSpace of the implicit Dimensions, if any
            intervals = [Interval(i, 0, 0) for i in idims]
            relations = (ispace0.itdimensions + idims,
                         idims + ispace1.itdimensions)
            ispaceN = IterationSpace(IntervalGroup(intervals, relations=relations))

            ispace = IterationSpace.union(ispace0, ispaceN)
            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

            ispace = IterationSpace.union(ispace0, ispaceN, ispace1)
            processed.append(c.rebuild(ispace=ispace))

        return processed
